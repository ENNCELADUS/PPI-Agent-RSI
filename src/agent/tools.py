"""The agent's tools for one round, served to pi over localhost HTTP (the TS extension only forwards calls).

Reads the sandbox directory (data/PRING/sandbox/) and the round state the evaluator
hands over: the pairs already tested and the agent-visible memory. It never reads labels or imports
the oracle. Every call is logged to tool_calls.jsonl; after CAP calls only submit_batch answers.
submit_batch accepts the first well-formed batch as final and writes it to submission.json.
"""

import csv
import json
import threading
import time
from bisect import bisect_left
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb

CAP = 80
K = 200
ANNOTATION = (
    "function",
    "subcellular_location",
    "tissue_specificity",
    "go_process",
    "go_component",
    "go_function",
    "keywords",
    "families",
    "pfam",
    "interpro",
)


def pair(a, b):
    return (a, b) if a < b else (b, a)


def tsv(header, rows):
    return "\t".join(header) + "\n" + "\n".join("\t".join(str(x) for x in r) for r in rows)


class ToolError(Exception):
    pass


@cache
class Sandbox:
    """The agent-visible files of one sandbox directory, loaded once per process and shared by rounds."""

    def __init__(self, root):
        root = Path(root)
        self.proteins = {r["accession"]: r for r in csv.DictReader(open(root / "proteins.tsv"), delimiter="\t")}
        for r in csv.DictReader(open(root / "annotations.tsv"), delimiter="\t"):
            self.proteins[r["accession"]] |= {k: r[k] for k in ANNOTATION}
        self.candidates = {x for x, r in self.proteins.items() if r["candidate"] == "1"}
        self.kb = {}
        for r in csv.DictReader(open(root / "kb.csv")):
            a, b = r["uniprotID_A"], r["uniprotID_B"]
            for x, y in ((a, b), (b, a)):
                self.kb.setdefault(x, []).append((y, r["isInteraction"], r["evidence"]))
        self.homologs = {}
        for r in csv.DictReader(open(root / "homologs.tsv"), delimiter="\t"):
            self.homologs.setdefault(r["query"], []).append((r["target"], r["pident"], r["evalue"], r["bits"]))
        # r: global rank by predictor score, ties broken by canonical pair id
        self.db = duckdb.connect()
        self.db.execute(
            "CREATE TABLE scores AS SELECT a, b, p, row_number() OVER (ORDER BY p DESC, a, b) AS r FROM read_csv("
            "?, delim='\t', header=true, columns={'a': 'VARCHAR', 'b': 'VARCHAR', 'p': 'DOUBLE'})",
            [str(root / "scores.tsv.gz")],
        )

    def gene(self, x):
        return self.proteins[x]["gene"]

    def query(self, sql, params=()):
        return self.db.cursor().execute(sql, params).fetchall()

    def ranks(self, pairs):
        """Global ranks of the given canonical pairs (unknown pairs are skipped)."""
        cur = self.db.cursor()
        cur.execute("CREATE TEMP TABLE q (a VARCHAR, b VARCHAR)")
        if pairs:
            cur.executemany("INSERT INTO q VALUES (?, ?)", sorted(pairs))
        return sorted(r for (r,) in cur.execute("SELECT r FROM scores JOIN q USING (a, b)").fetchall())


ID = {"type": "string"}
SPECS = {
    "top_candidates": (
        "Untested candidate pairs ranked by the predictor's calibrated probability p, best first. rank is the "
        "position among untested pairs; page with offset and limit.",
        {"offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
        ["offset", "limit"],
    ),
    "predict_ppi": (
        "The predictor's calibrated probability that two candidate proteins interact, the pair's rank among "
        "untested pairs, and whether it was already tested.",
        {"a": ID, "b": ID},
        ["a", "b"],
    ),
    "top_for_protein": (
        "The predictor's best untested partners of one candidate protein.",
        {"id": ID, "n": {"type": "integer", "minimum": 1, "maximum": 100}},
        ["id", "n"],
    ),
    "protein_info": (
        "What is known about up to 20 proteins apart from their interactions: names, function, location, tissue, "
        "GO terms, keywords, families, Pfam, InterPro; the sequence only if with_sequence is true.",
        {"ids": {"type": "array", "items": ID, "minItems": 1, "maxItems": 20}, "with_sequence": {"type": "boolean"}},
        ["ids"],
    ),
    "search_proteins": (
        "Case-insensitive text search over protein names and annotations, e.g. a GO term, Pfam id, keyword or "
        "location. scope: candidates (the proteins you can test), kb (knowledge-base proteins) or all.",
        {
            "text": {"type": "string", "minLength": 2},
            "scope": {"type": "string", "enum": ["candidates", "kb", "all"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        ["text", "scope", "limit"],
    ),
    "similar_proteins": (
        "Sequence homologs of a protein among knowledge-base proteins (mmseqs2 identity %, E-value, bit score), "
        "best first. Candidate proteins are at most ~40% identical to any knowledge-base protein.",
        {"id": ID},
        ["id"],
    ),
    "kb_partners": (
        "Labelled knowledge-base pairs of a knowledge-base protein: partner, label (1 interacts, 0 does not) and "
        "evidence source. Candidate proteins have none; reach them through similar_proteins.",
        {"id": ID},
        ["id"],
    ),
    "memory": (
        "Your experimental memory: every pair tested in earlier rounds, with its round and measured result. "
        "Filter by protein and/or to hits only.",
        {"protein": ID, "hits_only": {"type": "boolean"}},
        [],
    ),
    "submit_batch": (
        f"Send this round's {K} pairs to the lab. Final: call it once, with exactly {K} untested candidate pairs.",
        {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "a": ID,
                        "b": ID,
                        "ranking_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "your probability that the pair interacts",
                        },
                        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence_ids": {"type": "array", "items": ID},
                        "rationale": {"type": "string"},
                    },
                    "required": ["a", "b", "ranking_score", "uncertainty", "evidence_ids", "rationale"],
                },
            }
        },
        ["pairs"],
    ),
}


def specs():
    """JSON-schema tool definitions for the pi extension."""
    return [
        {
            "name": n,
            "description": d,
            "terminate": n == "submit_batch",
            "parameters": {"type": "object", "properties": p, "required": r},
        }
        for n, (d, p, r) in SPECS.items()
    ]


class RoundTools:
    """Tool calls for one round. tested: canonical pairs already submitted; memory: agent-visible facts
    as dicts with a, b, round, result ("hit", "not detected" or "withheld")."""

    def __init__(self, sandbox, round_dir, tested, memory, cap=CAP):
        self.s, self.dir, self.tested, self.facts, self.cap = sandbox, Path(round_dir), set(tested), memory, cap
        self.calls, self.submitted, self.lock = 0, None, threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tested_ranks = sandbox.ranks(self.tested)

    def call(self, name, args):
        """Run one tool call; returns its text, raises ToolError with a message for the model."""
        start, text, ok = time.time(), "", False
        try:
            with self.lock:
                self.calls += 1
                over = self.calls > self.cap
            if name not in SPECS:
                raise ToolError(f"unknown tool {name}")
            if over and name != "submit_batch":
                raise ToolError(f"tool budget of {self.cap} calls used up: call submit_batch now")
            text = getattr(self, name)(**args)
            ok = True
            return text
        except ToolError as e:
            text = f"error: {e}"
            raise
        except TypeError as e:  # bad argument names reach here
            text = f"error: {e}"
            raise ToolError(str(e)) from None
        finally:
            record = {
                "t": round(start, 3),
                "tool": name,
                "args": args,
                "ok": ok,
                "chars": len(text),
                "ms": round(1000 * (time.time() - start)),
            }
            with self.lock, open(self.dir / "tool_calls.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

    def _protein(self, x):
        if x not in self.s.proteins:
            raise ToolError(f"unknown protein {x}")
        return x

    def _candidate(self, x):
        if self._protein(x) not in self.s.candidates:
            raise ToolError(f"{x} is not a candidate protein")
        return x

    def _untested_rank(self, r):
        return r - bisect_left(self.tested_ranks, r)

    def _pairs(self, rows):
        return tsv(
            ["rank", "a", "gene_a", "b", "gene_b", "p"],
            [(self._untested_rank(r), a, self.s.gene(a), b, self.s.gene(b), f"{p:.4g}") for a, b, p, r in rows],
        )

    def top_candidates(self, offset, limit):
        # untested rank k sits at a global rank in [k, k + len(tested)]
        rows = self.s.query(
            "SELECT a, b, p, r FROM scores WHERE r > ? AND r <= ? ORDER BY r",
            [offset, offset + limit + len(self.tested)],
        )
        rows = [x for x in rows if (x[0], x[1]) not in self.tested and self._untested_rank(x[3]) > offset]
        return self._pairs(rows[:limit])

    def predict_ppi(self, a, b):
        a, b = pair(self._candidate(a), self._candidate(b))
        if a == b:
            raise ToolError("self pairs are not candidates")
        ((p, r),) = self.s.query("SELECT p, r FROM scores WHERE a = ? AND b = ?", [a, b])
        if (a, b) in self.tested:
            return f"{a}\t{b}\tp={p:.4g}\talready tested"
        return f"{a}\t{b}\tp={p:.4g}\trank={self._untested_rank(r)} of untested"

    def top_for_protein(self, id, n):
        x = self._candidate(id)
        rows = self.s.query(
            "SELECT a, b, p, r FROM scores WHERE a = ? OR b = ? ORDER BY r LIMIT ?", [x, x, n + len(self.tested)]
        )
        return self._pairs([r for r in rows if (r[0], r[1]) not in self.tested][:n])

    def protein_info(self, ids, with_sequence=False):
        out = []
        for x in ids:
            r = self.s.proteins[self._protein(x)]
            fields = {"id": x, "candidate": r["candidate"] == "1", "length": len(r["sequence"])}
            fields |= {"gene": r["gene"], "protein_name": r["protein_name"]}
            fields |= {k: r.get(k, "") for k in ANNOTATION if r.get(k)}
            if with_sequence:
                fields["sequence"] = r["sequence"]
            out.append(fields)
        return json.dumps(out, indent=0)

    def search_proteins(self, text, scope, limit):
        if scope not in ("candidates", "kb", "all"):
            raise ToolError("scope must be candidates, kb or all")
        needle, out = text.lower(), []
        names = ("gene", "protein_name", *ANNOTATION)
        for x, r in sorted(self.s.proteins.items()):
            if scope != "all" and (x in self.s.candidates) != (scope == "candidates"):
                continue
            hit = next((k for k in names if needle in r.get(k, "").lower()), None)
            if hit:
                value = r[hit]
                i = value.lower().index(needle)
                out.append((x, self.s.gene(x), int(x in self.s.candidates), hit, value[max(0, i - 60) : i + 80]))
        header = ["id", "gene", "candidate", "field", "match"]
        return f"{len(out)} matches\n" + tsv(header, out[:limit])

    def similar_proteins(self, id):
        rows = self.s.homologs.get(self._protein(id), [])
        if not rows:
            return f"no knowledge-base homolog found for {id}"
        return tsv(
            ["kb_protein", "gene", "identity_pct", "evalue", "bits"],
            [(t, self.s.gene(t), pid, e, bits) for t, pid, e, bits in rows],
        )

    def kb_partners(self, id):
        rows = self.s.kb.get(self._protein(id), [])
        if not rows:
            return f"{id} has no knowledge-base pairs"
        return tsv(["partner", "gene", "label", "evidence"], [(y, self.s.gene(y), lab, ev) for y, lab, ev in rows])

    def memory(self, protein=None, hits_only=False):
        rows = [m for m in self.facts if protein in (None, m["a"], m["b"])]
        if hits_only:
            rows = [m for m in rows if m["result"] == "hit"]
        if not rows:
            return "no matching tests yet"
        return f"{len(rows)} tests\n" + tsv(
            ["round", "a", "b", "result"], [(m["round"], m["a"], m["b"], m["result"]) for m in rows]
        )

    def submit_batch(self, pairs):
        with self.lock:
            if self.submitted is not None:
                raise ToolError("this round's batch was already submitted")
            self.submitted = pairs
        (self.dir / "submission.json").write_text(json.dumps(pairs, indent=0))
        return f"received {len(pairs)} pairs; results arrive next round"


def serve(tools):
    """Serve tools on an ephemeral localhost port in a daemon thread; returns (server, url)."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            name = self.path.rsplit("/", 1)[-1]
            args = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            try:
                body = {"ok": True, "text": tools.call(name, args)}
            except ToolError as e:
                body = {"ok": False, "text": str(e)}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/tool/"
