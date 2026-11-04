"""The agent's tools for one session, served to pi over localhost HTTP (the TS extension only forwards calls).

Reads the sandbox directory (data/PRING/sandbox/) and the state the evaluator hands over: for a
discovery round (RoundTools) the pairs already tested and the agent-visible memory; for a
classification batch (PairTools) nothing. It never reads labels or imports the oracle. Every call is
logged to tool_calls.jsonl; after CAP calls only the submit tool answers, and it accepts the first
well-formed submission as final and writes it to submission.json.
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

from src.data.ids import pair

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
        if (root / "scores.tsv.gz").exists():  # absent where the agent has no predictor (sandbox_cls)
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
    "predict_pairs": (
        "The predictor's calibrated probability p for up to 200 pairs of candidate proteins in one call.",
        {
            "pairs": {
                "type": "array",
                "items": {"type": "object", "properties": {"a": ID, "b": ID}, "required": ["a", "b"]},
                "minItems": 1,
                "maxItems": K,
            }
        },
        ["pairs"],
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
    "cases": (
        "Your case memory: every pair tested in earlier rounds, with its round, measured result, the predictor's "
        "p before the test, your ranking_score and uncertainty, evidence and rationale. Filters combine: protein "
        "(cases with this protein), similar_to (cases with this protein or one of its sequence homologs among case "
        "proteins; via shows which), query (text in either protein's names or annotations), pair (case id A-B), "
        "result, round. Page with offset and limit.",
        {
            "protein": ID,
            "similar_to": ID,
            "query": {"type": "string", "minLength": 2},
            "pair": {"type": "string", "description": "case id, e.g. P12345-Q67890"},
            "result": {"type": "string", "enum": ["hit", "not detected"]},
            "round": {"type": "integer", "minimum": 1},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        [],
    ),
    "update_lessons": (
        "Change the lessons. Final: call it once with all operations, applied in order. add: text, scope, "
        "supporting, contrary (case ids). revise: id and the fields that change (supporting or contrary replace "
        "the old list). retire: id and reason.",
        {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["add", "revise", "retire"]},
                        "id": {"type": "string", "description": "lesson id, for revise and retire"},
                        "text": {"type": "string"},
                        "scope": {"type": "string", "description": "the pairs or proteins the lesson applies to"},
                        "supporting": {"type": "array", "items": ID},
                        "contrary": {"type": "array", "items": ID},
                        "reason": {"type": "string"},
                    },
                    "required": ["op"],
                },
            }
        },
        ["operations"],
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
    "submit_predictions": (
        "Send your prediction for the pairs of this batch. Final: call it once, with one row per listed pair.",
        {
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "a": ID,
                        "b": ID,
                        "p": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "your probability that the pair interacts",
                        },
                        "evidence_ids": {"type": "array", "items": ID},
                    },
                    "required": ["a", "b", "p", "evidence_ids"],
                },
            }
        },
        ["predictions"],
    ),
}
PROTEIN_TOOLS = ("protein_info", "search_proteins", "similar_proteins", "kb_partners")
PREDICTOR_TOOLS = ("top_candidates", "predict_ppi", "top_for_protein")
ROUND_TOOLS = (*PREDICTOR_TOOLS, *PROTEIN_TOOLS, "memory", "submit_batch")
PAIR_TOOLS = (*PROTEIN_TOOLS, "submit_predictions")
# the two-layer memory agent: selector round, reviewer, final test (the empty-memory arm drops cases)
CASE_ROUND_TOOLS = (*PREDICTOR_TOOLS, *PROTEIN_TOOLS, "cases", "submit_batch")
REVIEW_TOOLS = ("predict_ppi", *PROTEIN_TOOLS, "cases", "update_lessons")
TEST_TOOLS = ("predict_pairs", *PROTEIN_TOOLS, "cases", "submit_predictions")
FINAL = ("submit_batch", "submit_predictions", "update_lessons")


def specs(names=ROUND_TOOLS, descriptions=None):
    """JSON-schema tool definitions for the pi extension; descriptions overrides some tools' text."""
    return [
        {
            "name": n,
            "description": (descriptions or {}).get(n, d),
            "terminate": n in FINAL,
            "parameters": {"type": "object", "properties": p, "required": r},
        }
        for n, (d, p, r) in SPECS.items()
        if n in names
    ]


class RoundTools:
    """Tool calls for one round. tested: canonical pairs already submitted; memory: agent-visible facts
    as dicts with a, b, round, result ("hit", "not detected" or "withheld")."""

    names, final = ROUND_TOOLS, "submit_batch"
    descriptions, show_b0 = {}, True  # the `cases` table shows the predictor's p unless the agent has no predictor

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
            if name not in self.names:
                raise ToolError(f"unknown tool {name}")
            if over and name != self.final:
                raise ToolError(f"tool budget of {self.cap} calls used up: call {self.final} now")
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

    def cases(self, protein=None, similar_to=None, query=None, pair=None, result=None, round=None, offset=0, limit=50):
        for x in (protein, similar_to):  # in the final test, case proteins are not in the sandbox
            if x is not None and x not in self.s.proteins and x not in self.casebook.proteins:
                raise ToolError(f"unknown protein {x}")
        if pair is not None:
            ends = pair.removeprefix("case:").split("-")
            if len(ends) != 2:
                raise ToolError("pair must be a case id A-B")
            pair = "-".join(sorted(ends))
        rows = self.casebook.find(protein, similar_to, query, pair, result, round)
        if not rows:
            return "no matching cases"
        gene = lambda x: self.casebook.proteins.get(x, {}).get("gene", "")  # noqa: E731
        b0 = ["b0_p"] if self.show_b0 else []
        header = ["case", "round", "gene_a", "gene_b", "result", *b0, "ranking_score", "uncertainty", "via"]
        body = [
            (
                f"{c['a']}-{c['b']}",
                c["round"],
                gene(c["a"]),
                gene(c["b"]),
                c["result"],
                *([f"{c['b0_p']:.4g}"] if self.show_b0 else []),
                c["ranking_score"],
                c["uncertainty"],
                c["via"],
                " ".join(map(str, c["evidence_ids"] or [])),
                " ".join(str(c["rationale"] or "").split()),
            )
            for c in rows[offset : offset + limit]
        ]
        return f"{len(rows)} cases\n" + tsv([*header, "evidence_ids", "rationale"], body)

    def _submit(self, rows):
        with self.lock:
            if self.submitted is not None:
                raise ToolError("already submitted")
            self.submitted = rows
        (self.dir / "submission.json").write_text(json.dumps(rows, indent=0))

    def submit_batch(self, pairs):
        self._submit(pairs)
        return f"received {len(pairs)} pairs; results arrive next round"


class PairTools(RoundTools):
    """Tool calls for one classification batch: protein and KB tools only (no predictor, no memory)."""

    names, final = PAIR_TOOLS, "submit_predictions"

    def __init__(self, sandbox, agent_dir, cap=CAP):
        self.s, self.dir, self.cap = sandbox, Path(agent_dir), cap
        self.calls, self.submitted, self.lock = 0, None, threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)

    def submit_predictions(self, predictions):
        self._submit(predictions)
        return f"received {len(predictions)} predictions"


class CaseRoundTools(RoundTools):
    """One selector round of the two-layer memory agent: the round tools with `cases` instead of `memory`."""

    names = CASE_ROUND_TOOLS

    def __init__(self, sandbox, round_dir, tested, casebook, cap=CAP):
        super().__init__(sandbox, round_dir, tested, [], cap)
        self.casebook = casebook


class ReviewTools(CaseRoundTools):
    """The reviewer session after a round: evidence tools, cases, and one final update_lessons call."""

    names, final = REVIEW_TOOLS, "update_lessons"

    def update_lessons(self, operations):
        self._submit(operations)
        return f"received {len(operations)} operations"


class TestTools(PairTools):
    """One final-test batch: predictor score lookup (unless predictor=False), protein and KB tools, and `cases` if
    casebook is given. Without the predictor the `cases` table also hides the predictor's p of each case."""

    def __init__(self, sandbox, agent_dir, casebook=None, cap=CAP, predictor=True):
        super().__init__(sandbox, agent_dir, cap)
        self.casebook = casebook
        drop = ({"cases"} if casebook is None else set()) | (set() if predictor else {"predict_pairs"})
        self.names = tuple(n for n in TEST_TOOLS if n not in drop)
        if not predictor:
            self.show_b0 = False
            self.descriptions = {"cases": SPECS["cases"][0].replace("the predictor's p before the test, ", "")}
        self.scored = set()  # pairs whose p the agent received; the evaluator reports coverage per batch

    def predict_pairs(self, pairs):
        if len(pairs) > K:
            raise ToolError(f"at most {K} pairs per call")
        rows = []
        for r in pairs:
            a, b = pair(self._candidate(r.get("a")), self._candidate(r.get("b")))
            if a == b:
                raise ToolError(f"{a}-{b}: self pairs are not candidates")
            found = self.s.query("SELECT p FROM scores WHERE a = ? AND b = ?", [a, b])
            if not found:  # the 22T sandbox scores the test pairs only
                raise ToolError(f"{a}-{b}: no predictor score; only the listed test pairs have one")
            rows.append((a, b, f"{found[0][0]:.4g}"))
        with self.lock:
            self.scored.update((a, b) for a, b, _ in rows)
        return tsv(["a", "b", "p"], rows)


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
