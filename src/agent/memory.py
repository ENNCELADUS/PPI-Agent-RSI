"""The agent's two memory layers: case memory (tested pairs, written by the system) and lessons (written by the
reviewer session). Design: docs/superpowers/specs/2026-11-02-two-layer-memory-agent-design.md.

A case id is the canonical pair "A-B". Lessons are hypotheses: the reviewer adds, revises or retires them; a
retired lesson stays in the record but is not shown. The system checks every operation and rejects one that
cites an unknown case, names an unknown or retired lesson, or would exceed MAX_ACTIVE active lessons.
"""

import copy
import csv

from src.agent.tools import ANNOTATION

MAX_ACTIVE = 30
CASE_FIELDS = ("round", "a", "b", "result", "b0_p", "ranking_score", "uncertainty", "rationale", "evidence_ids")


def case_id(a, b):
    return "-".join(sorted((a, b)))


def read_homologs(path):
    """{query: [(target, pident), ...]} best first, from a case_homologs.tsv."""
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        out.setdefault(r["query"], []).append((r["target"], r["pident"]))
    return out


class Cases:
    """Case records with the lookups of the `cases` tool. proteins: Sandbox.proteins of the discovery view (the
    case proteins' names and annotations); homologs: read_homologs of the current view's case_homologs.tsv."""

    def __init__(self, records, proteins, homologs):
        self.records, self.proteins, self.homologs = records, proteins, homologs
        self.ids = {case_id(c["a"], c["b"]) for c in records}

    def text(self, x):
        r = self.proteins.get(x, {})
        return " ".join(r.get(k, "") for k in ("gene", "protein_name", *ANNOTATION)).lower()

    def find(self, protein=None, similar_to=None, query=None, pair=None, result=None, round=None):
        """Matching records in canonical order, each with `via`: why it matched similar_to (else empty)."""
        near = {}
        if similar_to is not None:
            near = {similar_to: "self"} | {t: f"{t} {pid}%" for t, pid in self.homologs.get(similar_to, [])}
        out = []
        for c in sorted(self.records, key=lambda c: (c["a"], c["b"])):
            ends = (c["a"], c["b"])
            if protein is not None and protein not in ends:
                continue
            if pair is not None and case_id(*ends) != pair:
                continue
            if result is not None and c["result"] != result:
                continue
            if round is not None and c["round"] != round:
                continue
            if query is not None and not any(query.lower() in self.text(x) for x in ends):
                continue
            via = [near[x] for x in ends if x in near]
            if similar_to is not None and not via:
                continue
            out.append(c | {"via": "; ".join(via)})
        return out


class Lessons:
    """All lessons (active and retired) as dicts: id, text, scope, supporting, contrary, created, history,
    retired (None or {round, reason})."""

    def __init__(self, lessons=()):
        self.all = [copy.deepcopy(x) for x in lessons]

    def active(self):
        return [x for x in self.all if x["retired"] is None]

    def apply(self, ops, case_ids, t):
        """Apply the reviewer's operations in order for round t; returns one report line per operation."""
        report = []
        for i, op in enumerate(ops, 1):
            try:
                report.append(f"{i}: {self._apply(op, case_ids, t)}")
            except ValueError as e:
                report.append(f"{i}: rejected: {e}")
        return report

    def _cases(self, op, key, case_ids):
        ids = []
        for x in op.get(key) or []:
            parts = str(x).removeprefix("case:").split("-")  # PRING accessions have no "-"
            ids.append(case_id(*parts) if len(parts) == 2 else str(x))
        unknown = [x for x in ids if x not in case_ids]
        if unknown:
            raise ValueError(f"unknown case ids in {key}: {', '.join(unknown[:5])}")
        return sorted(set(ids))

    def _lesson(self, op):
        hit = [x for x in self.active() if x["id"] == op.get("id")]
        if not hit:
            raise ValueError(f"no active lesson {op.get('id')}")
        return hit[0]

    def _apply(self, op, case_ids, t):
        kind = op.get("op")
        if kind == "add":
            if not (op.get("text") and op.get("scope")):
                raise ValueError("add needs text and scope")
            if len(self.active()) >= MAX_ACTIVE:
                raise ValueError(f"{MAX_ACTIVE} lessons are active: retire or revise one first")
            lesson = {
                "id": f"L{len(self.all) + 1}",
                "text": op["text"],
                "scope": op["scope"],
                "supporting": self._cases(op, "supporting", case_ids),
                "contrary": self._cases(op, "contrary", case_ids),
                "created": t,
                "history": [],
                "retired": None,
            }
            self.all.append(lesson)
            return f"added {lesson['id']}"
        if kind == "revise":
            lesson = self._lesson(op)
            new = {k: op[k] for k in ("text", "scope") if op.get(k)}
            new |= {k: self._cases(op, k, case_ids) for k in ("supporting", "contrary") if op.get(k) is not None}
            if not new:
                raise ValueError("revise changes nothing")
            lesson["history"].append({"round": t, "op": "revise", "before": {k: lesson[k] for k in new}})
            lesson |= new
            return f"revised {lesson['id']}"
        if kind == "retire":
            lesson = self._lesson(op)
            if not op.get("reason"):
                raise ValueError("retire needs a reason")
            lesson["retired"] = {"round": t, "reason": op["reason"]}
            return f"retired {lesson['id']}"
        raise ValueError(f"unknown op {kind}")

    def render(self):
        """The active lessons as prompt text."""
        if not self.active():
            return "No lessons yet."
        out = []
        for x in self.active():
            sup, con = x["supporting"], x["contrary"]
            out.append(
                f"- {x['id']} (since round {x['created']}): {x['text']}\n  scope: {x['scope']}\n"
                f"  supporting cases ({len(sup)}): {', '.join(sup[:10])}{' ...' if len(sup) > 10 else ''}\n"
                f"  contrary cases ({len(con)}): {', '.join(con[:10])}{' ...' if len(con) > 10 else ''}"
            )
        return "\n".join(out)
