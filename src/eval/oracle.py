"""The offline lab: answers submitted pairs from a sealed labels file. Never imported by src/agent.

A labels file has uniprotID_A, uniprotID_B and optionally isInteraction (absent: every row is a
positive). PRING is closed-world: a listed positive is a hit, every other pair is not detected.
id_map (agent id -> accession, e.g. hidden/anon_map.tsv) translates anonymised submissions.
"""

import csv


def pair(a, b):
    return (a, b) if a < b else (b, a)


class Oracle:
    def __init__(self, labels, id_map=None):
        self.hits = {
            pair(r["uniprotID_A"], r["uniprotID_B"])
            for r in csv.DictReader(open(labels))
            if r.get("isInteraction", "1") == "1"
        }
        self.accession = (
            {r["id"]: r["accession"] for r in csv.DictReader(open(id_map), delimiter="\t")} if id_map else {}
        )

    def __call__(self, pairs):
        """{pair: 1 hit / 0 not detected} for valid agent-id pairs (an unmapped anonymised id raises)."""
        acc = self.accession.__getitem__ if self.accession else str
        return {p: int(pair(acc(p[0]), acc(p[1])) in self.hits) for p in pairs}
