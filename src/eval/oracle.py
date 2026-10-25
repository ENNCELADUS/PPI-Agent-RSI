"""The offline lab: answers submitted pairs from a sealed labels file. Never imported by src/agent.

A labels file has uniprotID_A, uniprotID_B and optionally isInteraction (absent: every row is a
positive). PRING is closed-world: a listed positive is a hit, every other pair is not detected.
"""

import csv


def pair(a, b):
    return (a, b) if a < b else (b, a)


class Oracle:
    def __init__(self, labels):
        self.hits = {
            pair(r["uniprotID_A"], r["uniprotID_B"])
            for r in csv.DictReader(open(labels))
            if r.get("isInteraction", "1") == "1"
        }

    def __call__(self, pairs):
        """{pair: 1 hit / 0 not detected} for valid canonical pairs."""
        return {p: int(p in self.hits) for p in pairs}
