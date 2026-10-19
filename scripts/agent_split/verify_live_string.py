"""Spot-check that sampled test pairs are absent from the live STRING API.

Usage: python verify_live_string.py [n_pairs] [seed] [label]   (label 1 = positives, 0 = negatives)
Prints every sampled pair that live STRING links at any score.
"""
import csv
import random
import sys
import urllib.parse
import urllib.request

from common import OUT

API = "https://string-db.org/api/tsv/"


def post(endpoint, ids):
    body = urllib.parse.urlencode({
        "identifiers": "\r".join(ids), "species": 9606, "required_score": 0,
        "caller_identity": "ppi-agent-rsi",
    }).encode()
    return urllib.request.urlopen(API + endpoint, body, timeout=300).read().decode().splitlines()


def main(n=100, seed=7, label=1):
    with open(OUT / "test.csv") as f:
        pairs = [(r["uniprotID_A"], r["uniprotID_B"]) for r in csv.DictReader(f) if r["isInteraction"] == str(label)]
    random.seed(seed)
    sample = random.sample(pairs, n)
    accs = sorted({x for pair in sample for x in pair})

    acc2string = {}
    for line in post("get_string_ids", accs)[1:]:
        index, string_id = line.split("\t")[:2]
        acc2string.setdefault(accs[int(index)], string_id)

    rows = [line.split("\t") for line in post("network", sorted(set(acc2string.values())))]
    a, b, score = rows[0].index("stringId_A"), rows[0].index("stringId_B"), rows[0].index("score")
    edges = {frozenset((r[a], r[b])): r[score] for r in rows[1:]}

    hits = [(x, y, edges[k]) for x, y in sample
            if (k := frozenset((acc2string.get(x), acc2string.get(y)))) in edges]
    print(f"proteins mapped by STRING: {len(acc2string)}/{len(accs)}")
    print(f"sampled test pairs (label {label}) linked in live STRING: {len(hits)}/{n}")
    for hit in hits:
        print(*hit)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
