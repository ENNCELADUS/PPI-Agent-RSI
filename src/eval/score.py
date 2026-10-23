"""Score a global top-N ranking against a labels file.

Run: uv run python -m src.eval.score <labels.csv> <ranking.csv>
labels.csv has uniprotID_A, uniprotID_B, isInteraction (a sandbox hidden/ file, or the private
test.csv); ranking.csv has uniprotID_A, uniprotID_B, best first, and may be shorter than the pool.
Pairs are unordered. Rows that are not in the pool or repeat an earlier pair are invalid and count
as misses, as do missing rows when the ranking is shorter than N. Enhancement is precision over
pool prevalence (1.0 = random).
"""
import csv
import json
import sys

NS = (50, 100, 250)


def pair(a, b):
    return (a, b) if a < b else (b, a)


def read_pairs(path):
    return [(pair(r["uniprotID_A"], r["uniprotID_B"]), r) for r in csv.DictReader(open(path))]


def score(labels, ranking, ns=NS):
    seen, hits, invalid = set(), [], 0
    for p in ranking:
        valid = p in labels and p not in seen
        invalid += not valid
        seen.add(p)
        hits.append(valid and labels[p] == 1)
    positives = sum(labels.values())
    prevalence = positives / len(labels)
    out = {"rows": len(ranking), "invalid_rate": invalid / len(ranking) if ranking else 0.0}
    for n in ns:
        h = sum(hits[:n])
        out |= {f"precision@{n}": h / n, f"recall@{n}": h / positives, f"enhancement@{n}": h / n / prevalence}
    return out


def main(labels_path, ranking_path):
    labels = {p: int(r["isInteraction"]) for p, r in read_pairs(labels_path)}
    ranking = [p for p, _ in read_pairs(ranking_path)]
    print(json.dumps(score(labels, ranking), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
