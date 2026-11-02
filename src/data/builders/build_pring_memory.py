"""Build the two-layer memory agent's split: discovery and final-test views of the PRING test proteins.

Run: uv run python -m src.data.builders.build_pring_memory   (after build_pring_tools; needs mmseqs)

Design: docs/superpowers/specs/2026-11-02-two-layer-memory-agent-design.md.

- Split: the 2,018 candidate proteins of sandbox/ are sorted and 40% (807) are drawn with random.Random(0) as
  final-test proteins; the other 60% are discovery proteins. Pairs with one end in each set are in neither view.
- sandbox_disc/: sandbox/ with the final-test proteins removed (rows leave proteins.tsv, annotations.tsv,
  homologs.tsv as query, and every pair with one of them leaves scores.tsv.gz). The discovery rounds run here.
- sandbox_final/: sandbox/ with the discovery proteins removed in the same way. The final test runs here.
- case_homologs.tsv in each view: mmseqs2 hits of that view's candidate proteins against the discovery proteins
  (the only proteins that can be in a case), best 20 non-self per query. The `cases` tool's similar_to reads it.
- hidden/final_test.csv: PRING's labelled test pairs (hidden/test.csv) with both ends among final-test proteins,
  no self pairs, balanced 1:1 by sampling the larger class down with random.Random(0) (sorted first).
- data/PRING/memory_split.json: counts and checksums. The ribosome stays in both views (user decision).
"""

import csv
import gzip
import hashlib
import json
import random
import shutil

from src.data.builders.build_pring_noribo import drop_rows
from src.data.builders.build_pring_tools import homologs
from src.data.paths import HIDDEN, PRING, SANDBOX

SEED = 0
FINAL_SHARE = 0.4
DISC, FINAL = PRING / "sandbox_disc", PRING / "sandbox_final"
FINAL_TEST = HIDDEN / "final_test.csv"


def split(candidates, seed=SEED):
    """(discovery, final-test) protein sets."""
    ordered = sorted(candidates)
    final = set(random.Random(seed).sample(ordered, int(FINAL_SHARE * len(ordered))))
    return set(ordered) - final, final


def final_pairs(rows, final, seed=SEED):
    """Labelled non-self pairs with both ends in final, balanced 1:1; rows are (a, b, label) with label 0/1."""
    inside = sorted(r for r in rows if r[0] in final and r[1] in final and r[0] != r[1])
    pos, neg = [r for r in inside if r[2] == 1], [r for r in inside if r[2] == 0]
    n, rng = min(len(pos), len(neg)), random.Random(seed)
    return sorted(rng.sample(pos, n) + rng.sample(neg, n))


def view(out, drop):
    out.mkdir(exist_ok=True)
    shutil.copy(SANDBOX / "kb.csv", out / "kb.csv")
    counts = {
        "proteins.tsv": drop_rows("proteins.tsv", lambda f: f[0] not in drop, out),
        "annotations.tsv": drop_rows("annotations.tsv", lambda f: f[0] not in drop, out),
        "homologs.tsv": drop_rows("homologs.tsv", lambda f: f[0] not in drop, out),
        "scores.tsv.gz": drop_rows(
            "scores.tsv.gz", lambda f: f[0] not in drop and f[1] not in drop, out, opener=gzip.open
        ),
    }
    n = sum(r["candidate"] == "1" for r in csv.DictReader(open(out / "proteins.tsv"), delimiter="\t"))
    kept = sum(1 for _ in gzip.open(out / "scores.tsv.gz", "rt")) - 1
    assert kept == n * (n - 1) // 2, (out, kept, n)  # every pair among the view's candidates, nothing else
    return n, counts


def write_homologs(path, hits, queries):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query", "target", "pident", "evalue", "bits"])
        w.writerows(h for h in hits if h[0] in queries)


def main():
    assert shutil.which("mmseqs"), "mmseqs not on PATH"
    proteins = {r["accession"]: r for r in csv.DictReader(open(SANDBOX / "proteins.tsv"), delimiter="\t")}
    candidates = {x for x, r in proteins.items() if r["candidate"] == "1"}
    assert len(candidates) == 2018, len(candidates)
    disc, final = split(candidates)

    stats = {"seed": SEED, "discovery_proteins": len(disc), "final_proteins": len(final)}
    for out, drop, name in ((DISC, final, "discovery"), (FINAL, disc, "final")):
        n, counts = view(out, drop)
        stats[f"{name}_view"] = {"candidates": n, "pairs": n * (n - 1) // 2, "rows_dropped": counts}

    hits = homologs({x: proteins[x]["sequence"] for x in candidates}, disc)
    write_homologs(DISC / "case_homologs.tsv", hits, disc)
    write_homologs(FINAL / "case_homologs.tsv", hits, final)

    labelled = [
        (r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])) for r in csv.DictReader(open(HIDDEN / "test.csv"))
    ]
    test = final_pairs(labelled, final)
    with open(FINAL_TEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uniprotID_A", "uniprotID_B", "isInteraction"])
        w.writerows(test)
    hits_disc = sum(
        1
        for r in csv.DictReader(open(HIDDEN / "test_positives.csv"))
        if r["uniprotID_A"] in disc and r["uniprotID_B"] in disc
    )
    stats |= {
        "discovery_hits": hits_disc,
        "final_test_pairs": len(test),
        "final_test_positives": sum(r[2] for r in test),
        "case_homolog_queries": {
            "discovery": len({h[0] for h in hits if h[0] in disc}),
            "final": len({h[0] for h in hits if h[0] in final}),
        },
        "outputs": {
            p.relative_to(PRING).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in (FINAL_TEST, DISC / "case_homologs.tsv", FINAL / "case_homologs.tsv")
        },
    }
    (PRING / "memory_split.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
