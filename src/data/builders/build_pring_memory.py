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
- Validation gate (user decision 2026-11-24): hidden/val_gate.csv = 200 positives and 200 negatives of
  hidden/val.csv, each sorted and shuffled with random.Random(0); after every round the agent classifies them in
  sandbox_val/ (build_pring_sandbox, build_pring_tools) with its candidate memory. sandbox_val/case_homologs.tsv
  holds the hits of the validation proteins against the discovery proteins. No validation protein is in a view.
- Development set (user decisions 2026-11-24): hidden/dev.csv = PRING's labelled test pairs with both ends among the
  discovery proteins, no self pairs, balanced 1:1 like the final test (20,668 pairs). All pairs of the discovery
  proteins (732,655, 1.41% positive) stay the candidate space of the discovery track only (src.eval.memory_agent
  rounds, metric: hits): a random draw from them reveals almost only negatives.
- Random acquisition, the baseline rule: hidden/dev_random.csv = 10 rounds of 200 development pairs
  (build_mus22t_rsi.draw: sorted by pair, shuffled with random.Random(0); the labels play no part in the draw).
- data/PRING/memory_split.json: counts and checksums. The ribosome stays in both views (user decision).
"""

import csv
import gzip
import hashlib
import json
import random
import shutil

from src.data.builders.build_mus22t_rsi import draw, val_gate
from src.data.builders.build_pring_noribo import drop_rows
from src.data.builders.build_pring_tools import homologs
from src.data.paths import HIDDEN, PRING, SANDBOX
from src.data.paths import SANDBOX_VAL as VAL

SEED = 0
FINAL_SHARE = 0.4
DISC, FINAL = PRING / "sandbox_disc", PRING / "sandbox_final"
FINAL_TEST = HIDDEN / "final_test.csv"
VAL_GATE = HIDDEN / "val_gate.csv"
DEV, DEV_RANDOM = HIDDEN / "dev.csv", HIDDEN / "dev_random.csv"
ROUNDS, BATCH = 10, 200
HEADER = ["uniprotID_A", "uniprotID_B", "isInteraction"]


def labelled(path):
    return [(r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])) for r in csv.DictReader(open(path))]


def gate_pairs(val, n=400, seed=SEED):
    """n / 2 positives and n / 2 negatives of the validation set, fixed for every round and every run."""
    return val_gate(val, val, n, seed)


def write_pairs(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def split(candidates, seed=SEED):
    """(discovery, final-test) protein sets."""
    ordered = sorted(candidates)
    final = set(random.Random(seed).sample(ordered, int(FINAL_SHARE * len(ordered))))
    return set(ordered) - final, final


def balanced_pairs(rows, proteins, seed=SEED):
    """Labelled non-self pairs with both ends in proteins, balanced 1:1; rows are (a, b, label) with label 0/1."""
    inside = sorted(r for r in rows if r[0] in proteins and r[1] in proteins and r[0] != r[1])
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

    val_rows = csv.DictReader(open(VAL / "proteins.tsv"), delimiter="\t")
    val_seqs = {r["accession"]: r["sequence"] for r in val_rows if r["candidate"] == "1"}
    assert not val_seqs.keys() & proteins.keys(), "validation protein in the sandbox of the rounds"
    hits = homologs({x: proteins[x]["sequence"] for x in candidates} | val_seqs, disc)
    write_homologs(DISC / "case_homologs.tsv", hits, disc)
    write_homologs(FINAL / "case_homologs.tsv", hits, final)
    write_homologs(VAL / "case_homologs.tsv", hits, val_seqs.keys())

    released = labelled(HIDDEN / "test.csv")
    test, dev = balanced_pairs(released, final), balanced_pairs(released, disc)
    assert not {r[:2] for r in test} & {r[:2] for r in dev}, "pair in the development set and the final test"
    write_pairs(FINAL_TEST, test)
    write_pairs(DEV, dev)
    gate = gate_pairs(labelled(HIDDEN / "val.csv"))
    write_pairs(VAL_GATE, gate)
    hit_pairs = {
        tuple(sorted((r["uniprotID_A"], r["uniprotID_B"]))) for r in csv.DictReader(open(HIDDEN / "test_positives.csv"))
    }
    drawn = draw(dev, ROUNDS, BATCH)
    with open(DEV_RANDOM, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*HEADER, "round"])
        w.writerows((*r, t) for t in sorted(drawn) for r in drawn[t])
    hits_disc = sum(1 for a, b in hit_pairs if a in disc and b in disc)
    stats |= {
        "discovery_hits": hits_disc,
        "final_test_pairs": len(test),
        "final_test_positives": sum(r[2] for r in test),
        "val_gate_pairs": len(gate),
        "dev": {"pairs": len(dev), "pos": sum(r[2] for r in dev)},
        "dev_random": {
            "pairs": sum(map(len, drawn.values())),
            "pos": sum(r[2] for rows in drawn.values() for r in rows),
        },
        "case_homolog_queries": {
            "discovery": len({h[0] for h in hits if h[0] in disc}),
            "final": len({h[0] for h in hits if h[0] in final}),
        },
        "outputs": {
            p.relative_to(PRING).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in (FINAL_TEST, VAL_GATE, DEV, DEV_RANDOM, *(d / "case_homologs.tsv" for d in (DISC, FINAL, VAL)))
        },
    }
    (PRING / "memory_split.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
