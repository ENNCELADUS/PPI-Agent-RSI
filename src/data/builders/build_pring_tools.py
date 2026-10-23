"""Build the agent's tool files for PRING wave discovery: calibrated predictor scores and KB homologs.

Run: uv run python -m src.data.builders.build_pring_tools   (after build_pring_sandbox; needs mmseqs)
Inputs: data/PRING/predictor/pring_candidates.tsv.gz (B0 logits for every candidate pair, with labels)
and pring_val.tsv.gz (B0 logits for hidden/val.csv), both from `src.predictor.score` on H20:

    awk -F, 'NR>1' data/PRING/hidden/val.csv  # mapped to node ids via raw/node_sequences.tsv -> pring_val_pairs.tsv
    python -m src.predictor.score score --checkpoint best.pt --features <store> --pairs pring_val_pairs.tsv \
        --out-dir pring_val && python -m src.predictor.score merge pring_val --output pring_val.tsv

Design: docs/superpowers/specs/2026-10-23-pring-wave-discovery-design.md.

- Calibration: Platt scaling (p = sigmoid(a * logit + b)) fitted on the 1:1 val pairs, then a prior
  shift from val prevalence (0.5) to the candidate hit rate, so p_uv reads as a probability among
  candidates. Monotone in the logit, so bf16 ties stay ties (the tools break them by pair id).
- sandbox/scores.tsv.gz: a, b, p for every non-self candidate pair, agent ids, a < b, sorted. No label.
- sandbox/homologs.tsv: mmseqs2 easy-search of every protein against the KB proteins, best 20 non-self
  hits per query (query, target, pident, evalue, bits). Test proteins are <= 40% identical to train, so
  their hits are remote homologs.
- anon/ gets the same files in opaque ids; data/PRING/tools.json records calibration and checksums.
"""

import csv
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.data.paths import HIDDEN, PRING, SANDBOX
from src.data.paths import PRING_RAW as RAW

PREDICTOR = PRING / "predictor"
CANDIDATE_PAIRS = 2_035_153
HOMOLOGS = 20


def platt(xs, ys, steps=50):
    """Fit p = sigmoid(a x + b) by Newton's method on the log-loss."""
    a, b = 1.0, 0.0
    for _ in range(steps):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-(a * x + b)))
            w = p * (1 - p)
            ga, gb = ga + (p - y) * x, gb + (p - y)
            haa, hab, hbb = haa + w * x * x, hab + w * x, hbb + w
        det = haa * hbb - hab * hab
        a, b = a - (hbb * ga - hab * gb) / det, b - (haa * gb - hab * ga) / det
    return a, b


def logit(p):
    return math.log(p / (1 - p))


def calibrate(a, b, shift):
    return lambda x: 1 / (1 + math.exp(-(a * x + b + shift)))


def homologs(proteins, kb):
    """Best HOMOLOGS non-self mmseqs hits of every protein against the KB proteins, sorted."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, ids in (("query", proteins), ("target", kb)):
            with open(tmp / f"{name}.fasta", "w") as f:
                f.writelines(f">{x}\n{proteins[x]}\n" for x in sorted(ids))
        subprocess.run(
            [
                "mmseqs",
                "easy-search",
                tmp / "query.fasta",
                tmp / "target.fasta",
                tmp / "hits.m8",
                tmp / "work",
                "--format-output",
                "query,target,pident,evalue,bits",
                "--threads",
                "1",
                "-v",
                "1",
            ],
            check=True,
        )
        best = {}
        for q, t, pident, evalue, bits in csv.reader(open(tmp / "hits.m8"), delimiter="\t"):
            if q != t:
                best.setdefault(q, []).append((-float(bits), t, pident, evalue, bits))
    return [(q, *h[1:]) for q in sorted(best) for h in sorted(best[q])[:HOMOLOGS]]


def main():
    assert shutil.which("mmseqs"), "mmseqs not on PATH"
    node2acc = {r["node_id"]: r["uniprot_id"] for r in csv.DictReader(open(RAW / "node_sequences.tsv"), delimiter="\t")}
    acc2anon = {r["accession"]: r["id"] for r in csv.DictReader(open(HIDDEN / "anon_map.tsv"), delimiter="\t")}

    val = list(csv.DictReader(gzip.open(PREDICTOR / "pring_val.tsv.gz", "rt"), delimiter="\t"))
    xs, ys = [float(r["logit"]) for r in val], [int(r["label"]) for r in val]
    assert len(val) == sum(1 for _ in open(HIDDEN / "val.csv")) - 1, "val scores do not cover hidden/val.csv"
    a, b = platt(xs, ys)
    n_hits = sum(1 for _ in open(HIDDEN / "test_positives.csv")) - 1
    prevalence = n_hits / CANDIDATE_PAIRS
    shift = logit(prevalence) - logit(sum(ys) / len(ys))
    p_of = calibrate(a, b, shift)

    proteins = {r["accession"]: r for r in csv.DictReader(open(SANDBOX / "proteins.tsv"), delimiter="\t")}
    candidates = {x for x, r in proteins.items() if r["candidate"] == "1"}
    scores = {"": [], "anon": []}
    for r in csv.DictReader(gzip.open(PREDICTOR / "pring_candidates.tsv.gz", "rt"), delimiter="\t"):
        u, v = node2acc[r["node_a"]], node2acc[r["node_b"]]
        if u == v:
            continue
        assert u in candidates and v in candidates, f"{u}-{v} is not a candidate pair"
        p = f"{p_of(float(r['logit'])):.6g}"
        scores[""].append((*sorted((u, v)), p))
        scores["anon"].append((*sorted((acc2anon[u], acc2anon[v])), p))
    assert len(set(scores[""])) == len(scores[""]) == CANDIDATE_PAIRS, "candidate scores incomplete or repeated"

    seqs = {x: r["sequence"] for x, r in proteins.items()}
    kb = {x for x, r in proteins.items() if r["candidate"] == "0"}
    hits = {"": homologs(seqs, kb)}
    hits["anon"] = sorted(
        ((acc2anon[q], acc2anon[t], *rest) for q, t, *rest in hits[""]), key=lambda h: (h[0], -float(h[4]), h[1])
    )

    checksums = {}
    for sub in scores:
        out = SANDBOX / sub
        with gzip.GzipFile(out / "scores.tsv.gz", "wb", mtime=0) as f:  # mtime=0: byte-identical rebuilds
            f.write("a\tb\tp\n".encode())
            f.write("".join(f"{u}\t{v}\t{p}\n" for u, v, p in sorted(scores[sub])).encode())
        with open(out / "homologs.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["query", "target", "pident", "evalue", "bits"])
            w.writerows(hits[sub])
        for name in ("scores.tsv.gz", "homologs.tsv"):
            path = out / name
            checksums[str(path.relative_to(PRING))] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    stats = {
        "platt_a": a,
        "platt_b": b,
        "prior_shift": shift,
        "candidate_prevalence": prevalence,
        "val_pairs": len(val),
        "mean_p_candidates": sum(float(p) for *_, p in scores[""]) / CANDIDATE_PAIRS,
        "proteins_with_homologs": len({h[0] for h in hits[""]}),
        "inputs": {
            f"predictor/{f}": hashlib.sha256((PREDICTOR / f).read_bytes()).hexdigest()[:16]
            for f in ("pring_candidates.tsv.gz", "pring_val.tsv.gz")
        },
        "outputs": checksums,
    }
    (PRING / "tools.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
