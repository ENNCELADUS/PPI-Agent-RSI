"""Parity of the vendored B0 scorer against a reference motif-graph scores artifact (e.g. the PRING test.npz).

Reference format: motif-graph-prompt-tuning-for-inductive-protein-protein-interaction-prediction@bb01ee1
src/score_universe.py (save_scores): node_ids, u_idx/v_idx (canonical pairs, row order), logit, label, meta.

    python -m src.predictor.parity pairs <reference.npz> <pairs.tsv>     # its rows, as a pairs TSV
    python -m src.predictor.parity compare <reference.npz> <scores.tsv>  # after score + merge

`compare` prints max/mean absolute logit difference, the identical fraction, Spearman rank
correlation (ties averaged) and both checkpoint ids. It reports; it does not pass or fail.
"""

import json
import sys
from pathlib import Path

import numpy as np


def reference_rows(path):
    z = np.load(path)
    ids = z["node_ids"]
    return ids[z["u_idx"]], ids[z["v_idx"]], z["logit"], z["label"], json.loads(str(z["meta"]))


def average_ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def spearman(x, y):
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def compare(reference, scores):
    u, v, ref_logit, _, ref_meta = reference_rows(reference)
    rows = [line.split("\t") for line in Path(scores).read_text(encoding="utf-8").splitlines()[1:]]
    if [(r[0], r[1]) for r in rows] != list(zip(u, v, strict=True)):
        raise ValueError(f"{scores} rows do not match the reference pairs in order")
    logit = np.array([np.float32(r[2]) for r in rows], dtype=np.float32)
    diff = np.abs(logit.astype(np.float64) - ref_logit.astype(np.float64))
    meta_path = Path(f"{scores}.meta.json")
    return {
        "rows": len(rows),
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "identical_fraction": float((diff == 0).mean()),
        "spearman": spearman(logit, ref_logit),
        "reference_checkpoint_id": ref_meta.get("checkpoint_id"),
        "checkpoint_id": json.loads(meta_path.read_text())["checkpoint_id"] if meta_path.exists() else None,
    }


def main(argv):
    command, reference, path = argv
    if command == "pairs":
        u, v, _, label, _ = reference_rows(reference)
        Path(path).write_text("".join(f"{a}\t{b}\t{y}\n" for a, b, y in zip(u, v, label, strict=True)))
    elif command == "compare":
        print(json.dumps(compare(reference, path), indent=2))
    else:
        raise SystemExit(f"unknown command {command!r}; expected pairs or compare")


if __name__ == "__main__":
    main(sys.argv[1:])
