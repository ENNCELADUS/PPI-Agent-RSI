"""Split a PPI dataset with the SEAL PPI baseline protocol, so its numbers compare with SEAL's.

Run: uv run --extra predictor python -m src.data.builders.build_seal_split mus22t|tmp   (needs torch)
- mus22t (~10 s): data/MUS22T/raw/ = 5_14_finetune_22T/ from custom_ppi.zip (private mouse membrane PPIs).
- tmp (~2 min, ~6 GB RAM): data/TMP/raw/, the human data SEAL's own split is built from (inferred from protein IDs;
  SEAL's 20250703 positive file is not available). Positives = pretrain + finetune + old-test positives, unordered
  duplicates across the three files dropped (self pairs kept); negatives = pretrain negatives only, whose 6,172
  proteins cover SEAL's 6,168 negative proteins. Nodes = data/TMP/processed/all_proteins.fasta.

Protocol = provenance/prepare_corrected.py in seal_ppi_code_20261101 (the human baseline, 764,488 / 95,561 /
95,562 positives at 1:1):
- Node IDs are sorted; positive rows whose ends are both nodes are kept in file order (self pairs kept).
- Positives: torch.randperm(seed 42), first 80% train, next 10% valid, rest test (random edge split).
- Negatives: unique unordered pairs, minus self pairs and any positive, shuffled once with
  np.random.RandomState(20261027), then assigned 1:1 to test, valid, train in that order. The rest are unused.
- Endpoint propensity baseline (SEAL diagnose.py): per protein log((train pos + 1) / (train neg + 1)), pair
  score = sum over both ends. SEAL reports it beats its model; it measures the protein-frequency shortcut.

Output data/<MUS22T|TMP>/seal_split/: train.csv, valid.csv, test.csv (uniprotID_A <= uniprotID_B, isInteraction),
split_stats.json (counts, positives per source stage, the SEAL reference numbers, the baseline on valid and test).
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

from src.data.paths import MUS22T as MUS
from src.data.paths import RAW_FILES, TMP

SPLITS = ("train", "valid", "test")
SEAL_HUMAN = {"train": 764488, "valid": 95561, "test": 95562, "val_baseline": {"auprc": 0.8829, "auroc": 0.8565}}
SEAL_MODEL = {"valid": {"auprc": 0.8606, "auroc": 0.8454}, "test": {"auprc": 0.8595, "auroc": 0.8439}}


def seal_split(ids, pos, neg):
    """ids: node IDs; pos, neg: (A, B) ID arrays. Returns {split: (pos index pairs, neg index pairs)}, sorted ids
    and the kept positive row numbers in shuffled order."""
    ids = np.array(sorted(set(ids)))
    n, index = len(ids), pd.Series(np.arange(len(ids)), index=ids)
    pa, pb = (pd.Series(pos[:, j]).map(index) for j in (0, 1))
    keep = np.flatnonzero((pa.notna() & pb.notna()).to_numpy())
    raw = torch.tensor(np.stack([pa[keep].to_numpy(np.int64), pb[keep].to_numpy(np.int64)]))
    perm = torch.randperm(raw.shape[1], generator=torch.Generator().manual_seed(42))
    shuf = raw[:, perm].numpy()
    nt, nv = int(0.8 * raw.shape[1]), int(0.1 * raw.shape[1])
    edges = {"train": shuf[:, :nt], "valid": shuf[:, nt : nt + nv], "test": shuf[:, nt + nv :]}
    allpos = np.unique(np.minimum(*shuf) * n + np.maximum(*shuf))
    na, nb = (pd.Series(neg[:, j]).map(index) for j in (0, 1))
    ok = (na.notna() & nb.notna()).to_numpy()
    a, b = na[ok].to_numpy(np.int64), nb[ok].to_numpy(np.int64)
    pool = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    pool = pool[(pool // n != pool % n) & ~np.isin(pool, allpos)]
    np.random.RandomState(20261027).shuffle(pool)
    assert len(pool) >= raw.shape[1], "insufficient negatives for 1:1"
    out, offset = {}, 0
    for s in ("test", "valid", "train"):
        k = pool[offset : offset + edges[s].shape[1]]
        offset += len(k)
        out[s] = (edges[s], np.stack([k // n, k % n]))
    return out, ids, keep[perm.numpy()]


def auroc_auprc(score, label):
    df = pd.DataFrame({"s": score, "y": label})
    pos, neg = df.y.sum(), len(df) - df.y.sum()
    auroc = (df.s.rank()[df.y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    g = df.groupby("s").y.agg(["sum", "count"]).sort_index(ascending=False)  # one threshold per distinct score
    tp = g["sum"].cumsum()
    precision, recall = tp / g["count"].cumsum(), tp / pos
    auprc = (recall.diff().fillna(recall.iloc[0]) * precision).sum()
    return {"auprc": round(float(auprc), 4), "auroc": round(float(auroc), 4)}


def propensity(split, n):
    counts = [np.bincount(split["train"][i].ravel(), minlength=n) for i in (0, 1)]
    rate = np.log((counts[0] + 1) / (counts[1] + 1))
    out = {}
    for s in ("valid", "test"):
        p, q = split[s]
        out[s] = auroc_auprc(np.r_[rate[p].sum(0), rate[q].sum(0)], np.r_[np.ones(p.shape[1]), np.zeros(q.shape[1])])
    return out


def load_mus22t():
    raw = MUS / "raw"
    nodes = pd.read_csv(raw / "raw_22T_node_sequences.csv", usecols=["id"]).id
    pos = pd.read_csv(raw / "raw_22T_positive_pair.txt", sep="\t").to_numpy()
    neg = pd.read_csv(raw / "raw_22T_negative_pair.csv").to_numpy()
    return MUS / "seal_split", nodes, pos, np.full(len(pos), "22T"), neg


def load_tmp():
    nodes = [line[1:].strip() for line in open(TMP / "processed" / "all_proteins.fasta") if line[0] == ">"]
    test = pd.read_csv(RAW_FILES["test"][0])
    stages = {
        "pretrain": pd.read_csv(RAW_FILES["pretrain_pos"][0], sep="\t", usecols=[0, 1]).to_numpy(),
        "finetune": pd.read_csv(RAW_FILES["finetune_pos"][0], sep="\t", usecols=[0, 1]).to_numpy(),
        "old_test": test[test.truth_de == "positive"][["p1", "p2"]].to_numpy(),
    }
    pos = np.concatenate(list(stages.values())).astype(str)
    stage = np.concatenate([np.full(len(v), k) for k, v in stages.items()])
    first = ~pd.DataFrame(np.sort(pos, axis=1)).duplicated().to_numpy()  # repeated pair keeps its earliest-stage row
    neg = pd.read_csv(RAW_FILES["pretrain_neg"][0], usecols=[0, 1]).to_numpy()
    return TMP / "seal_split", nodes, pos[first], stage[first], neg


def main(dataset):
    out_dir, nodes, pos, stage, neg = {"mus22t": load_mus22t, "tmp": load_tmp}[dataset]()
    split, ids, rows = seal_split(nodes, pos, neg)
    out_dir.mkdir(exist_ok=True)
    stats = {"nodes": len(ids), "positive_rows": len(pos), "negative_rows": len(neg), "splits": {}}
    seen, offset = set(np.unique(split["train"][0])), 0
    for s in SPLITS:
        p, q = split[s]
        e = np.concatenate([p, q], axis=1)
        a, b = ids[e.min(0)], ids[e.max(0)]
        y = np.r_[np.ones(p.shape[1], int), np.zeros(q.shape[1], int)]
        pd.DataFrame({"uniprotID_A": a, "uniprotID_B": b, "isInteraction": y}).to_csv(out_dir / f"{s}.csv", index=False)
        st = stage[rows[offset : offset + p.shape[1]]]
        offset += p.shape[1]
        stats["splits"][s] = {
            "pos": p.shape[1],
            "neg": q.shape[1],
            "self_pos": int((p[0] == p[1]).sum()),
            "pos_by_stage": dict(zip(*np.unique(st, return_counts=True), strict=True)),
        }
        if s != "train":
            both = np.isin(p[0], list(seen)) & np.isin(p[1], list(seen))
            stats["splits"][s]["pos_both_ends_in_train_pos"] = round(float(both.mean()), 4)
    stats |= {
        "endpoint_propensity_baseline": propensity(split, len(ids)),
        "seal_human": SEAL_HUMAN,
        "seal_model": SEAL_MODEL,
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=1, default=int))
    print(json.dumps(stats, indent=1, default=int))


if __name__ == "__main__":
    main(*sys.argv[1:])
