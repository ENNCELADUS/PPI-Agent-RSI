"""Split the 22T mouse membrane PPIs transductively 8:1:1 by positive pairs, with new PRING negatives.

Run: uv run python -m src.data.builders.build_mus22t_agent_split   (~5 s)
Input: data/MUS22T/raw/raw_22T_positive_pair.txt (private). The shipped 22T negatives are not used. Label
availability in public databases is ignored (user decision 2026-11-02): the agent sandbox bans web tools.

Protocol (user decisions 2026-11-02: 8:1:1 by positive pairs, transductive):
- Positives: unordered, self pairs dropped (22T has none); random.Random(42) shuffle, first 80% train, next 10%
  val, rest test. Transductive: a protein may appear in every split; no pair is dropped.
- Negatives: PRING negative_sample.py (github.com/SophieSarceau/PRING) via src.data.negatives.pring_negatives,
  random.Random(0), split by split in the order train, val, test, 1:1: both ends drawn from that split's positive
  endpoints (degree-preserving), redrawn until not self, not any 22T positive, not a negative of any split.
Roles (user decision 2026-11-06): train = KB, val = DL predictor validation (hyperparameters, architecture), test = final;
RSI rounds reveal labels from a part of test, never from val.

Output data/MUS22T/agent_split/: train.csv, val.csv, test.csv (uniprotID_A < uniprotID_B, isInteraction),
split_stats.json (counts, proteins, val/test positives with both ends in train, endpoint propensity baseline).
"""

import json
import random

import numpy as np
import pandas as pd

from src.data.builders.build_seal_split import auroc_auprc
from src.data.ids import pair
from src.data.negatives import IdentityKeys, pring_negatives
from src.data.paths import MUS22T as MUS

OUT = MUS / "agent_split"  # hidden labels; the agent sees sandbox/ only
SPLITS = ("train", "val", "test")


def transductive_split(positives, seed_split=42, seed_neg=0):
    """positives: unordered pairs. Returns {split: [(pair, label)]}, positives first."""
    shuffled = list(positives)
    random.Random(seed_split).shuffle(shuffled)
    nt, nv = int(0.8 * len(shuffled)), int(0.1 * len(shuffled))
    side = {"train": shuffled[:nt], "val": shuffled[nt : nt + nv], "test": shuffled[nt + nv :]}
    known, rng, labelled = set(positives), random.Random(seed_neg), {}
    for s in SPLITS:
        neg = pring_negatives(side[s], len(side[s]), lambda ks: any(k in known for k in ks), rng, IdentityKeys())
        known.update(neg)
        labelled[s] = [(p, 1) for p in side[s]] + [(p, 0) for p in neg]
    return labelled


def propensity(labelled):
    pos, neg = {}, {}
    for (a, b), y in labelled["train"]:
        for x in (a, b):
            (pos if y else neg)[x] = (pos if y else neg).get(x, 0) + 1
    rate = lambda x: np.log((pos.get(x, 0) + 1) / (neg.get(x, 0) + 1))  # noqa: E731
    return {
        s: auroc_auprc([rate(a) + rate(b) for (a, b), _ in labelled[s]], [y for _, y in labelled[s]])
        for s in ("val", "test")
    }


def main():
    raw = pd.read_csv(MUS / "raw" / "raw_22T_positive_pair.txt", sep="\t").to_numpy()
    positives = list(dict.fromkeys(pair(a, b) for a, b in raw if a != b))
    labelled = transductive_split(positives)
    OUT.mkdir(exist_ok=True)
    train_proteins = {x for (a, b), y in labelled["train"] if y for x in (a, b)}
    stats = {"positives": len(positives)}
    for s in SPLITS:
        rows = [(a, b, y) for (a, b), y in labelled[s]]
        pd.DataFrame(rows, columns=["uniprotID_A", "uniprotID_B", "isInteraction"]).to_csv(
            OUT / f"{s}.csv", index=False
        )
        pos = [(a, b) for a, b, y in rows if y]
        stats[s] = {"pos": len(pos), "neg": len(rows) - len(pos), "proteins": len({x for p in pos for x in p})}
        if s != "train":
            both = sum(a in train_proteins and b in train_proteins for a, b in pos)
            stats[s]["pos_both_ends_in_train_pos"] = round(both / len(pos), 4)
    stats["endpoint_propensity_baseline"] = propensity(labelled)
    (OUT / "split_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
