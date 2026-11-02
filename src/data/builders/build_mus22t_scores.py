"""Put V3's test-pair scores in the 22T agent sandbox, for the predictor-assisted classification test.

Run: uv run python -m src.data.builders.build_mus22t_scores   (~1 s; after build_mus22t_sandbox)
Input: data/MUS22T/predictor/v3_test.tsv.gz (a, b, logit), V3 = B0 v3_1 trained on agent_split/train.csv in
grasppi-mus22t 7a06976 (configs/mus22t/b0_v31.yaml, seed 0, epoch 17 selected on validation AUPRC; checkpoint
sha256 e5026e41e624ff21, scores/test.npz feadcce786595f39). Export it on H20 from outputs/mus22t/b0_v31/:

    python -c "import numpy as np, gzip; d = np.load('scores/test.npz'); n = [x.removeprefix('x_') for x in
      d['node_ids']]; f = gzip.open('v3_test.tsv.gz', 'wt'); f.write('a\\tb\\tlogit\\n'); f.writelines(
      f'{n[u]}\\t{n[v]}\\t{float(l)!r}\\n' for u, v, l in zip(d['u_idx'], d['v_idx'], d['logit']))"

Output: data/MUS22T/sandbox/scores.tsv.gz (a, b, p): canonical pairs, p = raw sigmoid of the logit (V3 is
calibrated: validation ECE 0.081). Only the 9,366 test pairs have a score, and they are exactly agent_split/test.csv.
The scores carry no labels; the agent reads them through predict_pairs.
"""

import csv
import gzip
import math

from src.data.builders.build_mus22t_agent_split import OUT as SPLIT
from src.data.ids import pair
from src.data.paths import MUS22T

EXPORT = MUS22T / "predictor" / "v3_test.tsv.gz"
OUT = MUS22T / "sandbox" / "scores.tsv.gz"


def main():
    with gzip.open(EXPORT, "rt") as f:
        scores = {
            pair(r["a"], r["b"]): 1 / (1 + math.exp(-float(r["logit"]))) for r in csv.DictReader(f, delimiter="\t")
        }
    test = {pair(r["uniprotID_A"], r["uniprotID_B"]) for r in csv.DictReader(open(SPLIT / "test.csv"))}
    assert scores.keys() == test, f"{len(scores.keys() ^ test)} pairs differ between the V3 export and test.csv"
    with gzip.open(OUT, "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["a", "b", "p"])
        w.writerows((a, b, f"{p:.6g}") for (a, b), p in sorted(scores.items()))
    print(f"{OUT}: {len(scores):,} test pairs")


if __name__ == "__main__":
    main()
