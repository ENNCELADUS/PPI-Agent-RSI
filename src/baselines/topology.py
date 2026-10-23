"""Non-LLM baseline rankings for the PRING sandbox, scored like any agent output.

Run: uv run python -m src.baselines.topology
- degree: deg(a) * deg(b) over KB positives, the popularity floor.
- l3: paths of length 3 over KB positives, each weighted 1/sqrt(deg(u) deg(v)) (Kovacs et al. 2019).
Writes data/PRING/baselines/<pool>_<method>.csv and prints src.eval.score metrics. Ties keep the shuffled
pool order. Inductive test proteins have no KB edges, so both baselines must sit at random there;
anything better means the split leaks.
"""
import csv
import math
from collections import defaultdict

from src.data.builders.build_pring_sandbox import PAIR, write
from src.data.paths import HIDDEN, PRING, SANDBOX
from src.eval.score import pair, read_pairs, score


def main():
    nbrs = defaultdict(set)
    for r in csv.DictReader(open(SANDBOX / "kb.csv")):
        if r["isInteraction"] == "1":
            nbrs[r["uniprotID_A"]].add(r["uniprotID_B"])
            nbrs[r["uniprotID_B"]].add(r["uniprotID_A"])
    deg = {x: len(n) for x, n in nbrs.items()}
    methods = {
        "degree": lambda a, b: deg.get(a, 0) * deg.get(b, 0),
        "l3": lambda a, b: sum(1 / math.sqrt(deg[u] * deg[v])
                               for u in nbrs.get(a, ()) for v in nbrs[u] & nbrs.get(b, set())),
    }
    for pool in ("inductive", "transductive"):
        pairs = [p for p, _ in read_pairs(SANDBOX / f"pool_{pool}.csv")]
        labels = {p: int(r["isInteraction"]) for p, r in read_pairs(HIDDEN / f"{pool}.csv")}
        for method, f in methods.items():
            ranking = sorted(pairs, key=lambda p: -f(*p))
            write(PRING / "baselines" / f"{pool}_{method}.csv", PAIR, ranking)
            s = score(labels, [pair(*p) for p in ranking])
            print(pool, method, {k: round(v, 3) for k, v in s.items() if "@" in k})


if __name__ == "__main__":
    main()
