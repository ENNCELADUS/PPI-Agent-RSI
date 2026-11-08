"""Build the agent's sandboxes for the 22T reveal rounds with the DL predictor: each round sees the newest checkpoint.

Run: uv run python -m src.data.builders.build_mus22t_rsi_sandbox   (~2 s; after build_mus22t_rsi, build_mus22t_sandbox)

Inputs: each DL checkpoint's logits for the 9,366 test pairs (a, b, logit): checkpoint 0 is V3
(data/MUS22T/predictor/v3_test.tsv.gz), checkpoint t is V3 retrained on train plus the kept rounds up to t
(data/MUS22T/predictor/al/round<t>.tsv.gz from train_mus22t_al.sh on H20); and each checkpoint's logits for the val
pairs (al/val/round<t>.tsv.gz, t = 0..10). Copy them with:

    rsync -a root@10.15.171.204:/2023533015/PPI-Agent-RSI/data/MUS22T/predictor/al/ -e 'ssh -p 30757' \
      data/MUS22T/predictor/al/

Output data/MUS22T/sandbox_rsi/:
- round<t>/ (t = 1..10): symlinks to data/MUS22T/sandbox/ (kb.csv, proteins.tsv, annotations.tsv, homologs.tsv) and
  scores.tsv.gz (a, b, p): round t's 100 pairs scored by checkpoint t - 1, trained on every pair revealed before
  round t. p = raw sigmoid of the logit, as for V3.
- final/: the same symlinks and the final pairs scored by checkpoint 10.
- val<t>/ (t = 0..10): the same symlinks and the val gate pairs (rsi/val_gate.csv) scored by checkpoint t: the view
  in which the agent's memory after round t is checked.
A view whose checkpoint is not there yet is skipped. The scores carry no labels.
"""

import csv
import gzip
import math

from src.data.builders.build_mus22t_rsi import OUT as RSI
from src.data.builders.build_mus22t_rsi import ROUNDS, read
from src.data.ids import pair
from src.data.paths import MUS22T

OUT = MUS22T / "sandbox_rsi"
SHARED = ("kb.csv", "proteins.tsv", "annotations.tsv", "homologs.tsv")


def checkpoint_path(t):
    return MUS22T / "predictor" / ("v3_test.tsv.gz" if t == 0 else f"al/round{t:02d}.tsv.gz")


def val_path(t):
    return MUS22T / "predictor" / "al" / "val" / f"round{t:02d}.tsv.gz"


def checkpoint_p(t, path=None):
    """{canonical pair: p} of checkpoint t over the test pairs (path: its val logits instead)."""
    with gzip.open(path or checkpoint_path(t), "rt") as f:
        return {pair(r["a"], r["b"]): 1 / (1 + math.exp(-float(r["logit"]))) for r in csv.DictReader(f, delimiter="\t")}


def reveal_rounds():
    """{round: [(a, b, label)]} in reveal.csv order."""
    out = {}
    for r in csv.DictReader(open(RSI / "reveal.csv")):
        out.setdefault(int(r["round"]), []).append((r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])))
    return out


def view(name, rows, p):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for f in SHARED:
        if not (d / f).is_symlink():
            (d / f).symlink_to(f"../../sandbox/{f}")
    with gzip.open(d / "scores.tsv.gz", "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["a", "b", "p"])
        w.writerows((a, b, f"{p[(a, b)]:.6g}") for a, b, _ in sorted(rows))


def main():
    rounds, final = reveal_rounds(), read(RSI / "final.csv")
    test = {(a, b) for a, b, _ in read(MUS22T / "agent_split" / "test.csv")}
    views = {f"round{t:02d}": (t - 1, rounds[t]) for t in range(1, ROUNDS + 1)} | {"final": (ROUNDS, final)}
    for name, (t, rows) in views.items():
        if not checkpoint_path(t).exists():
            print(f"{name}: checkpoint {t} missing, skipped")
            continue
        p = checkpoint_p(t)
        assert p.keys() == test, f"checkpoint {t}: scored pairs differ from test.csv"
        view(name, rows, p)
        print(f"{name}: {len(rows):,} pairs scored by checkpoint {t}")
    gate = read(RSI / "val_gate.csv")
    for t in range(ROUNDS + 1):
        if not val_path(t).exists():
            print(f"val{t:02d}: checkpoint {t} val scores missing, skipped")
            continue
        p = checkpoint_p(t, val_path(t))
        assert all((a, b) in p for a, b, _ in gate), f"checkpoint {t}: val gate pairs without a val score"
        view(f"val{t:02d}", gate, p)


if __name__ == "__main__":
    main()
