"""Build the agent's sandboxes for the 22T rounds: the training set as the look-up file, the newest DL checkpoint.

Run: uv run python -m src.data.builders.build_mus22t_rsi_sandbox   (~2 s; after build_mus22t_rsi, build_mus22t_sandbox)

Inputs: each DL checkpoint's logits for the 18,366 development and final pairs (a, b, logit): checkpoint 0 is V3
retrained on the RSI training set (rsi/train.csv), checkpoint t the same on the training set plus the kept rounds up
to t (data/MUS22T/predictor/al/round<t>.tsv.gz, t = 0..10, from train_mus22t_al.sh on H20); and each checkpoint's
logits for the validation pairs (al/val/round<t>.tsv.gz). Copy them with:

    rsync -a root@10.15.171.204:/2023533015/PPI-Agent-RSI/data/MUS22T/predictor/al/ -e 'ssh -p 30757' \
      data/MUS22T/predictor/al/

Output data/MUS22T/sandbox_rsi/:
- kb.csv: rsi/train.csv with its `evidence` column. data/MUS22T/sandbox/kb.csv is not used here: it is all of
  agent_split/train.csv and so holds the labels of the 9,000 development pairs that came from train.
- Every view: a symlink to that kb.csv and symlinks to data/MUS22T/sandbox/ (proteins.tsv, annotations.tsv,
  homologs.tsv).
- plain/: no scores; the LLM-only arm.
- round<t>/ (t = 1..10): scores.tsv.gz (a, b, p): round t's 100 pairs scored by checkpoint t - 1, trained on every
  pair revealed before round t. p = raw sigmoid of the logit, as for V3.
- final/: the final pairs scored by checkpoint 10.
- val<t>/ (t = 0..10): the validation gate pairs (rsi/val_gate.csv) scored by checkpoint t: the view in which the
  agent's memory after round t is checked.
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
SHARED = ("proteins.tsv", "annotations.tsv", "homologs.tsv")


def checkpoint_path(t):
    return MUS22T / "predictor" / "al" / f"round{t:02d}.tsv.gz"


def val_path(t):
    return MUS22T / "predictor" / "al" / "val" / f"round{t:02d}.tsv.gz"


def checkpoint_p(t, path=None):
    """{canonical pair: p} of checkpoint t over the development and final pairs (path: its val logits instead)."""
    with gzip.open(path or checkpoint_path(t), "rt") as f:
        return {pair(r["a"], r["b"]): 1 / (1 + math.exp(-float(r["logit"]))) for r in csv.DictReader(f, delimiter="\t")}


def reveal_rounds():
    """{round: [(a, b, label)]} in reveal.csv order."""
    out = {}
    for r in csv.DictReader(open(RSI / "reveal.csv")):
        out.setdefault(int(r["round"]), []).append((r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])))
    return out


def write_kb():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "kb.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["uniprotID_A", "uniprotID_B", "isInteraction", "evidence"])
        w.writerows((a, b, y, "22t" if y else "pring_sampled") for a, b, y in read(RSI / "train.csv"))


def view(name, rows, p):
    """One agent view: the shared links and, if p is given, the scores of rows."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for f, target in (*((f, f"../../sandbox/{f}") for f in SHARED), ("kb.csv", "../kb.csv")):
        (d / f).unlink(missing_ok=True)  # a view built before 2026-11-24 linked the full train split as kb.csv
        (d / f).symlink_to(target)
    if p is None:
        return
    with gzip.open(d / "scores.tsv.gz", "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["a", "b", "p"])
        w.writerows((a, b, f"{p[(a, b)]:.6g}") for a, b, _ in sorted(rows))


def main():
    rounds, final = reveal_rounds(), read(RSI / "final.csv")
    scored = {(a, b) for a, b, _ in read(RSI / "dev.csv") + final}
    write_kb()
    view("plain", [], None)
    views = {f"round{t:02d}": (t - 1, rounds[t]) for t in range(1, ROUNDS + 1)} | {"final": (ROUNDS, final)}
    for name, (t, rows) in views.items():
        if not checkpoint_path(t).exists():
            print(f"{name}: checkpoint {t} missing, skipped")
            continue
        p = checkpoint_p(t)
        assert p.keys() == scored, f"checkpoint {t}: scored pairs differ from dev.csv + final.csv"
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
