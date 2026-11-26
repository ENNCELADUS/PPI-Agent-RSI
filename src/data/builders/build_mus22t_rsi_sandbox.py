"""Build the agent's sandboxes for the 22T rounds: the training set as the look-up file, one view per DL checkpoint.

Run: uv run python -m src.data.builders.build_mus22t_rsi_sandbox   (~2 s; after build_mus22t_rsi, build_mus22t_sandbox)

Inputs: the candidate predictors of the DL arm of the random rule (src.data.builders.dl_rounds, train_mus22t_al.sh
random on H20). Candidate t (t = 0..10) is V3 retrained from scratch on the RSI training set (rsi/train.csv) plus
every pair revealed in rounds 1..t; candidate 0 is checkpoint 0, trained before any reveal. Under the random rule the
batches are fixed, so the candidates do not depend on any gate. Files: al_random/candidate<t>.tsv.gz (a, b, logit for
the 18,366 development and final pairs) and al_random/val/candidate<t>.tsv.gz (the validation pairs). Copy them with:

    rsync -a root@10.15.171.204:/2023533015/PPI-Agent-RSI/data/MUS22T/predictor/al_random/ -e 'ssh -p 30757' \
      data/MUS22T/predictor/al_random/

Output data/MUS22T/sandbox_rsi/:
- kb.csv: rsi/train.csv with its `evidence` column. data/MUS22T/sandbox/kb.csv is not used here: it is all of
  agent_split/train.csv and so holds the labels of the 9,000 development pairs that came from train.
- Every view: a symlink to that kb.csv and symlinks to data/MUS22T/sandbox/ (proteins.tsv, annotations.tsv,
  homologs.tsv).
- plain/: no scores; the LLM-only arm.
- ckpt<c>_round/, ckpt<c>_val/, ckpt<c>_final/ (c = 0..10): scores.tsv.gz (a, b, p) of candidate c for the pairs of
  one stage: the reveal batches (rsi/reveal.csv), the validation gate pairs (rsi/val_gate.csv), the final pairs. p =
  raw sigmoid of the logit, as for V3. src.eval.reveal_rounds reads the view of the checkpoint in use in a round and
  at the final test, and the view of the round's candidate at the gate.
A checkpoint whose scores are not there yet is skipped. The scores carry no labels.
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
    """The agents read the candidate predictors of the random rule (dl_rounds), whose batches they also get."""
    return MUS22T / "predictor" / "al_random" / f"candidate{t:02d}.tsv.gz"


def val_path(t):
    return MUS22T / "predictor" / "al_random" / "val" / f"candidate{t:02d}.tsv.gz"


def checkpoint_p(t, path=None):
    """{canonical pair: p} of checkpoint t over the development and final pairs (path: its val logits instead)."""
    with gzip.open(path or checkpoint_path(t), "rt") as f:
        return {pair(r["a"], r["b"]): 1 / (1 + math.exp(-float(r["logit"]))) for r in csv.DictReader(f, delimiter="\t")}


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
    reveal, final, gate = (read(RSI / f) for f in ("reveal.csv", "final.csv", "val_gate.csv"))
    scored = {(a, b) for a, b, _ in read(RSI / "dev.csv") + final}
    write_kb()
    view("plain", [], None)
    for c in range(ROUNDS + 1):
        if not (checkpoint_path(c).exists() and val_path(c).exists()):
            print(f"checkpoint {c}: scores missing, skipped")
            continue
        p, q = checkpoint_p(c), checkpoint_p(c, val_path(c))
        assert p.keys() == scored, f"checkpoint {c}: scored pairs differ from dev.csv + final.csv"
        assert all((a, b) in q for a, b, _ in gate), f"checkpoint {c}: val gate pairs without a val score"
        for stage, rows, s in (("round", reveal, p), ("val", gate, q), ("final", final, p)):
            view(f"ckpt{c:02d}_{stage}", rows, s)
        print(f"checkpoint {c}: views of {len(reveal):,} batch, {len(gate):,} val gate, {len(final):,} final pairs")


if __name__ == "__main__":
    main()
