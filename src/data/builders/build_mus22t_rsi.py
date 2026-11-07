"""Split the 22T test set into fixed reveal rounds and a final set, for RSI under sparse labels (no acquisition).

Run: uv run python -m src.data.builders.build_mus22t_rsi   (~1 s; after build_mus22t_agent_split)
     python -m src.data.builders.build_mus22t_rsi al <t>          (train_mus22t_al.sh: round t's DL split)
     python -m src.data.builders.build_mus22t_rsi gate <t> <val_auprc>  (train_mus22t_al.sh: keep or drop round t)

Design: docs/superpowers/specs/2026-11-06-mus22t-reveal-rounds-design.md. Roles (user decisions 2026-11-06): train = KB,
val = DL predictor validation only, test = reveal rounds + final set. Every arm (DL active learning, the LLM-only
agent, the LLM agent with the DL predictor) gets the same pairs in the same rounds and is scored on the same final set.

- Reveal: agent_split/test.csv positives and negatives, each sorted and shuffled with random.Random(0); the first
  ROUNDS * BATCH / 2 of each are revealed. Round t gets positives and negatives [(t-1) * BATCH / 2, t * BATCH / 2),
  shuffled together with random.Random(t). 10 rounds of 100 pairs, 50 + 50 each.
- Final: the other test pairs (8,366: 4,183 + 4,183).
- al/round<t>/: the DL active-learning split of round t: train.csv = agent_split/train.csv plus the pairs of the kept
  rounds before t and of round t; val.csv and test.csv = agent_split's, unchanged (the scores of all 9,366 test pairs
  per checkpoint). Built per round by train_mus22t_al.sh, because it depends on the gate.
- al/gate.json: the val gate (user decision 2026-11-07). Round 0 is V3's val AUPRC. Round t is kept only if its
  checkpoint's val AUPRC is higher than that of every earlier kept checkpoint; else its pairs leave the DL training
  set and checkpoint t is a copy of checkpoint t - 1.

Output data/MUS22T/rsi/ (hidden labels; the agent never reads it): reveal.csv (uniprotID_A, uniprotID_B,
isInteraction, round), final.csv (uniprotID_A, uniprotID_B, isInteraction), al/round01..round10/, rsi.json (counts and
sha256 prefixes). Deterministic.

The DL checkpoints (H20, grasppi-mus22t branch mus22t, same config and seed as V3) and the agent's per-round
sandboxes come from build_mus22t_rsi_sandbox.
"""

import csv
import hashlib
import json
import random
import shutil
import sys

from src.data.paths import MUS22T

SPLIT, OUT = MUS22T / "agent_split", MUS22T / "rsi"  # stdlib only: H20's checkout venv has no numpy
SEED, ROUNDS, BATCH = 0, 10, 100
HEADER = ["uniprotID_A", "uniprotID_B", "isInteraction"]


def read(path):
    return [(r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])) for r in csv.DictReader(open(path))]


def split(test, rounds=ROUNDS, batch=BATCH, seed=SEED):
    """test: (a, b, label) rows. Returns ({round: rows}, final rows), every round half positive."""
    half = batch // 2
    by_label = {}
    for y in (1, 0):
        rows = sorted(r for r in test if r[2] == y)
        random.Random(seed).shuffle(rows)
        by_label[y] = rows
    assert min(map(len, by_label.values())) >= rounds * half, "too few test pairs for the reveal rounds"
    reveal = {}
    for t in range(1, rounds + 1):
        rows = [r for y in (1, 0) for r in by_label[y][(t - 1) * half : t * half]]
        random.Random(t).shuffle(rows)
        reveal[t] = rows
    final = sorted(r for y in (1, 0) for r in by_label[y][rounds * half :])
    return reveal, final


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    train, test = read(SPLIT / "train.csv"), read(SPLIT / "test.csv")
    reveal, final = split(test)
    shown = [r for t in sorted(reveal) for r in reveal[t]]
    assert len(set(shown) | set(final)) == len(test) and not set(shown) & set(final)
    assert not {r[:2] for r in train} & {r[:2] for r in test}
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    write(OUT / "reveal.csv", [*HEADER, "round"], [(*r, t) for t in sorted(reveal) for r in reveal[t]])
    write(OUT / "final.csv", HEADER, final)
    stats = {
        "seed": SEED,
        "rounds": ROUNDS,
        "batch": BATCH,
        "reveal": {"pairs": len(shown), "pos": sum(r[2] for r in shown)},
        "final": {"pairs": len(final), "pos": sum(r[2] for r in final)},
        "inputs": {f"agent_split/{s}.csv": sha(SPLIT / f"{s}.csv") for s in ("train", "val", "test")},
        "outputs": {p: sha(OUT / p) for p in ("reveal.csv", "final.csv")},
    }
    (OUT / "rsi.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


def read_gate():
    path = OUT / "al" / "gate.json"
    return json.loads(path.read_text()) if path.exists() else []


def kept(gate):
    return [g["round"] for g in gate if g["kept"] and g["round"] > 0]


def decide(gate, t, val_auprc):
    """gate plus round t's decision: kept only if val AUPRC beats every earlier kept checkpoint (round 0 = V3)."""
    assert [g["round"] for g in gate] == list(range(t)), f"gate.json must hold rounds 0..{t - 1}"
    best = max((g["val_auprc"] for g in gate if g["kept"]), default=None)
    return [*gate, {"round": t, "val_auprc": val_auprc, "kept": best is None or val_auprc > best}]


def write_al(t):
    """al/round<t>/: train + the kept rounds before t + round t; val and test unchanged."""
    gate = read_gate()
    assert [g["round"] for g in gate] == list(range(t)), f"gate.json must hold rounds 0..{t - 1}"
    reveal = {}
    for r in csv.DictReader(open(OUT / "reveal.csv")):
        reveal.setdefault(int(r["round"]), []).append((r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])))
    d = OUT / "al" / f"round{t:02d}"
    d.mkdir(parents=True, exist_ok=True)
    write(d / "train.csv", HEADER, read(SPLIT / "train.csv") + [r for s in [*kept(gate), t] for r in reveal[s]])
    for s in ("val", "test"):
        shutil.copy(SPLIT / f"{s}.csv", d / f"{s}.csv")


if __name__ == "__main__":
    if sys.argv[1:2] == ["al"]:
        write_al(int(sys.argv[2]))
    elif sys.argv[1:2] == ["gate"]:
        gate = decide(read_gate(), int(sys.argv[2]), float(sys.argv[3]))
        (OUT / "al").mkdir(parents=True, exist_ok=True)
        (OUT / "al" / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
        print("keep" if gate[-1]["kept"] else "drop")
    else:
        main()
