"""Split the 22T data for RSI under sparse labels: training set, development set, random batches and final set.

Run: uv run python -m src.data.builders.build_mus22t_rsi   (~1 s; after build_mus22t_agent_split)
The DL predictor rounds (each round's training data and gate) are in src.data.builders.dl_rounds.

Design: docs/superpowers/specs/2026-11-06-mus22t-reveal-rounds-design.md, changed by the user decisions of
2026-11-24 (docs/superpowers/plans/2026-11-24-node-disjoint-validation-and-development-set.md): the label budget
(10 rounds of 100 pairs) is 10% of the development set, so the choice of the pairs can matter. The validation set
(agent_split/val.csv) is for selection only and is not touched.

- Final: as before, agent_split/test.csv positives and negatives, each sorted and shuffled with random.Random(0),
  minus the first 500 of each (8,366 pairs: 4,183 + 4,183).
- Development set (10,000 pairs: 5,000 + 5,000): those 1,000 test pairs and 4,500 positives + 4,500 negatives of
  agent_split/train.csv, each label sorted and shuffled with random.Random(0). Every cut is a uniform random draw
  by pair, so the development set, the final set and the training set hold the same kind of pair (transductive).
- Training set (65,912 pairs: 32,956 + 32,956): the other train pairs. It replaces agent_split/train.csv for the DL
  predictor and for the agent's look-up file in every RSI run; the development labels are in neither.
- reveal.csv: the batches of the random acquisition rule, the control: the development set sorted by pair and shuffled
  with random.Random(0), round t = pairs [(t - 1) * 100, t * 100). The labels play no part in the draw. Every arm gets
  the same pairs in the same rounds. The greedy rule on the DL scores is in dl_rounds; the agent's own picks are not
  built yet.
- The validation gate of the DL rounds and of the agents: build_mus22t_rsi.gate_round (user decision 2026-11-26: a
  candidate replaces the state in use unless a paired bootstrap shows it clearly worse than the best kept state).
  build_mus22t_rsi.decide is the earlier rule (2026-11-07), still used by the discovery track (src.eval.memory_agent).
- val_gate.csv: the agents' validation gate (user decision 2026-11-07), fixed VAL_GATE / 2 positives and
  VAL_GATE / 2 negatives of agent_split/val.csv whose two proteins are both test proteins, each sorted and shuffled
  with random.Random(0). An agent round's memory is kept only if the agent's validation AUPRC on these pairs beats
  every earlier kept memory (src.eval.reveal_rounds). Validation labels never reach the agent.

Output data/MUS22T/rsi/ (hidden labels; the agent never reads it): train.csv, dev.csv, final.csv, val_gate.csv
(uniprotID_A, uniprotID_B, isInteraction), reveal.csv (the same and round), rsi.json (counts and sha256
prefixes). Deterministic.

The DL checkpoints (H20, grasppi-mus22t branch mus22t, same config and seed as V3) and the agent's per-round
sandboxes come from build_mus22t_rsi_sandbox.
"""

import csv
import hashlib
import json
import random
import shutil

from src.data.paths import MUS22T

SPLIT, OUT = MUS22T / "agent_split", MUS22T / "rsi"  # stdlib only: H20's checkout venv has no numpy
SEED, ROUNDS, BATCH, VAL_GATE = 0, 10, 100, 400
DEV_FROM_TRAIN = 9000  # train pairs that join the 1,000 test pairs in the development set
HEADER = ["uniprotID_A", "uniprotID_B", "isInteraction"]


def read(path):
    return [(r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])) for r in csv.DictReader(open(path))]


def split(test, rounds=ROUNDS, batch=BATCH, seed=SEED):
    """test: (a, b, label) rows. Returns ({block: rows}, final rows): the first rounds * batch test pairs, half
    positive, in blocks of batch (they join the development set), and the final set."""
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


def move(train, n=DEV_FROM_TRAIN, seed=SEED):
    """(kept rows, moved rows): n / 2 pairs of each label leave train for the development set."""
    kept_rows, moved = [], []
    for y in (1, 0):
        rows = sorted(r for r in train if r[2] == y)
        random.Random(seed).shuffle(rows)
        assert len(rows) >= n // 2, "too few train pairs for the development set"
        moved += rows[: n // 2]
        kept_rows += rows[n // 2 :]
    return sorted(kept_rows), sorted(moved)


def draw(dev, rounds=ROUNDS, batch=BATCH, seed=SEED):
    """{round: rows}: the random acquisition rule, uniform over the development set. The labels play no part."""
    rows = sorted(dev, key=lambda r: r[:2])
    random.Random(seed).shuffle(rows)
    assert len(rows) >= rounds * batch, "development set smaller than the label budget"
    return {t: rows[(t - 1) * batch : t * batch] for t in range(1, rounds + 1)}


def val_gate(val, test, n=VAL_GATE, seed=SEED):
    """n / 2 positives and n / 2 negatives of val whose proteins both occur in test."""
    proteins = {x for a, b, _ in test for x in (a, b)}
    out = []
    for y in (1, 0):
        rows = sorted(r for r in val if r[2] == y and r[0] in proteins and r[1] in proteins)
        random.Random(seed).shuffle(rows)
        assert len(rows) >= n // 2, "too few val pairs for the gate"
        out += rows[: n // 2]
    return sorted(out)


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    train, val, test = (read(SPLIT / f"{s}.csv") for s in ("train", "val", "test"))
    from_test, final = split(test)
    training, moved = move(train)
    dev = sorted(moved + [r for t in sorted(from_test) for r in from_test[t]])
    reveal = draw(dev)
    shown = [r for t in sorted(reveal) for r in reveal[t]]
    parts = [{r[:2] for r in rows} for rows in (training, dev, final, val)]
    assert sum(map(len, parts)) == len(set().union(*parts)), (
        "a pair is in two of training, development, final, validation"
    )
    assert len(training) + len(dev) + len(final) == len(train) + len(test) and set(shown) <= set(dev)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    write(OUT / "train.csv", HEADER, training)
    write(OUT / "dev.csv", HEADER, dev)
    write(OUT / "reveal.csv", [*HEADER, "round"], [(*r, t) for t in sorted(reveal) for r in reveal[t]])
    write(OUT / "final.csv", HEADER, final)
    write(OUT / "val_gate.csv", HEADER, val_gate(val, test))
    count = lambda rows: {"pairs": len(rows), "pos": sum(r[2] for r in rows)}  # noqa: E731
    files = ("train.csv", "dev.csv", "reveal.csv", "final.csv", "val_gate.csv")
    stats = {
        "seed": SEED,
        "rounds": ROUNDS,
        "batch": BATCH,
        "train": count(training),
        "dev": count(dev),
        "reveal": count(shown),
        "final": count(final),
        "inputs": {f"agent_split/{s}.csv": sha(SPLIT / f"{s}.csv") for s in ("train", "val", "test")},
        "outputs": {p: sha(OUT / p) for p in files},
    }
    (OUT / "rsi.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


def decide(gate, t, val_auprc):
    """gate plus round t's decision: kept only if val AUPRC beats every earlier kept state (round 0: always)."""
    assert [g["round"] for g in gate] == list(range(t)), f"gate.json must hold rounds 0..{t - 1}"
    best = max((g["val_auprc"] for g in gate if g["kept"]), default=None)
    return [*gate, {"round": t, "val_auprc": val_auprc, "kept": best is None or val_auprc > best}]


# the gate of 2026-11-26: docs/superpowers/specs/2026-11-26-archive-and-one-gate-design.md
RESAMPLES, CLEARLY_WORSE = 2000, 0.9


def _ap(y, s, order, w):
    """Average precision of scores s over the pairs in `order` (highest score first), pair i counted w[i] times; tied
    scores are one step, as in src.eval.pring_test.metrics."""
    tp = n = before = 0
    total = 0.0
    k = 0
    while k < len(order):
        j = k
        while j < len(order) and s[order[j]] == s[order[k]]:
            n += w[order[j]]
            tp += w[order[j]] * y[order[j]]
            j += 1
        if tp > before:
            total += (tp - before) * tp / n
        before, k = tp, j
    return total / tp if tp else 0.0


def average_precision(y, s):
    return _ap(y, s, sorted(range(len(y)), key=lambda i: -s[i]), [1] * len(y))


def best_kept(gate):
    """The round of the kept state with the highest validation AUPRC (the earliest one on a tie); None if no state."""
    kept = [g for g in gate if g["kept"]]
    return max(kept, key=lambda g: (g["val_auprc"], -g["round"]))["round"] if kept else None


def gate_round(gate, t, y, p, p_best, seed=SEED):
    """gate plus round t's decision under the rule of 2026-11-26: the candidate replaces the state in use unless it
    is clearly worse than the best kept state. y: the validation labels; p: the candidate's scores; p_best: the scores
    of state best_kept(gate) on the same pairs in the same order (None for round 0, which is always kept). Clearly
    worse: the lower AUPRC in at least CLEARLY_WORSE of RESAMPLES paired bootstrap resamples of the pairs. Stdlib
    only (H20's checkout venv has no numpy); about 30 s for 9,400 pairs."""
    assert [g["round"] for g in gate] == list(range(t)), f"gate.json must hold rounds 0..{t - 1}"
    assert (p_best is None) == (t == 0) and (t == 0 or len(y) == len(p) == len(p_best)), "scores of the same pairs"
    entry = {"round": t, "val_auprc": average_precision(y, p), "kept": True}
    if t:
        rng, n = random.Random(seed), len(y)
        orders = [sorted(range(n), key=lambda i: -s[i]) for s in (p, p_best)]
        worse = 0
        for _ in range(RESAMPLES):
            w = [0] * n
            for i in rng.choices(range(n), k=n):
                w[i] += 1
            worse += _ap(y, p, orders[0], w) < _ap(y, p_best, orders[1], w)
        share = worse / RESAMPLES
        entry |= {"against": best_kept(gate), "worse_share": share, "kept": share < CLEARLY_WORSE}
    return [*gate, entry]


if __name__ == "__main__":
    main()
