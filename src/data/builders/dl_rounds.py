"""DL predictor rounds on both benchmarks: an acquisition rule picks each round's batch from the development set, the
predictor is retrained from scratch on the training set plus the kept revealed labels, and the validation gate keeps
or drops the round. The baseline arm (update target: predictor weights) under two rules.

Run (train_mus22t_al.sh, train_pring_al.sh on H20; stdlib only: H20's checkout venv has no numpy):
    python -m src.data.builders.dl_rounds al <data> <rule> <t>                # round t's batch and training data
    python -m src.data.builders.dl_rounds gate <data> <rule> <t> <val_auprc>  # keep or drop round t
    python -m src.data.builders.dl_rounds pairs pring                         # node-id pairs every checkpoint scores
    python -m src.data.builders.dl_rounds logits pring <rule> <t> <merged.tsv>  # src.predictor.score output -> round<t>

data: mus22t (rounds of 100 from rsi/dev.csv, 10,000 pairs) or pring (rounds of 200 from hidden/dev.csv, 20,668).
rule (user decision 2026-11-24):
- random: the fixed batches of the baseline rule (rsi/reveal.csv, hidden/dev_random.csv), the same for every arm.
- greedy: the BATCH development pairs with the highest logit of checkpoint t - 1 that no earlier round revealed;
  ties broken by pair. A dropped round's pairs stay revealed (the budget is spent), so no later round picks them.

Checkpoint 0 (trained on the training set alone) is shared by both rules: on 22T V3 retrained on rsi/train.csv, on
PRING the frozen B0 (it was trained and selected on the node-disjoint training/validation split). Gate: round t is
kept only if its checkpoint's validation AUPRC (of the selected epoch) beats every earlier kept checkpoint
(build_mus22t_rsi.decide); else its pairs leave the training data and checkpoint t is a copy of checkpoint t - 1.

Files: <work>/al_<rule>/ = gate.json, batches.csv (uniprotID_A, uniprotID_B, isInteraction, round: every revealed
pair), round<t>/ = the training data of checkpoint t (22T: train.csv = training set + kept rounds before t + round t,
val.csv = agent_split's, test.csv = development + final; PRING: revealed_pairs.tsv, node_a, node_b, label, for the B0
trainer's data.extra_training_pairs). <predictor>/al_<rule>/round<t>.tsv.gz = checkpoint t's logits (a, b, logit,
accessions) on the development and final pairs. work = data/MUS22T/rsi or data/PRING/rsi; predictor =
data/<benchmark>/predictor.
"""

import csv
import gzip
import json
import shutil
import sys

from src.data.builders.build_mus22t_rsi import HEADER, decide, read, write
from src.data.ids import pair
from src.data.paths import HIDDEN, MUS22T, PRING, PRING_RAW

ROUNDS = 10
BENCH = {
    "mus22t": {
        "dev": MUS22T / "rsi" / "dev.csv",
        "random": MUS22T / "rsi" / "reveal.csv",
        "final": MUS22T / "rsi" / "final.csv",
        "batch": 100,
        "work": MUS22T / "rsi",
        "predictor": MUS22T / "predictor",
    },
    "pring": {
        "dev": HIDDEN / "dev.csv",
        "random": HIDDEN / "dev_random.csv",
        "final": HIDDEN / "final_test.csv",
        "batch": 200,
        "work": PRING / "rsi",
        "predictor": PRING / "predictor",
    },
}
RULES = ("random", "greedy")


def work(data, rule):
    return BENCH[data]["work"] / f"al_{rule}"


def checkpoint_path(data, rule, t):
    return BENCH[data]["predictor"] / f"al_{rule}" / f"round{t:02d}.tsv.gz"


def logits(path):
    """{canonical pair: logit} of a checkpoint file (a, b, logit)."""
    with gzip.open(path, "rt") as f:
        return {pair(r["a"], r["b"]): float(r["logit"]) for r in csv.DictReader(f, delimiter="\t")}


def rows_by_round(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(int(r["round"]), []).append((r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])))
    return out


def greedy(dev, revealed, scores, n):
    """The n development pairs not yet revealed with the highest logit; ties by pair. Labels play no part."""
    seen = {pair(a, b) for a, b, _ in revealed}
    left = [r for r in dev if pair(r[0], r[1]) not in seen]
    return sorted(left, key=lambda r: (-scores[pair(r[0], r[1])], pair(r[0], r[1])))[:n]


def read_gate(data, rule):
    path = work(data, rule) / "gate.json"
    return json.loads(path.read_text()) if path.exists() else []


def batch(data, rule, t, earlier):
    """Round t's pairs under the rule, given {round: rows} of rounds 1..t - 1."""
    if rule == "random":
        return rows_by_round(BENCH[data]["random"])[t]
    scores = logits(checkpoint_path(data, rule, t - 1))
    return greedy(read(BENCH[data]["dev"]), [r for s in earlier.values() for r in s], scores, BENCH[data]["batch"])


def nodes():
    rows = csv.DictReader(open(PRING_RAW / "node_sequences.tsv"), delimiter="\t")
    return {r["uniprot_id"]: r["node_id"] for r in rows}


def write_al(data, rule, t):
    """Round t's batch (appended to batches.csv; a rerun of round t replaces it) and checkpoint t's training data."""
    gate = read_gate(data, rule)[:t]  # a rerun of round t (stopped after its gate) ignores its old decision
    assert [g["round"] for g in gate] == list(range(t)), f"gate.json must hold rounds 0..{t - 1}"
    d = work(data, rule)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "batches.csv"
    revealed = {s: rows for s, rows in (rows_by_round(path) if path.exists() else {}).items() if s < t}
    assert sorted(revealed) == list(range(1, t)), f"batches.csv must hold rounds 1..{t - 1}"
    if t:
        revealed[t] = batch(data, rule, t, revealed)
        dev = {pair(a, b) for a, b, _ in read(BENCH[data]["dev"])}
        got = [pair(a, b) for a, b, _ in revealed[t]]
        assert len(got) == BENCH[data]["batch"] == len(set(got)) and set(got) <= dev, f"round {t}: bad batch"
        assert not set(got) & {pair(a, b) for s in range(1, t) for a, b, _ in revealed[s]}, "a pair revealed twice"
    write(path, [*HEADER, "round"], [(*r, s) for s in sorted(revealed) for r in revealed[s]])
    rounds = [*(g["round"] for g in gate if g["kept"] and g["round"] > 0), *([t] if t else [])]
    extra = [r for s in rounds for r in revealed[s]]
    out = d / f"round{t:02d}"
    out.mkdir(exist_ok=True)
    if data == "mus22t":
        rsi = BENCH[data]["work"]
        write(out / "train.csv", HEADER, read(rsi / "train.csv") + extra)
        shutil.copy(MUS22T / "agent_split" / "val.csv", out / "val.csv")
        write(out / "test.csv", HEADER, read(rsi / "dev.csv") + read(rsi / "final.csv"))
    else:
        node = nodes()
        with open(out / "revealed_pairs.tsv", "w") as f:
            f.writelines(f"{node[a]}\t{node[b]}\t{y}\n" for a, b, y in extra)
    print(f"round {t}: {len(revealed.get(t, []))} pairs revealed, {len(extra)} in the training data")


def write_gate(data, rule, t, val_auprc):
    gate = decide(read_gate(data, rule)[:t], t, val_auprc)
    work(data, rule).mkdir(parents=True, exist_ok=True)
    (work(data, rule) / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    return "keep" if gate[-1]["kept"] else "drop"


def write_pring_pairs():
    """rsi/score_pairs.tsv: node_a, node_b (no header, no label) of the development and final pairs."""
    node = nodes()
    rows = read(BENCH["pring"]["dev"]) + read(BENCH["pring"]["final"])
    BENCH["pring"]["work"].mkdir(parents=True, exist_ok=True)
    with open(BENCH["pring"]["work"] / "score_pairs.tsv", "w") as f:
        f.writelines(f"{node[a]}\t{node[b]}\n" for a, b, _ in rows)
    print(f"{len(rows)} pairs")


def write_pring_logits(rule, t, merged):
    """Checkpoint t's file from src.predictor.score's merged output (node_a, node_b, logit[, label]): accessions."""
    acc = {v: k for k, v in nodes().items()}
    rows = [
        (*pair(acc[r["node_a"]], acc[r["node_b"]]), r["logit"]) for r in csv.DictReader(open(merged), delimiter="\t")
    ]
    want = {pair(a, b) for a, b, _ in read(BENCH["pring"]["dev"]) + read(BENCH["pring"]["final"])}
    assert len(rows) == len(want) and {r[:2] for r in rows} == want, "scored pairs differ from dev + final"
    path = checkpoint_path("pring", rule, t)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write("a\tb\tlogit\n" + "".join(f"{a}\t{b}\t{x}\n" for a, b, x in sorted(rows)))


if __name__ == "__main__":
    cmd, data = sys.argv[1], sys.argv[2]
    assert data in BENCH, data
    if cmd == "pairs":
        write_pring_pairs()
    elif cmd == "logits":
        write_pring_logits(sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        rule, t = sys.argv[3], int(sys.argv[4])
        assert rule in RULES, rule
        if cmd == "al":
            write_al(data, rule, t)
        else:
            print(write_gate(data, rule, t, float(sys.argv[5])))
