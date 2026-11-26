"""DL predictor rounds on both benchmarks: an acquisition rule picks each round's batch from the development set, the
candidate predictor is retrained from scratch on the training set plus every revealed label, and the validation gate
decides whether the candidate replaces the predictor in use. The baseline arm (update target: predictor weights)
under two rules. Design: docs/superpowers/specs/2026-11-26-archive-and-one-gate-design.md (runs before 2026-11-26
trained on the kept rounds only and gated on one AUPRC number, build_mus22t_rsi.decide).

Run (train_mus22t_al.sh, train_pring_al.sh on H20; stdlib only: H20's checkout venv has no numpy):
    python -m src.data.builders.dl_rounds al <data> <rule> <t>       # round t's batch and candidate t's training data
    python -m src.data.builders.dl_rounds gate <data> <rule> <t>     # keep or drop candidate t; writes round<t> files
    python -m src.data.builders.dl_rounds pairs pring                # node-id pairs every candidate scores
    python -m src.data.builders.dl_rounds logits pring <rule> <t> <merged.tsv> <merged_val.tsv>  # -> candidate<t>

data: mus22t (rounds of 100 from rsi/dev.csv) or pring (rounds of 200 from hidden/dev.csv).
rule (user decision 2026-11-24):
- random: the fixed batches of the baseline rule (rsi/reveal.csv, hidden/dev_random.csv), the same for every arm.
- greedy: the BATCH development pairs with the highest logit of the predictor in use after round t - 1 that no
  earlier round revealed; ties broken by pair. Every revealed pair stays revealed, so no later round picks it again.

Candidate 0 (trained on the training set alone) is shared by both rules and always kept: on 22T V3 retrained on
rsi/train.csv, on PRING the frozen B0 (trained and selected on the node-disjoint training/validation split).
Candidate t = 1..10: trained from scratch on the training set plus all pairs revealed in rounds 1..t, kept or
dropped. Revealed labels are never dropped. Gate (build_mus22t_rsi.gate_round): candidate t replaces the predictor
in use unless it is clearly worse than the best kept predictor (best_kept) in a paired bootstrap on all validation
pairs (22T agent_split/val.csv, PRING hidden/val.csv). The predictor in use after round t is candidate t if kept,
else the predictor in use after round t - 1.

Files: <work>/al_<rule>/ = gate.json, batches.csv (uniprotID_A, uniprotID_B, isInteraction, round: every revealed
pair), round<t>/ = candidate t's training data (22T: train.csv = training set + rounds 1..t, val.csv = agent_split's,
test.csv = development + final; PRING: revealed_pairs.tsv, node_a, node_b, label, for the B0 trainer's
data.extra_training_pairs). <predictor>/al_<rule>/ (a, b, logit, accessions): candidate<t>.tsv.gz = candidate t's
logits on the development and final pairs, val/candidate<t>.tsv.gz on the validation pairs (never deleted);
round<t>.tsv.gz and val/round<t>.tsv.gz = the same of the predictor in use after round t (round<t>.tsv.gz is written
last: it marks the round done). work = data/MUS22T/rsi or data/PRING/rsi; predictor = data/<benchmark>/predictor.
"""

import csv
import gzip
import json
import shutil
import sys

from src.data.builders.build_mus22t_rsi import HEADER, best_kept, gate_round, read, write
from src.data.ids import pair
from src.data.paths import HIDDEN, MUS22T, PRING, PRING_RAW

ROUNDS = 10
BENCH = {
    "mus22t": {
        "dev": MUS22T / "rsi" / "dev.csv",
        "random": MUS22T / "rsi" / "reveal.csv",
        "final": MUS22T / "rsi" / "final.csv",
        "val": MUS22T / "agent_split" / "val.csv",
        "batch": 100,
        "work": MUS22T / "rsi",
        "predictor": MUS22T / "predictor",
    },
    "pring": {
        "dev": HIDDEN / "dev.csv",
        "random": HIDDEN / "dev_random.csv",
        "final": HIDDEN / "final_test.csv",
        "val": HIDDEN / "val.csv",
        "batch": 200,
        "work": PRING / "rsi",
        "predictor": PRING / "predictor",
    },
}
RULES = ("random", "greedy")


def work(data, rule):
    return BENCH[data]["work"] / f"al_{rule}"


def _logits_path(data, rule, name, val):
    d = BENCH[data]["predictor"] / f"al_{rule}"
    return (d / "val" if val else d) / f"{name}.tsv.gz"


def checkpoint_path(data, rule, t, val=False):
    """The logits of the predictor in use after round t (val: on the validation pairs)."""
    return _logits_path(data, rule, f"round{t:02d}", val)


def candidate_path(data, rule, t, val=False):
    """The logits of candidate t, kept or dropped (val: on the validation pairs)."""
    return _logits_path(data, rule, f"candidate{t:02d}", val)


def logits(path):
    """{canonical pair: logit} of a logits file (a, b, logit)."""
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
    """Round t's batch (appended to batches.csv; a rerun of round t replaces it) and candidate t's training data."""
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
    rows = [(*r, s) for s in sorted(revealed) for r in revealed[s]]
    write(path, [*HEADER, "round"], rows)
    extra = [r[:3] for r in rows]  # every revealed pair, kept round or not
    out = d / f"round{t:02d}"
    out.mkdir(exist_ok=True)
    if data == "mus22t":
        rsi = BENCH[data]["work"]
        write(out / "train.csv", HEADER, read(rsi / "train.csv") + extra)
        shutil.copy(BENCH[data]["val"], out / "val.csv")
        write(out / "test.csv", HEADER, read(rsi / "dev.csv") + read(rsi / "final.csv"))
    else:
        node = nodes()
        with open(out / "revealed_pairs.tsv", "w") as f:
            f.writelines(f"{node[a]}\t{node[b]}\t{y}\n" for a, b, y in extra)
    print(f"round {t}: {len(revealed.get(t, []))} pairs revealed, {len(extra)} in the training data")


def write_gate(data, rule, t):
    """Gate candidate t on its validation logits against the best kept predictor's; then the round<t> files of the
    predictor in use (round<t>.tsv.gz last). Returns keep or drop."""
    gate = read_gate(data, rule)[:t]
    rows = read(BENCH[data]["val"])
    y = {pair(a, b): label for a, b, label in rows}
    assert len(y) == len(rows), "validation pairs repeat"
    order = sorted(y)

    def scores(path):
        p = logits(path)
        assert set(p) == set(y), f"{path}: pairs differ from the validation pairs"
        return [p[k] for k in order]

    best = best_kept(gate)
    p_best = None if best is None else scores(candidate_path(data, rule, best, val=True))
    gate = gate_round(gate, t, [y[k] for k in order], scores(candidate_path(data, rule, t, val=True)), p_best)
    work(data, rule).mkdir(parents=True, exist_ok=True)
    (work(data, rule) / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    kept = gate[-1]["kept"]
    for val in (True, False):
        src = candidate_path(data, rule, t, val) if kept else checkpoint_path(data, rule, t - 1, val)
        shutil.copy(src, checkpoint_path(data, rule, t, val))
    return "keep" if kept else "drop"


def write_pring_pairs():
    """rsi/score_pairs.tsv (the development and final pairs) and rsi/val_pairs.tsv (the validation pairs): node_a,
    node_b, no header, no label."""
    node = nodes()
    BENCH["pring"]["work"].mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("score", read(BENCH["pring"]["dev"]) + read(BENCH["pring"]["final"])),
        ("val", read(BENCH["pring"]["val"])),
    ):
        with open(BENCH["pring"]["work"] / f"{name}_pairs.tsv", "w") as f:
            f.writelines(f"{node[a]}\t{node[b]}\n" for a, b, _ in rows)
        print(f"{name}: {len(rows)} pairs")


def write_pring_logits(rule, t, merged, merged_val):
    """Candidate t's files from src.predictor.score's merged outputs (node_a, node_b, logit[, label]): accessions."""
    acc = {v: k for k, v in nodes().items()}
    for src, sets, val in (
        (merged, ("dev", "final"), False),
        (merged_val, ("val",), True),
    ):
        rows = [
            (*pair(acc[r["node_a"]], acc[r["node_b"]]), r["logit"]) for r in csv.DictReader(open(src), delimiter="\t")
        ]
        want = {pair(a, b) for s in sets for a, b, _ in read(BENCH["pring"][s])}
        assert len(rows) == len(want) and {r[:2] for r in rows} == want, f"{src}: scored pairs differ from {sets}"
        path = candidate_path("pring", rule, t, val)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            f.write("a\tb\tlogit\n" + "".join(f"{a}\t{b}\t{x}\n" for a, b, x in sorted(rows)))


if __name__ == "__main__":
    cmd, data = sys.argv[1], sys.argv[2]
    assert data in BENCH, data
    if cmd == "pairs":
        write_pring_pairs()
    elif cmd == "logits":
        write_pring_logits(sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
    else:
        rule, t = sys.argv[3], int(sys.argv[4])
        assert rule in RULES, rule
        if cmd == "al":
            write_al(data, rule, t)
        else:
            print(write_gate(data, rule, t))
