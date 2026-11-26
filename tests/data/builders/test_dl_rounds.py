import csv
import gzip
import json

import pytest

from src.data.builders import dl_rounds
from src.data.builders.build_mus22t_rsi import HEADER, write


def test_greedy_takes_the_top_unrevealed_pairs_and_ignores_labels():
    dev = [(f"P{i}", f"Q{i}", i % 2) for i in range(10)]
    scores = {(f"P{i}", f"Q{i}"): float(i % 4) for i in range(10)}  # ties broken by pair
    got = dl_rounds.greedy(dev, [("P7", "Q7", 1)], scores, 3)
    assert [r[:2] for r in got] == [("P3", "Q3"), ("P2", "Q2"), ("P6", "Q6")]
    relabelled = [(a, b, 1 - y) for a, b, y in dev]
    assert [r[:2] for r in dl_rounds.greedy(relabelled, [("P7", "Q7", 0)], scores, 3)] == [r[:2] for r in got]


VAL = [(f"V{i}", f"W{i}", i % 2) for i in range(40)]


@pytest.fixture
def bench(tmp_path, monkeypatch):
    rsi = tmp_path / "rsi"
    rsi.mkdir()
    dev = [(f"D{i}", f"E{i}", i % 2) for i in range(8)]
    write(rsi / "dev.csv", HEADER, dev)
    write(rsi / "final.csv", HEADER, [("F0", "G0", 1), ("F1", "G1", 0)])
    write(rsi / "train.csv", HEADER, [("T0", "U0", 1), ("T1", "U1", 0)])
    write(rsi / "reveal.csv", [*HEADER, "round"], [(*r, 1 + i // 2) for i, r in enumerate(dev[:6])])
    write(tmp_path / "val.csv", HEADER, VAL)
    b = {"dev": rsi / "dev.csv", "random": rsi / "reveal.csv", "final": rsi / "final.csv", "val": tmp_path / "val.csv"}
    b |= {"batch": 2, "work": rsi, "predictor": tmp_path / "predictor"}
    monkeypatch.setitem(dl_rounds.BENCH, "mus22t", b)
    return rsi, dev


def logits_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write("a\tb\tlogit\n" + "".join(f"{b}\t{a}\t{x}\n" for a, b, x in rows))  # pairs in either order


def candidate(rule, t, dev_logit, val_good):
    """Candidate t's files: dev logits dev_logit(i); validation logits that rank the pairs right (val_good) or not."""
    logits_file(dl_rounds.candidate_path("mus22t", rule, t), [(f"D{i}", f"E{i}", dev_logit(i)) for i in range(8)])
    val = [(a, b, y if val_good else (i * 7) % 5) for i, (a, b, y) in enumerate(VAL)]
    logits_file(dl_rounds.candidate_path("mus22t", rule, t, val=True), val)


def train_pairs(rsi, rule, t):
    return {
        (r["uniprotID_A"], r["uniprotID_B"]) for r in csv.DictReader(open(rsi / f"al_{rule}/round{t:02d}/train.csv"))
    }


def test_greedy_rounds_train_on_a_dropped_round_and_never_repick_it(bench):
    rsi, _ = bench
    dl_rounds.write_al("mus22t", "greedy", 0)
    assert train_pairs(rsi, "greedy", 0) == {("T0", "U0"), ("T1", "U1")}
    candidate("greedy", 0, lambda i: i, True)  # round 1 takes D7, D6
    assert dl_rounds.write_gate("mus22t", "greedy", 0) == "keep"
    dl_rounds.write_al("mus22t", "greedy", 1)
    candidate("greedy", 1, lambda i: -i, False)  # clearly worse on validation; would take D0, D1
    assert dl_rounds.write_gate("mus22t", "greedy", 1) == "drop"
    in_use = dl_rounds.checkpoint_path
    assert dl_rounds.logits(in_use("mus22t", "greedy", 1)) == dl_rounds.logits(in_use("mus22t", "greedy", 0))
    assert dl_rounds.candidate_path("mus22t", "greedy", 1).exists()  # a dropped candidate's scores stay
    dl_rounds.write_al("mus22t", "greedy", 2)
    batches = dl_rounds.rows_by_round(rsi / "al_greedy" / "batches.csv")
    assert [r[:2] for r in batches[1]] == [("D7", "E7"), ("D6", "E6")]
    assert [r[:2] for r in batches[2]] == [("D5", "E5"), ("D4", "E4")]  # from the state in use, round 1 not repicked
    revealed = {("D7", "E7"), ("D6", "E6"), ("D5", "E5"), ("D4", "E4")}
    assert train_pairs(rsi, "greedy", 2) == {("T0", "U0"), ("T1", "U1")} | revealed  # the dropped round stays
    dl_rounds.write_al("mus22t", "greedy", 2)  # a rerun of round 2 replaces its batch, not appends
    assert len(dl_rounds.rows_by_round(rsi / "al_greedy" / "batches.csv")[2]) == 2
    gate = json.loads((rsi / "al_greedy" / "gate.json").read_text())
    assert [g["kept"] for g in gate] == [True, False] and gate[1]["against"] == 0 and gate[0]["val_auprc"] == 1.0


def test_gate_keeps_a_candidate_that_is_not_clearly_worse(bench):
    dl_rounds.write_al("mus22t", "random", 0)
    candidate("random", 0, lambda i: i, True)
    dl_rounds.write_gate("mus22t", "random", 0)
    dl_rounds.write_al("mus22t", "random", 1)
    candidate("random", 1, lambda i: -i, True)  # the same validation scores: kept, and now in use
    assert dl_rounds.write_gate("mus22t", "random", 1) == "keep"
    in_use = dl_rounds.logits(dl_rounds.checkpoint_path("mus22t", "random", 1))
    assert in_use == dl_rounds.logits(dl_rounds.candidate_path("mus22t", "random", 1))


def test_gate_fails_closed_on_other_validation_pairs(bench):
    dl_rounds.write_al("mus22t", "random", 0)
    logits_file(dl_rounds.candidate_path("mus22t", "random", 0, val=True), [(a, b, y) for a, b, y in VAL[1:]])
    with pytest.raises(AssertionError):
        dl_rounds.write_gate("mus22t", "random", 0)


def test_random_rounds_use_the_fixed_batches(bench):
    rsi, dev = bench
    dl_rounds.write_al("mus22t", "random", 0)
    candidate("random", 0, lambda i: i, True)
    dl_rounds.write_gate("mus22t", "random", 0)
    dl_rounds.write_al("mus22t", "random", 1)
    assert dl_rounds.rows_by_round(rsi / "al_random" / "batches.csv")[1] == dev[:2]
    with pytest.raises(AssertionError):
        dl_rounds.write_al("mus22t", "random", 3)  # rounds run in order
