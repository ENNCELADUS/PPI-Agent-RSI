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


@pytest.fixture
def bench(tmp_path, monkeypatch):
    rsi = tmp_path / "rsi"
    rsi.mkdir()
    dev = [(f"D{i}", f"E{i}", i % 2) for i in range(8)]
    write(rsi / "dev.csv", HEADER, dev)
    write(rsi / "final.csv", HEADER, [("F0", "G0", 1), ("F1", "G1", 0)])
    write(rsi / "train.csv", HEADER, [("T0", "U0", 1), ("T1", "U1", 0)])
    write(rsi / "reveal.csv", [*HEADER, "round"], [(*r, 1 + i // 2) for i, r in enumerate(dev[:6])])
    (tmp_path / "agent_split").mkdir()
    write(tmp_path / "agent_split" / "val.csv", HEADER, [("V0", "W0", 1)])
    b = {"dev": rsi / "dev.csv", "random": rsi / "reveal.csv", "final": rsi / "final.csv", "batch": 2, "work": rsi}
    monkeypatch.setitem(dl_rounds.BENCH, "mus22t", {**b, "predictor": tmp_path / "predictor"})
    monkeypatch.setattr(dl_rounds, "MUS22T", tmp_path)
    return rsi, dev


def checkpoint(path, logit):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write("a\tb\tlogit\n" + "".join(f"D{i}\tE{i}\t{logit(i)}\n" for i in range(8)))


def train_pairs(rsi, rule, t):
    return {
        (r["uniprotID_A"], r["uniprotID_B"]) for r in csv.DictReader(open(rsi / f"al_{rule}/round{t:02d}/train.csv"))
    }


def test_greedy_rounds_drop_a_gated_round_from_training_but_never_repick_it(bench):
    rsi, _ = bench
    dl_rounds.write_al("mus22t", "greedy", 0)
    assert train_pairs(rsi, "greedy", 0) == {("T0", "U0"), ("T1", "U1")}
    assert dl_rounds.write_gate("mus22t", "greedy", 0, 0.70) == "keep"
    checkpoint(dl_rounds.checkpoint_path("mus22t", "greedy", 0), lambda i: i)  # round 1 takes D7, D6
    dl_rounds.write_al("mus22t", "greedy", 1)
    assert dl_rounds.write_gate("mus22t", "greedy", 1, 0.69) == "drop"
    checkpoint(dl_rounds.checkpoint_path("mus22t", "greedy", 1), lambda i: i)
    dl_rounds.write_al("mus22t", "greedy", 2)
    batches = dl_rounds.rows_by_round(rsi / "al_greedy" / "batches.csv")
    assert [r[:2] for r in batches[1]] == [("D7", "E7"), ("D6", "E6")]
    assert [r[:2] for r in batches[2]] == [("D5", "E5"), ("D4", "E4")]
    assert train_pairs(rsi, "greedy", 2) == {("T0", "U0"), ("T1", "U1"), ("D5", "E5"), ("D4", "E4")}
    dl_rounds.write_al("mus22t", "greedy", 2)  # a rerun of round 2 replaces its batch, not appends
    assert len(dl_rounds.rows_by_round(rsi / "al_greedy" / "batches.csv")[2]) == 2
    assert [g["kept"] for g in json.loads((rsi / "al_greedy" / "gate.json").read_text())] == [True, False]


def test_random_rounds_use_the_fixed_batches(bench):
    rsi, dev = bench
    dl_rounds.write_al("mus22t", "random", 0)
    dl_rounds.write_gate("mus22t", "random", 0, 0.70)
    dl_rounds.write_al("mus22t", "random", 1)
    assert dl_rounds.rows_by_round(rsi / "al_random" / "batches.csv")[1] == dev[:2]
    with pytest.raises(AssertionError):
        dl_rounds.write_al("mus22t", "random", 3)  # rounds run in order
