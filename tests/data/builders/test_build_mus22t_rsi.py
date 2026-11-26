import pytest

from src.data.builders.build_mus22t_rsi import decide, draw, move


def test_move_takes_half_of_each_label_and_is_seeded():
    train = [(f"P{i}", f"Q{i}", i % 2) for i in range(40)]
    rest, moved = move(train, n=10)
    assert len(moved) == 10 and sum(r[2] for r in moved) == 5
    assert sorted(rest + moved) == sorted(train) and move(train, n=10) == (rest, moved)


def test_draw_gives_disjoint_batches_from_the_development_set_without_reading_labels():
    dev = [(f"P{i}", f"Q{i}", i % 2) for i in range(50)]
    got = draw(dev, rounds=3, batch=5)
    flat = [r for t in (1, 2, 3) for r in got[t]]
    assert len(set(flat)) == 15 and set(flat) <= set(dev)
    relabelled = [(a, b, 1 - y) for a, b, y in dev]
    assert [r[:2] for t in (1, 2, 3) for r in draw(relabelled, rounds=3, batch=5)[t]] == [r[:2] for r in flat]
    with pytest.raises(AssertionError):
        draw(dev, rounds=11, batch=5)  # the development set must cover the label budget


def test_gate_keeps_a_round_only_if_val_beats_every_kept_checkpoint():
    gate = []
    for t, v in enumerate([0.70, 0.72, 0.71, 0.72, 0.73]):
        gate = decide(gate, t, v)
    assert [g["kept"] for g in gate] == [True, True, False, False, True]


def test_gate_rejects_rounds_out_of_order():
    with pytest.raises(AssertionError):
        decide([], 1, 0.7)


def test_average_precision_matches_the_evaluator_with_tied_scores():
    import random

    from src.data.builders.build_mus22t_rsi import average_precision
    from src.eval.pring_test import metrics

    rng = random.Random(1)
    y = [rng.randrange(2) for _ in range(300)]
    s = [round(rng.random() * 0.3 + 0.2 * label, 1) for label in y]  # few distinct scores: many ties
    assert average_precision(y, s) == pytest.approx(metrics(y, s)["ap"])


def test_new_gate_drops_a_candidate_only_if_it_is_clearly_worse_than_the_best_kept_state():
    import random

    from src.data.builders.build_mus22t_rsi import best_kept, gate_round

    rng = random.Random(0)
    y = [i % 2 for i in range(400)]
    good = [0.6 * label + rng.random() for label in y]
    close = [s + rng.gauss(0, 0.05) for s in good]  # the same system with a little noise
    bad = [rng.random() for _ in y]  # no signal
    gate = gate_round([], 0, y, good, None)
    assert gate[0]["kept"] and best_kept(gate) == 0
    gate = gate_round(gate, 1, y, close, good)
    assert gate[1]["kept"] and gate[1]["against"] == 0 and gate[1]["worse_share"] < 0.9
    gate = gate_round(gate, 2, y, bad, good if best_kept(gate) == 0 else close)
    assert not gate[2]["kept"] and gate[2]["worse_share"] >= 0.9
    assert gate_round(gate[:2], 2, y, bad, good if best_kept(gate[:2]) == 0 else close) == gate  # seeded
    with pytest.raises(AssertionError):
        gate_round(gate, 3, y, good, None)  # every round after 0 needs the best kept state's scores
