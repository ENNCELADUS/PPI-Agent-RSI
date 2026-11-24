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
