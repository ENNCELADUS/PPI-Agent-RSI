import pytest

from src.data.builders.build_mus22t_rsi import decide, kept


def test_gate_keeps_a_round_only_if_val_beats_every_kept_checkpoint():
    gate = []
    for t, v in enumerate([0.70, 0.72, 0.71, 0.72, 0.73]):
        gate = decide(gate, t, v)
    assert [g["kept"] for g in gate] == [True, True, False, False, True]
    assert kept(gate) == [1, 4]  # round 0 is V3, never a reveal round


def test_gate_rejects_rounds_out_of_order():
    with pytest.raises(AssertionError):
        decide([], 1, 0.7)
