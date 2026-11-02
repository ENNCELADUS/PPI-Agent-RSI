from src.eval.pring_test import batches, classification, evaluate, metrics


def test_evaluate_keeps_first_valid_prediction_per_listed_pair():
    batch = [("A", "B", 1), ("A", "C", 0)]
    sub = [
        {"a": "B", "b": "A", "p": 0.9},  # reversed order is the same pair
        {"a": "A", "b": "B", "p": 0.1},  # repeat
        {"a": "A", "b": "D", "p": 0.5},  # not listed
        {"a": "A", "b": "C", "p": 1.5},  # out of range
    ]
    rows, rejected = evaluate(sub, batch)
    assert [r["p"] for r in rows] == [0.9, None] and rejected == 3
    assert evaluate(None, batch)[1] == 0


def test_metrics_count_ties_once():
    assert metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == {"auroc": 1.0, "ap": 1.0}
    # all tied: AUROC 0.5, AP = prevalence
    assert metrics([1, 0, 0, 0], [0.5] * 4) == {"auroc": 0.5, "ap": 0.25}


def test_batches_are_a_fixed_shuffle_of_all_pairs():
    pairs = [(f"P{i}", "Q", i % 2) for i in range(10)]
    a = batches(pairs, 3)
    assert a == batches(list(reversed(pairs)), 3) and [len(b) for b in a] == [3, 3, 3, 1]
    assert sorted(x for b in a for x in b) == sorted(pairs)


def test_classification_at_max_f1_threshold():
    m = classification([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])  # F1 by threshold: 0.9 .67, 0.6 .5, 0.4 .8, 0.1 .67
    assert m["threshold"] == 0.4 and abs(m["f1"] - 0.8) < 1e-12
    assert (m["accuracy"], m["precision"], m["recall"], m["specificity"]) == (0.75, 2 / 3, 1.0, 0.5)
    assert abs(m["mcc"] - 2 / 12**0.5) < 1e-12
    assert abs(m["brier"] - (0.01 + 0.36 + 0.36 + 0.01) / 4) < 1e-12


def test_summarise_without_b0_scores_the_agent_only():
    from src.eval.pring_test import summarise

    rows = [{"a": "A", "b": "B", "label": 1, "p": 0.9}, {"a": "A", "b": "C", "label": 0, "p": None}]
    result = {"rejected": 0, "submission": [], "tool_calls": 3, "stats": None}
    s = summarise(rows, [result], predictor=None)
    assert s["coverage"] == 0.5 and set(s["agent"]) >= {"auroc", "f1"} and "b0" not in s


def test_summarise_scores_the_predictor_on_the_same_pairs(monkeypatch):
    from src.eval import pring_test

    rows = [
        {"a": "A", "b": x, "label": y, "p": p}
        for x, y, p in (("B", 1, 0.9), ("C", 0, 0.2), ("D", 1, 0.1), ("E", 0, 0.6))
    ]
    monkeypatch.setattr(pring_test, "predictor_p", lambda name, pairs: {k: 0.5 for k in pairs} | {("A", "B"): 0.7})
    result = {"rejected": 0, "submission": [], "tool_calls": 2, "stats": None, "rows": rows, "predictor_pairs": 3}
    s = pring_test.summarise(rows, [result], predictor="v3")
    assert s["predictor_coverage"] == 0.75 and s["v3"]["auroc"] == 0.75  # B beats C, E; D ties both
    assert s["agent_minus_v3"]["auroc"][0] == s["agent"]["auroc"] - s["v3"]["auroc"]  # 4 pairs: no interval


def test_paired_bootstrap_of_identical_scores_is_zero():
    from src.eval.pring_test import paired_bootstrap

    y, p = [1, 0, 1, 0, 1, 0] * 10, [0.9, 0.1, 0.8, 0.3, 0.4, 0.6] * 10  # no one-class resample
    assert paired_bootstrap(y, p, p, n=50) == {"auroc": [0.0, 0.0, 0.0], "ap": [0.0, 0.0, 0.0]}
