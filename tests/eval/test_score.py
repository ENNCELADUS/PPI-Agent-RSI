from src.eval.score import score

LABELS = {("a", "b"): 1, ("a", "c"): 0, ("b", "c"): 1}


def test_perfect_ranking_scores_one():
    s = score(LABELS, [("a", "b"), ("b", "c")], ns=(2,))
    assert s["precision@2"] == 1.0 and s["recall@2"] == 1.0 and s["enhancement@2"] == 1.5


def test_invalid_and_repeated_rows_count_as_misses():
    s = score(LABELS, [("a", "b"), ("a", "b"), ("x", "y")], ns=(2,))
    assert s["precision@2"] == 0.5 and s["invalid_rate"] == 2 / 3


def test_short_ranking_counts_missing_rows_as_misses():
    assert score(LABELS, [("a", "b")], ns=(2,))["precision@2"] == 0.5
