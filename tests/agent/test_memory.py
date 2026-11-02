from src.agent.memory import MAX_ACTIVE, Cases, Lessons

CASES = [
    {"round": 1, "a": "A", "b": "B", "result": "hit"},
    {"round": 2, "a": "C", "b": "D", "result": "not detected"},
]
PROTEINS = {"A": {"gene": "GA", "families": "kinase"}, "B": {"gene": "GB"}, "C": {"gene": "GC"}, "D": {"gene": "GD"}}


def test_lessons_add_revise_retire_and_reject():
    book = Lessons()
    report = book.apply(
        [
            {"op": "add", "text": "kinases pair", "scope": "kinase pairs", "supporting": ["case:B-A"]},
            {"op": "add", "text": "x", "scope": "y", "supporting": ["A-Z"]},  # unknown case
            {"op": "revise", "id": "L1", "contrary": ["C-D"]},
            {"op": "retire", "id": "L9", "reason": "r"},  # unknown lesson
            {"op": "add", "text": "second", "scope": "s"},
            {"op": "retire", "id": "L2", "reason": "contradicted"},
            {"op": "revise", "id": "L2", "text": "late"},  # retired lessons cannot change
        ],
        {"A-B", "C-D"},
        t=3,
    )
    assert [line.split(": ", 1)[1].split(":")[0] for line in report] == [
        "added L1",
        "rejected",
        "revised L1",
        "rejected",
        "added L2",
        "retired L2",
        "rejected",
    ]
    (l1,) = book.active()
    assert l1["supporting"] == ["A-B"] and l1["contrary"] == ["C-D"]
    assert l1["history"] == [{"round": 3, "op": "revise", "before": {"contrary": []}}]
    assert book.all[1]["retired"] == {"round": 3, "reason": "contradicted"}
    assert "L1" in book.render() and "L2" not in book.render()


def test_active_lesson_cap():
    book = Lessons()
    report = book.apply([{"op": "add", "text": f"t{i}", "scope": "s"} for i in range(MAX_ACTIVE + 1)], set(), t=1)
    assert len(book.active()) == MAX_ACTIVE and "rejected" in report[-1]
    report = book.apply(
        [{"op": "retire", "id": "L1", "reason": "r"}, {"op": "add", "text": "t", "scope": "s"}], set(), 2
    )
    assert report[1].endswith(f"added L{MAX_ACTIVE + 1}")


def test_cases_filters_and_homolog_reach():
    cases = Cases(CASES, PROTEINS, {"X": [("A", "38.0")]})
    assert [c["b"] for c in cases.find(query="KINASE")] == ["B"]
    assert [(c["a"], c["via"]) for c in cases.find(similar_to="X")] == [("A", "A 38.0%")]
    assert cases.find(similar_to="C")[0]["via"] == "self"
    assert [c["a"] for c in cases.find(result="not detected", round=2)] == ["C"]
    assert cases.find(protein="Z") == [] and cases.find(pair="A-B")[0]["result"] == "hit"
