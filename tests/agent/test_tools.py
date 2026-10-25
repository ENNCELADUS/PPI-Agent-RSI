import gzip
import json

import pytest

from src.agent.tools import Sandbox, ToolError, WaveTools, specs

ROWS = [("A", 1), ("B", 1), ("C", 1), ("D", 1), ("K1", 0), ("K2", 0)]
# A-C and B-C tie on p: ties go by pair id
P = {("A", "B"): 0.9, ("A", "C"): 0.5, ("B", "C"): 0.5, ("A", "D"): 0.4, ("B", "D"): 0.2, ("C", "D"): 0.1}
FIELDS = "function\tsubcellular_location\ttissue_specificity\tgo_process\tgo_component\tgo_function\tkeywords"


@pytest.fixture
def sandbox(tmp_path):
    head = "accession\tgene\tprotein_name\tcandidate\tsequence"
    body = [f"{x}\tG{x}\tname {x}\t{c}\tMKV" for x, c in ROWS]
    (tmp_path / "proteins.tsv").write_text("\n".join([head, *body]) + "\n")
    ann = [f"{x}\t\tMembrane\t\t\t\t\tKinase\tfam\tPF00069\tIPR1" for x, _ in ROWS]
    head = f"accession\t{FIELDS}\tfamilies\tpfam\tinterpro"
    (tmp_path / "annotations.tsv").write_text("\n".join([head, *ann]) + "\n")
    (tmp_path / "kb.csv").write_text("uniprotID_A,uniprotID_B,isInteraction,evidence\nK1,K2,1,pring\n")
    (tmp_path / "homologs.tsv").write_text("query\ttarget\tpident\tevalue\tbits\nA\tK1\t35.0\t1e-9\t59\n")
    with gzip.open(tmp_path / "scores.tsv.gz", "wt") as f:
        f.write("a\tb\tp\n" + "".join(f"{a}\t{b}\t{p}\n" for (a, b), p in P.items()))
    return Sandbox(str(tmp_path))


def table(text):
    lines = text.splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def pairs(text):
    return [(r["rank"], r["a"], r["b"]) for r in table(text)]


def test_ranking_breaks_ties_by_pair_id_and_pages(sandbox, tmp_path):
    w = WaveTools(sandbox, tmp_path / "w", set(), [])
    top = pairs(w.call("top_candidates", {"offset": 0, "limit": 6}))
    assert top == [("1", "A", "B"), ("2", "A", "C"), ("3", "B", "C"), ("4", "A", "D"), ("5", "B", "D"), ("6", "C", "D")]
    assert pairs(w.call("top_candidates", {"offset": 2, "limit": 2})) == top[2:4]


def test_tested_pairs_leave_the_ranking_and_ranks_close_up(sandbox, tmp_path):
    w = WaveTools(sandbox, tmp_path / "w", {("A", "B"), ("B", "C")}, [])
    assert pairs(w.call("top_candidates", {"offset": 0, "limit": 10})) == [
        ("1", "A", "C"),
        ("2", "A", "D"),
        ("3", "B", "D"),
        ("4", "C", "D"),
    ]
    assert pairs(w.call("top_candidates", {"offset": 1, "limit": 1})) == [("2", "A", "D")]
    assert "already tested" in w.call("predict_ppi", {"a": "C", "b": "B"})
    assert "rank=2 of untested" in w.call("predict_ppi", {"a": "D", "b": "A"})
    assert pairs(w.call("top_for_protein", {"id": "B", "n": 5})) == [("3", "B", "D")]


def test_bad_ids_and_kb_proteins_are_errors(sandbox, tmp_path):
    w = WaveTools(sandbox, tmp_path / "w", set(), [])
    for args in ({"a": "A", "b": "Z"}, {"a": "A", "b": "K1"}, {"a": "A", "b": "A"}):
        with pytest.raises(ToolError):
            w.call("predict_ppi", args)


def test_no_tool_output_carries_a_label_field(sandbox, tmp_path):
    w = WaveTools(sandbox, tmp_path / "w", set(), [])
    outputs = [
        w.call("top_candidates", {"offset": 0, "limit": 6}),
        w.call("protein_info", {"ids": ["A", "K1"], "with_sequence": True}),
        w.call("search_proteins", {"text": "pf00069", "scope": "candidates", "limit": 10}),
        w.call("similar_proteins", {"id": "A"}),
    ]
    assert not any("label" in o or "isInteraction" in o for o in outputs)
    assert "label" in w.call("kb_partners", {"id": "K1"})  # KB labels are the agent's by design


def test_search_scopes(sandbox, tmp_path):
    w = WaveTools(sandbox, tmp_path / "w", set(), [])

    def ids(scope):
        text = w.call("search_proteins", {"text": "kinase", "scope": scope, "limit": 9})
        return {r["id"] for r in table(text.split("\n", 1)[1])}

    assert ids("candidates") == {"A", "B", "C", "D"} and ids("kb") == {"K1", "K2"}


def test_memory_filters(sandbox, tmp_path):
    facts = [
        {"a": "A", "b": "B", "wave": 1, "result": "hit"},
        {"a": "C", "b": "D", "wave": 1, "result": "not detected"},
    ]
    w = WaveTools(sandbox, tmp_path / "w", {("A", "B"), ("C", "D")}, facts)
    assert "2 tests" in w.call("memory", {})
    assert "1 tests" in w.call("memory", {"hits_only": True})
    assert "C\tD" in w.call("memory", {"protein": "D"}) and "A\tB" not in w.call("memory", {"protein": "D"})


def test_cap_leaves_only_submit_and_submit_is_final(sandbox, tmp_path):
    w = WaveTools(sandbox, tmp_path / "w", set(), [], cap=1)
    w.call("memory", {})
    with pytest.raises(ToolError, match="submit_batch now"):
        w.call("memory", {})
    batch = [{"a": "A", "b": "B", "ranking_score": 0.9, "uncertainty": 0.1, "evidence_ids": [], "rationale": "x"}]
    w.call("submit_batch", {"pairs": batch})
    with pytest.raises(ToolError, match="already submitted"):
        w.call("submit_batch", {"pairs": batch})
    assert json.loads((tmp_path / "w" / "submission.json").read_text()) == batch
    log = [json.loads(line) for line in open(tmp_path / "w" / "tool_calls.jsonl")]
    assert [(r["tool"], r["ok"]) for r in log] == [
        ("memory", True),
        ("memory", False),
        ("submit_batch", True),
        ("submit_batch", False),
    ]


def test_specs_mark_only_submit_as_terminating():
    assert [s["name"] for s in specs() if s["terminate"]] == ["submit_batch"]
