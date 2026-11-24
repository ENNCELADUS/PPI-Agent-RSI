import argparse
import gzip
import json

import pytest

from src.agent.tools import ANNOTATION, ReviewTools

PROTEINS = ("A", "B", "C", "D")
ORDER = [("A", "B"), ("A", "C"), ("C", "D"), ("B", "D"), ("A", "D"), ("B", "C")]


def disc_view(root):
    root.mkdir()
    rows = "".join(f"{x}\tG{x}\tn\t1\tMKV\n" for x in PROTEINS)
    (root / "proteins.tsv").write_text("accession\tgene\tprotein_name\tcandidate\tsequence\n" + rows)
    head = "\t".join(("accession", *ANNOTATION))
    (root / "annotations.tsv").write_text(head + "\n" + "".join(x + "\t" * len(ANNOTATION) + "\n" for x in PROTEINS))
    (root / "kb.csv").write_text("uniprotID_A,uniprotID_B,isInteraction,evidence\n")
    for name in ("homologs.tsv", "case_homologs.tsv"):
        (root / name).write_text("query\ttarget\tpident\tevalue\tbits\n")
    with gzip.open(root / "scores.tsv.gz", "wt") as f:
        f.write("a\tb\tp\n" + "".join(f"{a}\t{b}\t0.{i + 1}\n" for i, (a, b) in enumerate(sorted(ORDER))))


def setup(tmp_path, monkeypatch, aps):
    """The round harness on a four-protein sandbox; the validation check of round t returns aps[t]."""
    from src.eval import memory_agent

    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "test_positives.csv").write_text("uniprotID_A,uniprotID_B\nA,B\nC,D\n")
    (hidden / "val_gate.csv").write_text("uniprotID_A,uniprotID_B,isInteraction\nA,B,1\nA,C,0\n")
    (tmp_path / "memory_split.json").write_text('{"outputs": {}}')
    disc_view(tmp_path / "sandbox_disc")
    disc_view(tmp_path / "sandbox_val")
    for name, value in (
        ("DISC", tmp_path / "sandbox_disc"),
        ("VAL", tmp_path / "sandbox_val"),
        ("VAL_GATE", hidden / "val_gate.csv"),
        ("HIDDEN", hidden),
        ("PRING", tmp_path),
    ):
        monkeypatch.setattr(memory_agent, name, value)
    monkeypatch.setattr(memory_agent, "get_pi_version", lambda: "test")
    calls, checked = [], []

    def fake_session(tools, agent_dir, system, message, nudge, **_):
        calls.append((type(tools).__name__, message))
        if isinstance(tools, ReviewTools):
            first = message.split("This round's cases:\n")[1].splitlines()[1].split("\t")[0]
            tools.update_lessons([{"op": "add", "text": f"from {first}", "scope": "s", "supporting": [first]}])
        else:
            picks = [p for p in ORDER if p not in tools.tested][:2]
            row = {"ranking_score": 0.5, "uncertainty": 0.5, "evidence_ids": ["pred"], "rationale": "why"}
            tools.submit_batch([{"a": a, "b": b} | row for a, b in picks])
        return {"submission": tools.submitted, "error": None, "tool_calls": 1}

    def fake_val_check(t, args, records, lessons, run_dir):
        checked.append((t, len(records)))
        return {"ap": aps[t]}

    monkeypatch.setattr(memory_agent, "run_session", fake_session)
    monkeypatch.setattr(memory_agent, "val_check", fake_val_check)
    return memory_agent, calls, checked


def cases_of(run_dir, t):
    return [json.loads(line) for line in open(run_dir / "memory" / f"round{t:02d}" / "cases.jsonl")]


def test_rounds_add_cases_update_lessons_and_resume(tmp_path, monkeypatch):
    memory_agent, calls, checked = setup(tmp_path, monkeypatch, [0.5, 0.6, 0.7])
    args = argparse.Namespace(rounds=2, until=None, model="m", thinking="t")
    run_dir = tmp_path / "run"
    memory_agent.run_rounds(run_dir, args)

    assert [c[0] for c in calls] == ["CaseRoundTools", "ReviewTools", "CaseRoundTools", "ReviewTools"]
    assert "No lessons yet." in calls[0][1] and "L1 (since round 1): from A-B" in calls[2][1]
    cases = cases_of(run_dir, 2)
    assert [(c["a"], c["b"], c["result"], c["round"]) for c in cases] == [
        ("A", "B", "hit", 1),
        ("A", "C", "not detected", 1),
        ("C", "D", "hit", 2),
        ("B", "D", "not detected", 2),
    ]
    assert cases[0]["b0_p"] == 0.1 and cases[0]["rationale"] == "why"
    mem = run_dir / "memory" / "round02"
    assert [x["supporting"] for x in json.loads((mem / "lessons.json").read_text())] == [["A-B"], ["C-D"]]
    assert checked == [(0, 0), (1, 2), (2, 4)]  # the empty memory, then each candidate memory
    assert [g["kept"] for g in json.loads((run_dir / "gate.json").read_text())] == [True, True, True]

    memory_agent.run_rounds(run_dir, args)  # resume replays rounds and memory without any session
    assert len(calls) == 4 and len(checked) == 3


def test_a_dropped_round_leaves_the_memory_but_its_pairs_stay_tested(tmp_path, monkeypatch):
    memory_agent, calls, checked = setup(tmp_path, monkeypatch, [0.5, 0.6, 0.4, 0.7])
    args = argparse.Namespace(rounds=3, until=None, model="m", thinking="t")
    run_dir = tmp_path / "run"
    memory_agent.run_rounds(run_dir, args)

    assert [g["kept"] for g in json.loads((run_dir / "gate.json").read_text())] == [True, True, False, True]
    assert [(c["a"], c["b"]) for c in cases_of(run_dir, 2)] == [("A", "B"), ("A", "C")]  # round 2's cases dropped
    # round 3 could not pick round 2's pairs again, and its memory has no round-2 case or lesson
    assert [(c["a"], c["b"], c["round"]) for c in cases_of(run_dir, 3)] == [
        ("A", "B", 1),
        ("A", "C", 1),
        ("A", "D", 3),
        ("B", "C", 3),
    ]
    lessons = json.loads((run_dir / "memory" / "round03" / "lessons.json").read_text())
    assert [x["supporting"] for x in lessons] == [["A-B"], ["A-D"]]
    assert "from C-D" not in calls[4][1]  # the selector of round 3 does not read the dropped lesson

    memory_agent.run_rounds(run_dir, args)  # resume after a dropped round: no session, no validation check
    assert len(calls) == 6 and len(checked) == 4


def test_rounds_stop_without_the_validation_view(tmp_path, monkeypatch):
    memory_agent, calls, _ = setup(tmp_path, monkeypatch, [0.5])
    monkeypatch.setattr(memory_agent, "VAL", tmp_path / "missing")
    with pytest.raises(AssertionError, match="no validation gate"):
        memory_agent.run_rounds(tmp_path / "run", argparse.Namespace(rounds=1, until=None, model="m", thinking="t"))
    assert not calls


def test_val_check_classifies_the_gate_pairs_with_the_memory(tmp_path, monkeypatch):
    import src.eval.pring_test as pring_test
    from src.agent.memory import Lessons
    from src.eval.memory_agent import val_check  # the real check; setup replaces only the module attribute

    setup(tmp_path, monkeypatch, [])
    seen = []

    def fake_session(tools, agent_dir, system, message, nudge, **_):
        seen.append((type(tools).__name__, system))
        pairs = [(r[0], r[2]) for r in (line.split("\t") for line in message.split("\n")[3:] if line)]
        tools.predict_pairs([{"a": a, "b": b} for a, b in pairs])
        p = {("A", "B"): 0.9, ("A", "C"): 0.2}
        tools.submit_predictions([{"a": a, "b": b, "p": p[(a, b)], "evidence_ids": ["pred"]} for a, b in pairs])
        return {"submission": tools.submitted, "error": None, "tool_calls": 1, "stats": {}}

    monkeypatch.setattr(pring_test, "run_session", fake_session)
    args = argparse.Namespace(model="m", thinking="t", workers=1)
    case = {"round": 1, "a": "C", "b": "D", "result": "hit", "b0_p": 0.3, "ranking_score": 0.5}
    case |= {"uncertainty": None, "rationale": None, "evidence_ids": None}
    run_dir = tmp_path / "run"
    assert val_check(0, args, [], Lessons(), run_dir)["ap"] == 1.0
    assert val_check(1, args, [case], Lessons(), run_dir)["ap"] == 1.0
    assert [name for name, _ in seen] == ["TestTools", "TestTools"]
    assert "earlier screen" not in seen[0][1] and "earlier screen" in seen[1][1]  # round 0 is the empty memory
    assert (run_dir / "val" / "round01" / "batch0001" / "result.json").exists()
