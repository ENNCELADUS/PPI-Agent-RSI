import argparse
import gzip
import json

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


def test_rounds_add_cases_update_lessons_and_resume(tmp_path, monkeypatch):
    from src.eval import memory_agent

    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "test_positives.csv").write_text("uniprotID_A,uniprotID_B\nA,B\nC,D\n")
    (tmp_path / "memory_split.json").write_text('{"outputs": {}}')
    disc_view(tmp_path / "sandbox_disc")
    for name, value in (("DISC", tmp_path / "sandbox_disc"), ("HIDDEN", hidden), ("PRING", tmp_path)):
        monkeypatch.setattr(memory_agent, name, value)
    monkeypatch.setattr(memory_agent, "get_pi_version", lambda: "test")
    calls = []

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

    monkeypatch.setattr(memory_agent, "run_session", fake_session)
    args = argparse.Namespace(rounds=2, until=None, model="m", thinking="t")
    run_dir = tmp_path / "run"
    memory_agent.run_rounds(run_dir, args)

    assert [c[0] for c in calls] == ["CaseRoundTools", "ReviewTools", "CaseRoundTools", "ReviewTools"]
    assert "No lessons yet." in calls[0][1] and "L1 (since round 1): from A-B" in calls[2][1]
    mem = run_dir / "memory" / "round02"
    cases = [json.loads(line) for line in open(mem / "cases.jsonl")]
    assert [(c["a"], c["b"], c["result"], c["round"]) for c in cases] == [
        ("A", "B", "hit", 1),
        ("A", "C", "not detected", 1),
        ("C", "D", "hit", 2),
        ("B", "D", "not detected", 2),
    ]
    assert cases[0]["b0_p"] == 0.1 and cases[0]["rationale"] == "why"
    assert [x["supporting"] for x in json.loads((mem / "lessons.json").read_text())] == [["A-B"], ["C-D"]]

    memory_agent.run_rounds(run_dir, args)  # resume replays rounds and memory without any session
    assert len(calls) == 4
