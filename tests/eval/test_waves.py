from src.eval.oracle import Oracle
from src.eval.waves import K, classify, feedback

KNOWN = {"A": True, "B": True, "C": True, "D": True, "K1": False}


def row(a, b):
    return {"a": a, "b": b}


def test_classify_statuses():
    rows = [row("B", "A"), row("A", "B"), row("A", "A"), row("A", "Z"), row("A", "K1"), row("C", "D"), {"a": 1}]
    got = classify(rows, KNOWN, tested={("C", "D")})
    assert [s for _, s in got] == [
        "valid",
        "duplicate",
        "self",
        "unknown id",
        "not a candidate",
        "already tested",
        "unknown id",
    ]
    assert got[0][0] == ("A", "B")


def test_rows_beyond_k_are_over_budget():
    rows = [row("A", "B")] * (K + 1)
    assert classify(rows, KNOWN, set())[-1][1] == "over budget"


def evaluated(hits):
    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")]
    rows = [{"a": a, "b": b, "status": "valid", "hit": h} for (a, b), h in zip(pairs, hits)]
    return rows + [{"a": "A", "b": "A", "status": "self"}]


def test_feedback_reports_measured_results_of_valid_pairs():
    facts = feedback(3, evaluated([1, 0, 0, 1, 0]))
    assert [(f["a"], f["b"]) for f in facts if f["result"] == "hit"] == [("A", "B"), ("B", "C")]
    assert len(facts) == 5 and all(f["wave"] == 3 and f["source"] == "oracle" for f in facts)


def test_oracle_is_closed_world_and_unordered(tmp_path):
    labels = tmp_path / "pos.csv"
    labels.write_text("uniprotID_A,uniprotID_B\nP2,P1\n")
    assert Oracle(labels)([("P1", "P2"), ("P1", "P3")]) == {("P1", "P2"): 1, ("P1", "P3"): 0}


def test_run_feeds_back_results_resumes_and_sets_aside_interrupted_attempts(tmp_path, monkeypatch):
    import argparse

    from src.eval import waves

    sandbox, hidden = tmp_path / "sandbox", tmp_path / "hidden"
    sandbox.mkdir(), hidden.mkdir()
    rows = "".join(f"{x}\tG\tn\t{int(x != 'K1')}\tMKV\n" for x in ("A", "B", "C", "D", "K1"))
    (sandbox / "proteins.tsv").write_text("accession\tgene\tprotein_name\tcandidate\tsequence\n" + rows)
    (hidden / "test_positives.csv").write_text("uniprotID_A,uniprotID_B\nA,B\nC,D\n")
    (tmp_path / "tools.json").write_text('{"outputs": {}}')
    for name, value in (("SANDBOX", sandbox), ("HIDDEN", hidden), ("PRING", tmp_path), ("RUNS", tmp_path)):
        monkeypatch.setattr(waves, name, value)
    monkeypatch.setattr(waves, "get_pi_version", lambda: "test")
    calls = []

    def fake_wave(sandbox_dir, wave_dir, tested, memory, **_):
        calls.append((wave_dir.name, set(tested), list(memory)))
        order = [("A", "B"), ("A", "C"), ("C", "D"), ("B", "D"), ("A", "D"), ("B", "C")]
        picks = [p for p in order if p not in tested][:2]
        (wave_dir / "agent").mkdir(parents=True, exist_ok=True)
        return {"submission": [{"a": a, "b": b} for a, b in picks], "error": None, "nudged": False}

    monkeypatch.setattr(waves, "run_wave", fake_wave)
    args = argparse.Namespace(waves=3, until=None, model="m", thinking="t")
    run_dir = tmp_path / "r0"
    (run_dir / "wave02" / "agent").mkdir(parents=True)  # a run killed mid-wave 2
    waves.run(run_dir, 0, args)

    assert [c[0] for c in calls] == ["wave01", "wave02", "wave03"]
    assert calls[1][1] == {("A", "B"), ("A", "C")} and len(calls[2][1]) == 4  # U_t shrinks by valid picks
    assert [(m["a"], m["b"], m["result"]) for m in calls[1][2]] == [("A", "B", "hit"), ("A", "C", "not detected")]
    assert (run_dir / "wave02" / "agent_interrupted").exists()
    waves.run(run_dir, 0, args)  # resume replays without calling the agent
    assert len(calls) == 3


def test_repeated_infrastructure_failure_stops_the_run_unscored(tmp_path, monkeypatch):
    import argparse

    import pytest

    from src.eval import waves

    sandbox, hidden = tmp_path / "sandbox", tmp_path / "hidden"
    sandbox.mkdir(), hidden.mkdir()
    (sandbox / "proteins.tsv").write_text(
        "accession\tgene\tprotein_name\tcandidate\tsequence\nA\tG\tn\t1\tM\nB\tG\tn\t1\tM\n"
    )
    (hidden / "test_positives.csv").write_text("uniprotID_A,uniprotID_B\nA,B\n")
    (tmp_path / "tools.json").write_text('{"outputs": {}}')
    for name, value in (("SANDBOX", sandbox), ("HIDDEN", hidden), ("PRING", tmp_path), ("RUNS", tmp_path)):
        monkeypatch.setattr(waves, name, value)
    monkeypatch.setattr(waves, "get_pi_version", lambda: "test")

    def broke(sandbox_dir, wave_dir, *_, **__):
        (wave_dir / "agent").mkdir(parents=True, exist_ok=True)
        return {"submission": None, "error": "402: Insufficient Balance"}

    monkeypatch.setattr(waves, "run_wave", broke)
    args = argparse.Namespace(waves=2, until=None, model="m", thinking="t")
    with pytest.raises(RuntimeError, match="failed twice"):
        waves.run(tmp_path / "r0", 0, args)
    assert not (tmp_path / "r0" / "wave01" / "result.json").exists()
    assert (tmp_path / "r0" / "wave01" / "agent_failed").exists()
