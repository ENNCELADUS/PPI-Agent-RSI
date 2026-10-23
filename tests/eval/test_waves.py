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


def test_true_feedback_reports_measured_results():
    facts = feedback("true", 3, evaluated([1, 0, 0, 1, 0]), seed=0)
    assert [(f["a"], f["b"]) for f in facts if f["result"] == "hit"] == [("A", "B"), ("B", "C")]
    assert len(facts) == 5 and all(f["wave"] == 3 and f["source"] == "oracle" for f in facts)


def test_permuted_feedback_keeps_the_hit_count_on_valid_pairs_only():
    rows = evaluated([1, 0, 0, 1, 0])
    shown = [feedback("permuted", w, rows, seed=1) for w in range(1, 30)]
    assert all(sum(f["result"] == "hit" for f in facts) == 2 and len(facts) == 5 for facts in shown)
    assert len({tuple(f["result"] for f in facts) for facts in shown}) > 1  # placement varies by wave
    assert feedback("permuted", 4, rows, seed=1) == feedback("permuted", 4, rows, seed=1)


def test_no_feedback_withholds_results():
    assert {f["result"] for f in feedback("none", 2, evaluated([1, 1, 0, 0, 0]), seed=0)} == {"withheld"}


def test_oracle_is_closed_world_unordered_and_maps_anon_ids(tmp_path):
    labels = tmp_path / "pos.csv"
    labels.write_text("uniprotID_A,uniprotID_B\nP2,P1\n")
    assert Oracle(labels)([("P1", "P2"), ("P1", "P3")]) == {("P1", "P2"): 1, ("P1", "P3"): 0}
    id_map = tmp_path / "map.tsv"
    id_map.write_text("id\taccession\nX1\tP2\nX2\tP1\nX3\tP3\n")
    assert Oracle(labels, id_map)([("X1", "X2"), ("X1", "X3")]) == {("X1", "X2"): 1, ("X1", "X3"): 0}


def test_run_branches_from_shared_wave_1_resumes_and_sets_aside_interrupted_attempts(tmp_path, monkeypatch):
    import argparse
    import json

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
        calls.append((wave_dir.parent.name, wave_dir.name, set(tested), list(memory)))
        order = [("A", "B"), ("A", "C"), ("C", "D"), ("B", "D"), ("A", "D"), ("B", "C")]
        picks = [p for p in order if p not in tested][:2]
        (wave_dir / "agent").mkdir(parents=True, exist_ok=True)
        return {"submission": [{"a": a, "b": b} for a, b in picks], "error": None, "nudged": False}

    monkeypatch.setattr(waves, "run_wave", fake_wave)
    args = argparse.Namespace(waves=3, model="m", thinking="t")
    shared, branch = tmp_path / "shared", tmp_path / "permuted"
    waves.run(shared, "true", 0, args, 1)
    (branch / "wave02" / "agent").mkdir(parents=True)  # a run killed mid-wave 2
    waves.run(branch, "permuted", 0, args, 3, first_wave_from=shared / "wave01")

    assert [c[:2] for c in calls] == [("shared", "wave01"), ("permuted", "wave02"), ("permuted", "wave03")]
    assert calls[1][2] == {("A", "B"), ("A", "C")} and len(calls[2][2]) == 4  # U_t shrinks by valid picks
    assert sum(m["result"] == "hit" for m in calls[1][3]) == 1  # permuted keeps wave 1's one hit
    assert (branch / "wave02" / "agent_interrupted").exists()
    first = json.loads((branch / "wave01" / "result.json").read_text())
    assert first == json.loads((shared / "wave01" / "result.json").read_text()) and first["hits"] == 1
    waves.run(branch, "permuted", 0, args, 3)  # resume replays without calling the agent
    assert len(calls) == 3
