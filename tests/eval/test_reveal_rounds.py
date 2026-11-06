import argparse
import gzip
import json

from tests.eval.test_memory_agent import disc_view

from src.agent.tools import RevealReviewTools, RevealTools

ROUND = {1: [("A", "B", 1), ("A", "C", 0)], 2: [("C", "D", 1), ("B", "D", 0)]}
FINAL = [("A", "D", 1), ("B", "C", 0)]


def logits(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = [(a, b) for rows in (*ROUND.values(), FINAL) for a, b, _ in rows]
    with gzip.open(path, "wt") as f:
        f.write("a\tb\tlogit\n" + "".join(f"{a}\t{b}\t{value}\n" for a, b in pairs))


def setup(tmp_path, monkeypatch):
    from src.data.builders import build_mus22t_rsi_sandbox as views
    from src.eval import reveal_rounds

    rsi = tmp_path / "rsi"
    rsi.mkdir()
    (rsi / "reveal.csv").write_text(
        "uniprotID_A,uniprotID_B,isInteraction,round\n"
        + "".join(f"{a},{b},{y},{t}\n" for t, rows in ROUND.items() for a, b, y in rows)
    )
    (rsi / "final.csv").write_text(
        "uniprotID_A,uniprotID_B,isInteraction\n" + "".join(f"{a},{b},{y}\n" for a, b, y in FINAL)
    )
    (rsi / "rsi.json").write_text('{"outputs": {}}')
    disc_view(tmp_path / "sandbox")
    ckpt = lambda t: tmp_path / f"ckpt{t}.tsv.gz"  # noqa: E731
    for t in range(3):
        logits(ckpt(t), t)  # checkpoint t gives every pair p = sigmoid(t)
    for name, value in (("RSI", rsi), ("OUT", tmp_path / "views"), ("checkpoint_path", ckpt)):
        monkeypatch.setattr(views, name, value)
    for name, value in (
        ("RSI", rsi),
        ("VIEWS", tmp_path / "views"),
        ("PLAIN", tmp_path / "sandbox"),
        ("RUNS", tmp_path / "runs"),
        ("ROUNDS", 2),
        ("checkpoint_path", ckpt),
        ("get_pi_version", lambda: "test"),
    ):
        monkeypatch.setattr(reveal_rounds, name, value)
    return views, reveal_rounds


def test_views_rounds_and_final_test(tmp_path, monkeypatch):
    views, reveal_rounds = setup(tmp_path, monkeypatch)
    for name, (t, rows) in {"round01": (0, ROUND[1]), "round02": (1, ROUND[2]), "final": (2, FINAL)}.items():
        views.view(name, rows, views.checkpoint_p(t))
    calls = []

    def fake_session(tools, agent_dir, system, message, nudge, **_):
        calls.append((type(tools).__name__, system, message))
        if isinstance(tools, RevealReviewTools):
            first = message.split("This round's cases")[1].splitlines()[2].split("\t")[0]
            tools.update_lessons([{"op": "add", "text": f"from {first}", "scope": "s", "supporting": [first]}])
        else:
            assert isinstance(tools, RevealTools)
            pairs = [line.split("\t") for line in message.split("\n")[3:] if line]
            tools.predict_pairs([{"a": r[0], "b": r[2]} for r in pairs])  # each view scores its listed pairs
            tools.submit_predictions([{"a": r[0], "b": r[2], "p": 0.7, "evidence_ids": ["pred"]} for r in pairs])
        return {"submission": tools.submitted, "error": None, "tool_calls": 1, "stats": {}}

    import src.eval.memory_agent as memory_agent
    import src.eval.pring_test as pring_test

    monkeypatch.setattr(memory_agent, "run_session", fake_session)
    monkeypatch.setattr(pring_test, "run_session", fake_session)
    monkeypatch.setattr(pring_test, "predictor_p", lambda name, pairs: dict.fromkeys(pairs, 0.6))
    args = argparse.Namespace(predictor=True, until=None, model="m", thinking="t", batch=200, workers=1, limit=None)
    run_dir = tmp_path / "runs" / "x"
    reveal_rounds.run_rounds(run_dir, args)

    assert [c[0] for c in calls] == ["RevealTools", "RevealReviewTools", "RevealTools", "RevealReviewTools"]
    assert "No lessons yet." in calls[0][1] and "L1 (since round 1): from A-B" in calls[2][1]
    cases = [json.loads(line) for line in open(run_dir / "memory" / "round02" / "cases.jsonl")]
    assert [(c["a"], c["b"], c["result"], c["round"]) for c in cases] == [
        ("A", "B", "hit", 1),
        ("A", "C", "not detected", 1),
        ("C", "D", "hit", 2),
        ("B", "D", "not detected", 2),
    ]
    assert [c["b0_p"] for c in cases] == [0.5, 0.5, 0.731059, 0.731059]  # round t sees checkpoint t - 1
    assert cases[0]["ranking_score"] == 0.7
    curve = json.loads((run_dir / "curve.json").read_text())
    assert [c["round"] for c in curve] == [1, 2] and curve[1]["agent"]["auroc"] == 0.5

    reveal_rounds.run_rounds(run_dir, args)  # resume: no new session
    assert len(calls) == 4

    reveal_rounds.run_test(run_dir, args)
    summary = json.loads((run_dir / "final" / "summary.json").read_text())
    assert summary["pairs"] == 2 and summary["al_checkpoint10"]["auroc"] == 0.5  # ROUNDS = 2 here
    assert "Its labels are never revealed" in calls[-1][1] and "L2" in calls[-1][1]
