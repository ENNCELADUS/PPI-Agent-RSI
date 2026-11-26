import argparse
import csv
import gzip
import json

import pytest
from tests.eval.test_memory_agent import disc_view

from src.agent.tools import RevealReviewTools, RevealTools
from src.data.builders.build_mus22t_rsi import gate_round

ROUND = {1: [("A", "B", 1), ("A", "C", 0)], 2: [("C", "D", 1), ("B", "D", 0)]}
FINAL = [("A", "D", 1), ("B", "C", 0)]
# the val gate pairs (every pair of the four proteins; the checkpoint logits score them too), sorted
VAL = [("A", "B", 1), ("A", "C", 0), ("A", "D", 1), ("B", "C", 0), ("B", "D", 0), ("C", "D", 1)]
Y = [y for _, _, y in VAL]
PERFECT = [0.9 if y else 0.1 for y in Y]
INVERTED = [0.1 if y else 0.9 for y in Y]
SWAP = [0.9, 0.1, 0.3, 0.6, 0.1, 0.9]  # one positive below one negative: worse, but not clearly
TRAIN = [("A", "E", 1), ("B", "E", 0)]  # the RSI training set; the development set is the pairs of the two rounds
HEAD = "uniprotID_A,uniprotID_B,isInteraction\n"
# predict_pairs text of candidate t, which gives every pair p = sigmoid(t)
SIGMOID = {0: "0.5", 1: "0.7311", 2: "0.8808"}


def logits(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = [(a, b) for rows in (*ROUND.values(), FINAL) for a, b, _ in rows]
    with gzip.open(path, "wt") as f:
        f.write("a\tb\tlogit\n" + "".join(f"{a}\t{b}\t{value}\n" for a, b in pairs))


def csv_rows(path, rows, extra=""):
    path.write_text(HEAD.strip() + extra + "\n" + "".join(",".join(map(str, r)) + "\n" for r in rows))


def setup(tmp_path, monkeypatch):
    from src.data.builders import build_mus22t_rsi_sandbox as views
    from src.eval import reveal_rounds

    rsi = tmp_path / "rsi"
    rsi.mkdir()
    csv_rows(rsi / "reveal.csv", [(*r, t) for t, rows in ROUND.items() for r in rows], ",round")
    csv_rows(rsi / "final.csv", FINAL)
    csv_rows(rsi / "val_gate.csv", VAL)
    csv_rows(rsi / "train.csv", TRAIN)
    csv_rows(rsi / "dev.csv", [r for rows in ROUND.values() for r in rows])
    (rsi / "rsi.json").write_text('{"outputs": {}}')
    disc_view(tmp_path / "sandbox")
    ckpt = lambda t: tmp_path / f"candidate{t}.tsv.gz"  # noqa: E731
    val = lambda t: tmp_path / "val" / f"candidate{t}.tsv.gz"  # noqa: E731
    for t in range(3):
        logits(ckpt(t), t)
        logits(val(t), t)
    for name, value in (
        ("RSI", rsi),
        ("OUT", tmp_path / "views"),
        ("checkpoint_path", ckpt),
        ("val_path", val),
        ("ROUNDS", 2),
    ):
        monkeypatch.setattr(views, name, value)
    views.main()
    for name, value in (
        ("RSI", rsi),
        ("VIEWS", tmp_path / "views"),
        ("PLAIN", tmp_path / "views" / "plain"),
        ("RUNS", tmp_path / "runs"),
        ("ROUNDS", 2),
        ("checkpoint_path", ckpt),
        ("get_pi_version", lambda: "test"),
    ):
        monkeypatch.setattr(reveal_rounds, name, value)
    return views, reveal_rounds


def fake_sessions(monkeypatch, calls, val_p, val_pairs):
    """Fake pi sessions: each call is logged as [tools, system, message, predictor p strings it received]. The
    reviewer adds one lesson "review <n>"; the val checks answer val_p in turn (p per sorted val pair)."""
    import src.eval.memory_agent as memory_agent
    import src.eval.pring_test as pring_test

    def fake_session(tools, agent_dir, system, message, nudge, **_):
        call = [type(tools).__name__, system, message, set()]
        calls.append(call)
        if isinstance(tools, RevealReviewTools):
            reviews = sum(c[0] == "RevealReviewTools" for c in calls)
            first = message.split("before the reveal.")[1].splitlines()[2].split("\t")[0]
            tools.update_lessons([{"op": "add", "text": f"review {reviews}", "scope": "s", "supporting": [first]}])
        else:
            assert isinstance(tools, RevealTools)
            pairs = [line.split("\t") for line in message.split("\n")[3:] if line]
            out = tools.predict_pairs([{"a": r[0], "b": r[2]} for r in pairs])  # each view scores its listed pairs
            call[3] = {line.split("\t")[2] for line in out.splitlines()[1:]}
            p = dict.fromkeys(((r[0], r[2]) for r in pairs), 0.7)
            if "final test" in system and set(p) == set(val_pairs) and val_p:
                p = dict(zip(val_pairs, val_p.pop(0)))
            tools.submit_predictions([{"a": a, "b": b, "p": x, "evidence_ids": ["pred"]} for (a, b), x in p.items()])
        return {"submission": tools.submitted, "error": None, "tool_calls": 1, "stats": {}}

    monkeypatch.setattr(memory_agent, "run_session", fake_session)
    monkeypatch.setattr(pring_test, "run_session", fake_session)
    monkeypatch.setattr(pring_test, "predictor_p", lambda name, pairs: dict.fromkeys(pairs, 0.6))


def run_args(predictor=True):
    return argparse.Namespace(
        predictor=predictor, until=None, model="m", thinking="t", batch=200, workers=1, limit=None
    )


def test_archive_one_gate_and_the_checkpoint_in_use(tmp_path, monkeypatch):
    _, reveal_rounds = setup(tmp_path, monkeypatch)
    calls = []
    # the val checks after rounds 0, 1, 2: round 1 clearly worse than round 0 (dropped), round 2 not clearly worse
    fake_sessions(monkeypatch, calls, [PERFECT, INVERTED, SWAP], [(a, b) for a, b, _ in VAL])
    args = run_args()
    run_dir = tmp_path / "runs" / "x"
    reveal_rounds.run_rounds(run_dir, args)

    names = [c[0] for c in calls]
    assert names == ["RevealTools"] * 2 + ["RevealReviewTools"] + ["RevealTools"] * 2 + ["RevealReviewTools"] + [
        "RevealTools"
    ]
    val0, round1, review1, val1, round2, review2, val2 = calls
    # the gate compares each candidate's per-pair val p with those of the best kept state (round 0 both times)
    expected = gate_round([], 0, Y, PERFECT, None)
    expected = gate_round(expected, 1, Y, INVERTED, PERFECT)
    expected = gate_round(expected, 2, Y, SWAP, PERFECT)
    gate = json.loads((run_dir / "gate.json").read_text())
    assert gate == expected and [g["kept"] for g in gate] == [True, False, True]
    assert [g.get("against") for g in gate] == [None, 0, 0]
    # with the predictor: round 1 sees checkpoint 0, its gate candidate 1; round 1 dropped, so round 2 sees
    # checkpoint 0 again and its gate candidate 2
    assert [c[3] for c in (val0, round1, val1, round2, val2)] == [{SIGMOID[t]} for t in (0, 0, 1, 0, 2)]
    # the dropped candidate is a record; after it the lessons in use are still empty, but its cases are in use
    dropped = json.loads((run_dir / "memory" / "round01" / "lessons.json").read_text())
    assert [x["text"] for x in dropped] == ["review 1"]
    assert "review 1" in val1[1] and "No lessons yet." in round2[1] and "Cases: 2 pairs" in round2[1]
    # the reviewer of round 2 gets every case since the last kept round (0): rounds 1 and 2
    assert "The cases of round 1 and of this round" in review2[2] and "2 of these 4 pairs interact" in review2[2]
    assert "This round's cases." in review1[2]
    cases = [json.loads(line) for line in open(run_dir / "memory" / "round02" / "cases.jsonl")]
    assert [(c["a"], c["b"], c["result"], c["round"]) for c in cases] == [
        ("A", "B", "hit", 1),
        ("A", "C", "not detected", 1),
        ("C", "D", "hit", 2),
        ("B", "D", "not detected", 2),
    ]  # the archive keeps the dropped round's cases, and they come back in the next candidate
    assert [c["b0_p"] for c in cases] == [0.5] * 4 and cases[0]["ranking_score"] == 0.7
    assert "Cases: 4 pairs" in val2[1] and "review 2" in val2[1] and "review 1" not in val2[1]
    curve = json.loads((run_dir / "curve.json").read_text())
    assert [c["round"] for c in curve] == [1, 2] and curve[1]["agent"]["auroc"] == 0.5

    reveal_rounds.run_rounds(run_dir, args)  # resume: no new session
    assert len(calls) == 7

    reveal_rounds.run_test(run_dir, args)
    final = calls[-1]
    assert final[3] == {SIGMOID[2]}  # the memory and the checkpoint in use after round 2, kept together
    assert "Its labels are never revealed" in final[1] and "review 2" in final[1] and "Cases: 4 pairs" in final[1]
    summary = json.loads((run_dir / "final" / "summary.json").read_text())
    assert summary["pairs"] == 2 and summary["al_checkpoint10"]["auroc"] == 0.5  # ROUNDS = 2 here
    assert summary["v3"]["auroc"] == 0.5 and "agent_minus_v3" in summary  # "v3" is candidate 0
    assert json.loads((run_dir / "final" / "run.json").read_text())["state_round"] == 2


def test_baseline_is_the_final_test_with_the_empty_memory_and_checkpoint_0(tmp_path, monkeypatch):
    _, reveal_rounds = setup(tmp_path, monkeypatch)
    calls = []
    fake_sessions(monkeypatch, calls, [], [])
    args = run_args()
    run_dir = tmp_path / "runs" / "base"
    reveal_rounds.run_baseline(run_dir, args)
    assert len(calls) == 1 and calls[0][3] == {SIGMOID[0]} and "No lessons yet." in calls[0][1]
    assert not (run_dir / "rounds").exists() and not (run_dir / "gate.json").exists()
    summary = json.loads((run_dir / "final" / "summary.json").read_text())
    assert summary["pairs"] == 2 and "agent_minus_v3" in summary
    with pytest.raises(AssertionError, match="own stage"):  # a baseline run is not continued with rounds
        reveal_rounds.run_rounds(run_dir, args)


def test_views_by_checkpoint_show_the_rsi_training_set_and_never_a_development_label(tmp_path, monkeypatch):
    views, _ = setup(tmp_path, monkeypatch)  # setup ran views.main()
    dev = {(a, b) for rows in ROUND.values() for a, b, _ in rows}
    root = tmp_path / "views"
    for name in ("plain", "ckpt00_round", "ckpt02_val", "ckpt01_final"):
        kb = {(r["uniprotID_A"], r["uniprotID_B"]) for r in csv.DictReader(open(root / name / "kb.csv"))}
        assert kb == {(a, b) for a, b, _ in TRAIN} and not kb & dev
    assert not (root / "plain" / "scores.tsv.gz").exists()

    def scored(name):  # the pairs a view scores
        rows = csv.DictReader(gzip.open(root / name / "scores.tsv.gz", "rt"), delimiter="\t")
        return {(r["a"], r["b"]) for r in rows}

    assert scored("ckpt00_round") == dev and scored("ckpt02_final") == {r[:2] for r in FINAL}
    assert scored("ckpt01_val") == {r[:2] for r in VAL}

    with gzip.open(tmp_path / "candidate0.tsv.gz", "wt") as f:
        f.write("a\tb\tlogit\nA\tB\t0\n")  # a checkpoint that did not score every development and final pair
    with pytest.raises(AssertionError, match="dev.csv"):
        views.main()


def test_rounds_stop_without_the_development_set(tmp_path, monkeypatch):
    _, reveal_rounds = setup(tmp_path, monkeypatch)
    (tmp_path / "rsi" / "dev.csv").unlink()
    with pytest.raises(AssertionError, match="no development set"):
        reveal_rounds.run_rounds(tmp_path / "runs" / "x", run_args(False))


def pring_setup(tmp_path, monkeypatch):
    from src.eval import reveal_rounds

    hidden = tmp_path / "hidden"
    hidden.mkdir()
    # round 1 lists one pair in node order (B before A): the harness makes it canonical; round 2 has no positive
    (hidden / "dev_random.csv").write_text(HEAD.strip() + ",round\nB,A,1,1\nA,C,0,1\nC,D,0,2\nB,D,0,2\n")
    (hidden / "dev.csv").write_text(HEAD + "A,B,1\nA,C,0\nC,D,0\nB,D,0\n")
    for name in ("final_test.csv", "val_gate.csv"):
        csv_rows(hidden / name, FINAL)
    (tmp_path / "memory_split.json").write_text('{"outputs": {}}')
    for view in ("sandbox_disc", "sandbox_val", "sandbox_final"):
        disc_view(tmp_path / view)
    for name, value in (
        ("PRING", tmp_path),
        ("DISC", tmp_path / "sandbox_disc"),
        ("VAL", tmp_path / "sandbox_val"),
        ("FINAL", tmp_path / "sandbox_final"),
        ("DATA", reveal_rounds.pring),
        ("ROUNDS", 2),
        ("get_pi_version", lambda: "test"),
    ):
        monkeypatch.setattr(reveal_rounds, name, value)
    return reveal_rounds


def test_pring_rounds_gate_and_final_test_with_the_random_batches(tmp_path, monkeypatch):
    reveal_rounds = pring_setup(tmp_path, monkeypatch)
    calls = []
    # the gate after rounds 0, 1, 2 (p of A-D, B-C): two pairs never make a candidate clearly worse, so all kept
    fake_sessions(monkeypatch, calls, [[0.4, 0.6], [0.9, 0.1], [0.4, 0.6]], [("A", "D"), ("B", "C")])
    args = run_args()
    run_dir = tmp_path / "runs" / "x"
    reveal_rounds.run_rounds(run_dir, args)

    assert [c[0] for c in calls] == ["RevealTools"] * 2 + ["RevealReviewTools"] + ["RevealTools"] * 2 + [
        "RevealReviewTools",
        "RevealTools",
    ]
    assert all("human proteins" in c[1] for c in calls)  # the PRING prompts, not the mouse ones
    assert "use the cases of their homologs" in calls[0][1] and "This is round 1 of 2" in calls[1][1]
    assert json.loads((run_dir / "run.json").read_text())["data"] == "pring"
    gate = json.loads((run_dir / "gate.json").read_text())
    assert [(g["round"], g["val_auprc"], g["kept"]) for g in gate] == [(0, 0.5, True), (1, 1.0, True), (2, 0.5, True)]
    cases = [json.loads(line) for line in open(run_dir / "memory" / "round02" / "cases.jsonl")]
    assert [(c["a"], c["b"], c["result"], c["round"]) for c in cases[:2]] == [
        ("A", "B", "hit", 1),  # listed as B,A
        ("A", "C", "not detected", 1),
    ]
    assert cases[0]["b0_p"] == 0.1 and cases[0]["ranking_score"] == 0.7  # the frozen predictor's score of A-B
    assert "This round's cases. 0 of these 2 pairs interact" in calls[5][2]  # round 1 was kept
    curve = json.loads((run_dir / "curve.json").read_text())
    assert "agent" in curve[0] and curve[1] == {"round": 2, "pairs": 2, "positives": 0}  # one-class: no metrics

    reveal_rounds.run_rounds(run_dir, args)  # resume: no new session
    assert len(calls) == 7

    reveal_rounds.run_test(run_dir, args)
    summary = json.loads((run_dir / "final" / "summary.json").read_text())
    assert summary["pairs"] == 2 and "agent_minus_b0" in summary and "v3" not in summary
    assert "review 2" in calls[-1][1] and "review 1" in calls[-1][1]  # the state in use after round 2

    monkeypatch.setattr(reveal_rounds, "DATA", reveal_rounds.mus22t)
    with pytest.raises(AssertionError):  # a PRING run is not resumed as a 22T run
        reveal_rounds.run_test(run_dir, args)


def test_pring_rounds_stop_without_the_development_set(tmp_path, monkeypatch):
    reveal_rounds = pring_setup(tmp_path, monkeypatch)
    (tmp_path / "hidden" / "dev.csv").unlink()
    with pytest.raises(AssertionError, match="no development set: run build_pring_sandbox"):
        reveal_rounds.run_rounds(tmp_path / "runs" / "x", run_args(False))
