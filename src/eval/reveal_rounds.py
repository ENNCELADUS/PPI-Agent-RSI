"""22T reveal rounds: the agent classifies each round's batch, its labels become cases, a reviewer updates the lessons.

    uv run python -m src.eval.reveal_rounds rounds <name> [--predictor] [--model ...] [--thinking high] [--until T]
    uv run python -m src.eval.reveal_rounds test <name> [--batch 200] [--workers 8] [--limit N]
    uv run python -m src.eval.reveal_rounds al --data mus22t|pring --rule random|greedy  # DL predictor arm, no LLM
    uv run python -m src.eval.reveal_rounds rounds|test <name> --data pring [--predictor]   # the same on PRING

Design: docs/superpowers/specs/2026-11-06-mus22t-reveal-rounds-design.md. Data: build_mus22t_rsi (rsi/dev.csv: the
development set of 10,000 pairs; rsi/reveal.csv: 10 rounds of 100 pairs drawn from it at random; rsi/final.csv: 8,366
pairs), build_mus22t_rsi_sandbox (the views: the RSI training set as kb.csv, the checkpoint scores). The acquisition
rule is the random draw, the control: every arm gets the same pairs in the same rounds and is scored on the same final
set. The agent's own picks are not built yet.

rounds: for t = 1..10 one fresh pi session predicts round t's 100 pairs (the pring_test batch protocol, RevealTools:
protein and KB tools, `cases`, and with --predictor `predict_pairs` from sandbox_rsi/round<t>/, i.e. checkpoint t - 1
of the DL active-learning arm). Its p is scored before the labels are shown. The system then adds the 100 pairs as
cases (label, the agent's p, the predictor's p it could see, its evidence ids) and a reviewer session updates the
lessons, as in src.eval.memory_agent. Val gate (user decision 2026-11-07): the agent with this candidate memory then
classifies the val gate pairs (rsi/val_gate.csv, 400, read-only like the final test; with --predictor in
sandbox_rsi/val<t>/, checkpoint t). The memory is kept only if its val AUPRC beats every earlier kept memory (round 0 =
the empty memory); else the round's cases and lesson changes are dropped. memory/round<t>/ holds the kept memory,
gate.json the decisions, val/round<t>/ the val batches. The LLM-only arm (no --predictor) uses sandbox_rsi/plain/
with no predictor tool and no predictor p in the cases. A round with saved memory is replayed, so runs resume.
curve.json: per round, the agent's metrics on its batch before the reveal and the DL checkpoints' on the same pairs.

test: the final pairs (seed 0, batches of --batch) with the memory after round 10 (lessons in the prompt, cases
through the tool), read-only; with --predictor the run's agent sees checkpoint 10 (sandbox_rsi/final/). summary.json:
the pring_test metrics of the agent, checkpoint 0 ("v3": V3 trained on the RSI training set, before any reveal) and
the round-10 checkpoint on the same pairs, with paired bootstraps.

--data pring (user decision 2026-11-24): the same rounds, gate and final test on PRING with the random acquisition
rule. Data: build_pring_memory (hidden/dev_random.csv: 10 rounds of 200 pairs drawn at random from the development
set hidden/dev.csv; hidden/val_gate.csv; hidden/final_test.csv). Views: sandbox_disc/ in the rounds, sandbox_val/ at
the gate, sandbox_final/ at the final test; the cases come from the development proteins, so at the gate and the final
test only the cases of homologs apply (case_homologs.tsv). B0 stays frozen: every round reads the same scores, and
summary.json compares the agent with B0 ("b0"). Prompts: reveal_pring/. Runs go to data/PRING/runs/<name>/.

al: data/<benchmark>/runs/dl_<rule>/summary.json, the DL predictor arm (src.data.builders.dl_rounds; trained by
train_mus22t_al.sh or train_pring_al.sh on H20): every available checkpoint's metrics on the final test, the gate, and
per round the pairs and positives revealed and checkpoint t - 1's metrics on that batch, with the paired bootstrap of
checkpoint 10 - checkpoint 0.
"""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from types import SimpleNamespace

from src.agent.memory import MAX_ACTIVE, Cases, Lessons, case_id, read_homologs
from src.agent.session import DEFAULT_MODEL, PROMPTS, prompt
from src.agent.session import pi_version as get_pi_version
from src.agent.tools import CAP, K, RevealReviewTools, RevealTools, Sandbox, tsv
from src.data.builders.build_mus22t_rsi import OUT as RSI
from src.data.builders.build_mus22t_rsi import ROUNDS, decide, read
from src.data.builders import dl_rounds
from src.data.builders.build_mus22t_rsi_sandbox import OUT as VIEWS
from src.data.builders.build_mus22t_rsi_sandbox import checkpoint_p, checkpoint_path
from src.data.ids import pair
from src.data.paths import MUS22T, PRING
from src.eval.memory_agent import load_memory, save_memory, session_twice
from src.eval.pring_test import batches, classification, paired_bootstrap, run_batch, summarise, test_pairs
from src.eval.rounds import commit, set_aside

FINAL_STAGE = "This is the final test. Its labels are never revealed; the cases and lessons stay as they are."
RUNS = MUS22T / "runs"
PLAIN = VIEWS / "plain"  # the LLM-only arm: no predictor tool
# checkpoint 0's own validation numbers replace the ones below after its training (these are V3's on all of train)
PREDICTOR_TOOL = (
    "- A PPI predictor trained on the knowledge base and on every pair revealed before this batch (it is retrained "
    "after\n  each round): `predict_pairs` gives its probability p for up to 200 of the listed pairs in one call. "
    "It predicts\n  from the two protein sequences. Before any reveal it had validation AUPRC 0.718 and AUROC 0.712, "
    "and its p\n  was calibrated (expected calibration error 0.081).\n"
)


PRING_PREDICTOR_TOOL = (
    "- A fixed PPI predictor trained on the knowledge base: `predict_pairs` gives its score p for up to 200 of the "
    "listed\n  pairs in one call. It predicts from the two protein sequences. p ranks the pairs; it was calibrated for "
    "a setting\n  where about 1% of all pairs interact, so read it as a rank, not as the probability of these pairs.\n"
)
DISC, VAL, FINAL = PRING / "sandbox_disc", PRING / "sandbox_val", PRING / "sandbox_final"


def mus22t():
    """The 22T files and views, read at call time (tests replace RSI, VIEWS, PLAIN and RUNS). One DL checkpoint per
    round: with the predictor, each stage has its own view."""

    def view(stage, t, predictor):
        return VIEWS / {"round": f"round{t:02d}", "val": f"val{t:02d}", "final": "final"}[stage] if predictor else PLAIN

    return SimpleNamespace(
        name="mus22t",
        runs=RUNS,
        batches=RSI / "reveal.csv",
        final=RSI / "final.csv",
        gate=RSI / "val_gate.csv",
        dev=RSI / "dev.csv",
        split=RSI / "rsi.json",
        prompts="reveal",
        view=view,
        cases=lambda records, sandbox, view: Cases(records, sandbox.proteins, read_homologs(PLAIN / "homologs.tsv")),
        predictor_tool=PREDICTOR_TOOL,
        final_stage=FINAL_STAGE,
        checkpoints=True,  # DL checkpoints 0..10 score the batches and the final set
        reference=None,
        build="build_mus22t_rsi and build_mus22t_rsi_sandbox",
    )


def pring():
    """The PRING files and views. B0 stays frozen, so every round reads the same scores; the cases are pairs of the
    development proteins, and each view has its own homolog table into them."""
    hidden = PRING / "hidden"
    return SimpleNamespace(
        name="pring",
        runs=PRING / "runs",
        batches=hidden / "dev_random.csv",
        final=hidden / "final_test.csv",
        gate=hidden / "val_gate.csv",
        dev=hidden / "dev.csv",
        split=PRING / "memory_split.json",
        prompts="reveal_pring",
        view=lambda stage, t, predictor: {"round": DISC, "val": VAL, "final": FINAL}[stage],
        cases=lambda records, sandbox, view: Cases(
            records, Sandbox(str(DISC)).proteins, read_homologs(view / "case_homologs.tsv")
        ),
        predictor_tool=PRING_PREDICTOR_TOOL,
        final_stage=FINAL_STAGE + " None of these proteins is in a case: use the cases of their homologs.",
        checkpoints=False,
        reference="b0",  # the frozen predictor scored beside the agent
        build="build_pring_sandbox, build_pring_tools and build_pring_memory",
    )


DATA = mus22t  # main() sets pring for --data pring


def round_batches(path):
    """{round: [(a, b, label)]} in file order, pairs canonical (the PRING files order a pair by node id)."""
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(int(r["round"]), []).append((*pair(r["uniprotID_A"], r["uniprotID_B"]), int(r["isInteraction"])))
    return out


def stage_sandbox(stage, t, predictor):
    """(sandbox, view directory) of a round, of the validation check after round t, or of the final test."""
    view = DATA().view(stage, t, predictor)
    assert not predictor or (view / "scores.tsv.gz").exists(), f"no predictor scores in {view}: run {DATA().build}"
    return Sandbox(str(view)), view


def system_prompt(sandbox, predictor, records, lessons, stage):
    return prompt(
        f"{DATA().prompts}/system",
        proteins=len(sandbox.candidates),
        cap=CAP,
        stage=stage,
        cases=f"{len(records):,}",
        lessons=lessons.render(),
        predictor_tool=DATA().predictor_tool if predictor else "",
        predictor_step="the predictor's p, " if predictor else "",
        pred_example="`pred:A-B`, " if predictor else "",
    )


def new_cases(t, result, seen):
    """One case per pair of round t: its label, the agent's p (ranking_score) and evidence, the predictor's p shown."""
    evidence = {}
    for r in result["submission"] or []:
        if isinstance(r, dict) and isinstance(r.get("a"), str) and isinstance(r.get("b"), str):
            evidence.setdefault(case_id(r["a"], r["b"]), r.get("evidence_ids"))
    return [
        {
            "round": t,
            "a": r["a"],
            "b": r["b"],
            "result": "hit" if r["label"] else "not detected",
            "b0_p": seen.get((r["a"], r["b"])) if seen else None,
            "ranking_score": r["p"],
            "uncertainty": None,
            "rationale": None,
            "evidence_ids": evidence.get(case_id(r["a"], r["b"])),
        }
        for r in result["rows"]
    ]


def fmt(x):
    return "" if x is None else f"{x:.3g}"


def review(t, args, sandbox, view, records, fresh, lessons, run_dir):
    predictor = args.predictor
    system = prompt(
        f"{DATA().prompts}/review_system",
        k=len(fresh),
        rounds=ROUNDS,
        final=len(test_pairs(DATA().final)),
        max_active=MAX_ACTIVE,
        cap=CAP,
        pred_clause=", after seeing a sequence-based predictor's p" if predictor else "",
        pred_cases="the predictor's p the agent saw (b0_p), " if predictor else "",
    )
    head = ["case", "gene_a", "gene_b", "agent_p", *(["predictor_p"] if predictor else []), "result"]
    rows = [
        (
            case_id(c["a"], c["b"]),
            sandbox.gene(c["a"]),
            sandbox.gene(c["b"]),
            fmt(c["ranking_score"]),
            *([fmt(c["b0_p"])] if predictor else []),
            c["result"],
        )
        for c in fresh
    ]
    message = prompt(
        "reveal/review",
        round=t,
        rounds=ROUNDS,
        hits=sum(c["result"] == "hit" for c in fresh),
        n=len(fresh),
        cases=len(records),
        lessons=lessons.render(),
        new_cases=tsv(head, rows),
    )
    book = DATA().cases(records, sandbox, view)
    agent_dir = run_dir / "rounds" / f"review{t:02d}"
    out = session_twice(
        lambda: RevealReviewTools(sandbox, agent_dir, book, CAP, predictor),
        agent_dir,
        system,
        message,
        prompt("memory/review_nudge"),
        args,
        f"review of round {t}",
    )
    ops = out["submission"]
    report = lessons.apply(ops, book.ids, t) if ops is not None else ["no update_lessons call: lessons unchanged"]
    keep = ("nudged", "aborted", "error", "stop_reasons", "tool_calls", "stats", "failed_attempt")
    return {"round": t, "operations": ops, "report": report, **{k: out.get(k) for k in keep}}


def provenance(args, **extra):
    own = {*(PROMPTS / "reveal").glob("*.md"), *(PROMPTS / DATA().prompts).glob("*.md")}
    files = sorted(own) + [PROMPTS / "pring_test" / f for f in ("batch.md", "nudge.md")]
    return {
        "data": DATA().name,
        "model": args.model,
        "thinking": args.thinking,
        "predictor": args.predictor,
        "cap": CAP,
        "max_active_lessons": MAX_ACTIVE,
        "git": commit(),
        "pi": get_pi_version(),
        "prompts": {p.relative_to(PROMPTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files},
        "split": json.loads(DATA().split.read_text())["outputs"],
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }


def val_check(t, args, records, lessons, run_dir):
    """The agent's metrics on the val gate pairs with this memory, read-only, as in the final test (the agent sees the
    final-test stage); with the predictor on 22T, checkpoint t."""
    sandbox, view = stage_sandbox("val", t, args.predictor)
    book = DATA().cases(records, sandbox, view)
    system = system_prompt(sandbox, args.predictor, records, lessons, DATA().final_stage)
    todo = batches(test_pairs(DATA().gate), K)
    d = run_dir / "val" / f"round{t:02d}"
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [
            pool.submit(
                run_batch,
                i,
                b,
                len(todo),
                sandbox,
                system,
                d,
                args,
                lambda x: RevealTools(sandbox, x, book, CAP, args.predictor),
            )
            for i, b in enumerate(todo, 1)
        ]
        results = [j.result() for j in jobs]
    if None in results:
        raise RuntimeError(f"val check of round {t}: a batch failed twice")
    rows = [row for r in results for row in r["rows"]]
    return classification([r["label"] for r in rows], [0.5 if r["p"] is None else r["p"] for r in rows])


def run_rounds(run_dir, args):
    data = DATA()
    assert data.dev.exists() and (data.view("val", 0, False) / "kb.csv").exists(), (
        f"no development set: run {data.build}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run.json").exists():
        old = json.loads((run_dir / "run.json").read_text())
        assert old["predictor"] == args.predictor, "resume with the run's own --predictor"
        assert old.get("data", "mus22t") == data.name, "resume with the run's own --data"
        args.model, args.thinking = old["model"], old["thinking"]
    else:
        (run_dir / "run.json").write_text(json.dumps(provenance(args), indent=1) + "\n")
    rounds = round_batches(data.batches)
    records, lessons = [], Lessons()
    gate_path = run_dir / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else []
    if not gate:
        gate = decide([], 0, val_check(0, args, records, lessons, run_dir)["ap"])
        gate_path.write_text(json.dumps(gate, indent=1) + "\n")
    for t in range(1, min(ROUNDS, args.until or ROUNDS) + 1):
        mem_dir = run_dir / "memory" / f"round{t:02d}"
        if (mem_dir / "cases.jsonl").exists():
            records, lessons = load_memory(mem_dir)
            continue
        sandbox, view = stage_sandbox("round", t, args.predictor)
        stage = (
            f"This is round {t} of {ROUNDS}. After you submit, the labels of this batch are revealed and become "
            "cases; a reviewer then updates the lessons."
        )
        system = system_prompt(sandbox, args.predictor, records, lessons, stage)
        book = data.cases(records, sandbox, view)
        result = run_batch(
            t,
            rounds[t],
            ROUNDS,
            sandbox,
            system,
            run_dir / "rounds",
            args,
            lambda d: RevealTools(sandbox, d, book, CAP, args.predictor),  # noqa: B023 (called within this round)
        )
        if result is None:
            raise RuntimeError(f"round {t} failed twice")
        score = lambda a, b: sandbox.query("SELECT p FROM scores WHERE a = ? AND b = ?", [a, b])[0][0]  # noqa: E731, B023
        seen = {(a, b): score(a, b) for a, b, _ in rounds[t]} if args.predictor else None
        fresh = new_cases(t, result, seen)
        kept_records, kept_lessons = records, Lessons(lessons.all)
        records = records + fresh
        log = review(t, args, sandbox, view, records, fresh, lessons, run_dir)
        # a rerun reviewed again: earlier val batches checked another memory
        set_aside(run_dir / "val", f"round{t:02d}_stale", f"round{t:02d}")
        m = val_check(t, args, records, lessons, run_dir)
        gate = decide(gate[:t], t, m["ap"])
        gate_path.write_text(json.dumps(gate, indent=1) + "\n")
        if not gate[t]["kept"]:
            records, lessons = kept_records, kept_lessons
        log["gate"] = gate[t] | {"val": m}
        save_memory(mem_dir, records, lessons, log)
        print(
            f"{run_dir.name} round {t}: val AUPRC {m['ap']:.4f}, {'kept' if gate[t]['kept'] else 'dropped'}; "
            f"{len(lessons.active())} active lessons, {len(records)} cases"
        )
    write_curve(run_dir)


def curve_rows(rows, t):
    """Metrics on round t's pairs of the agent (if rows give p) and of DL checkpoints 0 and t - 1."""
    y = [r["label"] for r in rows]
    out = {"round": t, "pairs": len(rows)}
    if "p" in rows[0]:
        out["agent"] = classification(y, [0.5 if r["p"] is None else r["p"] for r in rows])
    for name, c in (("v3_frozen", 0), ("al_checkpoint", t - 1)) if DATA().checkpoints else ():
        if checkpoint_path(c).exists():
            p = checkpoint_p(c)
            out[name] = classification(y, [p[(r["a"], r["b"])] for r in rows])
    return out


def write_curve(run_dir):
    curve = []
    for t in range(1, ROUNDS + 1):
        path = run_dir / "rounds" / f"batch{t:04d}" / "result.json"
        if path.exists():
            curve.append(curve_rows(json.loads(path.read_text())["rows"], t))
    (run_dir / "curve.json").write_text(json.dumps(curve, indent=1) + "\n")


def run_test(run_dir, args):
    cfg = json.loads((run_dir / "run.json").read_text())
    mem_dir = run_dir / "memory" / f"round{ROUNDS:02d}"
    assert (mem_dir / "cases.jsonl").exists(), f"no memory after round {ROUNDS}: run the rounds first"
    records, lessons = load_memory(mem_dir)
    args.model, args.thinking, args.predictor = cfg["model"], cfg["thinking"], cfg["predictor"]
    predictor = args.predictor
    data = DATA()
    assert cfg.get("data", "mus22t") == data.name, "test with the run's own --data"
    sandbox, view = stage_sandbox("final", ROUNDS, predictor)
    book = data.cases(records, sandbox, view)
    test_dir = run_dir / "final"
    test_dir.mkdir(exist_ok=True)
    todo = batches(test_pairs(data.final), args.batch)
    if (test_dir / "run.json").exists():
        assert json.loads((test_dir / "run.json").read_text())["batch"] == args.batch, "resume with the same batch"
    else:
        extra = {"batch": args.batch, "batches": len(todo), "cases": len(records)}
        (test_dir / "run.json").write_text(json.dumps(provenance(args, **extra), indent=1) + "\n")
    system = system_prompt(sandbox, predictor, records, lessons, data.final_stage)
    n = len(todo)
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [
            pool.submit(
                run_batch,
                i,
                b,
                n,
                sandbox,
                system,
                test_dir,
                args,
                lambda d: RevealTools(sandbox, d, book, CAP, predictor),
            )
            for i, b in enumerate(todo[: args.limit], 1)
        ]
        results = [r for j in jobs if (r := j.result()) is not None]
    print(f"{len(results)} of {len(jobs)} batches scored")
    if not results:
        return
    rows = [row for r in results for row in r["rows"]]
    summary = summarise(rows, results, data.reference)
    y = [r["label"] for r in rows]
    p = [0.5 if r["p"] is None else r["p"] for r in rows]
    # 22T: the DL predictor before and after the rounds
    for name, c in (("v3", 0), ("al_checkpoint10", ROUNDS)) if data.checkpoints else ():
        if checkpoint_path(c).exists():
            scores = checkpoint_p(c)
            q = [scores[(r["a"], r["b"])] for r in rows]
            summary[name] = classification(y, q)
            summary[f"agent_minus_{name}"] = paired_bootstrap(y, p, q)
    (test_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def run_al(data, rule):
    bench = dl_rounds.BENCH[data]
    final = read(bench["final"])
    y = [label for _, _, label in final]
    p = {
        c: checkpoint_p(c, path)
        for c in range(ROUNDS + 1)
        if (path := dl_rounds.checkpoint_path(data, rule, c)).exists()
    }
    on = lambda c, rows: [p[c][pair(a, b)] for a, b, _ in rows]  # noqa: E731
    out = {"final_pairs": len(final), "gate": dl_rounds.read_gate(data, rule), "checkpoints": {}, "curve": []}
    out["checkpoints"] = {c: classification(y, on(c, final)) for c in p}
    path = dl_rounds.work(data, rule) / "batches.csv"
    for t, rows in sorted(dl_rounds.rows_by_round(path).items() if path.exists() else []):
        point = {"round": t, "pairs": len(rows), "positives": sum(r[2] for r in rows)}
        if t - 1 in p and 0 < point["positives"] < len(rows):  # one-class batch (greedy): no AUPRC
            point["checkpoint_before"] = classification([r[2] for r in rows], on(t - 1, rows))
        out["curve"].append(point)
    if ROUNDS in p and 0 in p:
        out["checkpoint10_minus_checkpoint0"] = paired_bootstrap(y, on(ROUNDS, final), on(0, final))
    d = (MUS22T if data == "mus22t" else PRING) / "runs" / f"dl_{rule}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({c: round(m["ap"], 4) for c, m in out["checkpoints"].items()}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("rounds", "test", "al"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--data", choices=("mus22t", "pring"), default="mus22t")
    parser.add_argument("--predictor", action="store_true", help="rounds: the agent sees the DL predictor's scores")
    parser.add_argument("--until", type=int, help="rounds: stop after this round")
    parser.add_argument("--batch", type=int, default=K, help="test: pairs per session")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="test: run only the first N batches")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--rule", choices=dl_rounds.RULES, default="random", help="al: the acquisition rule")
    args = parser.parse_args()
    if args.stage == "al":
        run_al(args.data, args.rule)
        return
    assert args.name, f"{args.stage} needs a run name"
    global DATA
    DATA = pring if args.data == "pring" else mus22t
    (run_rounds if args.stage == "rounds" else run_test)(DATA().runs / args.name, args)


if __name__ == "__main__":
    main()
