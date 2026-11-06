"""22T reveal rounds: the agent classifies fixed batches, their labels become cases, a reviewer updates the lessons.

    uv run python -m src.eval.reveal_rounds rounds <name> [--predictor] [--model ...] [--thinking high] [--until T]
    uv run python -m src.eval.reveal_rounds test <name> [--batch 200] [--workers 8] [--limit N]
    uv run python -m src.eval.reveal_rounds al                    # the DL active-learning arm (no LLM)

Design: docs/superpowers/specs/2026-11-06-mus22t-reveal-rounds-design.md. Data: build_mus22t_rsi (rsi/reveal.csv: 10
rounds of 100 pairs, 50 + 50; rsi/final.csv: 8,366 pairs), build_mus22t_rsi_sandbox (the checkpoint views). No
acquisition: every arm gets the same pairs in the same rounds and is scored on the same final set.

rounds: for t = 1..10 one fresh pi session predicts round t's 100 pairs (the pring_test batch protocol, RevealTools:
protein and KB tools, `cases`, and with --predictor `predict_pairs` from sandbox_rsi/round<t>/, i.e. checkpoint t - 1
of the DL active-learning arm). Its p is scored before the labels are shown. The system then adds the 100 pairs as
cases (label, the agent's p, the predictor's p it could see, its evidence ids) and a reviewer session updates the
lessons (memory/round<t>/, as in src.eval.memory_agent). The LLM-only arm (no --predictor) uses data/MUS22T/sandbox/
with no predictor tool and no predictor p in the cases. A round with saved memory is replayed, so runs resume.
curve.json: per round, the agent's metrics on its batch before the reveal and the DL checkpoints' on the same pairs.

test: the final pairs (seed 0, batches of --batch) with the memory after round 10 (lessons in the prompt, cases
through the tool), read-only; with --predictor the run's agent sees checkpoint 10 (sandbox_rsi/final/). summary.json:
the pring_test metrics of the agent, frozen V3 and the round-10 checkpoint on the same pairs, with paired bootstraps.

al: data/MUS22T/runs/al_v3/summary.json: every available checkpoint's metrics on the final set, and checkpoint t - 1
on round t's batch (the curve of the DL arm), from data/MUS22T/predictor/ (train_mus22t_al.sh on H20).
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from src.agent.memory import MAX_ACTIVE, Cases, Lessons, case_id, read_homologs
from src.agent.session import DEFAULT_MODEL, PROMPTS, prompt
from src.agent.session import pi_version as get_pi_version
from src.agent.tools import CAP, K, RevealReviewTools, RevealTools, Sandbox, tsv
from src.data.builders.build_mus22t_rsi import OUT as RSI
from src.data.builders.build_mus22t_rsi import ROUNDS, read
from src.data.builders.build_mus22t_rsi_sandbox import OUT as VIEWS
from src.data.builders.build_mus22t_rsi_sandbox import checkpoint_path, checkpoint_p, reveal_rounds
from src.data.paths import MUS22T
from src.eval.memory_agent import load_memory, save_memory, session_twice
from src.eval.pring_test import batches, classification, paired_bootstrap, run_batch, summarise
from src.eval.rounds import commit

RUNS = MUS22T / "runs"
PLAIN = MUS22T / "sandbox"  # the LLM-only arm: no predictor tool
PREDICTOR_TOOL = (
    "- A PPI predictor trained on the knowledge base and on every pair revealed before this batch (it is retrained "
    "after\n  each round): `predict_pairs` gives its probability p for up to 200 of the listed pairs in one call. "
    "It predicts\n  from the two protein sequences. Before any reveal it had validation AUPRC 0.718 and AUROC 0.712, "
    "and its p\n  was calibrated (expected calibration error 0.081).\n"
)


def system_prompt(sandbox, predictor, records, lessons, stage):
    return prompt(
        "reveal/system",
        proteins=len(sandbox.candidates),
        cap=CAP,
        stage=stage,
        cases=f"{len(records):,}",
        lessons=lessons.render(),
        predictor_tool=PREDICTOR_TOOL if predictor else "",
        predictor_step="the predictor's p, " if predictor else "",
        pred_example="`pred:A-B`, " if predictor else "",
    )


def casebook(records, sandbox):
    return Cases(records, sandbox.proteins, read_homologs(PLAIN / "homologs.tsv"))


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


def review(t, args, sandbox, records, fresh, lessons, run_dir):
    predictor = args.predictor
    system = prompt(
        "reveal/review_system",
        k=len(fresh),
        rounds=ROUNDS,
        final=len(read(RSI / "final.csv")),
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
    book = casebook(records, sandbox)
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
    files = sorted((PROMPTS / "reveal").glob("*.md")) + [PROMPTS / "pring_test" / f for f in ("batch.md", "nudge.md")]
    return {
        "model": args.model,
        "thinking": args.thinking,
        "predictor": args.predictor,
        "cap": CAP,
        "max_active_lessons": MAX_ACTIVE,
        "git": commit(),
        "pi": get_pi_version(),
        "prompts": {p.relative_to(PROMPTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files},
        "split": json.loads((RSI / "rsi.json").read_text())["outputs"],
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }


def round_sandbox(t, predictor):
    if not predictor:
        return Sandbox(str(PLAIN))
    view = VIEWS / f"round{t:02d}"
    assert (view / "scores.tsv.gz").exists(), f"no checkpoint {t - 1} view: train it and run build_mus22t_rsi_sandbox"
    return Sandbox(str(view))


def run_rounds(run_dir, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run.json").exists():
        old = json.loads((run_dir / "run.json").read_text())
        assert old["predictor"] == args.predictor, "resume with the run's own --predictor"
        args.model, args.thinking = old["model"], old["thinking"]
    else:
        (run_dir / "run.json").write_text(json.dumps(provenance(args), indent=1) + "\n")
    rounds = reveal_rounds()
    records, lessons = [], Lessons()
    for t in range(1, min(ROUNDS, args.until or ROUNDS) + 1):
        mem_dir = run_dir / "memory" / f"round{t:02d}"
        if (mem_dir / "cases.jsonl").exists():
            records, lessons = load_memory(mem_dir)
            continue
        sandbox = round_sandbox(t, args.predictor)
        stage = (
            f"This is round {t} of {ROUNDS}. After you submit, the labels of this batch are revealed and become "
            "cases; a reviewer then updates the lessons."
        )
        system = system_prompt(sandbox, args.predictor, records, lessons, stage)
        book = casebook(records, sandbox)
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
        seen = {(a, b): p for (a, b, p) in sandbox.query("SELECT a, b, p FROM scores")} if args.predictor else None
        fresh = new_cases(t, result, seen)
        records = records + fresh
        log = review(t, args, sandbox, records, fresh, lessons, run_dir)
        save_memory(mem_dir, records, lessons, log)
        print(f"{run_dir.name} round {t}: {len(lessons.active())} active lessons, {len(records)} cases")
    write_curve(run_dir)


def curve_rows(rows, t):
    """Metrics on round t's pairs of the agent (if rows give p) and of DL checkpoints 0 and t - 1."""
    y = [r["label"] for r in rows]
    out = {"round": t, "pairs": len(rows)}
    if "p" in rows[0]:
        out["agent"] = classification(y, [0.5 if r["p"] is None else r["p"] for r in rows])
    for name, c in (("v3_frozen", 0), ("al_checkpoint", t - 1)):
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
    if predictor:
        assert (VIEWS / "final" / "scores.tsv.gz").exists(), "no checkpoint 10 view: build_mus22t_rsi_sandbox"
    sandbox = Sandbox(str(VIEWS / "final" if predictor else PLAIN))
    book = casebook(records, sandbox)
    test_dir = run_dir / "final"
    test_dir.mkdir(exist_ok=True)
    todo = batches(read(RSI / "final.csv"), args.batch)
    if (test_dir / "run.json").exists():
        assert json.loads((test_dir / "run.json").read_text())["batch"] == args.batch, "resume with the same batch"
    else:
        extra = {"batch": args.batch, "batches": len(todo), "cases": len(records)}
        (test_dir / "run.json").write_text(json.dumps(provenance(args, **extra), indent=1) + "\n")
    stage = "This is the final test. Its labels are never revealed; the cases and lessons stay as they are."
    system = system_prompt(sandbox, predictor, records, lessons, stage)
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
    summary = summarise(rows, results, "v3")
    if checkpoint_path(ROUNDS).exists():
        al = checkpoint_p(ROUNDS)
        y = [r["label"] for r in rows]
        p = [0.5 if r["p"] is None else r["p"] for r in rows]
        q = [al[(r["a"], r["b"])] for r in rows]
        summary["al_checkpoint10"] = classification(y, q)
        summary["agent_minus_al_checkpoint10"] = paired_bootstrap(y, p, q)
    (test_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def run_al():
    final = read(RSI / "final.csv")
    y = [label for _, _, label in final]
    rounds = reveal_rounds()
    out = {"final_pairs": len(final), "checkpoints": {}, "curve": []}
    for c in range(ROUNDS + 1):
        if checkpoint_path(c).exists():
            p = checkpoint_p(c)
            out["checkpoints"][c] = classification(y, [p[(a, b)] for a, b, _ in final])
    for t in range(1, ROUNDS + 1):
        rows = [{"a": a, "b": b, "label": label} for a, b, label in rounds[t]]
        out["curve"].append(curve_rows(rows, t))
    if checkpoint_path(ROUNDS).exists():
        last, first = checkpoint_p(ROUNDS), checkpoint_p(0)
        out["checkpoint10_minus_v3"] = paired_bootstrap(
            y, [last[(a, b)] for a, b, _ in final], [first[(a, b)] for a, b, _ in final]
        )
    d = RUNS / "al_v3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({c: round(m["ap"], 4) for c, m in out["checkpoints"].items()}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("rounds", "test", "al"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--predictor", action="store_true", help="rounds: the agent sees the DL checkpoints")
    parser.add_argument("--until", type=int, help="rounds: stop after this round")
    parser.add_argument("--batch", type=int, default=K, help="test: pairs per session")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="test: run only the first N batches")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", default="high")
    args = parser.parse_args()
    if args.stage == "al":
        run_al()
        return
    assert args.name, f"{args.stage} needs a run name"
    (run_rounds if args.stage == "rounds" else run_test)(RUNS / args.name, args)


if __name__ == "__main__":
    main()
