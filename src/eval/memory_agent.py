"""Run the two-layer memory agent: discovery rounds with case memory and reviewer-updated lessons, then the final test.

    uv run python -m src.eval.memory_agent rounds <name> [--rounds 10] [--until T] [--model ...] [--thinking ...]
    uv run python -m src.eval.memory_agent test <name> --arm memory|cases|empty [--no-predictor] [--limit N]

Design: docs/superpowers/specs/2026-11-02-two-layer-memory-agent-design.md. Data: build_pring_memory
(sandbox_disc/, sandbox_final/, hidden/final_test.csv). Runs go to data/PRING/runs/<name>/.

rounds: for t = 1 .. --rounds, round<t>/ holds the selector session (agent/) and result.json as in src.eval.rounds
(same validity rules, failure handling and resume); the system then adds one case per valid row (B0 p from the
discovery view's scores, the selector's scores, evidence and rationale, the oracle result) and a reviewer session
(round<t>/review/) updates the lessons through one update_lessons call. memory/round<t>/cases.jsonl and
lessons.json are the state after round t; the selector of round t + 1 reads them. A round with saved memory is
replayed. A reviewer that fails twice on infrastructure stops the run; one that settles without a call keeps the
lessons unchanged (logged in memory/round<t>/review.json).

test: the final-test pairs (seed 0, batches of 200) in data/PRING/runs/<name>/test_<arm>/ with the pring_test
protocol and metrics. memory: lessons (system prompt) + cases from memory/round<--rounds>/; cases: cases only;
empty: nothing from the rounds. The memory is read-only. summary.json adds B0 on the same pairs and every metric
without ribosome-ribosome pairs. --no-predictor (test_<arm>_nopred/): no predict_pairs tool and no predictor p in
the prompt or the `cases` table.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from src.agent.memory import MAX_ACTIVE, Cases, Lessons, case_id, read_homologs
from src.agent.session import PROMPTS, prompt, run_session
from src.agent.session import pi_version as get_pi_version
from src.agent.tools import CAP, K, CaseRoundTools, ReviewTools, Sandbox, TestTools, tsv
from src.data.builders.build_pring_noribo import is_ribosomal
from src.data.paths import HIDDEN, PRING
from src.eval.oracle import Oracle
from src.eval.pring_test import batches, run_batch, summarise, test_pairs
from src.eval.rounds import RUNS, commit, evaluate, proteins, set_aside

DISC, FINAL = PRING / "sandbox_disc", PRING / "sandbox_final"
FINAL_TEST = HIDDEN / "final_test.csv"
MEMORY_PROMPTS = PROMPTS / "memory"


def setup(args):
    return {"model": args.model, "thinking": args.thinking, "rounds": args.rounds, "git": commit()}


def provenance(args, **extra):
    files = sorted(MEMORY_PROMPTS.glob("*.md")) + sorted((PROMPTS / "pring_test").glob("*.md"))
    return {
        **setup(args),
        **extra,
        "k": K,
        "cap": CAP,
        "max_active_lessons": MAX_ACTIVE,
        "pi": get_pi_version(),
        "prompts": {p.relative_to(PROMPTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in files},
        "split": json.loads((PRING / "memory_split.json").read_text())["outputs"],
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def new_cases(result, submission, b0):
    """One case per valid row of a round: system-written, from result.json and the selector's submission."""
    out = []
    for r, s in zip(result["rows"], submission or []):
        if r["status"] == "valid":
            out.append(
                {
                    "round": result["round"],
                    "a": r["a"],
                    "b": r["b"],
                    "result": "hit" if r["hit"] else "not detected",
                    "b0_p": b0(r["a"], r["b"]),
                    "ranking_score": s.get("ranking_score"),
                    "uncertainty": s.get("uncertainty"),
                    "rationale": s.get("rationale"),
                    "evidence_ids": s.get("evidence_ids"),
                }
            )
    return out


def casebook(records, view):
    return Cases(records, Sandbox(str(DISC)).proteins, read_homologs(view / "case_homologs.tsv"))


def case_table(records, gene):
    rows = [(case_id(c["a"], c["b"]), gene(c["a"]), gene(c["b"]), c["result"], f"{c['b0_p']:.4g}") for c in records]
    return tsv(["case", "gene_a", "gene_b", "result", "b0_p"], rows)


def load_memory(mem_dir):
    records = [json.loads(line) for line in open(mem_dir / "cases.jsonl")]
    return records, Lessons(json.loads((mem_dir / "lessons.json").read_text()))


def save_memory(mem_dir, records, lessons, review):
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "review.json").write_text(json.dumps(review, indent=1) + "\n")
    (mem_dir / "lessons.json").write_text(json.dumps(lessons.all, indent=1) + "\n")
    with open(mem_dir / "cases.jsonl", "w") as f:  # written last: its presence marks the round's memory as saved
        f.writelines(json.dumps(c) + "\n" for c in records)


def session_twice(make_tools, agent_dir, system, message, nudge, args, what):
    """Run a session in agent_dir, rerun once after an infrastructure failure without a submission; raise after
    the second. Earlier attempts move to <agent_dir>_interrupted (a killed run) and <agent_dir>_failed."""
    parent, sub = agent_dir.parent, agent_dir.name
    set_aside(parent, f"{sub}_interrupted", sub)
    for attempt in (1, 2):
        if attempt == 2:
            set_aside(parent, f"{sub}_failed", sub)
        out = run_session(make_tools(), agent_dir, system, message, nudge, model=args.model, thinking=args.thinking)
        out["failed_attempt"] = attempt == 2
        if out["submission"] is not None or not out["error"]:
            return out
    raise RuntimeError(f"{what} failed twice: {out['error']}")


def review(t, args, sandbox, records, fresh, lessons, tested, round_dir):
    """The reviewer session after round t; returns the review log (operations and the system's report)."""
    agent_dir = round_dir / "review"
    n = len(sandbox.candidates)
    system = prompt("memory/review_system", k=K, proteins=n, max_active=MAX_ACTIVE, cap=CAP)
    message = prompt(
        "memory/review",
        round=t,
        rounds=args.rounds,
        hits=sum(c["result"] == "hit" for c in fresh),
        valid=len(fresh),
        cases=len(records),
        lessons=lessons.render(),
        new_cases=case_table(fresh, sandbox.gene) if fresh else "(no valid pairs this round)",
    )
    book = casebook(records, DISC)
    out = session_twice(
        lambda: ReviewTools(sandbox, agent_dir, tested, book),
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


def run_rounds(run_dir, args):
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "rounds.json").exists():
        (run_dir / "rounds.json").write_text(json.dumps(provenance(args, sandbox="sandbox_disc"), indent=1) + "\n")
    sandbox = Sandbox(str(DISC))
    known = proteins(DISC)
    oracle = Oracle(HIDDEN / "test_positives.csv")
    b0 = lambda a, b: sandbox.query("SELECT p FROM scores WHERE a = ? AND b = ?", [a, b])[0][0]  # noqa: E731
    n = len(sandbox.candidates)
    system = prompt("memory/system", proteins=n, pairs=n * (n - 1) // 2, rounds=args.rounds, k=K, cap=CAP, window=5 * K)
    tested, records, lessons = set(), [], Lessons()
    for t in range(1, min(args.rounds, args.until or args.rounds) + 1):
        round_dir, mem_dir = run_dir / f"round{t:02d}", run_dir / "memory" / f"round{t:02d}"
        result_path = round_dir / "result.json"
        if not result_path.exists():
            book = casebook(records, DISC)
            message = prompt(
                "memory/round", round=t, rounds=args.rounds, tested=len(tested), k=K, lessons=lessons.render()
            )
            out = session_twice(
                lambda: CaseRoundTools(sandbox, round_dir / "agent", tested, book),
                round_dir / "agent",
                system,
                message,
                prompt("nudge", k=K),
                args,
                f"round {t}",
            )
            result = evaluate(out, known, tested, oracle, t) | {"setup": setup(args)}
            result_path.write_text(json.dumps(result, indent=1) + "\n")
        result = json.loads(result_path.read_text())
        submission = json.loads((round_dir / "agent" / "submission.json").read_text()) if result["submitted"] else []
        fresh = new_cases(result, submission, b0)
        tested |= {(c["a"], c["b"]) for c in fresh}
        records = records + fresh
        if (mem_dir / "cases.jsonl").exists():
            records, lessons = load_memory(mem_dir)
        else:
            log = review(t, args, sandbox, records, fresh, lessons, tested, round_dir)
            save_memory(mem_dir, records, lessons, log)
        print(
            f"{run_dir.name} round {t}: {result['hits']} hits / {result['valid']} valid / {result['submitted']} "
            f"submitted; {len(lessons.active())} active lessons"
        )


def memory_section(arm, records, lessons):
    if arm == "empty":
        return "", "", "", ""
    text = (
        f"\n## Memory from an earlier screen\n- An earlier screen of other human proteins (none of them a test "
        f"protein) tested {len(records):,} pairs; each tested pair is a case with its measured result (`hit` or "
        f"`not detected`; a pair not detected may still interact). Use the `cases` tool to find cases of similar "
        f"proteins.\n"
    )
    if arm == "memory":
        text += (
            "- A reviewer distilled these lessons from the cases. They are hypotheses, not facts: check them "
            f"against the cases and the evidence.\n\nLessons:\n{lessons.render()}\n"
        )
    cases_tool = "- `cases`: the earlier screen's tested pairs and results; similar_to finds cases of homologs.\n"
    return (
        text,
        cases_tool,
        ", and cases of similar proteins",
        "`case:A-B`, " + ("`lesson:L3`, " if arm == "memory" else ""),
    )


def run_test(run_dir, args):
    test_dir = run_dir / (f"test_{args.arm}" + ("" if args.predictor else "_nopred"))
    test_dir.mkdir(parents=True, exist_ok=True)
    records, lessons = [], Lessons()
    if args.arm != "empty":
        mem_dir = run_dir / "memory" / f"round{args.rounds:02d}"
        assert (mem_dir / "cases.jsonl").exists(), f"no memory after round {args.rounds}: run the rounds first"
        records, lessons = load_memory(mem_dir)
    sandbox = Sandbox(str(FINAL))
    book = casebook(records, FINAL) if args.arm != "empty" else None
    todo = batches(test_pairs(FINAL_TEST), args.batch)
    if (test_dir / "run.json").exists():
        assert json.loads((test_dir / "run.json").read_text())["batch"] == args.batch, (
            "resume with the run's batch size"
        )
    else:
        extra = {
            "arm": args.arm,
            "predictor_tool": args.predictor,
            "batch": args.batch,
            "batches": len(todo),
            "cases": len(records),
        }
        extra["test_csv"] = hashlib.sha256(FINAL_TEST.read_bytes()).hexdigest()[:16]
        (test_dir / "run.json").write_text(json.dumps(provenance(args, **extra), indent=1) + "\n")
    memory, cases_tool, cases_step, examples = memory_section(args.arm, records, lessons)
    system = prompt(
        "memory/test_system",
        k=K,
        proteins=len(sandbox.candidates),
        cap=CAP,
        memory=memory,
        cases_tool=cases_tool,
        cases_step=cases_step,
        evidence_examples=examples,
        predictor_tool=(
            "- A fixed PPI predictor trained on the knowledge base: `predict_pairs` gives its calibrated probability p "
            f"for up\n  to {K} pairs of test proteins in one call.\n"
        )
        if args.predictor
        else "",
        predictor_step="the predictor's p, " if args.predictor else "",
        pred_example="`pred:A-B`, " if args.predictor else "",
    )
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
                lambda d: TestTools(sandbox, d, book, predictor=args.predictor),
            )
            for i, b in enumerate(todo[: args.limit], 1)
        ]
        results = [r for j in jobs if (r := j.result()) is not None]
    print(f"{len(results)} of {len(jobs)} batches scored")
    if not results:
        return
    rows = [row for r in results for row in r["rows"]]
    ribo = {x for x in sandbox.candidates if is_ribosomal(sandbox.gene(x))}
    no_ribo = [r for r in rows if not (r["a"] in ribo and r["b"] in ribo)]
    summary = summarise(rows, results)
    summary["no_ribosome"] = {
        k: v for k, v in summarise(no_ribo, results).items() if k in ("pairs", "positives", "agent", "b0")
    }
    (test_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("rounds", "test"))
    parser.add_argument("name")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--until", type=int, help="rounds: stop after this round; prompts still say 'of --rounds'")
    parser.add_argument(
        "--arm", choices=("memory", "cases", "empty"), help="test: what the agent keeps from the rounds"
    )
    parser.add_argument(
        "--no-predictor", dest="predictor", action="store_false", help="test: no predict_pairs, no b0_p in cases"
    )
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="test: run only the first N batches")
    parser.add_argument("--model", default="deepseek/deepseek-flash")
    parser.add_argument("--thinking", default="high")
    args = parser.parse_args()
    run_dir = RUNS / args.name
    if args.stage == "rounds":
        run_rounds(run_dir, args)
    else:
        assert args.arm, "test needs --arm"
        run_test(run_dir, args)


if __name__ == "__main__":
    main()
