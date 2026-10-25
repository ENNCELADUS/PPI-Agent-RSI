"""Run PRING wave discovery with the pi agent: waves of K submitted pairs, oracle results, true feedback.

    uv run python -m src.eval.waves <name> <R> [--waves 10] [--until T] [--model ...] [--thinking ...]

Design: docs/superpowers/specs/2026-10-23-pring-wave-discovery-design.md. Runs replicate R from wave 1;
after each wave the agent's memory gets the measured result of every valid pair. --until T stops after
wave T (budget); the agent is still told the run has --waves waves, so a later resume continues it.
Runs go to data/PRING/runs/<name>/r<R>/wave<t>/: agent/ (pi session, tool calls, submission;
what the agent produced) and result.json (validity and oracle results; evaluator only). A wave whose
result.json exists is replayed, so a stopped run resumes.

- Validity: only the first K rows count; a row is valid if both ids are candidate proteins, it is not
  a self pair, not tested in an earlier wave and not repeated. Invalid rows are misses, never refilled.
- A wave that fails (provider or transport error, timeout) without a submission is rerun once; the
  failed attempt's agent/ is kept as agent_failed/. A second failure stops the run with no result.json,
  so it resumes at that wave: an infrastructure failure (e.g. an empty API balance) is not the agent's
  empty wave. Only a session that settles without a submission and without an error scores as empty.
  An attempt cut off by a killed run is kept as agent_interrupted/ and the wave reruns. result.json records the
  model, thinking level, wave count and git commit of the attempt it scores. DeepSeek's thinking mode ignores
  temperature and has no seed, so replicates differ by sampling.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime

from src.agent.session import PROMPTS, run_wave
from src.agent.session import pi_version as get_pi_version
from src.agent.tools import K, pair
from src.data.paths import HIDDEN, PRING, SANDBOX
from src.eval.oracle import Oracle

RUNS = PRING / "runs"


def proteins():
    rows = csv.DictReader(open(SANDBOX / "proteins.tsv"), delimiter="\t")
    return {r["accession"]: r["candidate"] == "1" for r in rows}


def classify(rows, known, tested):
    """(canonical pair or None, status) per submitted row."""
    seen, out = set(), []
    for i, r in enumerate(rows):
        a, b = r.get("a"), r.get("b")
        p = pair(a, b) if isinstance(a, str) and isinstance(b, str) else None
        if i >= K:
            status = "over budget"
        elif p is None or a not in known or b not in known:
            status = "unknown id"
        elif not (known[a] and known[b]):
            status = "not a candidate"
        elif a == b:
            status = "self"
        elif p in tested:
            status = "already tested"
        elif p in seen:
            status = "duplicate"
        else:
            status = "valid"
            seen.add(p)
        out.append((p, status))
    return out


def feedback(wave, rows):
    """Agent-visible memory facts from one wave's evaluated rows: the measured result of each valid pair."""
    result = {1: "hit", 0: "not detected"}
    return [
        {"a": r["a"], "b": r["b"], "wave": wave, "result": result[r["hit"]], "source": "oracle"}
        for r in rows
        if r["status"] == "valid"
    ]


def evaluate(out, known, tested, oracle, wave):
    rows = out["submission"] or []
    statuses = classify(rows, known, tested)
    hits = oracle([p for p, s in statuses if s == "valid"])
    evaluated = []
    for r, (p, status) in zip(rows, statuses):
        row = {"a": p[0] if p else r.get("a"), "b": p[1] if p else r.get("b"), "status": status}
        row |= {k: r.get(k) for k in ("ranking_score", "uncertainty")}
        if status == "valid":
            row["hit"] = hits[p]
        evaluated.append(row)
    return {
        "wave": wave,
        "submitted": len(rows),
        "valid": len(hits),
        "hits": sum(hits.values()),
        "rows": evaluated,
        **{
            k: out.get(k)
            for k in ("nudged", "aborted", "error", "stop_reasons", "tool_calls", "stats", "failed_attempt")
        },
    }


def set_aside(wave_dir, name):
    """Move an earlier attempt's agent/ to the first free <name>, <name>2, ... so logs never mix."""
    if (wave_dir / "agent").exists():
        target = next(wave_dir / f"{name}{i or ''}" for i in range(100) if not (wave_dir / f"{name}{i or ''}").exists())
        shutil.move(wave_dir / "agent", target)


def setup(args):
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {
        "model": args.model,
        "thinking": args.thinking,
        "waves": args.waves,
        "git": git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "src") else ""),
    }


def provenance(replicate, args):
    return {
        "replicate": replicate,
        "k": K,
        **setup(args),
        "sandbox": str(SANDBOX.relative_to(PRING)),
        "pi": get_pi_version(),
        "prompts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(PROMPTS.glob("*.md"))},
        "tools_build": json.loads((PRING / "tools.json").read_text())["outputs"],
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run(run_dir, replicate, args):
    """Run (or resume) waves 1..args.waves (or ..args.until) of one replicate in run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "run.json").exists():
        (run_dir / "run.json").write_text(json.dumps(provenance(replicate, args), indent=1) + "\n")
    known = proteins()
    oracle = Oracle(HIDDEN / "test_positives.csv")
    tested, memory = set(), []
    for t in range(1, min(args.waves, args.until or args.waves) + 1):
        wave_dir = run_dir / f"wave{t:02d}"
        result_path = wave_dir / "result.json"
        if not result_path.exists():
            set_aside(wave_dir, "agent_interrupted")  # a run killed mid-wave left this attempt unscored
            for attempt in (1, 2):
                if attempt == 2:
                    set_aside(wave_dir, "agent_failed")
                out = run_wave(
                    SANDBOX,
                    wave_dir,
                    tested,
                    memory,
                    wave=t,
                    waves=args.waves,
                    model=args.model,
                    thinking=args.thinking,
                )
                out["failed_attempt"] = attempt == 2
                if out["submission"] is not None or not out["error"]:
                    break
            else:
                raise RuntimeError(f"{run_dir.relative_to(RUNS)} wave {t} failed twice: {out['error']}")
            result = evaluate(out, known, tested, oracle, t) | {"setup": setup(args)}
            result_path.write_text(json.dumps(result, indent=1) + "\n")
        result = json.loads(result_path.read_text())
        tested |= {(r["a"], r["b"]) for r in result["rows"] if r["status"] == "valid"}
        memory += feedback(t, result["rows"])
        print(
            f"{run_dir.relative_to(RUNS)} wave {t}: {result['hits']} hits / {result['valid']} valid "
            f"/ {result['submitted']} submitted" + (f" error: {result['error']}" if result["error"] else "")
        )
    with open(run_dir / "memory.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name")
    parser.add_argument("replicate", type=int)
    parser.add_argument("--waves", type=int, default=10)
    parser.add_argument("--until", type=int, help="stop after this wave; prompts still say 'of --waves'")
    parser.add_argument("--model", default="deepseek/deepseek-flash")
    parser.add_argument("--thinking", default="high")
    args = parser.parse_args()
    run(RUNS / args.name / f"r{args.replicate}", args.replicate, args)


if __name__ == "__main__":
    main()
