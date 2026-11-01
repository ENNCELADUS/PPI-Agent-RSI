"""Classification test on PRING: the agent predicts every pair of PRING's human BFS test split.

    uv run python -m src.eval.pring_test <name> [--batch 200] [--workers 4] [--limit N]
        [--model deepseek/deepseek-flash] [--thinking high]

Design: docs/superpowers/specs/2026-11-01-pring-classification-test-design.md. The pairs are
hidden/test.csv (PRING's labelled test split as released: 32,019 positives incl. 1,891 self pairs, 32,019 negatives),
shuffled with seed 0 and cut into batches. The agent sees data/PRING/sandbox_cls/
(build_pring_cls: KB = train only). Each batch is one fresh pi session with protein and KB
tools only (PairTools: no predictor, no memory, no web) that calls submit_predictions once. --limit N
runs only the first N batches (a smoke run or a budget stop); a later call without it continues.

Runs go to data/PRING/runs/<name>/batch<i>/: agent/ (pi session, tool calls, submission) and
result.json (labels and predictions; evaluator only). A batch with result.json is replayed, so a
stopped run resumes. A missing or invalid prediction scores p = 0.5 and counts against coverage. A
batch that fails (provider or transport error, timeout) without a submission is rerun once; after a
second failure it has no result.json and reruns on the next call. summary.json: AUROC, AP and Brier
of the agent over the scored batches, and B0's AUROC and AP on the same pairs (evaluator-side logits).
"""

import argparse
import csv
import gzip
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np

from src.agent.session import PROMPTS, prompt, run_session
from src.agent.session import pi_version as get_pi_version
from src.agent.tools import CAP, PairTools, Sandbox, tsv
from src.data.ids import pair
from src.data.paths import HIDDEN, PRING, PRING_RAW
from src.eval.rounds import RUNS, commit, set_aside
from src.eval.score import read_pairs
from src.predictor.parity import average_ranks

SEED = 0
TEST_PROMPTS = PROMPTS / "pring_test"


def test_pairs(path=HIDDEN / "test.csv"):
    return [(*p, int(r["isInteraction"])) for p, r in read_pairs(path)]


def batches(pairs, size, seed=SEED):
    pairs = sorted(pairs)
    random.Random(seed).shuffle(pairs)
    return [pairs[i : i + size] for i in range(0, len(pairs), size)]


def evaluate(submission, batch):
    """Each listed pair with its label and the agent's p (None if missing or invalid), and the rejected row count."""
    listed = {(a, b) for a, b, _ in batch}
    p, rejected = {}, 0
    for r in submission or []:
        a, b, x = r.get("a"), r.get("b"), r.get("p")
        k = pair(a, b) if isinstance(a, str) and isinstance(b, str) else None
        if k in listed and k not in p and type(x) in (int, float) and 0 <= x <= 1:
            p[k] = float(x)
        else:
            rejected += 1  # unlisted, repeated or malformed
    return [{"a": a, "b": b, "label": y, "p": p.get((a, b))} for a, b, y in batch], rejected


def steps(y, s):
    """Each distinct score, best first (ties form one step), with true and predicted positives at or above it."""
    order = np.argsort(-s, kind="stable")
    ss = s[order]
    last = np.r_[ss[1:] != ss[:-1], True]
    return ss[last], np.cumsum(y[order])[last], np.arange(1, len(y) + 1)[last]


def metrics(y, s):
    """AUROC (ties count half) and average precision."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos = int(y.sum())
    neg = len(y) - pos
    auroc = (average_ranks(s)[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    _, tp, n = steps(y, s)
    ap = float((np.diff(np.r_[0, tp]) * tp / n).sum() / pos)
    return {"auroc": float(auroc), "ap": ap}


def classification(y, prob, bins=15):
    """PRING Table 8 metrics and MCC, specificity at the max-F1 threshold chosen on these labels (oracle: val is
    RSI data, not a selector), plus ECE and Brier."""
    y, prob = np.asarray(y), np.asarray(prob, dtype=float)
    t, tp_at, n_at = steps(y, prob)
    threshold = float(t[np.argmax(2 * tp_at / (n_at + y.sum()))])  # F1 = 2tp / (predicted + positives)
    pred = prob >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    which = np.minimum((prob * bins).astype(int), bins - 1)
    ece = sum(abs(prob[which == b].mean() - y[which == b].mean()) * (which == b).mean() for b in np.unique(which))
    return metrics(y, prob) | {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0,
        "ece": float(ece),
        "brier": float(np.mean((prob - y) ** 2)),
    }


def b0_logits(pairs):
    """B0's logit for each accession pair, from the evaluator-side candidate scores (node ids)."""
    node = {
        r["uniprot_id"]: r["node_id"] for r in csv.DictReader(open(PRING_RAW / "node_sequences.tsv"), delimiter="\t")
    }
    want = {pair(node[a], node[b]): (a, b) for a, b in pairs}
    out = {}
    with gzip.open(PRING / "predictor" / "pring_candidates.tsv.gz", "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if (k := pair(r["node_a"], r["node_b"])) in want:
                out[want[k]] = float(r["logit"])
    return out


def summarise(rows, results):
    scored = [r for r in rows if r["p"] is not None]
    b0 = b0_logits([(r["a"], r["b"]) for r in rows])
    stats = [x["stats"] or {} for x in results]

    def arms(rows):
        y = [r["label"] for r in rows]
        p = [0.5 if r["p"] is None else r["p"] for r in rows]
        b0_p = 1 / (1 + np.exp(-np.array([b0[(r["a"], r["b"])] for r in rows])))  # raw sigmoid: logit 0 = 0.5
        return {"agent": classification(y, p), "b0": classification(y, b0_p)}

    return {
        "batches": len(results),
        "pairs": len(rows),
        "positives": sum(r["label"] for r in rows),
        "self_pairs": sum(r["a"] == r["b"] for r in rows),
        "coverage": len(scored) / len(rows),
        "rejected_rows": sum(x["rejected"] for x in results),
        "failed_batches": sum(x["submission"] is None for x in results),
        **arms(rows),
        "non_self": arms([r for r in rows if r["a"] != r["b"]]),
        "tool_calls": sum(x["tool_calls"] for x in results),
        "tokens": sum((s.get("tokens") or {}).get("total", 0) for s in stats),
        "cost_usd": sum(s.get("cost", 0) for s in stats),
    }


def provenance(args, n_batches):
    return {
        "model": args.model,
        "thinking": args.thinking,
        "batch": args.batch,
        "batches": n_batches,
        "seed": SEED,
        "cap": CAP,
        "git": commit(),
        "pi": get_pi_version(),
        "prompts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(TEST_PROMPTS.glob("*.md"))},
        "test_csv": hashlib.sha256((HIDDEN / "test.csv").read_bytes()).hexdigest()[:16],
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_batch(i, batch, n_batches, sandbox, system, run_dir, args):
    batch_dir = run_dir / f"batch{i:04d}"
    result_path = batch_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    set_aside(batch_dir, "agent_interrupted")
    genes = [(a, sandbox.gene(a), b, sandbox.gene(b)) for a, b, _ in batch]
    message = prompt(
        "pring_test/batch", batch=i, batches=n_batches, k=len(batch), pairs=tsv(["a", "gene_a", "b", "gene_b"], genes)
    )
    nudge = prompt("pring_test/nudge", k=len(batch))
    for attempt in (1, 2):
        if attempt == 2:
            set_aside(batch_dir, "agent_failed")
        tools = PairTools(sandbox, batch_dir / "agent")
        out = run_session(tools, batch_dir / "agent", system, message, nudge, model=args.model, thinking=args.thinking)
        if out["submission"] is not None or not out["error"]:
            break
    else:
        print(f"batch {i} failed twice: {out['error']}")
        return None
    rows, rejected = evaluate(out["submission"], batch)
    result = {"batch": i, "rows": rows, "rejected": rejected, "failed_attempt": attempt == 2}
    result |= {
        k: out.get(k) for k in ("submission", "nudged", "aborted", "error", "stop_reasons", "tool_calls", "stats")
    }
    result_path.write_text(json.dumps(result, indent=1) + "\n")
    predicted = sum(r["p"] is not None for r in rows)
    print(f"batch {i}: {predicted}/{len(rows)} predicted, cost {(out['stats'] or {}).get('cost')}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="run only the first N batches")
    parser.add_argument("--model", default="deepseek/deepseek-flash")
    parser.add_argument("--thinking", default="high")
    args = parser.parse_args()

    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    todo = batches(test_pairs(), args.batch)
    if (run_dir / "run.json").exists():
        old = json.loads((run_dir / "run.json").read_text())
        assert old["batch"] == args.batch, "resume with the run's own batch size"
    else:
        (run_dir / "run.json").write_text(json.dumps(provenance(args, len(todo)), indent=1) + "\n")
    sandbox = Sandbox(str(PRING / "sandbox_cls"))
    system = prompt("pring_test/system", proteins=len(sandbox.candidates), cap=CAP)
    n = len(todo)
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [
            pool.submit(run_batch, i, b, n, sandbox, system, run_dir, args) for i, b in enumerate(todo[: args.limit], 1)
        ]
        results = [r for j in jobs if (r := j.result()) is not None]
    print(f"{len(results)} of {len(jobs)} batches scored")
    if not results:
        return
    summary = summarise([row for r in results for row in r["rows"]], results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
