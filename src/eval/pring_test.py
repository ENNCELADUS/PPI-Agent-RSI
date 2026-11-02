"""Classification test on PRING: the agent predicts every pair of PRING's human BFS test split.

    uv run python -m src.eval.pring_test <name> [--batch 200] [--workers 4] [--limit N]
        [--model deepseek/deepseek-flash] [--thinking high]

Design: docs/superpowers/specs/2026-11-01-pring-classification-test-design.md. The pairs are
hidden/test.csv (PRING's labelled test split as released: 32,019 positives incl. 1,891 self pairs, 32,019 negatives),
shuffled with seed 0 and cut into batches. The agent sees data/PRING/sandbox_cls/
(build_pring_cls: KB = train only). Each batch is one fresh pi session with protein and KB
tools only (PairTools: no predictor, no memory, no web) that calls submit_predictions once. --limit N
runs only the first N batches (a smoke run or a budget stop); a later call without it continues.

--data mus22t runs the same agent (same tools and backend) on the 22T mouse membrane split instead: pairs =
data/MUS22T/agent_split/test.csv (transductive, 4,683 positives, 4,683 PRING-sampled negatives), sandbox =
data/MUS22T/sandbox/ (build_mus22t_sandbox: KB = train), system prompt pring_test/system_mus22t.md, runs in
data/MUS22T/runs/; its predictor arm is V3 (B0 v3_1 trained on the 22T train split; build_mus22t_scores).

--predictor (mus22t only) gives the agent predict_pairs: V3's p for up to 200 pairs per call, from
sandbox/scores.tsv.gz, with system prompt pring_test/system_mus22t_predictor.md (design:
docs/superpowers/specs/2026-11-02-mus22t-llm-predictor-design.md). result.json records how many listed pairs
got a score. --cap sets the tool calls per batch (default 80, submit_predictions included).

Runs go to data/PRING/runs/<name>/batch<i>/: agent/ (pi session, tool calls, submission) and
result.json (labels and predictions; evaluator only). A batch with result.json is replayed, so a
stopped run resumes. A missing or invalid prediction scores p = 0.5 and counts against coverage. A
batch that fails (provider or transport error, timeout) without a submission is rerun once; after a
second failure it has no result.json and reruns on the next call. summary.json: AUROC, AP and Brier
of the agent over the scored batches, the predictor (B0 for PRING, V3 for 22T) on the same pairs, and the paired
bootstrap of agent - predictor (1,000 resamples, seed 0) for AUROC and AP.
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
from src.agent.tools import CAP, PairTools, Sandbox, TestTools, tsv
from src.data.ids import pair
from src.data.paths import HIDDEN, MUS22T, PRING, PRING_RAW
from src.eval.rounds import RUNS, commit, set_aside
from src.eval.score import read_pairs
from src.predictor.parity import average_ranks

SEED = 0
TEST_PROMPTS = PROMPTS / "pring_test"
DATA = {  # labelled pairs, agent-visible sandbox, run root, system prompt, predictor scored beside the agent
    "pring": (HIDDEN / "test.csv", PRING / "sandbox_cls", RUNS, "pring_test/system", "b0"),
    "mus22t": (
        MUS22T / "agent_split" / "test.csv",
        MUS22T / "sandbox",
        MUS22T / "runs",
        "pring_test/system_mus22t",
        "v3",
    ),
}


def test_pairs(path):
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


def predictor_p(name, pairs):
    """The predictor's probability for each accession pair: B0's raw sigmoid (PRING) or V3's (22T sandbox)."""
    if name == "b0":
        return {k: float(1 / (1 + np.exp(-x))) for k, x in b0_logits(pairs).items()}  # logit 0 = 0.5
    with gzip.open(MUS22T / "sandbox" / "scores.tsv.gz", "rt") as f:
        return {(r["a"], r["b"]): float(r["p"]) for r in csv.DictReader(f, delimiter="\t")}


def paired_bootstrap(y, p, q, n=1000, seed=SEED):
    """AUROC and AP of p minus q on the same pairs: [difference, 2.5 %, 97.5 %] over n resamples of the pairs."""
    y, p, q = np.asarray(y), np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    rng = np.random.default_rng(seed)
    diff = lambda i: {k: v - metrics(y[i], q[i])[k] for k, v in metrics(y[i], p[i]).items()}  # noqa: E731
    boot = [diff(rng.integers(0, len(y), len(y))) for _ in range(n)]
    whole = diff(np.arange(len(y)))
    return {k: [whole[k], *np.percentile([b[k] for b in boot], [2.5, 97.5]).tolist()] for k in whole}


def summarise(rows, results, predictor="b0"):
    """predictor: "b0", "v3" or None (the agent only)."""
    scored = [r for r in rows if r["p"] is not None]
    pred = predictor_p(predictor, [(r["a"], r["b"]) for r in rows]) if predictor else None
    stats = [x["stats"] or {} for x in results]
    used = [x["predictor_pairs"] for x in results if x.get("predictor_pairs") is not None]

    def arms(rows):
        y = [r["label"] for r in rows]
        p = [0.5 if r["p"] is None else r["p"] for r in rows]
        if pred is None:
            return {"agent": classification(y, p)}
        q = [pred[(r["a"], r["b"])] for r in rows]
        return {
            "agent": classification(y, p),
            predictor: classification(y, q),
            f"agent_minus_{predictor}": paired_bootstrap(y, p, q),
        }

    return {
        "batches": len(results),
        "pairs": len(rows),
        "positives": sum(r["label"] for r in rows),
        "self_pairs": sum(r["a"] == r["b"] for r in rows),
        "coverage": len(scored) / len(rows),
        "rejected_rows": sum(x["rejected"] for x in results),
        "failed_batches": sum(x["submission"] is None for x in results),
        "predictor_coverage": sum(used) / sum(len(x["rows"]) for x in results) if used else None,
        **arms(rows),
        "non_self": arms([r for r in rows if r["a"] != r["b"]]),
        "tool_calls": sum(x["tool_calls"] for x in results),
        "tokens": sum((s.get("tokens") or {}).get("total", 0) for s in stats),
        "cost_usd": sum(s.get("cost", 0) for s in stats),
    }


def provenance(args, n_batches, labels):
    return {
        "data": args.data,
        "model": args.model,
        "thinking": args.thinking,
        "batch": args.batch,
        "batches": n_batches,
        "seed": SEED,
        "cap": args.cap,
        "predictor_tool": args.predictor,
        "git": commit(),
        "pi": get_pi_version(),
        "prompts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(TEST_PROMPTS.glob("*.md"))},
        "test_csv": hashlib.sha256(labels.read_bytes()).hexdigest()[:16],
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_batch(i, batch, n_batches, sandbox, system, run_dir, args, make_tools):
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
        tools = make_tools(batch_dir / "agent")
        out = run_session(tools, batch_dir / "agent", system, message, nudge, model=args.model, thinking=args.thinking)
        if out["submission"] is not None or not out["error"]:
            break
    else:
        print(f"batch {i} failed twice: {out['error']}")
        return None
    rows, rejected = evaluate(out["submission"], batch)
    result = {"batch": i, "rows": rows, "rejected": rejected, "failed_attempt": attempt == 2}
    if isinstance(tools, TestTools):  # listed pairs whose predictor p the agent received
        result["predictor_pairs"] = len(tools.scored & {(a, b) for a, b, _ in batch})
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
    parser.add_argument("--data", choices=DATA, default="pring")
    parser.add_argument("--predictor", action="store_true", help="give the agent predict_pairs (22T only)")
    parser.add_argument("--cap", type=int, default=CAP, help="tool calls per batch")
    args = parser.parse_args()
    assert not args.predictor or args.data == "mus22t", (
        "--predictor needs --data mus22t (PRING sandbox_cls has no scores)"
    )

    labels, sandbox_dir, runs, system_prompt, predictor = DATA[args.data]
    if args.predictor:
        system_prompt += "_predictor"
    run_dir = runs / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    todo = batches(test_pairs(labels), args.batch)
    if (run_dir / "run.json").exists():
        old = json.loads((run_dir / "run.json").read_text())
        assert old["batch"] == args.batch, "resume with the run's own batch size"
        assert old.get("data", "pring") == args.data, "resume with the run's own data"
        assert (old.get("cap", CAP), old.get("predictor_tool", False)) == (args.cap, args.predictor), "same cap, arm"
    else:
        (run_dir / "run.json").write_text(json.dumps(provenance(args, len(todo), labels), indent=1) + "\n")
    sandbox = Sandbox(str(sandbox_dir))
    system = prompt(system_prompt, proteins=len(sandbox.candidates), cap=args.cap)
    make_tools = (
        (lambda d: TestTools(sandbox, d, cap=args.cap))
        if args.predictor
        else (lambda d: PairTools(sandbox, d, args.cap))
    )
    n = len(todo)
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [
            pool.submit(run_batch, i, b, n, sandbox, system, run_dir, args, make_tools)
            for i, b in enumerate(todo[: args.limit], 1)
        ]
        results = [r for j in jobs if (r := j.result()) is not None]
    print(f"{len(results)} of {len(jobs)} batches scored")
    if not results:
        return
    summary = summarise([row for r in results for row in r["rows"]], results, predictor)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
