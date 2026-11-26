"""22T reveal rounds: the agent classifies each round's batch, its labels become cases, a reviewer updates the lessons.

    uv run python -m src.eval.reveal_rounds rounds <name> [--predictor] [--model ...] [--thinking high] [--until T]
    uv run python -m src.eval.reveal_rounds test <name> [--batch 200] [--workers 8] [--limit N]
    uv run python -m src.eval.reveal_rounds baseline <name> [--predictor] [--batch 200] [--workers 8]
    uv run python -m src.eval.reveal_rounds al --data mus22t|pring --rule random|greedy  # DL predictor arm, no LLM
    uv run python -m src.eval.reveal_rounds rounds|test|baseline <name> --data pring [--predictor]   # on PRING

Design: docs/superpowers/specs/2026-11-06-mus22t-reveal-rounds-design.md, round rules of
docs/superpowers/specs/2026-11-26-archive-and-one-gate-design.md. Data: build_mus22t_rsi (rsi/dev.csv: the
development set of 10,000 pairs; rsi/reveal.csv: 10 rounds of 100 pairs drawn from it at random; rsi/final.csv: 8,366
pairs), build_mus22t_rsi_sandbox (the views: the RSI training set as kb.csv, one view per DL checkpoint and stage). The
acquisition rule is the random draw, the control: every arm gets the same pairs in the same rounds and is scored on
the same final set. The agent's own picks are not built yet.

rounds: the state in use is every revealed case (the archive), the lessons in use and, with --predictor on 22T, DL
checkpoint c of the random rule; it starts empty with checkpoint 0. The cases are facts and are never gated (user
decision 2026-11-26): the gate decides only the lessons and the checkpoint. For t = 1..10 one fresh pi session
predicts round t's 100 pairs with the state in use (the pring_test batch protocol, RevealTools: protein and KB tools,
`cases`, and with --predictor `predict_pairs` from sandbox_rsi/ckpt<c>_round/). Its p is scored before the labels
are shown. The labels go to the archive as cases (label, the agent's p, the predictor's p it could see, its evidence
ids); the archive only grows. The candidate: the lessons in use revised by a reviewer session that gets as new cases
all cases revealed since the last kept round (the cases are the archive, as in the state in use). Gate
(build_mus22t_rsi.gate_round): the agent with the candidate lessons (and with --predictor the candidate checkpoint t,
sandbox_rsi/ckpt<t>_val/) classifies the val gate pairs (rsi/val_gate.csv, 400, read-only like the final test); its
per-pair p are compared with the saved per-pair p of the best kept state on the same pairs (a paired bootstrap). The
candidate lessons replace the lessons in use (with --predictor: lessons and checkpoint together, c = t) unless the
candidate is clearly worse. Round 0, the empty memory, is always kept. The LLM-only arm (no --predictor) uses
sandbox_rsi/plain/ with no predictor tool and no predictor p in the cases.
Run directory: rounds/batch<t>/ the round sessions and their result.json, rounds/review<t>/ the reviewer;
memory/round<t>/ the candidate of round t, kept or dropped (cases.jsonl: the archive after round t; lessons.json;
review.json: the review log, the gate entry and the val metrics); val/round<t>/ the gate batches of candidate t;
gate.json the gate_round entries of rounds 0..t. The state in use after round t is the whole archive with the lessons
(and checkpoint) of the last kept round s <= t (s = 0: no lessons, checkpoint 0); after a drop this state has no
validation score of its own. A round with saved memory is replayed, so runs resume. curve.json: per round, the
agent's metrics on its batch before the reveal and the DL candidates' 0 and t - 1 on the same pairs; a one-class
batch has no metrics.

test: the final pairs (seed 0, batches of --batch) with the state in use after round 10 (lessons in the prompt, cases
through the tool), read-only; with --predictor its checkpoint (sandbox_rsi/ckpt<c>_final/). final/summary.json: the
pring_test metrics of the agent, candidate 0 ("v3": V3 trained on the RSI training set, before any reveal) and
candidate 10 ("al_checkpoint10") on the same pairs, with paired bootstraps. baseline: the same final test with the
empty memory and checkpoint 0 (PRING: B0) and no rounds, in its own run directory: the score of every arm of this
system before the rounds.

--data pring (user decision 2026-11-24): the same rounds, gate and final test on PRING with the random acquisition
rule. Data: build_pring_memory (hidden/dev_random.csv: 10 rounds of 200 pairs drawn at random from the development
set; hidden/val_gate.csv; hidden/final_test.csv). Views: sandbox_disc/ in the rounds, sandbox_val/ at the gate,
sandbox_final/ at the final test; the cases come from the development proteins, so at the gate and the final test
only the cases of homologs apply (case_homologs.tsv). B0 stays frozen: every round reads the same scores, and
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
from src.data.builders.build_mus22t_rsi import ROUNDS, best_kept, gate_round, read
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
    "- A PPI predictor trained on the knowledge base and on pairs revealed in earlier rounds. After each round it "
    "is\n  retrained on the knowledge base and every revealed pair, and the retrained predictor replaces it unless "
    "the\n  predictions made with it are clearly worse on a validation check. `predict_pairs` gives its probability "
    "p\n  for up to 200 of the listed pairs in one call. "
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
    """The 22T files and views, read at call time (tests replace RSI, VIEWS, PLAIN and RUNS). With the predictor,
    each stage of each DL checkpoint c has its own view."""

    def view(stage, c, predictor):
        return VIEWS / f"ckpt{c:02d}_{stage}" if predictor else PLAIN

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
        view=lambda stage, c, predictor: {"round": DISC, "val": VAL, "final": FINAL}[stage],
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


def stage_sandbox(stage, c, predictor):
    """(sandbox, view directory) of a stage ("round", "val" or "final") with DL checkpoint c (22T, --predictor)."""
    view = DATA().view(stage, c, predictor)
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
    """The reviewer after round t: records = the whole archive; fresh = the cases revealed since the last kept round,
    whose lessons (lessons, revised in place) are in use."""
    predictor = args.predictor
    first = min(c["round"] for c in fresh)
    earlier = f"round {first}" if first == t - 1 else f"rounds {first} to {t - 1}"
    since = (
        "This round's cases."
        if first == t
        else f"The cases of {earlier} and of this round: the lesson updates after {earlier} did not pass the "
        "validation check, so you review their cases again."
    )
    system = prompt(
        f"{DATA().prompts}/review_system",
        k=sum(c["round"] == t for c in fresh),
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
        since=since,
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
    """The val gate batches of candidate t (this memory; with the predictor on 22T, checkpoint t), read-only, as in the
    final test (the agent sees the final-test stage). Returns val_rows(t)."""
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
    return val_rows(t, run_dir)


def val_rows(t, run_dir):
    """{pair: (label, p)} of candidate t's saved val gate batches (a missing p counts as 0.5); every gate pair once."""
    rows = [
        r
        for f in sorted((run_dir / "val" / f"round{t:02d}").glob("batch*/result.json"))
        for r in json.loads(f.read_text())["rows"]
    ]
    out = {(r["a"], r["b"]): (r["label"], 0.5 if r["p"] is None else r["p"]) for r in rows}
    want = {(a, b) for a, b, _ in test_pairs(DATA().gate)}
    assert len(out) == len(rows) and out.keys() == want, f"val check of round {t}: not the val gate pairs"
    return out


def gate_step(gate, t, cand, run_dir):
    """gate plus round t's entry: candidate t's per-pair p against those of the best kept state, aligned by pair."""
    pairs = sorted(cand)
    y, p = [cand[x][0] for x in pairs], [cand[x][1] for x in pairs]
    p_best = None
    if t:
        best = val_rows(best_kept(gate), run_dir)
        p_best = [best[x][1] for x in pairs]
    return gate_round(gate, t, y, p, p_best), classification(y, p)


def run_rounds(run_dir, args):
    data = DATA()
    assert data.dev.exists() and (data.view("val", 0, False) / "kb.csv").exists(), (
        f"no development set: run {data.build}"
    )
    open_run(run_dir, args, "rounds")
    rounds = round_batches(data.batches)
    archive, kept, lessons = [], 0, Lessons()  # kept: the round whose candidate is in use (its checkpoint too)
    gate_path = run_dir / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else []
    if not gate:
        gate, _ = gate_step([], 0, val_check(0, args, archive, lessons, run_dir), run_dir)
        gate_path.write_text(json.dumps(gate, indent=1) + "\n")
    for t in range(1, min(ROUNDS, args.until or ROUNDS) + 1):
        mem_dir = run_dir / "memory" / f"round{t:02d}"
        if (mem_dir / "cases.jsonl").exists():
            archive, candidate = load_memory(mem_dir)
            if gate[t]["kept"]:
                kept, lessons = t, candidate
            continue
        records = archive  # every revealed case is in use; the gate decides the lessons and the checkpoint
        sandbox, view = stage_sandbox("round", kept, args.predictor)
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
        archive = archive + new_cases(t, result, seen)
        candidate = Lessons(lessons.all)
        fresh = [c for c in archive if c["round"] > kept]
        log = review(t, args, sandbox, view, archive, fresh, candidate, run_dir)
        # a rerun reviewed again: earlier val batches checked another memory
        set_aside(run_dir / "val", f"round{t:02d}_stale", f"round{t:02d}")
        gate, m = gate_step(gate[:t], t, val_check(t, args, archive, candidate, run_dir), run_dir)
        gate_path.write_text(json.dumps(gate, indent=1) + "\n")
        log["gate"] = gate[t] | {"val": m}
        save_memory(mem_dir, archive, candidate, log)
        if gate[t]["kept"]:
            kept, lessons = t, candidate
        print(
            f"{run_dir.name} round {t}: val AUPRC {m['ap']:.4f}, {'kept' if gate[t]['kept'] else 'dropped'} "
            f"(worse in {gate[t]['worse_share']:.0%} of resamples); in use: round {kept}, "
            f"{len(lessons.active())} active lessons; {len(archive)} cases revealed"
        )
    write_curve(run_dir)


def curve_rows(rows, t):
    """Metrics on round t's pairs of the agent (if rows give p) and of DL candidates 0 and t - 1; none for a one-class
    batch (a PRING batch of the development set can hold no positive)."""
    y = [r["label"] for r in rows]
    out = {"round": t, "pairs": len(rows), "positives": sum(y)}
    if not 0 < sum(y) < len(y):
        return out
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


def open_run(run_dir, args, stage):
    """run.json of a rounds or baseline run: written by the first call; a resume keeps its model and must keep its
    stage, --predictor and --data."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    if not path.exists():
        path.write_text(json.dumps(provenance(args, stage=stage), indent=1) + "\n")
    old = json.loads(path.read_text())
    assert (old.get("stage"), old["predictor"], old["data"]) == (stage, args.predictor, DATA().name), (
        "resume with the run's own stage, --predictor and --data"
    )
    args.model, args.thinking = old["model"], old["thinking"]


def run_test(run_dir, args):
    cfg = json.loads((run_dir / "run.json").read_text())
    assert (cfg.get("stage"), cfg["data"]) == ("rounds", DATA().name), "test a rounds run with its own --data"
    args.model, args.thinking, args.predictor = cfg["model"], cfg["thinking"], cfg["predictor"]
    gate = json.loads((run_dir / "gate.json").read_text())
    assert len(gate) == ROUNDS + 1, f"no gate decision of round {ROUNDS}: run the rounds first"
    kept = max(g["round"] for g in gate if g["kept"])  # the lessons (and checkpoint) in use: the last kept candidate
    records, _ = load_memory(run_dir / "memory" / f"round{ROUNDS:02d}")  # the whole archive
    _, lessons = load_memory(run_dir / "memory" / f"round{kept:02d}") if kept else ([], Lessons())
    final_test(run_dir, args, records, lessons, kept)


def run_baseline(run_dir, args):
    open_run(run_dir, args, "baseline")
    final_test(run_dir, args, [], Lessons(), 0)


def final_test(run_dir, args, records, lessons, kept):
    """The final test with this memory and, with the predictor on 22T, checkpoint `kept`; final/summary.json."""
    predictor = args.predictor
    data = DATA()
    sandbox, view = stage_sandbox("final", kept, predictor)
    book = data.cases(records, sandbox, view)
    test_dir = run_dir / "final"
    test_dir.mkdir(exist_ok=True)
    todo = batches(test_pairs(data.final), args.batch)
    if (test_dir / "run.json").exists():
        assert json.loads((test_dir / "run.json").read_text())["batch"] == args.batch, "resume with the same batch"
    else:
        extra = {"batch": args.batch, "batches": len(todo), "cases": len(records), "state_round": kept}
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
    # 22T: the DL candidates before any reveal and after all rounds
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
    parser.add_argument("stage", choices=("rounds", "test", "baseline", "al"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--data", choices=("mus22t", "pring"), default="mus22t")
    parser.add_argument(
        "--predictor", action="store_true", help="rounds, baseline: the agent sees the DL predictor's scores"
    )
    parser.add_argument("--until", type=int, help="rounds: stop after this round")
    parser.add_argument("--batch", type=int, default=K, help="test, baseline: pairs per session")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="test, baseline: run only the first N batches")
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
    run = {"rounds": run_rounds, "test": run_test, "baseline": run_baseline}[args.stage]
    run(DATA().runs / args.name, args)


if __name__ == "__main__":
    main()
