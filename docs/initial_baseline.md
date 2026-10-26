# Initial Baseline: Fixed Scientific Agent and PPI Predictor

**Goal.** Within a limited experimental budget, the agent should make progressively more accurate PPI
predictions and find valuable interactions more efficiently. At this stage the agent's prompts, tools and
workflow stay fixed: the first question is whether a fixed agent can use experimental feedback at all.
Failure modes come out of this evaluation; they are not assumed in advance. Only then does recursive
self-improvement (RSI) start.

## Task: static PPI discovery

- **Input:** protein pairs (u, v) among a given set of proteins.
- **Output:** a probability p_uv for each pair and the next batch of pairs to test.
- **Feedback:** newly measured PPI labels for the tested pairs.
- **Objective:** choose the pairs most worth testing, so that hits accumulate as fast as possible.
- **Data:** public PRING, adapted to this task (the dev sandbox), first; then the private TMP split, used
  only through the locally deployed LLM.

## System

A single agent with specialised tools, an experimental memory and an offline oracle. A separate evaluator
holds the sealed labels; the agent never reads them.

- **A. Predictor.** A fixed ESM-based PPI model (candidate: the motif-graph project's no-topology trunk),
  called as `predict_ppi(u, v) → score`. Fixed checkpoint, calibrated on a validation set, never retrained.
- **B. Evidence retriever and memory.** Background knowledge (function, localisation, GO, known
  interactions) and an experimental memory of revealed measurements (pair, result, round, source). Facts
  are stored apart from model predictions, and every entry carries its provenance.
- **C. Fixed LLM agent.** A pinned LLM (candidate: the latest DeepSeek API) following one workflow: read
  the task → call the predictor → retrieve evidence and memory → compare and rank → choose the next batch
  within budget → save → receive feedback.

Per chosen pair the agent records `pair_id`, `predictor_score`, `ranking_score` (p_uv), `evidence_ids`,
`uncertainty` and `rationale`.

## Evaluation: discovery in rounds

Each round, the predictor and the agent rank only the pairs whose labels are still unknown and submit the
top K. The oracle reveals which of them interact; all K then leave the unknown pool. The first round uses
background knowledge only, so it is the no-feedback starting point. Each round's picks are scored against
labels the agent could not see when it chose them, so no separate sealed pool is needed.

- **Budget:** 200 pairs per round × 10 rounds, 10 replicate runs per arm (sized after BioDiscoveryAgent,
  5 × 128, and Owkin's lab-in-the-loop study, 10 × 100).
- **Arms:** true feedback only for the baseline (user decision 2026-10-25); the permuted-feedback,
  no-feedback and anonymised-ID controls were run once and archived, and return only if a claim needs them.
- **Non-LLM baselines:** the predictor alone, and the predictor plus link prediction over revealed hits.
- **Metrics:** hits per round, cumulative hits against tests spent, and the gap over the predictor alone
  and random picks; also hits outside hub proteins, invalid or repeated picks, and cost per hit.
- **Failure analysis:** errors from the predictor, from conflicting evidence, from ignoring feedback (true
  ≈ permuted) and from over-following it; compare each stage with the predictor alone.

The PRING version of this protocol is specified in
[`docs/superpowers/specs/2026-10-23-pring-wave-discovery-design.md`](superpowers/specs/2026-10-23-pring-wave-discovery-design.md).
Its numbers are internal and never reported; only the private split and a future post-cutoff slice can
support a claim.

**Later, not in this baseline:** biological contexts and interventions (intervention-sensitive PPIs,
matched reference–intervention measurements), calibration of predicted responses, and results split by
familiar versus held-out receptors and by complete versus missing modalities.

**Expected deliverable:** a measured baseline and a reproducible set of failure cases showing where
workflow-level or model-level improvement is needed.
