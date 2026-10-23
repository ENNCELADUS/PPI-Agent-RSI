# PRING wave discovery benchmark — design

**Status:** drafted 2026-10-23 from the design agreed in conversation; awaiting the user's review; not built. Extends the PRING dev sandbox
([design](2026-10-23-pring-sandbox-design.md)). Calls marked *(my call)* are the designer's defaults, not
the user's decisions.

## Why
The first question for the fixed agent ([`docs/initial_baseline.md`](../../initial_baseline.md)) is whether
it uses experimental feedback: with prompts, tools and predictor held fixed, does seeing measured labels
make its next choices better? The sandbox's 1:1 labelled pools cannot ask that. A 1:1 pool reveals sampled
negatives as if they were measurements, and it does not resemble real discovery, where hits are rare among
all possible pairs. So the agent searches every pair among a set of proteins, as a lab would.

Prior work this follows (notes in `docs/research-reports/research_notes/Public PPI benchmarks for
agents/t3_discovery_with_feedback.md`): BioDiscoveryAgent (ICLR 2025; rounds × batch, cumulative hits, 10
runs), Owkin lab-in-the-loop (arXiv 2603.26177; zero-shot vs feedback vs permuted feedback), Gupta et al.
(EMNLP Findings 2025; permuted labels changed nothing; LLM-prior + nearest-neighbour baseline), PerturbTrace
(bioRxiv 2026; high recall without real feedback use). No published benchmark selects PPI pairs in rounds.

## Goals, non-goals, anti-goals
- **Goal:** measure, per wave, how many true interactions a fixed agent finds among its picks, and whether
  that exceeds the same agent without feedback or with permuted feedback.
- **Non-goals:** RSI of prompts or tools; retraining the predictor; web retrieval (closed-world only);
  intervention or context tasks; reporting any sandbox number outside the project.
- **Anti-goals:** assuming failure modes before the runs; claiming feedback use from final hit count alone
  (PerturbTrace); a candidate file or tool whose order or contents leak labels.

## Candidate space and labels
- **Proteins:** the 2,018 PRING BFS test proteins (none in the KB; ≤ 40% identity to train). Every one has
  at least one test positive.
- **Candidates:** all 2,035,153 unordered non-self pairs among them.
- **Hits:** the 30,128 PRING positives among them (1.48%). PRING has no positive among test proteins outside
  its test split, so the label is closed-world: a PRING positive is a hit, every other pair is "not
  detected". Some "not detected" pairs are real, undiscovered interactions; this caps measured precision
  equally for every arm.
- The 55 test-side tested negatives (Negatome, IntAct negative) change nothing here: they are "not detected"
  like any other non-hit. The 1:1 `hidden/test.csv` is no longer used and is replaced (see Package).

## One run
For wave t = 1 … 10, with U_t = all candidates not submitted in earlier waves (U_1 = all):
1. The predictor's score for every pair in U_t is available. The predictor is fixed, so all 2,035,153
   scores are computed once per checkpoint and cached.
2. The agent re-ranks the predictor's top 1,000 pairs in U_t *(my call: 5 × K)*, using the KB, retrieval
   tools and its experimental memory, and submits exactly K = 200 pairs from U_t. Pairs outside the top
   1,000 may be submitted if the agent finds them through its tools.
3. The oracle returns, for each submitted pair, hit or not detected. All 200 leave U_t, so nothing is
   tested twice.
4. The memory stores each result as a fact (pair, result, wave, source = oracle), apart from predictions.

Wave 1 has no feedback yet, so it is the prior-only starting point. Total budget: 2,000 tests per run =
0.1% of candidates, at most 6.6% of hits.

## What the agent sees and what is hidden
| Item | Agent | Evaluator only |
|---|---|---|
| KB (`kb.csv`: train + val, 1:1 labels) | yes, from the start | — |
| Test proteins: any per-protein information (sequence, names, function, localisation, GO, domains, structure, ESM features); anon arm: id-free features only | yes, from the start | — |
| Interaction-derived annotation of test proteins (UniProt interaction/subunit text, GO terms with IPI evidence, complex membership) | never: it is the label in another form | — |
| Predictor scores via `predict_ppi` and a top-N tool over U_t | yes | — |
| Results of its own submitted pairs | from the wave after submission | — |
| Labels of pairs never submitted | never | yes |
| PRING files, `hidden/`, degree of test proteins | never | yes |

Any candidate listing the agent sees is in canonical order (sorted ids), never PRING's
`candidate_test_edges.txt` order, which lists positives first.

## Arms
All arms use the same LLM, prompts, tools, predictor and budget, and are scored against the true labels.
- **True feedback:** results as measured.
- **Permuted feedback:** each wave reports the same number of hits as the true result, placed on a random
  subset of that wave's 200 submitted pairs *(my call; Owkin-style)*.
- **No feedback:** the agent is told which pairs were submitted, but not their results.
- **Random picks:** 200 uniform pairs from U_t each wave (no LLM).
- **Anonymised:** true feedback on `sandbox/anon/` (opaque ids, sequences only); the named-vs-anonymised gap
  estimates memorised PRING labels.
Each LLM arm runs 10 replicates; random picks 10 seeds; deterministic baselines once. Wave 1 has no
feedback in any arm, so within a replicate the true, permuted and no-feedback arms share one wave-1 run and
branch from wave 2; feedback gain is then a paired difference per replicate *(my call)*. The anonymised arm
runs its own wave 1.

## Non-LLM baselines (same K, waves and top-1,000 window)
- **Predictor alone:** top 200 of U_t by predictor score each wave.
- **Predictor + L3:** the predictor's top 1,000 of U_t re-ranked by L3 paths over the hits revealed so far,
  ties by predictor score *(my call)*.
- **Predictor + revealed degree:** the same window re-ranked by the product of each protein's revealed-hit
  counts, ties by predictor score *(my call)*.
These test whether the LLM adds anything beyond cheap use of the same feedback.

## Metrics
- **Hits per wave** (precision at 200) and **cumulative hits vs tests**; summary: area under the cumulative
  curve over 10 waves. Per-wave precision falls in every arm as easy hits run out, so the comparison is
  between arms, not across waves.
- **Feedback gain:** true − no-feedback and true − permuted, mean and 95% CI over replicates.
- **Non-hub hits** *(my call)*: hits where neither protein is a hub (top decile of test positive degree,
  ≥ 76 partners; 203 proteins touch 15,623 of the 30,128 hits). Hubs are found by degree for free.
- **Validity:** submissions out of U_t (already tested, self, unknown id, outside the protein set) and
  short batches, reported per wave; invalid picks count as misses and are not refilled.
- **Calibration:** Brier score of `ranking_score` (p_uv) on submitted pairs.
- **Cost:** tokens and dollars per run and per hit; tool calls logged in full.
Hit ratio (hits / all 30,128) is reported but not headline: its ceiling is 6.6%.

## Package (builder changes)
- `sandbox/proteins.tsv` and `sandbox/anon/proteins.tsv` gain the 2,018 test proteins (now KB-only).
- `sandbox/annotations.tsv` (built): UniProt function, localisation, tissue, GO, keywords, families, Pfam,
  InterPro for KB and test proteins (user decision 2026-10-23: everything except the label). Removed: any
  sentence naming another human gene symbol, GO terms under protein binding (GO:0005515) or
  protein-containing complex (GO:0032991); interaction and subunit comments are never fetched. The anon
  copy also drops function text and every sentence naming any symbol.
- The predictor (B0) is vendored into this repo and scores the candidates once on H20 (4 GPUs, one
  container); the agent's `predict_ppi` reads the cached scores.
- `hidden/test_positives.csv`: the 30,128 hits (replaces the 1:1 `hidden/test.csv`). Train and val stay 1:1.
- An oracle module reads `hidden/` and answers submitted pairs; the agent harness never imports it.
- A run log per wave: the submitted 200 with all output fields, oracle results, tool calls and cost.

## Predictor (built 2026-10-23)
B0 (`b0_v31`), the motif-graph project's no-topology trunk (V3.1 over per-residue ESM-3 features), chosen by
the user. Checkpoint on H20: `/2023533015/topology-conditioned-inductive-edge-prediction/outputs/split_seed42/
b0_v31/best.pt` (checkpoint id `7e78609f8c904284`, epoch 8, 2026-09-09). Trained on PRING BFS train positives
with 1:5 sampled negatives, selected on V_val (a node-held-out region of train); PRING val and test were never
trained on, so the sandbox's val split is clean for calibration.
- Vendored inference in `src/predictor/` reproduces the reference test scores bit for bit (64,038 rows, max
  abs logit diff 0). All 2,037,171 rows of `candidate_test_edges.txt` were scored once on 4 GPUs (container
  30757, ~1 min of GPU time at ~92% utilisation); the cached logits are `data/PRING/predictor/
  pring_candidates.tsv.gz` with provenance in `.meta.json`. `predict_ppi` reads them.
- Logits are bf16-quantised (3,376 distinct values), so rankings have ties: break them by canonical pair id.
- The raw sigmoid is fitted to a 1:5 stream: recalibrate p_uv on val with a prevalence correction to 1.48%.

## Reuse on the private split
The same harness later runs on the private TMP split, whose data may reach only the locally deployed LLM.
So the LLM is called through one OpenAI-compatible chat endpoint (DeepSeek's API and a local vLLM server
both speak it), the oracle reads any labels file through one interface, and run logs stay on the machine
*(my call)*. The private candidate space and its labelling are designed later; the PRING closed-world rule
is not assumed outside the sandbox.

## Left to the builder
How the agent re-ranks 1,000 pairs (batched prompts, shortlists, tool design); file formats; the L3
normalisation; LLM sampling settings, provided each replicate is seeded and logged.

## Open, needed before runs (not before building the data and harness)
- **Pinned LLM:** model and version (candidate: the latest DeepSeek API).

## Later, optional
A 30-wave × 32 run (BioDiscoveryAgent's long ablation) to see whether gains persist with smaller, more
frequent waves; prompt-only and results-only ablations.
