# 22T classification test: LLM agent + V3 predictor

Date: 2026-11-02. Status: built; smoke run not done.

## Question

On private 22T mouse membrane PPIs, does an LLM agent that sees a DL predictor's score classify test pairs better
than the predictor alone? The LLM-only agent has test AUPRC 0.587 and V3 (B0 v3_1) has 0.716 on the same 9,366
pairs ([results](../../../results/mus22t-classification-test/summary.html)). This experiment adds the third cell:
the agent with V3's score.

## Arms (all on `data/MUS22T/agent_split/test.csv`, same 9,366 pairs)

| Arm | What scores the pair | Status |
|---|---|---|
| The predictor-assisted agent (new) | DeepSeek-V4 Pro gets V3's p with `predict_pairs`, uses the tools, gives its own p | to run |
| V3 alone | V3's raw sigmoid, evaluator side | done (grasppi-mus22t `7a06976`, epoch 17) |
| The LLM-only agent | DeepSeek-V4 Pro, no predictor | done (`mus22t-llm-only-pro-high-8w-20261102T134702Z`) |
| Degree-only control | endpoint propensity | done (`split_stats.json`) |

The new arm keeps the settings of the LLM-only run: `custom/deepseek-pro`, thinking high, batch 200, seed 0 (so the
batches hold the same pairs), the same protein and KB tools, no web. It adds `predict_pairs` and a prompt that
describes V3. The cap is 100 tool calls (user decision), not 80; no LLM-only batch used more than 78 calls, so the
cap did not limit that arm.

## Design decisions

1. **V3's p comes from the tool `predict_pairs`** (user decision): up to 200 pairs per call, so one call covers a
   batch. The single-pair `predict_ppi` cannot: 200 pairs, and the two-layer memory smoke batch spent 79 of 80
   calls on it. `result.json` records `predictor_pairs` (listed pairs whose p the agent received), and
   `summary.json` gives `predictor_coverage`. Low coverage means that the arm is partly LLM-only.
2. **The prompt gives V3's validation numbers only**: trained on the knowledge base, sequence-based, validation
   AUPRC 0.718, AUROC 0.712, calibrated (ECE 0.081). Test numbers never go in the prompt.
3. **The prompt does not give the share of positives**, the same as for the LLM-only prompt. Because V3's p is
   calibrated, its mean tells the agent the share approximately; that is part of what the predictor gives.

## Build (done)

- `src/data/builders/build_mus22t_scores.py`: V3's exported test logits (`data/MUS22T/predictor/v3_test.tsv.gz`,
  export command in the docstring) to `data/MUS22T/sandbox/scores.tsv.gz` (`a, b, p`, raw sigmoid). Asserts that
  the scored pairs are exactly `agent_split/test.csv`. Only test pairs have a score; `predict_pairs` gives a
  clear error for any other pair.
- `src/eval/pring_test.py`: `--predictor` (22T only) gives the agent `TestTools` (`predict_pairs`, no `cases`) and
  `pring_test/system_mus22t_predictor.md`; `--cap` sets the tool calls (recorded in `run.json`, checked on
  resume). `summary.json` has V3 on the same pairs (for every 22T run, the LLM-only one too) and
  `agent_minus_v3`: the paired bootstrap of AUROC and AP (1,000 resamples, seed 0).
- Check on real data: V3's p from `scores.tsv.gz` gives test AUPRC 0.7155, AUROC 0.7175 (V3's own report).

## Run (WSL, `remote-api-agent` skill)

1. Commit and push; `git pull` in `~/PPI-Agent-RSI-22t`; copy `data/MUS22T/sandbox/scores.tsv.gz` there.
2. Smoke: `--limit 1`. Pass when `batch0001/result.json` has 200 non-null p, no `error` and no `length` stop.
   Also look at `predictor_pairs` (expect 200) and the copy rate (pairs where the agent's p equals V3's p).
3. Full run (`--data mus22t --predictor --cap 100`): 47 batches, 8 workers. Expected cost: about $50, as for the
   LLM-only run.

## Report

Add the new arm to `results/mus22t-classification-test/summary.html`:
- Table 1 (test AUPRC, AUROC): degree-only, V3, LLM-only, predictor-assisted agent.
- Table 2 (paired bootstrap 95% intervals): predictor-assisted − V3 (the main result) and predictor-assisted −
  LLM-only.
- Table 3 (threshold metrics at p = 0.5 and at max-F1).
- Diagnostics: predictor coverage, Spearman of the agent's p with V3's p, the copy rate, and ΔAUPRC in V3-score
  terciles (where the agent changes V3's ranking and if that helps).

## Claim boundary

One run: the intervals cover pair sampling, not LLM sampling variance. If the agent mostly copies V3's p, the
result is "no gain", not "the agent agrees with V3". The 22T labels are private, but the proteins are public
mouse proteins, so the LLM can still know some interactions from the literature.
