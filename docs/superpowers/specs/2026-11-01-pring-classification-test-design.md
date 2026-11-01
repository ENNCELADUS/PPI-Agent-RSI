# PRING classification test and RSI data — design

**Status:** 2026-11-01, user decisions in conversation. Classification arm built (`src/eval/pring_test.py`,
`src/data/builders/build_pring_cls.py`); RSI loop on val not built.

## Decisions (user, 2026-11-01)
- **Split roles.** PRING train = the agent's KB. PRING val = RSI data: in each RSI round the agent chooses
  val pairs to test and their labels are revealed (active, not a fixed schedule). PRING test = final test,
  first the classification test, later the topology (graph reconstruction) test. The classification test
  keeps PRING's self pairs (user decision 2026-11-01, reversing an earlier drop).
- **Val is transductive (accepted).** 6,531 of the 6,608 val proteins are train proteins, and 10,681 of the
  10,760 val positives have both ends in train; test shares no protein with train. Lessons learned on val
  can fail to transfer to test; measure that gap.
- **First arm:** one LLM-only arm. Protein and KB tools only (`protein_info`, `search_proteins`,
  `similar_proteins`, `kb_partners`); no DL predictor tools, no memory, no web search (label leakage).

## Classification test
- **Pairs:** `hidden/test.csv` = PRING's labelled BFS test split as released: 64,038 pairs, 32,019 positive
  and 32,019 negative (1:1). 1,891 positives are self pairs, and no negative is one, so "self pair" alone
  marks a positive (2.95% of pairs). The prompt says that a pair can name one protein twice.
- **Agent view:** `sandbox_cls/` = the sandbox with KB = train only (76,400 pairs, 7,750 proteins), no
  `scores.tsv.gz`; homologs only to train-KB proteins (998 of 2,018 test proteins have one).
- **Protocol:** pairs shuffled with seed 0, batches of 200; one fresh pi session per batch, 80 tool calls,
  one `submit_predictions` call with `p` and `evidence_ids` per pair. The prompt does not state the
  prevalence. Missing or invalid predictions score p = 0.5 and count against coverage.
- **Metrics:** AUROC (ties count half), AP (ties form one step), and PRING's Table 8 set (accuracy, precision,
  recall, F1) plus specificity and MCC at each model's max-F1 threshold chosen on the test labels (oracle,
  user decision 2026-11-01: val is RSI data, so nothing else selects it); ECE (15 bins) and Brier, as in the
  motif-graph test reports; all pairs and without self pairs. Also coverage, tool calls, tokens, cost.
  B0 (DL predictor, evaluator-side logits) on the same pairs: full test AUROC 0.701, AP 0.736 (without self
  pairs: 0.682, 0.697; B0's median self-pair logit is 6.75).
- **Claim boundary:** PRING labels are public, and a closed-world LLM can remember famous pairs, so the
  LLM-vs-B0 gap can be memory. Internal numbers only (project rule).

## Not built
RSI loop on val; topology test (needs up to ~2M pair scores); LLM + B0 arm; name-blind arm.
