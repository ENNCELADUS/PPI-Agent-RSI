# PRING dev sandbox — design

**Status:** KB built (`src/data/builders/build_pring_sandbox.py`; outputs in git-ignored `data/PRING/`). The
first inductive and transductive candidate pools and their degree/L3 baselines were removed on 2026-10-23
(not the design the user asked for). They are replaced by the wave discovery benchmark over all pairs among
the test proteins ([design](2026-10-23-pring-wave-discovery-design.md)).

## Why
The agent baseline is not built, and the private TMP test set must not be spent on debugging it. The
sandbox is an **internal** public-data benchmark for building and debugging the fixed agent. Its numbers
are never reported: every label is public, so a web-enabled agent would look it up and a closed-world LLM
may remember it.

Sources: data and split from PRING (Zheng et al., NeurIPS 2025 D&B); task packaging and graded scoring
from ScienceAgentBench (arXiv 2410.05080: hidden test labels, validity metric next to success metric,
cost); contamination protocol from Han et al., "Search-Time Data Contamination" (arXiv 2508.13180) and
Wang et al., "Search-Time Contamination in Deep Research Agents" (arXiv 2606.05241).

## What is special about an agent benchmark
- The test-taker is pretrained on the web and can use tools at test time, so answers leak three ways:
  split, pretraining, search time. ML benchmarks guard only the split.
- The training split becomes the environment: a reference KB the agent queries, not data it fits.
- Scoring covers validity and process (out-of-pool / repeated pairs, tool-call logs, cost), not only accuracy.
- The task runs in rounds (rank → reveal → re-rank); gains count only against permuted-label and
  no-feedback arms, with replicates.

## Contamination protocol
1. **Split.** PRING's BFS train/val/test, self pairs dropped (PRING keeps them as positives). Tested
   negatives (Negatome 2.0, IntAct negative) with both ends on the KB side of the protein split join train
   (114); others and PRING positives are skipped. Train and val are then filled to 1:1
   with a seeded sample of PRING's own negatives (which almost never coincide with tested ones: 1 of 85k).
   The KB is train + val (val also serves as the predictor's validation set); PRING's node split keeps
   test proteins out of it (≤ 40% identity to train), and the builder asserts it. Never use PRING
   `*_ratio5_exclusive.txt` (negatives leak across the node split).
2. **Search time.** Closed-world: the agent gets only `sandbox/` and fixed tools. Labels and the anon-ID map
   sit in `hidden/`. Per-protein annotation is allowed (user decision 2026-10-23: everything but the
   label), minus interaction-derived fields: UniProt interaction and subunit text, GO terms with IPI
   evidence, complex membership (PRING took positives partly from UniProt). The harness must log every
   tool call (2606.05241).
   Audit-only defences cannot rescue a web-enabled run: STC papers found leaks in 1–4% of QA items, but
   every public PPI positive is in a database by construction.
3. **Pretraining.** Not preventable for pre-cutoff public labels; measured as the named-vs-anonymised gap.

## Package
- `sandbox/`: `proteins.tsv` (accession, gene, protein_name, candidate, sequence; KB and test proteins) and
  `kb.csv`
  (`uniprotID_A,uniprotID_B,isInteraction,evidence`; 47,762 pos / 47,762 neg over 7,856 proteins; `evidence`
  is `pring`, `negatome` or `intact_negative`); `sandbox/anon/` the same with random opaque IDs and sequences
  only. `hidden/anon_map.tsv` maps them back. `annotations.tsv` (and an anon copy): per-protein UniProt and
  GO annotation minus interaction-derived text and terms (rules in the wave discovery design).
- `hidden/{train,val}.csv`: the 1:1 labelled splits (38,200 / 9,562 positives); `hidden/test_positives.csv`:
  the 30,128 PRING positives among test proteins (wave-discovery hits; test is no longer balanced).
- Column names follow the private split, so the harness and `src.eval.score` run unchanged on both.
- Protein names come from one bulk all-human UniProt download, so the same table serves the TMP proteins
  without sending a TMP list out.
- Seed 0; rebuild is byte-identical.

## Left open
Candidate space and reveal policy: see the wave discovery design. Skipped: TMP-stratified metrics, DFS/RW
splits, PRING graph-reconstruction metrics, post-cutoff temporal slice.
