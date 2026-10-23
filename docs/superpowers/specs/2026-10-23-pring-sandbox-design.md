# PRING dev sandbox — design

**Status:** built (`src/data/builders/build_pring_sandbox.py`, `src/eval/score.py`, `src/baselines/topology.py`; outputs in git-ignored `data/PRING/`).

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
1. **Split.** Inductive pool: PRING's node-disjoint BFS test proteins (≤ 40% identity to train); none is in
   the KB. Transductive pool: held-out KB positives, removed from the KB; negatives PRING-sampled over their
   endpoints, rejecting every PRING positive and every KB pair. Self pairs dropped everywhere (PRING keeps
   them as positives, so a self pair in a pool would be a guaranteed positive). Never use PRING
   `*_ratio5_exclusive.txt` (negatives leak across the node split). Identical ID format for both classes;
   pool rows shuffled (PRING's `candidate_test_edges.txt` lists positives first).
2. **Search time.** Closed-world: the agent gets only `sandbox/` and fixed tools. Labels, permuted labels and
   the anon-ID map sit in `hidden/`. The protein table has no UniProt interaction, subunit or function text
   (PRING took positives partly from UniProt). The harness must log every tool call (2606.05241).
   Audit-only defences cannot rescue a web-enabled run: STC papers found leaks in 1–4% of QA items, but
   every public PPI positive is in a database by construction.
3. **Pretraining.** Not preventable for pre-cutoff public labels; measured as the named-vs-anonymised gap on
   the same pool, next to the KB degree-product (popularity) baseline.

## Package
- `sandbox/`: `proteins.tsv` (accession, gene, protein_name, sequence), `kb.csv`
  (`uniprotID_A,uniprotID_B,isInteraction,evidence`), `pool_inductive.csv`, `pool_transductive.csv`
  (`uniprotID_A,uniprotID_B`); `sandbox/anon/` the same with random opaque IDs and sequences only.
- `hidden/`: `<pool>.csv` and `<pool>_permuted.csv` labels (also under `hidden/anon/`), `anon_map.tsv`.
- Column names follow the private split, so the harness and `src.eval.score` run unchanged on it once the private
  `test.csv` is split into a pool and a hidden labels file.
- Protein names come from one bulk all-human UniProt download, so the same table serves the TMP proteins
  without sending a TMP list out.
- BFS split only; 500 positives + 500 negatives per pool; seed 0; rebuild is byte-identical.

## First results (baselines, 2026-10-23)
| Pool | Method | P@50 | P@100 | P@250 |
|---|---|---|---|---|
| inductive | degree / L3 | 0.46 | 0.45 | 0.46 |
| transductive | degree | 0.90 | 0.82 | 0.58 |
| transductive | L3 | 1.00 | 0.98 | 0.97 |

The inductive baselines sit at random (test proteins have no KB edges), which confirms the split. On the
transductive pool, L3 over the KB nearly solves the task: PRING-sampled negatives over held-out endpoints
rarely share network neighbourhoods. An agent that uses the KB must beat L3 there, or its KB use is just L3.

## Left open
Reveal policy (which labels are revealed per round), decided with the harness. Skipped: TMP-stratified
metrics, DFS/RW splits, PRING graph-reconstruction metrics, feedback-round scoring, degree matching to the
private pool, post-cutoff temporal slice.
