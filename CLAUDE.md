# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research project: LLM-based scientific agents for membrane protein–protein interaction (PPI)
discovery, then recursive self-improvement (RSI) of the agent workflow and, later, the PPI predictor
under sparse experimental feedback. `docs/introduction.md` is the motivation (three research
directions), `docs/initial_baseline.md` the baseline design, `docs/literature.md` the related work,
`docs/research-reports/<date>-<slug>.md` dated deep-research reports (notes behind them in
`research_notes/`). PDFs live in `literature/<topic>/` (`ppi`, `science-agents`, `self-improvement`,
`feedback-benchmarks`), git-ignored except the index `literature/README.md`. When a change alters
behaviour a doc describes, update that doc in the same change.

**Core thesis — do not let it drift:** measure a *fixed*, expert-designed agent before building any
RSI. Failure modes must come out of evaluation, never be assumed in advance.

## Active work

- **Benchmark split (built):** `scripts/agent_split/` → `data/TMP/agent_split/`. Test pairs have no
  evidence in any public PPI database, because the agent can search online. Both files are 1:1.
  `train.csv` (public positives + experimentally tested negatives + PRING-sampled negatives,
  `evidence` column) is the agent's **reference KB**, not model-training data; `test.csv`
  (non-public positives + PRING-sampled negatives) is the held-out candidate pool. No validation split.
- **Agent baseline (not built):** task is global top-N — from an unlabeled candidate pool, output the
  N most likely interacting pairs; score, then reveal a limited number of labels as feedback and
  re-score with the same fixed procedure. Runs on LLM knowledge plus the reference KB.
  Still open, decide before building: which LLM is pinned, and whether the agent gets
  web/literature retrieval or is closed-world.
- **Predictor:** none yet. A trained predictor is added later as a tool; never
  vendor or retrain DIPPI here.
- **Public benchmarks (surveyed, not built):** `docs/research-reports/2026-10-19-public-ppi-benchmarks.md`.
  Every public PPI label is retrievable online, so public tracks run **closed-world** only: PRING human
  test subgraphs; BioDiscoveryAgent screens (feedback-loop calibration); SKEMPI 2.0 via the
  `ppi-mutation-ddg` component split (perturbation); a post-cutoff IntAct/BioGRID slice (to build).
  Agents often ignore revealed labels, so every feedback run needs permuted-label and no-feedback arms.

## Project rules

- No formalism gates: no digest pinning, contract verifiers, eligibility ceremony or any check that
  blocks a run. Record provenance and proceed.
- Update the CLAUDE.md and README.md of this repo every conversation. Replace deprecated text directly with 
  new ones. Keep these documents concise. Don't keep backward compatible unless explicitly asked.
- **Private data stays private.** Never commit, push, upload or send TMP data to external services,
  except to the LLM the agent is configured to call. `data/` is git-ignored.
- Ask when intent is ambiguous (the user prefers being questioned over silent assumptions); do not
  re-ask what a doc or earlier decision already settles.

## Commands

No tests or linters are configured yet. The scripts import `common` directly, so run them from
`scripts/agent_split/`.

```bash
uv sync
# Private data (from the DIPPI repo on the lab server):
rsync -az wangar2023@10.15.89.191:/public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions/data/TMP data/

cd scripts/agent_split
./fetch_public.sh idmap string_alias      # ID maps first; every other source depends on them
./fetch_public.sh string biogrid intact hippie bioplex corum negatome intact_negative
uv run python build_split.py              # ~30 s, ~3 GB RAM; deterministic (seed 0)
python3 verify_live_string.py 100 7 1     # n, seed, label (1 = test positives, 0 = negatives)
```

`fetch_public.sh` downloads to `public_db/_download/` with `curl -C -` retries (this network drops
long transfers; IntAct is 1.1 GB), runs `extract_public.py <source>`, then deletes the download.
Set `KEEP_DOWNLOADS=1` to keep raw files when iterating on a parser. `extract_public.py` and
`verify_live_string.py` are stdlib-only; `build_split.py` needs DuckDB.

## Architecture

`data/TMP/raw/` (DIPPI pretrain/finetune/GT-test pairs) → `extract_public.py` filters each public
source to TMP proteins as gene-level node pairs (`public_db/<db>.tsv.gz`: STRING v12.5, BioGRID,
IntAct, HIPPIE, BioPlex, CORUM; tested negatives as raw accession pairs in `public_db/negative/`) →
`build_split.py` → `train.csv`, `test.csv`, `pair_annotations.parquet` (per-pair `in_<db>`,
`<db>_score`, `public`), `split_stats.json`, plus a `README.md` with counts and verification.

- **Identity (`common.IdMap`):** a UniProt accession expands to *every* Entrez gene from UniProt
  idmapping and STRING aliases (`g<entrez>`), else its STRING protein ID, else the bare accession.
  A pair matches a DB if **any** node combination does. This is deliberately permissive: it is what
  keeps multi-gene proteins (histone H4 → 14 genes), TrEMBL and obsolete accessions from leaking.
- **Positives:** union of raw positives, unordered; `NA`, self pairs and 6 pairs also labelled
  negative are dropped. Raw negatives (18.4M, computationally generated) are unused. Public (any DB,
  STRING at any score) → train; non-public → test.
- **Tested negatives (train only):** Negatome 2.0 + IntAct negative, mapped onto TMP accessions,
  minus TMP positives, curated-DB pairs and STRING ≥ 700.
- **PRING sampling** (`pring_negatives`, after github.com/SophieSarceau/PRING
  `negative_sample.py` / Bernett 2024): both ends drawn from the split's positive-endpoint multiset
  (degree-preserving), the second redrawn until the pair is not a self/same-gene pair, not known and
  not already drawn. Test is sampled first and rejects any public evidence including tested
  negatives; train rejects PRING's known set (TMP positives, curated DBs, STRING > 700) plus all
  test pairs.

## Data traps

- `data/TMP/agent_split/README.md` and `split_stats.json` sit under git-ignored `data/`; the
  versioned record of the protocol is this file plus the `build_split.py` docstring. Rebuild instead
  of trusting a stale copy.
- `data/TMP/processed/` is DIPPI-era (TPPNI splits, its own README) and unused here; `raw/` file
  names carry upstream typos (`homo_negative_pair_fintune.csv`, `depedent_test.csv`).
- In the raw data, 97% of GT-test negatives also appear in the training negatives, and all finetune
  negatives are inside the pretrain negatives. That is why raw negatives are discarded.
- `public_db/*.tsv.gz` keep only pairs among the current TMP proteins, as gene-level nodes. Adding
  proteins or changing `IdMap` means re-running every extract (the downloads are gone).
- 1,320 TMP proteins have no Entrez gene and no STRING ID; they match by accession only.
- 65,047 PRING train negatives have a weak STRING link (≤ 700) by design (PRING-faithful rule);
  filter them with `pair_annotations.parquet` if a stricter KB is needed.
- IntAct pairs with both positive and negative evidence count as public. This only shrinks test,
  but such pairs are never clean tested negatives.
- HuRI's site has a broken TLS certificate; HuRI is covered through IntAct/BioGRID/HIPPIE. Do not
  bypass certificate checks.