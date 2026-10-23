# CLAUDE.md

## Research task

LLM scientific agents for membrane protein–protein interaction (PPI) discovery, then recursive self-improvement
(RSI) of the agent workflow and, later, the PPI predictor under sparse experimental feedback. Docs:
`docs/introduction.md` (motivation), `docs/initial_baseline.md` (baseline design), `docs/literature.md`,
`docs/research-reports/<date>-<slug>.md` (git-ignored deep research), `docs/superpowers/specs/` (dated designs).
PDFs live in git-ignored `literature/<topic>/`; the index `literature/README.md` is tracked.

**Core thesis — do not let it drift:** measure a *fixed*, expert-designed agent before building any RSI. Failure
modes must come out of evaluation, never be assumed in advance. Name every pool, arm, baseline and run by what
it is ("the inductive pool", "the permuted-label arm"), never by a bare label; give a code label once in parentheses.

## Current state (2026-10-23)

Replace this section on every update; history lives in git and the specs.

- **Private split (built):** `build_agent_split` → `data/TMP/agent_split/`. `train.csv` is the agent's reference
  KB (public positives, tested and PRING-sampled negatives); `test.csv` is the 1:1 held-out pool whose positives
  have no evidence in any public PPI database. No validation split.
- **PRING dev sandbox (built):** `build_pring_sandbox` → `data/PRING/`
  ([design](docs/superpowers/specs/2026-10-23-pring-sandbox-design.md)). Internal and closed-world, for
  debugging the agent before it touches `test.csv`. 1:1 labelled train/val in `hidden/` (val doubles as the
  predictor's validation set) form the KB; `hidden/test_positives.csv` holds the wave-discovery hits.
  The agent gets `sandbox/` or `sandbox/anon/` (memorisation arm), never `hidden/`.
- **Wave discovery (not built,** [spec](docs/superpowers/specs/2026-10-23-pring-wave-discovery-design.md)**):** all
  2.04M pairs among the 2,018 test proteins (1.48% hits); each wave predictor + agent pick 200 unknown pairs,
  the oracle reveals their hits; 10 waves × 10 replicates, true/permuted/no-feedback/random-pick arms. Open:
  predictor checkpoint, pinned LLM, web retrieval for the private run (public tracks are closed-world).
- **Predictor:** motif-graph's B0 `best.pt` on H20 (wave spec), being vendored. Never vendor or retrain DIPPI.
- **Other public tracks (surveyed):** `docs/research-reports/2026-10-19-public-ppi-benchmarks.md` —
  BioDiscoveryAgent screens, SKEMPI 2.0 via `ppi-mutation-ddg`, and a post-cutoff IntAct/BioGRID slice.

## Commands

Python ≥ 3.11 via `uv`; run from the repo root as modules; imports are `src.*`. Module map: `src/README.md`.

```bash
uv sync                                               # core deps + dev group (pytest, ruff)
uv run python -m pytest tests -q > pytest.txt 2>&1    # full suite; redirect (rtk note below)
.venv/bin/ruff check .                                # lint (E,W,F)
.venv/bin/ruff format <files you touched>             # never `ruff format .`
rsync -az wangar2023@10.15.89.191:/public/home/wangar2023/DIPPI-DNN-based-Interaction-Predictor-for-Protein-Protein-Interactions/data/TMP data/
src/data/builders/fetch_public.sh idmap string_alias  # ID maps first; every source depends on them
src/data/builders/fetch_public.sh string biogrid intact hippie bioplex corum negatome intact_negative
uv run python -m src.data.builders.build_agent_split  # ~30 s, ~3 GB RAM, deterministic
python3 -m src.eval.verify_live_string 100 7 1        # n, seed, label: sampled test pairs vs live STRING
uv run python -m src.data.builders.build_pring_sandbox  # ~1 s; inputs and fetch commands in its docstring
uv run python -m src.eval.score <labels.csv> <ranking.csv>  # private test.csv (sandbox pools: not built)
```

- `fetch_public.sh` resumes downloads (`curl -C -`; this network drops long transfers, IntAct is 1.1 GB), pipes
  each to `python3 -m src.data.public_db`, then deletes it (`KEEP_DOWNLOADS=1` keeps it).
- Tests use synthetic inputs; builders, fetch and live checks need real data or network, so "it imports" proves
  nothing: rebuild and compare checksums. **Baseline the suite before changing code.**
- A global `rtk` hook rewrites Bash output (`ruff check .` may print `[]`, foreground pytest may show fake errors):
  use `.venv/bin/ruff`, redirect pytest, `rtk proxy <cmd>` for exact output.

## Project rules

- When a change alters behaviour a doc describes, update that doc in the same change. Update this file and
  README.md every conversation; replace deprecated text; keep both concise (this file under 120 lines).
- Experiment code reports results; decisions are made by reading them, not in code.
- No formalism gates (digest pinning, contract verifiers, eligibility ceremony, run-blocking checks). Record
  provenance and proceed. The builders' asserts against silent wrong artifacts fail closed and stay.
- **Private data stays private:** never commit, push, upload or send TMP data anywhere except the LLM the agent
  is configured to call. `/data/` is git-ignored and no data is tracked.
- Sync code between machines through Git only; never rsync/scp/tar working trees. GPU work (the later predictor)
  runs on H20 (`ssh -p 30838 root@10.15.171.204`): finish local work, commit, push, end with the launch command.
- Merge finished branch or worktree work into `main` once suite and lint pass, review is adjudicated and any
  data-dependent path has run on real data; push `main`, delete the branch. Conventional Commits.
- Codex review only when the user asks: from Bash, fresh `CODEX_HOME` with only `auth.json` and a `config.toml`
  (`model = "gpt-6.1-sol"`, `model_reasoning_effort = "high"`, `approval_policy = "never"`), recreated per review;
  background `codex review --base <sha>` to a file and adjudicate findings against the code.
- Ask when intent is ambiguous; do not re-ask what a doc or earlier decision settles.
- `graphify-out/` (git-ignored; `.graphifyignore` drops `/data/`, `literature/`) maps the repo: query it before a
  change, run `/graphify . --update` after edits, rebuild in full after moves or deletions. Check health on the
  built `graph.json`: the raw extraction lists stdlib imports as dangling (they become stub nodes), ignore those.

## Architecture

`src/data` (paths, identity, sampling, public-DB parsers) never imports `eval` or `baselines`; dataset builders
live in `src/data/builders/`; `tests/` mirrors `src`. Each builder's docstring is the versioned protocol.

- **Private split:** `data/TMP/raw/` (DIPPI pairs) → `public_db` keeps each public source's pairs among TMP
  proteins as gene-level nodes (STRING v12.5, BioGRID, IntAct, HIPPIE, BioPlex, CORUM; tested negatives as raw
  accessions in `public_db/negative/`) → `build_agent_split` → `train.csv`, `test.csv`,
  `pair_annotations.parquet` (per-pair DB evidence), `split_stats.json`.
- **Identity (`ids.IdMap`):** an accession expands to *every* Entrez gene from UniProt idmapping and STRING
  aliases, else its STRING ID, else itself; a pair matches a DB if any node combination does. Deliberately
  permissive, so multi-gene proteins (histone H4 → 14 genes), TrEMBL and obsolete accessions cannot leak.
- **Positives:** union of raw positives, unordered; `NA`, self pairs and 6 pairs also labelled negative dropped.
  Public (any DB, STRING at any score) → train; non-public → test.
- **Tested negatives (train only):** Negatome 2.0 + IntAct negative on TMP accessions, minus TMP positives,
  curated-DB pairs and STRING ≥ 700.
- **PRING sampling (`negatives.pring_negatives`):** both ends drawn from the positive-endpoint multiset
  (degree-preserving), redrawn until not self/same-gene, not known, not drawn. Test is sampled first and rejects
  any public evidence; train rejects TMP positives, curated DBs, STRING > 700 and all test pairs.
- **PRING sandbox:** PRING BFS splits minus self pairs (match PRING's GitHub release). KB-side tested negatives
  (114) join train; train/val filled to 1:1 from PRING negatives (38,200 / 9,562 positives, 7,856 proteins).
  Candidates: all 2,035,153 pairs among the 2,018 test proteins (`candidate` = 1), 30,128 hits.
  `annotations.tsv`: UniProt + GO minus sentences naming another gene and GO protein-binding/complex terms.

## Pitfalls and claim boundaries

Mistakes here produce a complete-looking **wrong artifact**, not an exception.

- Sandbox numbers are never reported (public, likely memorised labels). Only the private `test.csv` and a future
  post-cutoff slice can support a claim.
- Raw negatives (18.4M, computational) are unused: 97% of GT-test negatives recur in training negatives.
- `public_db/*.tsv.gz` keep only pairs among current TMP proteins; adding proteins or changing `IdMap` means
  re-running every extract. 1,320 TMP proteins have no Entrez gene or STRING ID and match by accession only.
- 65,047 PRING train negatives have a weak STRING link (≤ 700) by design; filter with `pair_annotations.parquet`
  for a stricter KB. IntAct pairs with both positive and negative evidence count as public.
- PRING: never use `*_ratio5_exclusive.txt` (negatives leak across the node split); it keeps self pairs as
  positives (dropped here); motif `node_*` ids follow accession order (anon arm uses fresh ids);
  `candidate_test_edges.txt` lists positives first.
- `data/TMP/processed/` is DIPPI-era and unused; raw file names carry upstream typos (`depedent_test.csv`).
- HuRI's site has a broken TLS certificate; it is covered via IntAct/BioGRID/HIPPIE. Never bypass certificate checks.
