# PPI-Agent-RSI

Research project: LLM-based scientific agents for membrane protein–protein interaction (PPI) discovery, and recursive self-improvement (RSI) of the agent workflow and, later, the PPI predictor under sparse experimental feedback. Full motivation in `docs/introduction.md`; baseline design in `docs/initial_baseline.md`; related work in `docs/literature.md` (PDFs in `literature/`, indexed by `literature/README.md`).

## Current phase: fixed agent baseline

Measure how well a fixed, expert-designed LLM agent ranks PPIs before building any RSI. Failure modes must come out of evaluation, not be assumed.

- **Task:** global top-N. Given a candidate pair pool, output the N most likely interacting pairs; scored by precision@N and related ranking/calibration metrics, then re-scored after revealing a limited number of labels as feedback.
- **No predictor yet.** The baseline agent runs on LLM knowledge plus the reference KB. A trained predictor (e.g. our GraspPI) is added later as a tool; do not vendor or retrain DIPPI here.
- **Open, decide before building the agent:** which LLM is pinned, and whether the agent gets web/literature retrieval or is closed-world.
- **Future:** survey public PPI benchmarks for agents to complement the private one.

## Data

- `data/TMP/` is **our private dataset** (human transmembrane-protein PPIs, synced from the DIPPI repo on `10.15.89.191`). `raw/` holds the original pretrain/finetune/GT-test pairs; `processed/` holds the DIPPI-era splits, which this project does not use.
- `scripts/agent_split/` builds the agent benchmark in `data/TMP/agent_split/`:
  - `extract_public.py <source>` streams a public DB (STRING, BioGRID, IntAct, HIPPIE, HuRI, BioPlex, CORUM; `idmap` for the UniProt→Entrez map) into `public_db/<source>.tsv.gz`, restricted to TMP proteins.
  - `build_split.py` pools all raw pairs at gene level; `test.csv` = every pair absent from all public DBs (all non-public positives + negatives at up to 38:1); `train.csv` = everything else, used as the agent's **reference KB**, not to train a model. No validation split.
  - Run from `scripts/agent_split/` (modules import `common` directly).

## Hard rules

- **No label leakage.** The agent never reads `agent_split/test.csv` labels or `data/TMP/raw|processed`. Candidates reach it unlabeled; labels enter only through the evaluation script and the explicit feedback-reveal step.
- **Private data stays private.** Never commit, push, upload, or send TMP data to external services, except to the LLM the agent is configured to call. `data/` is git-ignored.
- Keep the baseline fixed: prompts, tools, and model are pinned per run so results are comparable to later RSI versions.

## Engineering

- `uv`-managed Python project with one `pyproject.toml`; add dependencies there rather than `uv run --with`.
- One-off pipelines live in `scripts/`; reusable agent/eval code goes in a package under `src/` once it exists.
- DuckDB for joins over the ~19M raw pairs; stream large public files instead of storing them.
