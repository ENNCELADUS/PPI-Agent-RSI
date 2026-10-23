# Module map

One line per module, from its header. Commands and rules are in [`CLAUDE.md`](../CLAUDE.md). Run any
module from the repo root as `python -m src.<path>`. `data` never imports `eval` or `baselines`.

### `data/`

- `paths`: Data locations, all under the git-ignored `data/` directory.
- `ids`: TMP protein identity: accessions as gene-level nodes, and the raw TMP universe.
- `negatives`: PRING negative sampling, shared by the agent split and the PRING sandbox.
- `public_db`: Filter one public PPI source (text on stdin) to pairs among TMP proteins.

### `data/builders/`

- `build_agent_split`: Build the private agent-benchmark train/test split from raw TMP positives and public DB tables.
- `build_pring_sandbox`: Build the PRING dev sandbox, a closed-world internal benchmark for debugging the agent.
- `fetch_public.sh`: Fetch each public PPI source (resumable) and pipe it to `public_db`.

### `eval/`

- `score`: Score a global top-N ranking against a labels file (sandbox `hidden/` or the private `test.csv`).
- `verify_live_string`: Spot-check that sampled test pairs are absent from the live STRING API.

### `baselines/`

- `topology`: Non-LLM baseline rankings (KB degree product, L3) for the PRING sandbox.
