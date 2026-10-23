# Module map

One line per module, from its header. Commands and rules are in [`CLAUDE.md`](../CLAUDE.md). Run any
module from the repo root as `python -m src.<path>`. `data` never imports `eval` or `baselines`.

### `data/`

- `paths`: Data locations, all under the git-ignored `data/` directory.
- `ids`: TMP protein identity: accessions as gene-level nodes, and the raw TMP universe.
- `negatives`: PRING negative sampling for the agent split.
- `public_db`: Filter one public PPI source (text on stdin) to pairs among TMP proteins.

### `data/builders/`

- `build_agent_split`: Build the private agent-benchmark train/test split from raw TMP positives and public DB tables.
- `build_pring_sandbox`: Build the PRING dev sandbox, a closed-world internal reference KB for debugging the agent.
- `fetch_public.sh`: Fetch each public PPI source (resumable) and pipe it to `public_db`.

### `eval/`

- `score`: Score a global top-N ranking against a labels file (the private `test.csv`; sandbox pools not built yet).
- `verify_live_string`: Spot-check that sampled test pairs are absent from the live STRING API.

### `predictor/` (needs `uv sync --extra predictor`: torch, numpy)

- `model`: The B0 PPI predictor (motif-graph `V3_1`, config b0_v31), inference only; built from the config `best.pt` embeds.
- `features`: Read-only accessor for the motif-graph per-node feature store (per-residue ESM-3, keys node_*).
- `score`: Score protein pairs with the frozen B0 checkpoint: one process per GPU (`score --shard K`), then `merge`.
- `parity`: Parity of the vendored B0 scorer against a reference motif-graph scores artifact (e.g. the PRING test.npz).
