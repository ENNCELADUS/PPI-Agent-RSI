# Module map

One line per module, from its header. Commands and rules are in [`CLAUDE.md`](../CLAUDE.md). Run any
module from the repo root as `python -m src.<path>`. `data` never imports `eval` or `baselines`; `agent` never imports `eval`.

### `data/`

- `paths`: Data locations, all under the git-ignored `data/` directory.
- `ids`: TMP protein identity: accessions as gene-level nodes, and the raw TMP universe.
- `negatives`: PRING negative sampling for the agent split.
- `public_db`: Filter one public PPI source (text on stdin) to pairs among TMP proteins.

### `data/builders/`

- `build_agent_split`: Build the private agent-benchmark train/test split from raw TMP positives and public DB tables.
- `build_pring_sandbox`: Build the PRING dev sandbox, a closed-world internal reference KB for debugging the agent.
- `build_pring_tools`: Build the agent's tool files for PRING closed-loop discovery: calibrated predictor scores and KB homologs.
- `build_pring_noribo`: Build the ribosome-ablation sandbox: the PRING sandbox with the cytosolic ribosome removed from the candidates.
- `build_pring_cls`: Build the classification-test sandbox: the PRING sandbox with the KB cut to train and no predictor.
- `build_pring_memory`: Build the two-layer memory agent's split: discovery and final-test views of the PRING test proteins.
- `build_mus22t_agent_split`: Split the 22T mouse membrane PPIs transductively 8:1:1 by positive pairs, with new PRING negatives.
- `build_mus22t_sandbox`: Build the 22T agent sandbox: the agent-visible files for the 22T mouse membrane split, PRING-sandbox layout.
- `build_seal_split`: Split a PPI dataset (22T mouse, or human TMP) with the SEAL PPI baseline protocol, so its numbers compare with SEAL's.
- `fetch_public.sh`: Fetch each public PPI source (resumable) and pipe it to `public_db`.

### `agent/` (pi runtime: `npm ci` in `agent/pi/`)

- `tools`: The agent's tools for one round, served to pi over localhost HTTP (the TS extension only forwards calls).
- `session`: Run one round of the agent: a fresh pi session (RPC mode) whose only tools are the round's tools.
- `memory`: The agent's two memory layers: case memory (tested pairs, written by the system) and lessons (written by the reviewer session).
- `pi/ppi_tools.ts`: pi extension registering the tools from the server's spec file; `prompts/`: system, round, nudge.

For a DeepSeek V4 Pro endpoint, set `LLM_API_KEY` and `LLM_API_BASE` in the shell that
starts the run. Use `--model custom/<model-id> --thinking high`. The `pi/provider.ts` extension reads
the connection settings from the process environment. It does not write them to configuration files.
Set the complete SDK base URL in the shell; the client appends `/chat/completions`. The server must
support streaming and function tools. The extension reuses pi's built-in DeepSeek V4 Pro definition;
only the model ID and endpoint change. Cost estimates use the built-in official API prices and do not
show the endpoint's actual charges. Other model families need their own model definition.
Remote runs follow the local, untracked `remote-api-agent` skill.

### `eval/`

- `oracle`: The offline lab: answers submitted pairs from a sealed labels file. Never imported by src/agent.
- `rounds`: Run PRING closed-loop discovery with the pi agent: rounds of K submitted pairs, oracle results, true feedback.
- `pring_test`: Classification test on PRING: the agent predicts every pair of PRING's human BFS test split. `--data mus22t`: the same agent on the 22T mouse test split.
- `memory_agent`: Run the two-layer memory agent: discovery rounds with case memory and reviewer-updated lessons, then the final test.
- `al_predictor`: Active-learning B0 arm of the memory agent's final test: B0 retrained on the knowledge base plus the round labels.
- `score`: Score a global top-N ranking against a labels file (the private `test.csv`; sandbox pools not built yet).
- `verify_live_string`: Spot-check that sampled test pairs are absent from the live STRING API.

### `predictor/` (needs `uv sync --extra predictor`: torch, numpy)

- `model`: The B0 PPI predictor (motif-graph `V3_1`, config b0_v31), inference only; built from the config `best.pt` embeds.
- `features`: Read-only accessor for the motif-graph per-node feature store (per-residue ESM-3, keys node_*).
- `score`: Score protein pairs with the frozen B0 checkpoint: one process per GPU (`score --shard K`), then `merge`.
- `parity`: Parity of the vendored B0 scorer against a reference motif-graph scores artifact (e.g. the PRING test.npz).
