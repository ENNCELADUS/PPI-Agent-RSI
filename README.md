# PPI-Agent-RSI

LLM scientific agents for membrane protein–protein interaction (PPI) discovery, and recursive
self-improvement of the agent workflow under sparse experimental feedback. Current phase: measure a
fixed, expert-designed agent baseline before building any RSI.

## Layout

| Path | Contents |
| --- | --- |
| `docs/introduction.md` | Motivation and the three research directions |
| `docs/initial_baseline.md` | Baseline agent design and evaluation protocol |
| `docs/task-setting.html` | Problem formulation: train / validation / acquisition / final test splits |
| `docs/literature.md` | Related work, with links to local PDFs |
| `docs/research-reports/` | Dated deep-research reports; `research_notes/` holds the notes behind each |
| `literature/` | PDFs by topic (`ppi`, `science-agents`, `self-improvement`, `feedback-benchmarks`); index in `literature/README.md` |
| `docs/superpowers/specs/` | Dated design documents |
| `results/` | Tracked readable summaries, one folder per run (raw artifacts in `data/PRING/runs/`) |
| `src/` | The code, one package; module map in [`src/README.md`](src/README.md) |
| `tests/` | pytest suite mirroring `src/` |
| `data/` | Private TMP data, git-ignored; never shared outside the configured LLM |

## Benchmarks

- **Private (built):** human transmembrane PPIs; test positives have no evidence in any public PPI
  database, with 1:1 degree-preserving negatives. See `CLAUDE.md` for the build commands.
- **22T mouse agent split and sandbox (built; run with `src.eval.pring_test --data mus22t`):** the private 22T mouse membrane positives split at
  random 8:1:1 (transductive: proteins appear in every split) with new PRING degree-preserving negatives at 1:1;
  train = KB, val = DL predictor validation, test = final (RSI reveals part of test). Labels may be public; the agent's sandbox has no web tools. The agent
  sees `data/MUS22T/sandbox/` (PRING-sandbox layout, mouse UniProt annotations, KB = train). See
  `build_mus22t_agent_split.py` and `build_mus22t_sandbox.py` in `src/data/builders/`. Test AUPRC: LLM-only
  agent 0.587, the V3 predictor 0.716, and the agent with V3's scores (`--predictor`, `predict_pairs`, `build_mus22t_scores.py`)
  0.708 (`results/classification-test/summary.html`, with the PRING test).
- **SEAL-protocol splits (built):** the SEAL PPI baseline protocol (random edge 80/10/10, 1:1), so DL numbers compare
  with SEAL's: the private 22T mouse membrane data (22-tissue proteomics) and the human TMP data (all stages). Most
  positives are public or orthologs of public pairs, so these are not agent test pools. See
  `src/data/builders/build_seal_split.py`.
- **PRING dev sandbox (built):** an internal, closed-world benchmark built from PRING human data, for
  debugging the agent before it sees the private test set. Its 1:1 train and val splits form the agent's
  reference KB; the test proteins are the closed-loop discovery candidates. Its numbers are never reported. See the [design](docs/superpowers/specs/2026-10-23-pring-sandbox-design.md).
- **Closed-loop discovery benchmark (true-feedback baseline run on DeepSeek, internal):** the agent searches all
  pairs among the sandbox's test proteins in rounds of 200 picks and sees the measured result of each of its
  own picks before the next round. See the [design](docs/superpowers/specs/2026-10-23-pring-wave-discovery-design.md).
  Ribosome ablation (done): the same agent with the 71 cytosolic ribosomal proteins removed from the candidates. The
  agent's prompt no longer states the hit rate among candidate pairs (unknown in a real screen).
- **PRING classification test (implemented):** PRING train is the agent's KB, val is kept for the RSI
  part, and an LLM-only agent (no DL predictor, no web) predicts each pair of PRING's test split as released
  (self pairs included). Full test, DeepSeek-V4 Pro: AUPRC 0.799 vs B0 0.736
  ([results](results/classification-test/summary.html)). See the [design](docs/superpowers/specs/2026-11-01-pring-classification-test-design.md).
- **Two-layer memory agent (run, DeepSeek-V4 Pro):** the PRING test proteins are split 60/40. Ten discovery rounds
  on the 60% build a case memory and reviewer-updated lessons. A fixed 1:1 test (~10.3k pairs) on the 40% compares
  memory (AUPRC 0.803), cases-only (0.804), cases without the predictor (0.811), empty-memory (0.775), B0 (0.695)
  and B0 retrained on the 1,895 revealed pairs (0.717, `src.eval.al_predictor`). See the [design](docs/superpowers/specs/2026-11-02-two-layer-memory-agent-design.md).
- **Other public tracks (surveyed):** [2026-10-19 survey](docs/research-reports/2026-10-19-public-ppi-benchmarks.md):
  BioDiscoveryAgent screens, SKEMPI 2.0 via `ppi-mutation-ddg`, and a post-cutoff IntAct/BioGRID slice.
  All are run closed-world, because their labels are retrievable online.

## Setup

```bash
uv sync
(cd src/agent/pi && npm ci)   # the pi agent runtime (Node), for closed-loop discovery runs
```

For a DeepSeek V4 Pro endpoint, set `LLM_API_KEY` and `LLM_API_BASE` in the shell, then use
`--model custom/<model-id> --thinking high`. The provider reuses pi's built-in model settings.
Use `--workers 8` for eight concurrent classification batches. Remote runs follow the local, untracked `remote-api-agent` skill.

## Model API gateway

The default model is `gateway/deepseek-pro`. GPT uses the streaming `/api/v1/response`
endpoint (the gateway restricts reasoning with tools on `/start`); `gateway/deepseek-pro`
and `gateway/kimi` use `/api/v1/start`. pi still manages the tool loop and Python tools.
Set `GATEWAY_API_KEY` and `GATEWAY_API_BASE` in the environment. Both are required;
the adapter has no default endpoint. Credentials are never written to run artifacts.
On Windows, `run-agent.ps1` also reads the key from the current user's environment:

```powershell
.\run-agent.ps1 -m src.eval.memory_agent rounds gateway-run --model gateway/deepseek-pro --thinking high
.\run-agent.ps1 -m src.eval.pring_test gateway-test --model gateway/deepseek-pro --limit 1
```

Build the corresponding sandbox data before running experiments. Gateway model limits use conservative
local budgets (128k context, 16k output), not advertised server limits. Reported gateway costs are zero
placeholders because actual pricing is unknown. Kimi's thinking level is not mapped to an effort parameter.
The original `custom/` provider remains available with `LLM_API_KEY` and `LLM_API_BASE`.
