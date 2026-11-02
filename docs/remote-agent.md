# API agent experiments on remote WSL

Use this guide for API agent experiments on a remote WSL host. Replace every uppercase placeholder
before use. Host details, API credentials and endpoint addresses belong in the remote shell, not Git.
The API performs model inference; these runs do not need a GPU or PyTorch.

## Prepare the environment

Connect to the remote host and clone the repository. Use Git to sync later code changes.

```bash
ssh USER@REMOTE_HOST
git clone REPOSITORY_URL ~/PPI-Agent-RSI
cd ~/PPI-Agent-RSI
uv sync --python 3.11
(cd src/agent/pi && npm ci)
mkdir -p data/PRING/runs
```

Install Git, uv, Python 3.11, Node.js 22.19 or later, npm and tmux on the remote host first.
The classification evaluator needs NumPy. The commands below use `uv run --with numpy` to supply it
without installing the predictor's PyTorch dependencies or changing the project dependency files.

Provision the prepared PRING files from an approved data source. Git does not contain these files:

| Path | Purpose |
| --- | --- |
| `data/PRING/sandbox_cls/proteins.tsv` | Agent protein records |
| `data/PRING/sandbox_cls/annotations.tsv` | Agent annotations |
| `data/PRING/sandbox_cls/kb.csv` | Train-only knowledge base |
| `data/PRING/sandbox_cls/homologs.tsv` | Homologs to knowledge-base proteins |
| `data/PRING/hidden/test.csv` | Evaluator labels |
| `data/PRING/raw/node_sequences.tsv` | Evaluator node mapping |
| `data/PRING/predictor/pring_candidates.tsv.gz` | Evaluator-side B0 comparison |

Keep labels and B0 scores outside the agent sandbox. `sandbox_cls/` must not contain `scores.tsv.gz`.

## Configure the remote Bash shell

Replace the key and URL placeholders, then execute this block once in the remote shell:

```bash
cat >> ~/.bashrc <<'EOF'

export LLM_API_KEY='REPLACE_WITH_API_KEY'
export LLM_API_BASE='https://api.example.invalid/v1'
EOF
chmod 600 ~/.bashrc
source ~/.bashrc
```

The key is stored as plain text in this shell file. Do not copy it into the repository or share the file.
Run `source ~/.bashrc` in an existing terminal after changing the values. New interactive Bash sessions
read the file automatically. The Python process passes these variables to pi.

`LLM_API_BASE` is the complete SDK base URL. The client appends `/chat/completions`; do not append that
suffix twice. Use the base URL supplied by the service operator. If direct routing is required, add
the API host, `127.0.0.1` and `localhost` to both `NO_PROXY` and `no_proxy` in the remote shell.

The `custom/MODEL_ID` selector uses `MODEL_ID` as the request's model field. The current extension
reuses pi's built-in DeepSeek V4 Pro definition, including reasoning and compatibility settings.
It changes only the model ID and base URL; the key comes from the shell. This definition is not a
generic template for other model families. Do not replace its token limits with small fixed values.
The endpoint must support streaming, function tools and the selected reasoning parameters.

## Validate one complete batch

Use a new run name when the model, reasoning level, prompts, test data or batch size changes. The
example below starts a new LLM-only classification run: protein and KB tools, no predictor, no web,
no memory. The evaluator computes the B0 comparison separately.

```bash
uv run --with numpy python -u -m src.eval.pring_test pring-test-pro \
  --batch 200 --workers 1 --limit 1 \
  --model custom/MODEL_ID --thinking high
```

Check `data/PRING/runs/pring-test-pro/batch0001/result.json`: it must contain 200 rows with valid `p`
values, no final `error`, and no `length` stop. A successful one-pair request does not validate a
200-pair batch. A missing submission can be scored with missing predictions even when the command
exits successfully; inspect the result and coverage before the full run.

## Run the full test with eight workers

Start a persistent terminal on the remote host:

```bash
tmux new-session -s ppi-agent
```

Inside that terminal, run the same configuration without `--limit`:

```bash
cd ~/PPI-Agent-RSI
source ~/.bashrc
set -o pipefail
uv run --with numpy python -u -m src.eval.pring_test pring-test-pro \
  --batch 200 --workers 8 \
  --model custom/MODEL_ID --thinking high \
  2>&1 | tee -a data/PRING/runs/pring-test-pro.log
```

The full released test contains 64,038 pairs in 321 batches at batch size 200. The completed validation
batch is reused. Eight workers run eight independent batches; they do not change the batch size.
Detach with `Ctrl-b`, then `d`. Disconnecting SSH does not stop tmux. A WSL or host shutdown does.

```bash
tmux attach-session -t ppi-agent
tail -f ~/PPI-Agent-RSI/data/PRING/runs/pring-test-pro.log
```

## Resume and interpret outputs

After a stopped process has exited, repeat the full command with the same run name and settings.
Do not start two processes for the same run directory. Existing `batch*/result.json` files are loaded;
batches without a result are retried. Keep model, reasoning, prompts and data unchanged: the current
resume check enforces batch size only. A completed result with missing predictions is also reused;
it is not automatically repaired. Use a separate run for a changed model or protocol.

`summary.json` is written when the command finishes. A summary from the validation batch can remain
while the full run is active; it is not a live progress file. Check the final batch count, pair count,
coverage and failed-batch count. A zero process exit code alone does not prove full coverage.

AUROC and AP use continuous scores. F1 and related classification metrics use the threshold that
maximizes F1 on the test labels; these are oracle metrics, not a threshold validated on unseen data.
Missing predictions are scored as 0.5 and reduce coverage. Cost estimates use official API prices;
they do not establish the endpoint's actual charges. See the
[classification protocol](superpowers/specs/2026-11-01-pring-classification-test-design.md).
