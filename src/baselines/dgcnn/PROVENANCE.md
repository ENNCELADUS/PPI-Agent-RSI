# Vendored SEAL DGCNN baseline

Source: `seal_ppi_code_20261101(1).tar.gz` (SEAL PPI fixed-data baseline package, 2026-11-01; README in Chinese).
The user supplied it as the earlier DL PPI baseline. Copied files (sha256 of the originals):

| File | sha256 (prefix) | Change |
|---|---|---|
| `model.py` | `6fa2b79104af231a` | none (the DGCNN class: GCN layers, sort pooling k = 285, 1D conv, 2560 → 512 ESM projection) |
| `core.py` | `a41b68c7834d3ff1` | `import model as mod` → `from src.baselines.dgcnn import model as mod` |
| `run.py` | `d1566143b530e0bc` | rewritten for data-parallel training with Accelerate (user decision 2026-11-02): same model, loss, optimiser, selection and outputs; scores gathered across processes in row order |

Model, loss (focal, alpha 0.25), optimiser (Adam 1e-4), masking of the target edge, one-hop subgraphs on the train
positive graph and val-AUPRC selection are unchanged. The original ran on PyTorch 2.6 + PyG 2.0.4 on one GPU; on H20 it runs in the existing GraSPPI environment
(torch 2.10 cu128, PyG 2.8, accelerate 1.13) with `fair-esm` added as a `--target` overlay on `PYTHONPATH`. `prepare_mus22t.py` is new: it builds the
`fixed_data.pt` payload for the 22T agent split with SEAL's ESM2-3B feature recipe. `model.py` and `core.py` are
excluded from ruff to keep them byte-comparable with the source.
