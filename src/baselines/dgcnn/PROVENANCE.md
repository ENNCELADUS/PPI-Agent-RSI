# Vendored SEAL DGCNN baseline

Source: `seal_ppi_code_20261101(1).tar.gz` (SEAL PPI fixed-data baseline package, 2026-11-01; README in Chinese).
The user supplied it as the earlier DL PPI baseline. Copied files (sha256 of the originals):

| File | sha256 (prefix) | Change |
|---|---|---|
| `model.py` | `6fa2b79104af231a` | none (the DGCNN class: GCN layers, sort pooling k = 285, 1D conv, 2560 → 512 ESM projection) |
| `core.py` | `a41b68c7834d3ff1` | `import model as mod` → `from src.baselines.dgcnn import model as mod` |
| `run.py` | `d1566143b530e0bc` | `from core import` → `from src.baselines.dgcnn.core import` |

Model, loss (focal, alpha 0.25), optimiser (Adam 1e-4), masking of the target edge, one-hop subgraphs on the train
positive graph and val-AUPRC selection are unchanged. The original ran on PyTorch 2.6 + PyG 2.0.4; here it runs on the
`dgcnn` extra (PyG ≥ 2.5, where `global_sort_pool` still exists). `prepare_mus22t.py` is new: it builds the
`fixed_data.pt` payload for the 22T agent split with SEAL's ESM2-3B feature recipe. The three vendored files are
excluded from ruff to keep them byte-comparable with the source.
