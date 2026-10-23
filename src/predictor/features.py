"""Read-only accessor for the motif-graph per-node feature store (per-residue ESM-3, keys node_*).

Vendored from motif-graph-prompt-tuning-for-inductive-protein-protein-interaction-prediction @ bb01ee1:
src/data/features.py (FeatureStore). Layout: `metadata.json` (format torch_pt_per_node, input_dim),
`index.json` (node id -> relative .pt path, in the order the source's bf16 pack used), one
(L, input_dim) float32 tensor per node, BOS/EOS included. Tensors are memory-mapped, so a node's
length is known without reading its data.
"""

import json
from pathlib import Path

import torch

_EXPECTED_FORMAT = "torch_pt_per_node"


class FeatureStore:
    def __init__(self, root):
        self.root = Path(root)
        metadata = json.loads((self.root / "metadata.json").read_text())
        if metadata.get("format") != _EXPECTED_FORMAT:
            raise ValueError(
                f"Unexpected feature store format {metadata.get('format')!r}; expected {_EXPECTED_FORMAT!r}"
            )
        if "input_dim" not in metadata:
            raise ValueError("metadata.json is missing required field 'input_dim'")
        self.input_dim = int(metadata["input_dim"])
        self.index = json.loads((self.root / "index.json").read_text())

    def load_tokens(self, node_id):
        """(L, input_dim) float32 tensor, memory-mapped; KeyError if the node has no features."""
        if node_id not in self.index:
            raise KeyError(node_id)
        tensor = torch.load(self.root / self.index[node_id], map_location="cpu", weights_only=True, mmap=True)
        if tensor.ndim != 2 or tensor.size(1) != self.input_dim:
            raise ValueError(
                f"Feature tensor for {node_id!r} has shape {tuple(tensor.shape)}, expected (L, {self.input_dim})"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(f"Feature tensor for {node_id!r} has dtype {tensor.dtype}, expected torch.float32")
        return tensor
