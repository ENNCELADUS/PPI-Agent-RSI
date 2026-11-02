import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.baselines.dgcnn.prepare_mus22t import payload  # noqa: E402


def test_payload_maps_pairs_to_sorted_node_rows_by_label():
    ids = ["A", "B", "C"]
    x = np.ones((3, 2560), dtype=np.float32)
    d = payload(ids, x, {"train": [("A", "B", 1), ("A", "C", 0)], "valid": [("B", "C", 1), ("A", "C", 0)]})
    assert d["split"]["train"]["edge"].tolist() == [[0], [1]] and d["split"]["train"]["edge_neg"].tolist() == [[0], [2]]
    assert d["split"]["valid"]["edge"].tolist() == [[1], [2]]
    assert d["num_nodes"] == 3 and d["id_to_protein"][2] == "C" and d["x"].shape == (3, 2560)
