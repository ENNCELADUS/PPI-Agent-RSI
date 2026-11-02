import itertools

import numpy as np
import pytest

pytest.importorskip("torch")
from src.data.builders.build_seal_split import auroc_auprc, seal_split  # noqa: E402


def test_seal_split_is_80_10_10_and_1_to_1_with_clean_negatives():
    ids = [f"p{i:02d}" for i in range(30)]
    pairs = list(itertools.combinations(ids, 2))
    pos = np.array(pairs[::4] + [("p00", "zz")])  # last row has an unknown end and is dropped
    neg = np.array(pairs + [("p01", "p01")] + [p[::-1] for p in pairs[:5]])  # all pairs, a self pair, duplicates
    split, sorted_ids, _ = seal_split(ids, pos, neg)
    n_pos = len(pos) - 1
    assert [split[s][0].shape[1] for s in ("train", "valid", "test")] == [
        int(0.8 * n_pos),
        int(0.1 * n_pos),
        n_pos - int(0.8 * n_pos) - int(0.1 * n_pos),
    ]
    posset = {frozenset(sorted_ids[e]) for s in split.values() for e in s[0].T}
    negs = [frozenset(sorted_ids[e]) for s in split.values() for e in s[1].T]
    assert all(split[s][0].shape == split[s][1].shape for s in split)
    assert len(set(negs)) == len(negs) and not set(negs) & posset and all(len(x) == 2 for x in negs)


def test_auroc_auprc_handles_ties():
    assert auroc_auprc([1, 1, 0, 0], [1, 0, 1, 0]) == {"auprc": 0.5, "auroc": 0.5}
    assert auroc_auprc([3, 2, 1], [1, 1, 0]) == {"auprc": 1.0, "auroc": 1.0}
