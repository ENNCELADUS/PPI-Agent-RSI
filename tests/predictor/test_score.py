import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.predictor.model import build_model  # noqa: E402
from src.predictor.score import main  # noqa: E402

CONFIG = {
    "input_dim": 8,
    "d_model": 16,
    "encoder_layers": 1,
    "cross_attn_layers": 1,
    "n_heads": 2,
    "mlp_head": {"hidden_dims": [8], "dropout": 0.2, "activation": "gelu", "norm": "layernorm"},
    "pair_readout": {"mode": "pair_context_gated", "order_aggregation": "abba_max"},
    "mixing": {"mode": "none"},
}
LENGTHS = {"node_1": 5, "node_2": 7, "node_3": 130, "node_4": 300, "node_5": 4}
# reversed orientation of row 0, a self pair, and pairs spanning three length buckets
PAIRS = [
    ("node_1", "node_2", 1),
    ("node_2", "node_1", 1),
    ("node_3", "node_3", 1),
    ("node_4", "node_1", 0),
    ("node_5", "node_3", 0),
    ("node_2", "node_4", 0),
    ("node_3", "node_1", 1),
]


@pytest.fixture
def setup(tmp_path):
    generator = torch.Generator().manual_seed(0)
    store = tmp_path / "features"
    (store / "embeddings").mkdir(parents=True)
    (store / "metadata.json").write_text(json.dumps({"format": "torch_pt_per_node", "input_dim": 8}))
    (store / "index.json").write_text(json.dumps({node: f"embeddings/{node}.pt" for node in LENGTHS}))
    for node, length in LENGTHS.items():
        torch.save(torch.randn(length, 8, generator=generator), store / "embeddings" / f"{node}.pt")
    torch.manual_seed(0)
    checkpoint = tmp_path / "best.pt"
    state = build_model(CONFIG).state_dict()
    torch.save({"model_state": state, "model_family": "v3_1", "model_config": CONFIG}, checkpoint)
    pairs = tmp_path / "pairs.tsv"
    pairs.write_text("".join(f"{a}\t{b}\t{y}\n" for a, b, y in PAIRS))
    return tmp_path, ["--checkpoint", str(checkpoint), "--features", str(store)]


def run(tmp_path, common, name, num_shards, pairs="pairs.tsv", amp="off"):
    out_dir = tmp_path / name
    for shard in range(num_shards):
        main(["score", *common, "--pairs", str(tmp_path / pairs), "--out-dir", str(out_dir)]
             + ["--shard", str(shard), "--num-shards", str(num_shards), "--amp", amp]
             + ["--token-budget", "1024", "--device", "cpu"])  # fmt: skip
    main(["merge", str(out_dir), "--output", str(tmp_path / f"{name}.tsv")])
    lines = (tmp_path / f"{name}.tsv").read_text().splitlines()
    assert lines[0] == "node_a\tnode_b\tlogit\tlabel"
    return [line.split("\t") for line in lines[1:]]


@pytest.mark.parametrize("amp", ["off", "bf16"])
def test_rows_keep_input_order_and_sharding_does_not_change_scores(setup, amp):
    tmp_path, common = setup
    whole = run(tmp_path, common, "whole", 1, amp=amp)
    sharded = run(tmp_path, common, "sharded", 3, amp=amp)
    assert [(a, b, int(y)) for a, b, _, y in whole] == PAIRS
    assert [r[:2] for r in sharded] == [r[:2] for r in whole]
    logits = np.array([np.float32(r[2]) for r in whole])
    assert np.isfinite(logits).all()
    np.testing.assert_allclose([np.float32(r[2]) for r in sharded], logits, atol=1e-5)
    assert logits[0] == logits[1]  # endpoint order does not matter


def test_missing_features_fail_closed(setup):
    tmp_path, common = setup
    (tmp_path / "bad.tsv").write_text("node_1\tnode_9\n")
    with pytest.raises(ValueError, match="missing 1 nodes"):
        run(tmp_path, common, "bad", 1, pairs="bad.tsv")


def test_merge_refuses_incomplete_shards(setup):
    tmp_path, common = setup
    out_dir = tmp_path / "partial"
    main(["score", *common, "--pairs", str(tmp_path / "pairs.tsv"), "--out-dir", str(out_dir)]
         + ["--shard", "0", "--num-shards", "2", "--amp", "off", "--device", "cpu"])  # fmt: skip
    with pytest.raises(ValueError, match="exactly shards"):
        main(["merge", str(out_dir), "--output", str(tmp_path / "partial.tsv")])


def test_unsupported_model_option_is_rejected():
    with pytest.raises(ValueError, match="mixing.mode"):
        build_model({**CONFIG, "mixing": {"mode": "bidirectional_cross"}})
