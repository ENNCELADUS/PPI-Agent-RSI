"""Score protein pairs with the frozen B0 checkpoint: one process per GPU, then merge.

Vendored from motif-graph-prompt-tuning-for-inductive-protein-protein-interaction-prediction @ bb01ee1:
src/score_universe.py (_score_v3_1_packed, _shard_range, _read_pairs_tsv, _checkpoint_id), src/data/pairs.py
(BUCKET_BOUNDARIES, LengthBucketedBatchSampler with shuffle=False), src/data/packed_features.py (bf16 pack).

    python -m src.predictor.score score --checkpoint best.pt --features <store dir> --pairs pairs.tsv \
        --out-dir <dir> [--shard K --num-shards N] [--amp bf16|off] [--token-budget 131072] [--device auto]
    python -m src.predictor.score merge <dir> --output scores.tsv

pairs.tsv has no header: `node_a<TAB>node_b[<TAB>label]`. Each pair is scored as (min, max), as the
source canonicalises; the output keeps the input ids and row order. `score` writes
<dir>/shard-K-of-N.npz for the contiguous rows [K*ceil(n/N), (K+1)*ceil(n/N)); `merge` checks that all
N shards are present and writes `node_a, node_b, logit[, label]` (logit = raw float32, not sigmoid)
plus <output>.meta.json. A node without features fails the run before any scoring.

Numerics match the reference test.npz (score_universe via score_fanout: --amp bf16, default token
budget 131072, --pack-dir b0_v31_bf16, 4 shards): features rounded to bf16 as the pack stores them,
each used node encoded once per shard in feature-store order and bucket batches of
token_budget // boundary nodes, an fp32 encoding cache, then pair batches of
token_budget // (2 * boundary) rows within a length bucket, all under bf16 autocast. Self pairs are
scored like any other pair. Parity holds only with the same shard count as the reference.
"""

import argparse
import hashlib
import json
import logging
import math
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from src.predictor.features import FeatureStore
from src.predictor.model import load_model

BUCKET_BOUNDARIES = (128, 256, 384, 512, 768, 1024)
SOURCE = "motif-graph-prompt-tuning-for-inductive-protein-protein-interaction-prediction@bb01ee1"
logger = logging.getLogger(__name__)


def bucket(length):
    for boundary in BUCKET_BOUNDARIES:
        if length <= boundary:
            return boundary
    raise ValueError(f"length {length} exceeds the largest bucket boundary {BUCKET_BOUNDARIES[-1]}")


def autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if amp == "bf16" else nullcontext()


def read_pairs(path):
    """Rows of the pairs TSV as ([(a, b)], int8 labels with -1 where absent)."""
    pairs, labels = [], []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) not in (2, 3):
            raise ValueError(f"{path}:{line_number}: expected 2 or 3 tab-separated fields, got {len(parts)}")
        pairs.append((parts[0], parts[1]))
        labels.append(int(parts[2]) if len(parts) == 3 else -1)
    return pairs, np.array(labels, dtype=np.int8)


def shard_range(num_rows, shard, num_shards):
    chunk = math.ceil(num_rows / num_shards) if num_rows else 0
    start = min(shard * chunk, num_rows)
    return start, min(start + chunk, num_rows)


def checkpoint_id(state):
    """The source's 16-hex state digest, equal to `checkpoint_id` in the reference scores' meta."""
    hasher = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(tensor.numpy(force=True).tobytes())
    return hasher.hexdigest()[:16]


def score_pairs(model, pairs, store, *, device, amp="bf16", token_budget=131_072):
    """Float32 logits for `pairs` (already canonical), in input order."""
    if not pairs:  # a trailing shard of a short pairs file
        return np.empty(0, dtype=np.float32)
    used = {node for pair in pairs for node in pair}
    missing = sorted(used - store.index.keys())
    if missing:
        raise ValueError(f"feature store {store.root} is missing {len(missing)} nodes: {missing[:5]}")
    nodes = [node for node in store.index if node in used]  # store order = the source pack's node order
    position = {node: i for i, node in enumerate(nodes)}
    tokens = [store.load_tokens(node) for node in nodes]
    lengths = torch.tensor([t.size(0) for t in tokens], dtype=torch.int64)
    node_lengths = lengths.tolist()

    encoded = torch.zeros(
        (len(nodes), max(bucket(n) for n in node_lengths), model.d_model), dtype=torch.float32, device=device
    )
    lengths = lengths.to(device)
    previous = 0
    for boundary in BUCKET_BOUNDARIES:
        members = [i for i, n in enumerate(node_lengths) if previous < n <= boundary]
        previous = boundary
        step = max(token_budget // boundary, 1)
        for start in range(0, len(members), step):
            chunk = members[start : start + step]
            raw = torch.zeros((len(chunk), boundary, store.input_dim), dtype=torch.bfloat16)
            for row, i in enumerate(chunk):
                raw[row, : node_lengths[i]] = tokens[i].to(torch.bfloat16)  # the pack's bf16 rounding
            raw = raw.to(device) if amp == "bf16" else raw.to(device).float()
            index = torch.tensor(chunk, dtype=torch.int64, device=device)
            with torch.inference_mode(), autocast(device, amp):
                node_encoded = model.encoder(raw, lengths.index_select(0, index))
            encoded[index, :boundary] = node_encoded.float()
    logger.info("encoded %d unique nodes", len(nodes))
    del tokens

    node_a = torch.tensor([position[a] for a, _ in pairs], dtype=torch.int64)
    node_b = torch.tensor([position[b] for _, b in pairs], dtype=torch.int64)
    pair_bucket = [bucket(max(node_lengths[position[a]], node_lengths[position[b]])) for a, b in pairs]
    out = np.empty(len(pairs), dtype=np.float32)
    done = 0
    for boundary in BUCKET_BOUNDARIES:
        members = [i for i, b in enumerate(pair_bucket) if b == boundary]
        step = max(1, token_budget // (2 * boundary))
        for start in range(0, len(members), step):
            rows = torch.tensor(members[start : start + step], dtype=torch.int64)
            a, b = node_a[rows].to(device), node_b[rows].to(device)
            with torch.inference_mode(), autocast(device, amp):
                logits = model.logits_from_encoded(
                    encoded.index_select(0, a)[:, :boundary],
                    encoded.index_select(0, b)[:, :boundary],
                    lengths.index_select(0, a),
                    lengths.index_select(0, b),
                )
            out[rows.numpy()] = logits.detach().to(torch.float32).cpu().numpy().reshape(-1)
            done += len(rows)
        if members:
            logger.info("scored %d/%d rows", done, len(pairs))
    return out


def run_score(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, state = load_model(args.checkpoint)
    model.to(device)
    pairs, labels = read_pairs(args.pairs)
    start, end = shard_range(len(pairs), args.shard, args.num_shards)
    rows = pairs[start:end]
    logger.info("shard %d/%d: rows [%d, %d) on %s", args.shard, args.num_shards, start, end, device)
    store = FeatureStore(args.features)
    canonical = [(min(a, b), max(a, b)) for a, b in rows]
    logits = score_pairs(model, canonical, store, device=device, amp=args.amp, token_budget=args.token_budget)
    meta = {
        "source": SOURCE,
        "checkpoint": str(args.checkpoint),
        "checkpoint_id": checkpoint_id(state),
        "features": str(args.features),
        "pairs": str(args.pairs),
        "num_rows": len(pairs),
        "num_shards": args.num_shards,
        "amp": args.amp,
        "token_budget": args.token_budget,
        "torch_version": torch.__version__,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"shard-{args.shard}-of-{args.num_shards}.npz"
    np.savez(
        path,
        row_start=np.int64(start),
        node_a=np.array([a for a, _ in rows], dtype=str),
        node_b=np.array([b for _, b in rows], dtype=str),
        logit=logits,
        label=labels[start:end],
        meta=np.array(json.dumps(meta)),
    )
    logger.info("wrote %d rows to %s", len(rows), path)


def run_merge(args):
    shards = sorted(args.out_dir.glob("shard-*-of-*.npz"))
    if not shards:
        raise ValueError(f"no shard-*-of-*.npz files in {args.out_dir}")
    loaded = [np.load(path) for path in shards]
    metas = [json.loads(str(z["meta"])) for z in loaded]
    num_shards, num_rows = metas[0]["num_shards"], metas[0]["num_rows"]
    expected = {args.out_dir / f"shard-{k}-of-{num_shards}.npz" for k in range(num_shards)}
    if set(shards) != expected:
        raise ValueError(f"{args.out_dir} must hold exactly shards 0..{num_shards - 1} of {num_shards}, got {shards}")
    if any(
        (m["num_rows"], m["checkpoint_id"], m["pairs"]) != (num_rows, metas[0]["checkpoint_id"], metas[0]["pairs"])
        for m in metas
    ):
        raise ValueError("shards disagree on num_rows, checkpoint_id or pairs")
    loaded.sort(key=lambda z: int(z["row_start"]))
    cursor = 0
    for z in loaded:
        if int(z["row_start"]) != cursor:
            raise ValueError(f"shard rows are not contiguous at row {cursor}")
        cursor += len(z["logit"])
    if cursor != num_rows:
        raise ValueError(f"shards cover {cursor} rows, expected {num_rows}")
    labelled = all((z["label"] >= 0).all() for z in loaded)
    with open(args.output, "w", encoding="utf-8") as out:
        out.write("node_a\tnode_b\tlogit" + ("\tlabel" if labelled else "") + "\n")
        for z in loaded:
            for i, (a, b, logit) in enumerate(zip(z["node_a"], z["node_b"], z["logit"], strict=True)):
                out.write(f"{a}\t{b}\t{logit}" + (f"\t{z['label'][i]}" if labelled else "") + "\n")
    meta = {key: value for key, value in metas[0].items() if key != "created_utc"}
    Path(f"{args.output}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("merged %d shards (%d rows) into %s", num_shards, num_rows, args.output)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.predictor.score")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--features", type=Path, required=True, help="feature store dir (metadata.json, index.json)")
    score.add_argument("--pairs", type=Path, required=True, help="headerless node_a<TAB>node_b[<TAB>label]")
    score.add_argument("--out-dir", type=Path, required=True)
    score.add_argument("--shard", type=int, default=0)
    score.add_argument("--num-shards", type=int, default=1)
    score.add_argument("--amp", choices=["bf16", "off"], default="bf16")
    score.add_argument("--token-budget", type=int, default=131_072)
    score.add_argument("--device", default="auto")
    merge = commands.add_parser("merge")
    merge.add_argument("out_dir", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "score" and not 0 <= args.shard < args.num_shards:
        parser.error(f"--shard must be in [0, {args.num_shards})")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "score":
        run_score(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
