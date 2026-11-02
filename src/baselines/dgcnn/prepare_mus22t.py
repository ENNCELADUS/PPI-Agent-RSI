"""Build the SEAL DGCNN input (fixed_data.pt) for the 22T agent split: ESM2-3B node features + the split's pairs.

Run on a GPU host (H20): accelerate launch --num_processes 4 -m src.baselines.dgcnn.prepare_mus22t
(each process embeds every num_processes-th node; the main process merges the shards and writes the payload).
Then: accelerate launch --num_processes 4 -m src.baselines.dgcnn.run train --data data/MUS22T/dgcnn/fixed_data.pt \
    --output data/MUS22T/dgcnn/train_<date> --workers 8 --microbatch 64 --epochs 20

Inputs: data/MUS22T/raw/raw_22T_node_sequences.csv, data/MUS22T/agent_split/{train,val,test}.csv (the same 1:1 pairs
the agent is scored on). Features follow SEAL provenance/rebuild_esm_and_train.py: esm2_t36_3B_UR50D (fp16), layer 35,
mean over real residue tokens (no BOS/EOS/padding), sequences cut at the 99th length percentile, every node. Node rows
are the sorted node IDs (SEAL's canonical order). The SEAL payload keys are kept: x, split[train|valid|test][edge|
edge_neg] (2 x n index tensors), id_to_protein, num_nodes. The ESM checkpoint is fetched by fair-esm's hub loader.
"""

import argparse
import csv
import hashlib
import json

import numpy as np
import torch

from src.data.paths import MUS22T

OUT = MUS22T / "dgcnn"
LAYER, TOKEN_BUDGET = 35, 2048


def esm_means(seqs, device, rows):
    """Mean layer-35 embedding over the residues of seqs[rows], as SEAL computes it (OOM halves the batch); other
    rows stay zero."""
    import esm

    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.half().to(device).eval()
    convert = alphabet.get_batch_converter()
    out = np.zeros((len(seqs), 2560), dtype=np.float32)
    order = sorted(rows, key=lambda i: len(seqs[i]))
    batches, batch, longest = [], [], 0
    for i in order:
        longest = max(longest, len(seqs[i]) + 2)
        if batch and (longest * (len(batch) + 1) > TOKEN_BUDGET or len(batch) >= 8):
            batches.append(batch)
            batch, longest = [], len(seqs[i]) + 2
        batch.append(i)
    batches.append(batch)

    def run(idx):
        try:
            _, _, tokens = convert([(str(i), seqs[i]) for i in idx])
            with torch.inference_mode():
                rep = model(tokens.to(device), repr_layers=[LAYER])["representations"][LAYER]
                vec = torch.stack([rep[j, 1 : len(seqs[i]) + 1].float().mean(0) for j, i in enumerate(idx)])
            out[idx] = vec.cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            run(idx[: len(idx) // 2])
            run(idx[len(idx) // 2 :])

    for k, idx in enumerate(batches, 1):
        run(idx)
        if k % 100 == 0 or k == len(batches):
            print(f"esm {k}/{len(batches)} batches", flush=True)
    bad = [i for i in rows if not np.isfinite(out[i]).all() or not np.linalg.norm(out[i]) > 0]
    assert not bad, (
        f"invalid ESM output for {len(bad)} rows: {[(i, len(seqs[i]), out[i][:2].tolist()) for i in bad[:5]]}"
    )
    return out


def payload(ids, x, splits):
    """SEAL fixed_data.pt layout from node IDs, features and {split: [(a, b, label)]}."""
    index = {v: i for i, v in enumerate(ids)}
    split = {}
    for name, rows in splits.items():
        edges = {y: [(index[a], index[b]) for a, b, label in rows if label == y] for y in (1, 0)}
        split[name] = {"edge": torch.tensor(edges[1]).T.contiguous(), "edge_neg": torch.tensor(edges[0]).T.contiguous()}
    return {"x": torch.from_numpy(x), "split": split, "id_to_protein": dict(enumerate(ids)), "num_nodes": len(ids)}


def main():
    from accelerate import Accelerator

    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    accelerator = Accelerator()
    rank, world = accelerator.process_index, accelerator.num_processes
    seq = {r["id"]: r["sequence"].upper() for r in csv.DictReader(open(MUS22T / "raw" / "raw_22T_node_sequences.csv"))}
    ids = sorted(seq)
    maxlen = int(np.percentile([len(s) for s in seq.values()], 99))
    seqs = [seq[i][:maxlen] for i in ids]
    splits = {}
    for name, file in (("train", "train"), ("valid", "val"), ("test", "test")):
        rows = csv.DictReader(open(MUS22T / "agent_split" / f"{file}.csv"))
        splits[name] = [(r["uniprotID_A"], r["uniprotID_B"], int(r["isInteraction"])) for r in rows]
    OUT.mkdir(exist_ok=True)
    rows = list(range(rank, len(ids), world))
    np.save(OUT / f"esm_shard{rank}.npy", esm_means(seqs, accelerator.device, rows)[rows])
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    x = np.zeros((len(ids), 2560), dtype=np.float32)
    for r in range(world):
        x[r::world] = np.load(OUT / f"esm_shard{r}.npy")
        (OUT / f"esm_shard{r}.npy").unlink()
    assert (np.linalg.norm(x, axis=1) > 0).all(), "a node has no ESM feature"
    data = payload(ids, x, splits)
    torch.save(data, OUT / "fixed_data.pt")
    meta = {
        "nodes": len(ids),
        "max_length": maxlen,
        "layer": LAYER,
        "checkpoint": "esm2_t36_3B_UR50D (fair-esm hub)",
        "processes": world,
        "pairs": {s: {k: int(v.shape[1]) for k, v in d.items()} for s, d in data["split"].items()},
        "fixed_data_sha256": hashlib.sha256((OUT / "fixed_data.pt").read_bytes()).hexdigest(),
    }
    (OUT / "prepare.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
