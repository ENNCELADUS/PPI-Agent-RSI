"""Train, evaluate or smoke-test the SEAL DGCNN on a fixed_data.pt payload; data-parallel over GPUs with Accelerate.

    accelerate launch --num_processes 4 --mixed_precision no -m src.baselines.dgcnn.run train \
        --data data/MUS22T/dgcnn/fixed_data.pt --output data/MUS22T/dgcnn/train_<date> --microbatch 64

SEAL's run.py with the device loop replaced by Accelerate (PROVENANCE.md): the same model (core.create_model),
focal loss, Adam 1e-4, masked one-hop subgraphs, val-AUPRC checkpoint selection and one final test. Each process
gets --microbatch graphs per step; DDP averages gradients, so the effective batch is microbatch x accumulate x
processes (SEAL: 128 x 2 x 1 = 256; default here 64 x 1 x 4 = 256). Evaluation gathers every process's scores
(gather_for_metrics drops the sampler's padding) before AUPRC/AUROC. Outputs as SEAL: config.json, graph_audit.json,
history.json, best.pt, last.pt, status.json, result.json, test_predictions.npz.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.baselines.dgcnn.core import audit, build_adjacency, create_model, focal, load, make_dataset, write_json


class Indexed(torch.utils.data.Dataset):
    """A dataset whose graphs carry their row index, so gathered scores can be put back in row order."""

    def __init__(self, ds):
        self.ds, self.pairs = ds, ds.pairs

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        d.idx = torch.tensor([i])
        return d


def forward(model, b):
    return model(b.z, b.edge_index, b.batch, b.x, None, b.node_id).view(-1)


@torch.no_grad()
def evaluate(model, loader, accelerator):
    model.eval()
    scores, labels, rows = [], [], []
    for b in loader:
        p, y, i = accelerator.gather_for_metrics((forward(model, b).sigmoid(), b.y, b.idx))
        scores.append(p.float().cpu())
        labels.append(y.cpu())
        rows.append(i.cpu())
    order = torch.cat(rows).argsort()
    p, y = torch.cat(scores)[order].numpy(), torch.cat(labels)[order].numpy()
    assert len(p) == len(loader.dataset), "gathered scores do not cover the split once"
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p))}, p, y


def main():
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["train", "evaluate", "smoke"])
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--checkpoint", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--microbatch", type=int, default=64)
    a.add_argument("--accumulate", type=int, default=1)
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--seed", type=int, default=42)
    args = a.parse_args()
    assert args.microbatch > 0 and args.accumulate > 0 and args.workers >= 0
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulate)
    main_process, o = accelerator.is_main_process, args.output
    if main_process:
        if o.exists() and any(o.iterdir()):
            raise RuntimeError("Output directory must be empty to preserve prior results")
        o.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # same initial weights on every process (DDP also broadcasts them)

    payload = load(args.data)
    A = build_adjacency(payload["split"]["train"]["edge"], payload["num_nodes"])
    if main_process:
        write_json(o / "graph_audit.json", audit(payload, A))
    active = torch.unique(torch.cat([e.flatten() for d in payload["split"].values() for e in d.values()]))
    assert torch.isfinite(payload["x"][active]).all() and (payload["x"][active].norm(dim=1) > 0).all()
    model = create_model(payload, accelerator.device)
    ds = {s: Indexed(make_dataset(payload, s, A)) for s in ["train", "valid", "test"]}
    kw = {"num_workers": args.workers, "pin_memory": True}
    if args.workers:
        kw["prefetch_factor"] = 2

    def loader(s, shuffle=False):
        return DataLoader(ds[s], batch_size=args.microbatch, shuffle=shuffle, **kw)

    checksum = hashlib.sha256(args.data.read_bytes()).hexdigest()
    effective = args.microbatch * args.accumulate * accelerator.num_processes
    if main_process:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        write_json(
            o / "config.json",
            config | {"data_sha256": checksum, "processes": accelerator.num_processes, "effective_batch": effective},
        )
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    if args.mode == "evaluate":
        state = load(args.checkpoint)
        model.load_state_dict(state.get("model", state), strict=True)
        model, test = accelerator.prepare(model, loader("test"))
        metrics, scores, labels = evaluate(model, test, accelerator)
        if main_process:
            np.savez_compressed(
                o / "test_predictions.npz", pairs=ds["test"].pairs.numpy(), scores=scores, labels=labels
            )
            write_json(o / "result.json", {"test": metrics, "data_sha256": checksum})
            print(metrics)
        return
    model, opt, train, valid, test = accelerator.prepare(
        model, opt, loader("train", True), loader("valid"), loader("test")
    )
    if args.mode == "smoke":
        model.train()
        b = next(iter(train))
        loss = focal(forward(model, b), b.y)
        assert torch.isfinite(loss)
        accelerator.backward(loss)
        opt.step()
        if main_process:
            write_json(o / "smoke.json", {"loss": float(loss), "finite": True, "processes": accelerator.num_processes})
        return

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = torch.zeros(2, device=accelerator.device)  # summed loss x graphs, graphs (this process)
        for step, b in enumerate(train):
            with accelerator.accumulate(model):
                loss = focal(forward(model, b), b.y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                accelerator.backward(loss)
                opt.step()
                opt.zero_grad(set_to_none=True)
            total += torch.stack([loss.detach() * b.num_graphs, torch.tensor(b.num_graphs, device=total.device)])
            if main_process and step % 100 == 0:
                write_json(
                    o / "status.json",
                    {"stage": "training", "epoch": epoch, "step": step + 1, "loss": float(total[0] / total[1])},
                )
        total = accelerator.reduce(total, "sum")
        val, _, _ = evaluate(model, valid, accelerator)
        if main_process:
            row = {"epoch": epoch, "train_loss": float(total[0] / total[1]), "valid": val}
            history.append(row)
            write_json(o / "history.json", history)
            state = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "valid": val,
                "data_sha256": checksum,
            }
            torch.save(state, o / "last.pt")
            if val["auprc"] > best:
                best = val["auprc"]
                torch.save(state, o / "best.pt")
            print(json.dumps(row), flush=True)
        accelerator.wait_for_everyone()

    state = load(o / "best.pt")
    accelerator.unwrap_model(model).load_state_dict(state["model"])
    if main_process:
        write_json(o / "status.json", {"stage": "test", "best_epoch": state["epoch"]})
    metrics, scores, labels = evaluate(model, test, accelerator)
    if main_process:
        np.savez_compressed(o / "test_predictions.npz", pairs=ds["test"].pairs.numpy(), scores=scores, labels=labels)
        write_json(
            o / "result.json",
            {"best_epoch": state["epoch"], "validation": state["valid"], "test": metrics, "data_sha256": checksum},
        )
        write_json(o / "status.json", {"stage": "complete"})


if __name__ == "__main__":
    main()
