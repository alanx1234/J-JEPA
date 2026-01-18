#!/usr/bin/env python3
import os, json, glob, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

from src.options import Options
from src.models.jjepa import JJEPA
from src.dataset.ParticleDataset import ParticleDataset

def collate_eval_fn(batch):
    p_spatial = default_collate([b[0] for b in batch])
    p4        = default_collate([b[1] for b in batch])
    mask      = default_collate([b[2] for b in batch])
    return p_spatial, p4, mask

@torch.no_grad()
def encode_batch(encoder, p4, p4_spatial, particle_mask, stats, use_parT: bool):
    if use_parT:
        reps = encoder(p4, p4_spatial, particle_mask, split_mask=None, stats=stats)
    else:
        reps = encoder(p4, particle_mask, split_mask=None, stats=stats)
    return reps.mean(dim=1)

def rankme_from_Z(Z: torch.Tensor, eps: float = 1e-12) -> float:
    Z = Z.to(torch.float64)
    Z = Z - Z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Z)
    s = torch.clamp(s, min=0.0)
    p = torch.clamp(s / (torch.sum(s) + eps), min=eps, max=1.0)
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H).cpu().item())

def fixed_subset_indices(total, n, seed=123):
    rng = np.random.RandomState(seed)
    n = min(int(n), int(total))
    return rng.choice(total, size=n, replace=False).tolist()

def load_checkpoint_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--n_jets", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    options = Options.load(args.config)
    options.batch_size = args.batch_size

    data_path = args.data_path
    if args.split == "val":
        data_path = data_path.replace("train", "val")

    ds = ParticleDataset(
        data_path,
        num_jets=None,
        compute_subjets=False,
        return_labels=False,
        shuffle_files_each_epoch=False,
    )

    idxs = fixed_subset_indices(len(ds), args.n_jets, seed=args.seed)

    loader = DataLoader(
        Subset(ds, idxs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_eval_fn,
        persistent_workers=True,
    )

    stats = ds.stats

    model = JJEPA(options).to(device).float()
    model.eval()

    ckpt_paths = sorted(glob.glob(args.checkpoints))
    if not ckpt_paths:
        raise SystemExit(f"No checkpoints matched: {args.checkpoints}")

    results = {}
    for ckpt_path in ckpt_paths:
        load_checkpoint_into_model(model, ckpt_path, device)
        enc = model.target_transformer
        enc.eval()

        Z_chunks = []
        for (p4_spatial, p4, mask) in loader:
            p4 = p4.to(device, non_blocking=True).float()
            p4_spatial = p4_spatial.to(device, non_blocking=True).float()
            particle_mask = mask.squeeze(-1).to(device, non_blocking=True).float()

            z = encode_batch(
                enc,
                p4=p4,
                p4_spatial=p4_spatial,
                particle_mask=particle_mask,
                stats=stats,
                use_parT=options.use_parT_encoder,
            )
            Z_chunks.append(z.detach().cpu())

        Z = torch.cat(Z_chunks, dim=0)
        score = rankme_from_Z(Z)

        results[ckpt_path] = {
            "rankme": score,
            "N": int(Z.shape[0]),
            "D": int(Z.shape[1]),
            "split": args.split,
            "seed": args.seed,
        }

        print(f"{ckpt_path}: RankMe={score:.3f} (N={Z.shape[0]}, D={Z.shape[1]})")

        if args.out is None:
            outp = Path(ckpt_path).with_suffix("").as_posix() + "_rankme.json"
            with open(outp, "w") as f:
                json.dump(results[ckpt_path], f, indent=2)

    if args.out is not None:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
