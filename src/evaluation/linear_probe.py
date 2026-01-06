#!/bin/env python3
import sys
sys.path.insert(0, "../src")

import os
import re
import argparse
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from sklearn.metrics import accuracy_score
from sklearn import metrics

from src.models.jjepa import JJEPA
from src.options import Options
from src.dataset.ParticleDataset import ParticleDataset

torch.set_num_threads(2)


def collate_ptcl_keep_subjets_last(batch):
    tensors = default_collate([b[:-1] for b in batch])  # includes labels if present
    subjets = [b[-1] for b in batch]                    # keep python list
    return (*tensors, subjets)


def load_split(dataset_path, batch_size, num_workers):
    dataset = ParticleDataset(
        dataset_path,
        return_labels=True,
        num_jets=None,
        compute_subjets=True,
    )
    stats = dataset.stats
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_ptcl_keep_subjets_last,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return dataloader, stats


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_perf_stats(labels, measures):
    measures = np.nan_to_num(measures)
    auc = metrics.roc_auc_score(labels, measures)
    fpr, tpr, _ = metrics.roc_curve(labels, measures)

    fpr2 = [fpr[i] for i in range(len(fpr)) if tpr[i] >= 0.5]
    tpr2 = [tpr[i] for i in range(len(tpr)) if tpr[i] >= 0.5]

    epsilon = 1e-8
    try:
        if len(tpr2) > 0 and len(fpr2) > 0:
            nearest_tpr_idx = list(tpr2).index(find_nearest(list(tpr2), 0.5))
            imtafe = np.nan_to_num(1 / (fpr2[nearest_tpr_idx] + epsilon))
            if imtafe > 1e4:
                imtafe = 1
        else:
            imtafe = 1
    except (ValueError, IndexError):
        imtafe = 1

    return float(auc), float(imtafe)


def extract_epoch(path):
    m = re.search(r"checkpoint_epoch_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else -1


@torch.no_grad()
def embed_split(net, args, loader, stats):
    zs, ys = [], []
    net.eval()

    for batch in tqdm.tqdm(loader, leave=False):
        if len(batch) == 5:
            p4_spatial, p4, particle_mask, labels, _subjets = batch
        elif len(batch) == 4:
            p4_spatial, p4, particle_mask, labels = batch
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}; expected 4 or 5")

        y = labels.to(args.device)
        particle_mask = particle_mask.squeeze(-1).bool()

        p4 = p4.to(dtype=torch.float32)
        p4_spatial = p4_spatial.to(dtype=torch.float32)

        p4 = p4.to(args.device, non_blocking=True)
        p4_spatial = p4_spatial.to(args.device, non_blocking=True)
        particle_mask = particle_mask.to(args.device, non_blocking=True, dtype=torch.float32)

        if args.use_parT:
            reps = net(p4, p4_spatial, particle_mask, split_mask=None, stats=stats)
        else:
            reps = net(p4, particle_mask, split_mask=None, stats=stats)

        if args.flatten:
            reps = reps.view(reps.shape[0], -1)
        elif args.sum:
            reps = reps.sum(dim=1)
        else:
            raise ValueError("No aggregation method specified (use --sum 1 or --flatten 1)")

        zs.append(reps.detach().cpu())
        ys.append(y.detach().cpu())

    Z = torch.cat(zs, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()

    # Ensure binary labels {0,1} if dataset uses other conventions
    uniq = np.unique(Y)
    if len(uniq) == 2 and not np.array_equal(uniq, np.array([0, 1])):
        mapping = {int(uniq[0]): 0, int(uniq[1]): 1}
        Y = np.vectorize(lambda v: mapping[int(v)])(Y).astype(np.int64)

    return Z, Y


def train_linear_head(Ztr, Ytr, Zva, Yva, device, lr, epochs, batch_size, weight_decay, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = Ztr.shape[1]
    head = nn.Linear(in_dim, 2).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    Xtr = torch.from_numpy(Ztr).to(device)
    ytr = torch.from_numpy(Ytr).long().to(device)

    n = Xtr.shape[0]
    for _ in range(epochs):
        head.train()
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            j = idx[start:start + batch_size]
            out = head(Xtr[j])
            loss = loss_fn(out, ytr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        Xva = torch.from_numpy(Zva).to(device)
        out = head(Xva)
        probs = softmax(out).detach().cpu().numpy()
        scores = probs[:, 1]
        ytrue = Yva

    acc = accuracy_score(ytrue, scores > 0.5)
    auc, imtafe = get_perf_stats(ytrue, scores)
    return float(acc), float(auc), float(imtafe)


def list_checkpoints(ckpt_dir):
    ckpts = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
    ]
    ckpts = sorted(ckpts, key=extract_epoch)
    return ckpts


def main(args):
    options = Options.load(args.option_file)
    args.use_parT = options.use_parT_encoder
    args.device = torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")

    train_loader, train_stats = load_split(args.train_dataset_path, args.batch_size, args.num_workers)
    val_loader, val_stats = load_split(args.val_dataset_path, args.batch_size, args.num_workers)

    ckpts = list_checkpoints(args.ckpt_dir)
    if not ckpts:
        raise FileNotFoundError("No checkpoint_epoch_*.pth found in ckpt directory")

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_csv, "w") as f:
        f.write("epoch,acc,auc,imtafe,ckpt\n")

    for ckpt_path in ckpts:
        epoch = extract_epoch(ckpt_path)
        print(f"\nprobing checkpoint epoch {epoch}: {ckpt_path}")

        model = JJEPA(options).to(args.device)
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        if "model" not in checkpoint:
            raise KeyError(f"Expected key 'model' in checkpoint dict. Keys: {list(checkpoint.keys())}")
        model.load_state_dict(checkpoint["model"])
        net = model.target_transformer

        net.eval()
        for p in net.parameters():
            p.requires_grad = False

        Ztr, Ytr = embed_split(net, args, train_loader, train_stats)
        Zva, Yva = embed_split(net, args, val_loader, val_stats)

        acc, auc, imtafe = train_linear_head(
            Ztr, Ytr, Zva, Yva,
            device=args.device,
            lr=args.probe_lr,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            weight_decay=args.probe_weight_decay,
            seed=args.seed,
        )

        with open(args.out_csv, "a") as f:
            f.write(f"{epoch},{acc:.6g},{auc:.6g},{imtafe:.6g},{ckpt_path}\n")

        print(f"VAL acc={acc:.4f} auc={auc:.4f} imtafe={imtafe:.2f}")

    print("\nwrote:", args.out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option-file", type=str, required=True)
    parser.add_argument("--train-dataset-path", type=str, required=True)
    parser.add_argument("--val-dataset-path", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--sum", type=int, default=1)
    parser.add_argument("--flatten", type=int, default=0)

    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
