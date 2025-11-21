#!/bin/env python3.7

import sys
sys.path.insert(0, "../src")

import os
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
from src.evaluation.ClassificationHead import ClassificationHead
import json

torch.set_num_threads(2)


def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def collate_drop_subjets(batch):
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 5:
        batch = [(b[0], b[1], b[2], b[4]) for b in batch]
    return default_collate(batch)


def load_test_data(args, dataset_path):
    dataset = ParticleDataset(
        dataset_path,
        return_labels=True,
        num_jets=None,
        compute_subjets=False,
    )
    stats = dataset.stats
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_drop_subjets,
        shuffle=False,
    )
    return dataloader, stats
def load_checkpoint_weights(net, proj, out_dir, ckpt_type, device):
    if ckpt_type == "last":
        checkpoint_path = os.path.join(out_dir, "last_checkpoint.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["encoder"])
        proj.load_state_dict(checkpoint["projector"])
        return checkpoint_path

    suffix_map = {
        "best_acc": "acc",
        "best_loss": "loss",
        "best_rej": "rej",
    }
    if ckpt_type not in suffix_map:
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")

    suffix = suffix_map[ckpt_type]
    enc_path = os.path.join(out_dir, f"jjepa_finetune_best_{suffix}.pt")
    proj_path = os.path.join(out_dir, f"projector_finetune_best_{suffix}.pt")

    if not os.path.isfile(enc_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {enc_path}")
    if not os.path.isfile(proj_path):
        raise FileNotFoundError(f"Projector checkpoint not found: {proj_path}")

    net.load_state_dict(torch.load(enc_path, map_location=device))
    proj.load_state_dict(torch.load(proj_path, map_location=device))
    return f"{enc_path} + {proj_path}"

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
    return auc, imtafe


def eval_single_trial(options, args, out_dir):
    test_dataloader, test_stats = load_test_data(args, args.test_dataset_path)

    model = JJEPA(options).to(args.device)
    net = model.target_transformer

    finetune_mlp_dim = args.output_dim
    if args.finetune_mlp:
        finetune_mlp_dim = f"{args.output_dim}-{args.finetune_mlp}"

    if args.cls:
        proj = ClassificationHead(finetune_mlp_dim).to(args.device)
    else:
        proj = Projector(2, finetune_mlp_dim).to(args.device)

    src = load_checkpoint_weights(net, proj, out_dir, args.checkpoint_type, args.device)
    print(f"loaded checkpoint from {src}")

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    softmax = torch.nn.Softmax(dim=1)

    losses_e = []
    predicted_e = []
    correct_e = []

    net.eval()
    proj.eval()
    with torch.no_grad():
        pbar = tqdm.tqdm(test_dataloader)
        for i, (p4_spatial, p4, particle_mask, labels) in enumerate(pbar):
            y = labels.to(args.device)
            particle_mask = particle_mask.squeeze(-1).bool()
            p4 = p4.to(dtype=torch.float32)
            p4_spatial = p4_spatial.to(dtype=torch.float32)
            p4 = p4.to(args.device, non_blocking=True)
            p4_spatial = p4_spatial.to(args.device, non_blocking=True)
            particle_mask = particle_mask.to(
                args.device, non_blocking=True, dtype=torch.float32
            )

            if args.use_parT:
                reps = net(
                    p4, p4_spatial, particle_mask, split_mask=None, stats=test_stats
                )
            else:
                reps = net(p4, particle_mask, split_mask=None, stats=test_stats)

            if not args.cls:
                if args.flatten:
                    reps = reps.view(reps.shape[0], -1)
                elif args.sum:
                    reps = reps.sum(dim=1)
                else:
                    raise ValueError("No aggregation method specified")
                out = proj(reps)
            else:
                out = proj(reps.transpose(0, 1), padding_mask=particle_mask == 0)

            batch_loss = loss_fn(out, y.long()).detach().cpu().item()
            losses_e.append(batch_loss)
            predicted_e.append(softmax(out).cpu().data.numpy())
            correct_e.append(y.cpu().data)
            pbar.set_description(f"test loss: {batch_loss}")

    loss_test = float(np.mean(np.array(losses_e)))
    predicted = np.concatenate(predicted_e)
    target = np.concatenate(correct_e)

    acc = accuracy_score(target, predicted[:, 1] > 0.5)
    auc, imtafe = get_perf_stats(target, predicted[:, 1])

    np.save(os.path.join(out_dir, "test_target_vals.npy"), target)
    np.save(os.path.join(out_dir, "test_predicted_vals.npy"), predicted)

    return loss_test, acc, auc, imtafe


def main(args):
    options = Options.load(args.option_file)
    args.use_parT = options.use_parT_encoder
    args.output_dim = options.emb_dim
    if args.flatten and not args.cls:
        args.output_dim *= 128

    if torch.cuda.device_count():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    if args.parent_dir:
        trial_dirs = [
            os.path.join(args.parent_dir, d)
            for d in sorted(os.listdir(args.parent_dir))
            if d.startswith("trial-") and os.path.isdir(os.path.join(args.parent_dir, d))
        ]
        all_losses = []
        all_accs = []
        all_aucs = []
        all_imtafes = []

        for d in trial_dirs:
            print("evaluating", d)
            loss_test, acc, auc, imtafe = eval_single_trial(options, args, d)
            print("trial:", d)
            print("  test loss:", loss_test)
            print("  test acc :", acc)
            print("  test auc :", auc)
            print("  test imtafe:", imtafe)
            all_losses.append(loss_test)
            all_accs.append(acc)
            all_aucs.append(auc)
            all_imtafes.append(imtafe)

        losses = np.array(all_losses)
        accs = np.array(all_accs)
        aucs = np.array(all_aucs)
        imtafes = np.array(all_imtafes)

        print("summary over", len(trial_dirs), "trials")
        print("loss   mean:", losses.mean(), "std:", losses.std(ddof=1))
        print("acc    mean:", accs.mean(), "std:", accs.std(ddof=1))
        print("auc    mean:", aucs.mean(), "std:", aucs.std(ddof=1))
        print("imtafe mean:", imtafes.mean(), "std:", imtafes.std(ddof=1))

        summary = {
            "trials": [
                {
                    "trial": os.path.basename(d),
                    "loss": float(l),
                    "acc": float(a),
                    "auc": float(au),
                    "imtafe": float(im),
                }
                for d, l, a, au, im in zip(
                    trial_dirs, losses, accs, aucs, imtafes
                )
            ],
            "mean": {
                "loss": float(losses.mean()),
                "acc": float(accs.mean()),
                "auc": float(aucs.mean()),
                "imtafe": float(imtafes.mean()),
            },
            "std": {
                "loss": float(losses.std(ddof=1)),
                "acc": float(accs.std(ddof=1)),
                "auc": float(aucs.std(ddof=1)),
                "imtafe": float(imtafes.std(ddof=1)),
            },
        }

        summary_path = os.path.join(args.parent_dir, "test_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("wrote test summary to", summary_path)


    else:
        if not args.out_dir:
            raise ValueError("either --out-dir or --parent-dir must be specified")
        loss_test, acc, auc, imtafe = eval_single_trial(options, args, args.out_dir)
        print("test loss:", loss_test)
        print("test acc :", acc)
        print("test auc :", auc)
        print("test imtafe:", imtafe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option-file", type=str, required=True)
    parser.add_argument("--test-dataset-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--finetune-mlp", type=str, default="")
    parser.add_argument("--flatten", type=int, default=0)
    parser.add_argument("--sum", type=int, default=1)
    parser.add_argument("--cls", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--parent-dir", type=str, default="")
    parser.add_argument( "--checkpoint-type", type=str,default="last", choices=["last", "best_acc", "best_loss", "best_rej"])
    args = parser.parse_args()
    main(args)
