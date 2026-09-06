# -*- coding: utf-8 -*-
"""Downstream utility experiment: the filling-rate sensitivity sweep.

This is the measurement the method's core claim rests on. A from-scratch
classifier is trained once per filling rate on the real imbalanced subset plus
that ratio's share of synthetic images, and every run is evaluated on the *same*
held-out real test set. The paper reports that low filling rates can hurt - too
few synthetic samples add noise without fixing the imbalance - that performance
recovers through the mid range, and that it peaks at 1.0 ("balance-to-max").

Protocol notes:
- The test split is real-only, stratified, and identical for every ratio.
- The real training subset is reconstructed with the same seed and the same
  name-keyed counts the generator was trained on, so condition r=0.0 is exactly
  the dataset the generator saw and nothing else.
- Synthetic images for a given ratio are a sorted prefix of one generated pool,
  so ratios are nested and the curve reflects augmentation volume rather than a
  change of sample.
- Every ratio uses identical hyperparameters and epochs; the only difference is
  the training set. Higher ratios therefore also perform proportionally more
  optimization steps, which is the augmentation treatment being measured.
- Known limitation, disclosed in README.md: the papers evaluate LSTM/GRU
  sequence models on 21-frame molten-pool sequences. Static weld-bead images
  have no temporal axis, so an image classifier is substituted. The *protocol*
  is reproduced; the numbers are not comparable to the papers'.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.models import resnet18

from augment import (filling_rate_counts, generated_by_class, synthetic_prefix,
                     verify_generation_meta)
from data import (ClassFolderDataset, ListDataset, count_by_class, make_splits,
                  parse_name_counts, sample_named_subset)


def build_model(num_classes, channels=1):
    """ResNet-18 trained from scratch.

    From scratch rather than ImageNet-pretrained on purpose: the papers choose
    lightweight models over heavy pretrained ones for in-situ edge computing, and
    a pretrained backbone saturates on this task, which hides the augmentation
    effect being measured.
    """
    model = resnet18(weights=None)
    if channels != 3:
        model.conv1 = nn.Conv2d(channels, model.conv1.out_channels,
                                model.conv1.kernel_size, model.conv1.stride,
                                model.conv1.padding, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one(model, loader, epochs, lr, device, tag):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
        if ep % 10 == 0 or ep == epochs:
            print(f"[{tag}] epoch {ep}/{epochs} loss={tot / n:.4f}")


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p in zip(y.numpy(), pred.numpy()):
            cm[t, p] += 1
    return cm


def metrics_from_cm(cm, class_names):
    """Per-class P/R/F1 plus accuracy and macro/weighted F1, from the confusion
    matrix directly - keeps the dependency footprint at requirements.txt."""
    per_class, f1s, supports = [], [], []
    for i, cname in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1); supports.append(support)
        per_class.append((cname, float(prec), float(rec), float(f1), support))
    total = int(cm.sum())
    summary = {
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.sum(np.array(f1s) * np.array(supports)) / total) if total else 0.0,
    }
    return per_class, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--gen_root", required=True,
                    help="generate.py output tree holding the ratio-1.0 pool")
    ap.add_argument("--subset", required=True,
                    help="real training subset by class NAME, must match what the "
                         "generator was trained on, e.g. 'pore=40,deposit=150,discontinuity=300,stain=600'")
    ap.add_argument("--ratios", default="0.0,0.25,0.5,0.75,1.0",
                    help="comma-separated filling rates to sweep")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--fft_denoise", action="store_true")
    ap.add_argument("--fft_cutoff", type=float, default=0.25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default="runs/classifier_sweep")
    args = ap.parse_args()

    ratios = [float(r) for r in args.ratios.split(",") if r.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ClassFolderDataset(args.data_root, args.img_size, args.channels,
                            args.fft_denoise, args.fft_cutoff)
    classes, num_classes = ds.classes, len(ds.classes)

    # Same split and same seed as train_joint.py, so the generator never saw the
    # test images and condition r=0.0 is exactly the generator's training set.
    train_pool, test_idx = make_splits(ds.samples, args.test_frac, args.seed)
    subset_idx = sample_named_subset(ds, parse_name_counts(args.subset), train_pool, args.seed)
    real_counts = count_by_class(ds, subset_idx)
    print(f"classes ({num_classes}): {classes}")
    print(f"real training subset: {real_counts} -> {len(subset_idx)}")
    print(f"held-out real test:   {count_by_class(ds, test_idx)} -> {len(test_idx)}")

    pool = generated_by_class(args.gen_root, classes)
    verify_generation_meta(args.gen_root, args.subset, args.seed, args.test_frac)
    print("generated pool available:", {k: len(v) for k, v in pool.items()})
    n_max = max(real_counts.values())
    print(f"N_max (majority class count) = {n_max}; balance-to-max fills every class to it")

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 pin_memory=True, persistent_workers=args.num_workers > 0)
    loader_test = DataLoader(Subset(ds, test_idx), shuffle=False, drop_last=False, **dl_kw)

    rows = []
    for ratio in ratios:
        tag = f"r={ratio:g}"
        need = filling_rate_counts(real_counts, ratio)
        synthetic = []
        for label, name in enumerate(classes):
            picked = synthetic_prefix(pool[name], need[name])
            synthetic.extend((p, label) for p in picked)
        train_ds = (Subset(ds, subset_idx) if not synthetic
                    else ConcatDataset([Subset(ds, subset_idx),
                                        ListDataset(synthetic, ds.tf, args.channels)]))
        total_counts = {n: real_counts[n] + need[n] for n in classes}
        print(f"\n=== {tag} === synthetic +{len(synthetic)} -> {len(train_ds)} total; "
              f"per class {total_counts}")

        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = build_model(num_classes, args.channels).to(device)
        if len(train_ds) <= args.batch_size:
            raise SystemExit(
                f"{tag}: only {len(train_ds)} training images for batch size "
                f"{args.batch_size}; lower --batch_size")
        # drop_last: a final batch of one image makes BatchNorm raise in train mode.
        loader_train = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
        train_one(model, loader_train, args.epochs, args.lr, device, tag)

        cm = evaluate(model, loader_test, device, num_classes)
        np.save(out / f"cm_r{ratio:g}.npy", cm)
        per_class, summary = metrics_from_cm(cm, classes)
        for cname, prec, rec, f1, support in per_class:
            print(f"  {cname:<6} P={prec:.4f} R={rec:.4f} F1={f1:.4f} support={support}")
            rows.append((ratio, f"precision:{cname}", prec))
            rows.append((ratio, f"recall:{cname}", rec))
            rows.append((ratio, f"f1:{cname}", f1))
            rows.append((ratio, f"support:{cname}", float(support)))
        print(f"  accuracy={summary['accuracy']:.4f} macroF1={summary['macro_f1']:.4f} "
              f"weightedF1={summary['weighted_f1']:.4f}")
        for metric, value in summary.items():
            rows.append((ratio, metric, value))
        rows.append((ratio, "n_synthetic", float(len(synthetic))))
        rows.append((ratio, "n_train", float(len(train_ds))))

        with open(out / "sweep_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ratio", "metric", "value"])
            writer.writerows(rows)

    print("\n================ FILLING-RATE SWEEP (held-out real test) ================")
    header = f"{'ratio':>6} {'acc':>7} {'macroF1':>8} {'wF1':>7} " + " ".join(
        f"{c + ':F1':>9}" for c in classes)
    print(header)
    by_ratio = {}
    for ratio, metric, value in rows:
        by_ratio.setdefault(ratio, {})[metric] = value
    lines = [header]
    for ratio in ratios:
        m = by_ratio[ratio]
        row = (f"{ratio:>6g} {m['accuracy']:>7.4f} {m['macro_f1']:>8.4f} "
               f"{m['weighted_f1']:>7.4f} "
               + " ".join(f"{m[f'f1:{c}']:>9.4f}" for c in classes))
        print(row); lines.append(row)

    with open(out / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nresults -> {out / 'results.txt'}; metrics -> {out / 'sweep_metrics.csv'}")


if __name__ == "__main__":
    main()
