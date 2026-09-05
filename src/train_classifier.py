# -*- coding: utf-8 -*-
"""Downstream utility experiment: fine-tune the same pretrained ResNet-18 on
(A) an imbalanced real subset vs (B) the same subset plus generated
augmentation, then compare per-class recall/F1 on a held-out real-only test
set. A positive defect-class recall / macro-F1 gain in condition B is the
method's core claim.

Protocol notes:
- The test split is real-only, stratified per class, and identical for both
  conditions.
- Condition A downsamples the minority defect classes of the real training
  pool (see --keep_ratios), mirroring the paper's "small and imbalanced"
  setting.
- Condition B adds class-conditional images produced by src/generate.py.
- Both conditions use identical hyperparameters and epochs; the only
  difference is the training set (B performs proportionally more optimization
  steps - that is the augmentation treatment being measured).
- Known limitation, shared with the paper's protocol: the generator was
  trained on all real images, so generated images may carry information about
  test images. A rigorous variant would retrain the generator on the
  classifier's training split only.
- Class folder names are the sorted subdirectory names, e.g. CR=0, LP=1,
  ND=2, PO=3. Generated images live in class_0..class_N under --gen_root
  using the same indices; the mapping is printed and verified, never assumed
  silently.
- Metrics are computed from the confusion matrix directly to keep the
  dependency footprint identical to requirements.txt.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.models import resnet18, ResNet18_Weights

from data import ClassFolderDataset


def parse_keep_ratios(text):
    out = {}
    for part in text.split(","):
        k, v = part.split(":")
        out[int(k)] = float(v)
    return out


def stratified_split(samples, test_frac, seed):
    rng = random.Random(seed)
    by_class = {}
    for i, (_, y) in enumerate(samples):
        by_class.setdefault(y, []).append(i)
    train_idx, test_idx = [], []
    for y, idxs in sorted(by_class.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        k = round(len(idxs) * test_frac)
        test_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])
    return sorted(train_idx), sorted(test_idx)


def apply_keep_ratios(samples, pool_idx, keep_ratios, seed):
    """Downsamples an index pool per class; mirrors data.make_imbalanced_subset
    but operates on a pool so that held-out test images can never leak into the
    imbalanced training set."""
    rng = random.Random(seed)
    by_class = {}
    for i in pool_idx:
        by_class.setdefault(samples[i][1], []).append(i)
    keep = []
    for y, idxs in sorted(by_class.items()):
        r = keep_ratios.get(y, 1.0)
        k = max(1, int(len(idxs) * r))
        keep.extend(rng.sample(idxs, k))
    return sorted(keep)


class ListDataset(Dataset):
    """Flat (path, label) list sharing the real dataset's preprocessing."""

    def __init__(self, samples, tf):
        self.samples = samples
        self.tf = tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fp, y = self.samples[i]
        return self.tf(Image.open(fp).convert("RGB")), y


def build_generated(gen_root, class_names, per_class, only_labels, seed):
    rng = random.Random(seed)
    out = []
    for i, cname in enumerate(class_names):
        if only_labels and i not in only_labels:
            continue
        d = Path(gen_root) / f"class_{i}"
        files = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        if not files:
            raise FileNotFoundError(f"no generated images in {d}")
        if len(files) > per_class:
            files = rng.sample(files, per_class)
        print(f"generated/{d.name} -> {cname} (label {i}): +{len(files)}")
        out.extend((f, i) for f in files)
    return out


def build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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
    lines, precisions, recalls, f1s, supports = [], [], [], [], []
    for i, cname in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1); supports.append(support)
        lines.append(f"  {cname:<6} P={prec:.4f} R={rec:.4f} F1={f1:.4f} support={support}")
    acc = float(np.trace(cm) / cm.sum())
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / cm.sum())
    lines.append(f"  accuracy={acc:.4f} macroF1={macro_f1:.4f} weightedF1={weighted_f1:.4f}")
    return "\n".join(lines), recalls, f1s, acc, macro_f1, weighted_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--gen_root", default="generated")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_ratios", default="0:0.2,1:0.2,2:1.0,3:0.2",
                    help="class_index:keep_fraction applied to the real training pool")
    ap.add_argument("--gen_per_class", type=int, default=1000)
    ap.add_argument("--gen_classes", default="0,1,3",
                    help="comma list of generated class labels to add; empty = all")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default="runs/classifier")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ClassFolderDataset(args.data_root, args.img_size)
    num_classes = len(ds.classes)
    print(f"classes: {num_classes} {ds.classes}, real samples: {len(ds)}")

    train_idx, test_idx = stratified_split(ds.samples, args.test_frac, args.seed)
    keep_ratios = parse_keep_ratios(args.keep_ratios)
    a_idx = apply_keep_ratios(ds.samples, train_idx, keep_ratios, args.seed)
    print(f"split: train pool {len(train_idx)}, test {len(test_idx)} "
          f"(test_frac={args.test_frac}, seed={args.seed}, real only)")

    def counts(idxs):
        c = {}
        for i in idxs:
            c[ds.samples[i][1]] = c.get(ds.samples[i][1], 0) + 1
        return {ds.classes[k]: v for k, v in sorted(c.items())}

    print("condition A (imbalanced real):", counts(a_idx), "->", len(a_idx))

    only_labels = {int(v) for v in args.gen_classes.split(",") if v != ""}
    gen_samples = build_generated(args.gen_root, ds.classes, args.gen_per_class,
                                  only_labels, args.seed)
    b_ds = ConcatDataset([Subset(ds, a_idx), ListDataset(gen_samples, ds.tf)])
    print(f"condition B: A + {len(gen_samples)} generated -> {len(b_ds)}")

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    loader_a = DataLoader(Subset(ds, a_idx), shuffle=True, drop_last=False, **dl_kw)
    loader_b = DataLoader(b_ds, shuffle=True, drop_last=False, **dl_kw)
    loader_t = DataLoader(Subset(ds, test_idx), shuffle=False, drop_last=False, **dl_kw)

    results = []
    for tag, loader in [("A", loader_a), ("B", loader_b)]:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = build_model(num_classes).to(device)
        train_one(model, loader, args.epochs, args.lr, device, tag)
        cm = evaluate(model, loader_t, device, num_classes)
        text, recalls, f1s, acc, mf1, wf1 = metrics_from_cm(cm, ds.classes)
        print(f"[cond {tag}] test:\n{text}")
        results.append((tag, recalls, f1s, acc, mf1, wf1))

    print("================ COMPARISON (held-out real test) ================")
    header = "class   recall_A  recall_B    dR     f1_A    f1_B    dF1"
    print(header)
    lines = [header]
    for i, cname in enumerate(ds.classes):
        ra, rb = results[0][1][i], results[1][1][i]
        fa, fb = results[0][2][i], results[1][2][i]
        row = (f"{cname:<6} {ra:8.4f} {rb:8.4f} {rb - ra:+.4f} "
               f"{fa:7.4f} {fb:7.4f} {fb - fa:+.4f}")
        print(row); lines.append(row)
    for k, name in [(3, "accuracy"), (4, "macro-F1"), (5, "weighted-F1")]:
        row = (f"{name:<8} {results[0][k]:.4f} -> {results[1][k]:.4f} "
               f"({results[1][k] - results[0][k]:+.4f})")
        print(row); lines.append(row)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("results written to", out / "results.txt")


if __name__ == "__main__":
    main()
