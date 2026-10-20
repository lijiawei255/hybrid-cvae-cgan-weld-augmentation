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
- Default scoring uses the final epoch (``--selection final``), which is how
  every published table in this repository was produced. ``--selection
  best_val`` instead restores the epoch with the lowest loss on a real-only
  leftover validation pool (train images not in ``--subset``). A late loss
  spike can otherwise invalidate a single filling-rate arm under ``final``.
- Disclosure: when ``--selection best_val`` is used, that leftover validation
  pool is the same pool ``train_joint.py`` draws its generator validation / FID
  reference set from, so images used to select the generator checkpoint (early
  stopping, LR schedule) also sit inside classifier validation here. This
  follows from the deliberate one-train-pool design - the generator never trains
  on them either, and the held-out test set is untouched by everything.
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

from augment import (
    filling_rate_counts,
    generated_by_class,
    synthetic_prefix,
    verify_generation_meta,
)
from data import (
    ClassFolderDataset,
    ListDataset,
    assert_channels_match_data,
    count_by_class,
    make_splits,
    parse_name_counts,
    sample_named_subset,
)


def build_model(num_classes, channels=1):
    """ResNet-18 trained from scratch.

    From scratch rather than ImageNet-pretrained on purpose: the papers choose
    lightweight models over heavy pretrained ones for in-situ edge computing, and
    a pretrained backbone saturates on this task, which hides the augmentation
    effect being measured.
    """
    model = resnet18(weights=None)
    if channels != 3:
        model.conv1 = nn.Conv2d(
            channels,
            model.conv1.out_channels,
            model.conv1.kernel_size,
            model.conv1.stride,
            model.conv1.padding,
            bias=False,
        )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Mean cross-entropy on a loader; used to pick the best validation epoch."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        tot += loss_fn(model(x), y).item() * y.size(0)
        n += y.size(0)
    return tot / n if n else float("inf")


def train_one(
    model, loader, epochs, lr, device, tag, val_loader=None, selection="best_val"
):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
        if val_loader is not None and selection == "best_val":
            val_loss = evaluate_loss(model, val_loader, device)
            if val_loss < best["loss"]:
                best = {
                    "loss": val_loss,
                    "epoch": ep,
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                }
            if ep % 10 == 0 or ep == epochs:
                print(
                    f"[{tag}] epoch {ep}/{epochs} loss={tot / n:.4f} val={val_loss:.4f}"
                )
        elif ep % 10 == 0 or ep == epochs:
            print(f"[{tag}] epoch {ep}/{epochs} loss={tot / n:.4f}")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(
            f"[{tag}] restored best val-loss epoch {best['epoch']} ({best['loss']:.4f})"
        )


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
        f1s.append(f1)
        supports.append(support)
        per_class.append((cname, float(prec), float(rec), float(f1), support))
    total = int(cm.sum())
    summary = {
        "accuracy": float(np.trace(cm) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.sum(np.array(f1s) * np.array(supports)) / total)
        if total
        else 0.0,
    }
    return per_class, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument(
        "--gen_root",
        default=None,
        help="generate.py output tree holding the ratio-1.0 pool. Optional "
        "when every --ratio is 0.0, because a real-only baseline needs "
        "no synthetic images.",
    )
    ap.add_argument(
        "--subset",
        required=True,
        help="real training subset by class NAME, must match what the "
        "generator was trained on, e.g. 'pore=40,deposit=150,discontinuity=300,stain=600'. "
        "By default this must name every dataset class with a positive count.",
    )
    ap.add_argument(
        "--allow_partial_subset",
        action="store_true",
        help="allow --subset to omit a class or set it to 0. The selection "
        "pool is restricted to the classes actually trained on; the "
        "held-out test set still contains every class.",
    )
    ap.add_argument(
        "--ratios",
        default="0.0,0.25,0.5,0.75,1.0",
        help="comma-separated filling rates to sweep",
    )
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument(
        "--channels",
        type=int,
        default=3,
        choices=(1, 3),
        help="3 for RGB (LoHi-WELD default); 1 for grayscale. "
        "Refuses --channels 1 on colour files.",
    )
    ap.add_argument("--fft_denoise", action="store_true")
    ap.add_argument("--fft_cutoff", type=float, default=0.25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--selection",
        choices=("best_val", "final"),
        default="final",
        help="final (default since v0.6.0) scores the last epoch, which is how "
        "every published table in this repository was produced. best_val "
        "restores the lowest-loss epoch on a real-only leftover validation "
        "pool and is a different protocol.",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="request deterministic CuDNN algorithms; slower, and still "
        "not a guarantee of bit-identical GPU results",
    )
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument(
        "--split_by",
        choices=("crop", "source"),
        default="crop",
        help="must match the generator's --split_by, and the pool's; 'crop' is the "
        "published protocol. See train_joint.py --split_by.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--allow_missing_meta",
        action="store_true",
        help="sweep a generated pool that has no meta.json, so its split "
        "configuration cannot be checked against this run. Only for pools "
        "written before meta.json existed.",
    )
    ap.add_argument(
        "--strict_meta",
        action="store_true",
        help="treat a meta.json that predates split_by/fft_denoise/fft_cutoff as "
        "an error instead of a warning. New pools should always use this; old "
        "pools remain sweepable without it.",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default="runs/classifier_sweep")
    args = ap.parse_args()

    if not Path(args.data_root).is_dir():
        raise SystemExit(f"--data_root is not a directory: {args.data_root}")
    if args.gen_root is not None and not Path(args.gen_root).is_dir():
        raise SystemExit(f"--gen_root is not a directory: {args.gen_root}")
    if not 0.0 < args.test_frac < 1.0:
        raise SystemExit(f"--test_frac must be in (0, 1), got {args.test_frac}")
    if args.img_size < 32:
        raise SystemExit(
            f"--img_size must be >= 32 for ResNet-18's five stride-2 stages, "
            f"got {args.img_size}"
        )
    if args.batch_size < 2:
        raise SystemExit(
            f"--batch_size must be >= 2 (BatchNorm needs more than one sample), "
            f"got {args.batch_size}"
        )
    if args.epochs < 1:
        raise SystemExit(f"--epochs must be >= 1, got {args.epochs}")
    if args.num_workers < 0:
        raise SystemExit(f"--num_workers must be >= 0, got {args.num_workers}")
    if args.lr <= 0:
        raise SystemExit(f"--lr must be positive, got {args.lr}")
    if not 0.0 < args.fft_cutoff <= 1.0:
        raise SystemExit(f"--fft_cutoff must be in (0, 1], got {args.fft_cutoff}")

    ratios = [float(r) for r in args.ratios.split(",") if r.strip()]
    if not ratios:
        raise SystemExit("--ratios is empty; give at least one filling rate")
    for ratio in ratios:
        if not 0.0 <= ratio <= 1.0:
            raise SystemExit(f"filling rates must be in [0, 1], got {ratio}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    assert_channels_match_data(args.data_root, args.channels)

    ds = ClassFolderDataset(
        args.data_root, args.img_size, args.channels, args.fft_denoise, args.fft_cutoff
    )
    classes, num_classes = ds.classes, len(ds.classes)

    # Same split and same seed as train_joint.py, so the generator never saw the
    # test images and condition r=0.0 is exactly the generator's training set.
    train_pool, test_idx = make_splits(
        ds.samples, args.test_frac, args.seed, args.split_by
    )
    subset_counts = parse_name_counts(args.subset)
    unknown = sorted(set(subset_counts) - set(classes))
    if unknown:
        raise SystemExit(
            f"--subset names unknown class(es) {unknown}; dataset classes are {classes}"
        )
    missing = [c for c in classes if c not in subset_counts]
    zero = [c for c in classes if c in subset_counts and subset_counts[c] == 0]
    if missing or zero:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if zero:
            detail.append(f"zero {zero}")
        message = (
            f"--subset does not give every dataset class a positive count "
            f"({'; '.join(detail)})."
        )
        if not args.allow_partial_subset:
            raise SystemExit(
                message
                + " Pass --allow_partial_subset to train on the selected classes only."
            )
        print(
            f"warning: {message}; selection and synthetic filling are restricted to "
            f"{sorted(set(subset_counts) & set(classes))}"
        )
    active_names = [c for c in classes if subset_counts.get(c, 0) > 0]
    if not active_names:
        raise SystemExit("--subset selects no images; nothing to train on")
    active_label_set = {classes.index(c) for c in active_names}

    subset_idx = sample_named_subset(ds, subset_counts, train_pool, args.seed)
    all_counts = count_by_class(ds, subset_idx)
    real_counts = {name: all_counts[name] for name in active_names}
    used = set(subset_idx)
    val_idx = [
        i for i in train_pool if i not in used and ds.samples[i][1] in active_label_set
    ]
    selection = args.selection
    if selection == "best_val" and len(val_idx) < max(2, args.batch_size):
        print(
            f"note: leftover real pool has {len(val_idx)} images, too few for "
            f"validation selection; scoring the final epoch"
        )
        selection = "final"
    print(f"classes ({num_classes}): {classes}")
    print(f"real training subset: {real_counts} -> {len(subset_idx)}")
    print(f"held-out real test:   {count_by_class(ds, test_idx)} -> {len(test_idx)}")
    print(
        f"selection={selection} leftover val: {count_by_class(ds, val_idx)} -> {len(val_idx)}"
    )
    print(f"split_by={args.split_by}")

    n_max = max(real_counts.values())
    max_ratio_counts = filling_rate_counts(real_counts, max(ratios))
    needs_synthetic = any(max_ratio_counts.values())
    pool = {}
    if needs_synthetic:
        if args.gen_root is None:
            raise SystemExit(
                "--gen_root is required because at least one --ratio requests "
                "synthetic images; only a real-only sweep (all ratios 0.0) can "
                "omit it"
            )
        pool = generated_by_class(args.gen_root, classes)
        verify_generation_meta(
            args.gen_root,
            args.subset,
            args.seed,
            args.test_frac,
            img_size=args.img_size,
            channels=args.channels,
            split_by=args.split_by,
            fft_denoise=args.fft_denoise,
            fft_cutoff=args.fft_cutoff if args.fft_denoise else None,
            allow_missing_meta=args.allow_missing_meta,
            strict_meta=args.strict_meta,
        )
        print("generated pool available:", {k: len(v) for k, v in pool.items()})
    else:
        print("no synthetic images requested; --gen_root is not required")
    print(
        f"N_max (majority class count) = {n_max}; balance-to-max fills every class to it"
    )

    dl_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    loader_test = DataLoader(
        Subset(ds, test_idx), shuffle=False, drop_last=False, **dl_kw
    )
    loader_val = (
        DataLoader(Subset(ds, val_idx), shuffle=False, drop_last=False, **dl_kw)
        if selection == "best_val"
        else None
    )

    rows = []
    for ratio in ratios:
        tag = f"r={ratio:g}"
        need = filling_rate_counts(real_counts, ratio)
        synthetic = []
        for label, name in enumerate(classes):
            count = need.get(name, 0)
            if not count:
                continue
            picked = synthetic_prefix(pool[name], count)
            synthetic.extend((p, label) for p in picked)
        train_ds = (
            Subset(ds, subset_idx)
            if not synthetic
            else ConcatDataset(
                [Subset(ds, subset_idx), ListDataset(synthetic, ds.tf, args.channels)]
            )
        )
        total_counts = {n: real_counts.get(n, 0) + need.get(n, 0) for n in classes}
        print(
            f"\n=== {tag} === synthetic +{len(synthetic)} -> {len(train_ds)} total; "
            f"per class {total_counts}"
        )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = build_model(num_classes, args.channels).to(device)
        if len(train_ds) <= args.batch_size:
            raise SystemExit(
                f"{tag}: only {len(train_ds)} training images for batch size "
                f"{args.batch_size}; lower --batch_size"
            )
        # drop_last: a final batch of one image makes BatchNorm raise in train mode.
        loader_train = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
        train_one(
            model,
            loader_train,
            args.epochs,
            args.lr,
            device,
            tag,
            val_loader=loader_val,
            selection=selection,
        )

        cm = evaluate(model, loader_test, device, num_classes)
        np.save(out / f"cm_r{ratio:g}.npy", cm)
        per_class, summary = metrics_from_cm(cm, classes)
        for cname, prec, rec, f1, support in per_class:
            print(
                f"  {cname:<6} P={prec:.4f} R={rec:.4f} F1={f1:.4f} support={support}"
            )
            rows.append((ratio, f"precision:{cname}", prec))
            rows.append((ratio, f"recall:{cname}", rec))
            rows.append((ratio, f"f1:{cname}", f1))
            rows.append((ratio, f"support:{cname}", float(support)))
        print(
            f"  accuracy={summary['accuracy']:.4f} macroF1={summary['macro_f1']:.4f} "
            f"weightedF1={summary['weighted_f1']:.4f}"
        )
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
        f"{c + ':F1':>9}" for c in classes
    )
    print(header)
    by_ratio = {}
    for ratio, metric, value in rows:
        by_ratio.setdefault(ratio, {})[metric] = value
    lines = [header]
    for ratio in ratios:
        m = by_ratio[ratio]
        row = (
            f"{ratio:>6g} {m['accuracy']:>7.4f} {m['macro_f1']:>8.4f} "
            f"{m['weighted_f1']:>7.4f} "
            + " ".join(f"{m[f'f1:{c}']:>9.4f}" for c in classes)
        )
        print(row)
        lines.append(row)

    with open(out / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nresults -> {out / 'results.txt'}; metrics -> {out / 'sweep_metrics.csv'}")


if __name__ == "__main__":
    main()
