# -*- coding: utf-8 -*-
"""Paper-style figures, drawn from scratch from this repo's own artifacts.

Nothing here copies a figure from either reference paper; these are original
plots of our own measurements, in the same spirit as the paper's figures so the
results are recognisable to someone who has read it.

Data sources, all produced by this repo:
    runs/joint/history.csv            training + FID curves
    runs/classifier_sweep/sweep_metrics.csv   filling-rate sensitivity
    runs/classifier_sweep/cm_r*.npy   confusion matrices
    runs/joint/joint.pt               real / reconstruction / generation, latent t-SNE

Every plotting function is pure - it takes plain arrays and writes a file - so
each one is unit-testable without training a model. Only the CLI touches
checkpoints and the GPU.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from augment import filling_rate_counts
from data import ClassFolderDataset, count_by_class, make_splits, parse_name_counts, sample_named_subset
from models import Decoder, Encoder, reparameterize


# --------------------------------------------------------------------------- #
# artifact readers
# --------------------------------------------------------------------------- #
def read_history_csv(path):
    """history.csv -> {column: [values]}; 'epoch' as ints, everything else floats."""
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    out = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            out[key].append(int(float(value)) if key == "epoch" else float(value))
    return out


def read_sweep_csv(path):
    """sweep_metrics.csv -> {ratio: {metric: value}}."""
    out = {}
    for row in csv.DictReader(Path(path).open(encoding="utf-8")):
        out.setdefault(float(row["ratio"]), {})[row["metric"]] = float(row["value"])
    if not out:
        raise ValueError(f"{path} has no data rows")
    return out


# --------------------------------------------------------------------------- #
# dataset / class distribution
# --------------------------------------------------------------------------- #
def build_distribution_series(class_names, real_counts, synthetic_counts):
    """Per-class real counts, and the same plus the balance-to-max synthetic fill."""
    return {
        "real (imbalanced)": [real_counts[name] for name in class_names],
        "real + synthetic (balance-to-max)":
            [real_counts[name] + synthetic_counts[name] for name in class_names],
    }


def save_class_distribution(class_names, series, output):
    x = np.arange(len(class_names))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.set_xticks(x, class_names)
    ax.set_ylabel("images")
    ax.set_title("Class distribution: original imbalance vs balance-to-max augmentation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, output)


# --------------------------------------------------------------------------- #
# training curves
# --------------------------------------------------------------------------- #
def save_training_curves(history, output):
    """Six panels of training history: the loss components and FID per epoch."""
    panels = [("recon", "reconstruction (MSE)", False),
              ("kl", "KL divergence", False),
              ("perc", "VGG19 perceptual", False),
              ("loss_g", "generator loss (total)", False),
              ("loss_d", "discriminator loss", False),
              ("fid", "FID vs real validation set", True)]
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (key, label, lower_better) in zip(axes.ravel(), panels):
        values = np.array(history[key], dtype=float)
        finite = np.isfinite(values)
        ax.plot(np.array(epochs)[finite], values[finite], marker="o", ms=2.5, lw=1.2)
        if key == "recon" and "val_recon" in history:
            val = np.array(history["val_recon"], dtype=float)
            ax.plot(np.array(epochs)[finite], val[finite], lw=1.2, ls="--",
                    color="tab:orange", label="validation")
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{label}{' (lower is better)' if lower_better else ''}", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Joint CVAE-CGAN training history", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, output)


# --------------------------------------------------------------------------- #
# filling-rate sensitivity
# --------------------------------------------------------------------------- #
def save_filling_rate_curve(sweep, class_names, output):
    """The key result: how downstream performance depends on the filling rate.

    Left panel is the overall metrics, right panel is per-class F1, so the
    minority classes that augmentation is supposed to rescue are visible on their
    own rather than averaged away.
    """
    ratios = sorted(sweep)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for metric, style in [("accuracy", "-o"), ("macro_f1", "-s"), ("weighted_f1", "-^")]:
        axes[0].plot(ratios, [sweep[r][metric] for r in ratios], style, ms=4, label=metric)
    axes[0].set_xlabel("filling rate (fraction of the gap to $N_{max}$ closed)")
    axes[0].set_ylabel("score")
    axes[0].set_title("Overall metrics on the held-out real test set")
    axes[0].set_xticks(ratios)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    for name in class_names:
        axes[1].plot(ratios, [sweep[r][f"f1:{name}"] for r in ratios], "-o", ms=4, label=name)
    axes[1].set_xlabel("filling rate")
    axes[1].set_ylabel("F1")
    axes[1].set_title("Per-class F1")
    axes[1].set_xticks(ratios)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, output)


def aggregate_seed_metrics(seeds, metric):
    """[{ratio: {metric: value}}, ...] -> {ratio: (mean, sample std, n seeds)}.

    The spread is the sample standard deviation (ddof=1), the same convention as
    the table in docs/CALIBRATION.md section 9, so the plotted band and the
    tabulated +/- values cannot disagree. A ratio with one seed has no sample
    std and reports 0.0 rather than NaN.

    A ratio reported by only some seeds still aggregates over the seeds that have
    it: an arm excluded for a training fault must not delete its ratio from the
    figure, or the curve would silently change shape.
    """
    per_ratio = {}
    for sweep in seeds:
        for ratio, metrics in sweep.items():
            if metric in metrics:
                per_ratio.setdefault(float(ratio), []).append(float(metrics[metric]))
    if not per_ratio:
        raise ValueError(f"no seed reports metric {metric!r}")
    return {ratio: (float(np.mean(values)),
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    len(values))
            for ratio, values in per_ratio.items()}


def save_multi_seed_filling_rate(seeds, references, minority, output):
    """Seed-aggregated filling-rate curve against labelled reference arms.

    The single-seed curve cannot distinguish a real effect from seed noise, so the
    mean carries a +/- 1 std band and any ratio with fewer seeds is marked.
    """
    ref_colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown"]
    panels = [("macro_f1", "macro-F1"),
              (f"f1:{minority}", f"{minority} F1 (minority class)")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for ax, (metric, label) in zip(axes, panels):
        agg = aggregate_seed_metrics(seeds, metric)
        ratios = sorted(agg)
        mean = np.array([agg[r][0] for r in ratios])
        std = np.array([agg[r][1] for r in ratios])
        counts = [agg[r][2] for r in ratios]
        full = max(counts)

        ax.plot(ratios, mean, "-o", ms=4, color="tab:blue",
                label=f"GroupNorm + balanced sampler, mean of {full} seeds")
        ax.fill_between(ratios, mean - std, mean + std, color="tab:blue", alpha=0.18,
                        label="$\\pm$1 sample std over seeds")
        for ratio, value, count in zip(ratios, mean, counts):
            if count < full:
                ax.annotate(f"n={count}", (ratio, value), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=7, color="tab:red")

        for color, (name, sweep) in zip(ref_colors, references.items()):
            ref_ratios = sorted(r for r in sweep if metric in sweep[r])
            ax.plot(ref_ratios, [sweep[r][metric] for r in ref_ratios], "--s", ms=3.5,
                    lw=1.1, color=color, label=name)

        ax.set_xlabel("filling rate (fraction of the gap to $N_{max}$ closed)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(ratios)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7.5)

    fig.suptitle("Filling-rate sensitivity across seeds, against single-seed reference arms",
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, output)


# --------------------------------------------------------------------------- #
# confusion matrices
# --------------------------------------------------------------------------- #
def normalize_confusion_matrix(matrix):
    """Row-normalise so each cell reads as recall for that true class."""
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)


def save_confusion_matrices(class_names, condition_a, condition_b, output,
                            labels=("r = 0.0: real only", "r = 1.0: balance-to-max")):
    """Side-by-side row-normalised confusion matrices for two conditions."""
    matrices = [normalize_confusion_matrix(condition_a), normalize_confusion_matrix(condition_b)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    image = None
    for axis, matrix, title in zip(axes, matrices, labels):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("predicted class")
        axis.set_title(title)
        for row, column in np.ndindex(matrix.shape):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                      color="white" if matrix[row, column] > 0.5 else "black")
    axes[0].set_ylabel("true class")
    fig.suptitle("Held-out real-test confusion matrices")
    # Reserve the right-hand gutter BEFORE creating the colorbar: creating it first
    # and adjusting afterwards leaves the colorbar positioned over the right matrix.
    # tight_layout is avoided because a shared colorbar makes it emit a UserWarning.
    fig.subplots_adjust(top=0.84, bottom=0.22, wspace=0.30, right=0.80)
    fig.colorbar(image, ax=axes.tolist(), pad=0.02, label="row-normalized recall")
    _save(fig, output)


# --------------------------------------------------------------------------- #
# image and latent figures
# --------------------------------------------------------------------------- #
def _imshow(axis, image):
    """Display one CHW image in [0, 1]; grayscale is squeezed to 2-D."""
    if image.ndim == 3 and image.shape[0] == 1:
        axis.imshow(image[0], cmap="gray", vmin=0, vmax=1)
    elif image.ndim == 3 and image.shape[0] == 3:
        axis.imshow(np.transpose(image, (1, 2, 0)))
    else:
        raise ValueError(f"expected a CHW image with 1 or 3 channels, got {image.shape}")
    axis.axis("off")


def save_reconstruction_comparison(real, reconstructed, generated, output,
                                   class_names=None, n_per_class=None):
    """Three rows - real, CVAE reconstruction, generation from z ~ N(0, I).

    Columns are expected to be grouped by class. When `class_names` and
    `n_per_class` are given, each group is labelled above the top row; without
    that annotation a reader cannot tell which column belongs to which class.
    """
    rows = [real, reconstructed, generated]
    labels = ["real", "CVAE reconstruction", "generated from z ~ N(0,I)"]
    n = min(len(row) for row in rows)
    fig, axes = plt.subplots(3, n, figsize=(max(1.5 * n, 4), 5.2), squeeze=False)
    for row_index, (images, label) in enumerate(zip(rows, labels)):
        display = images[:n].detach().cpu().clamp(0, 1).float()
        for column in range(n):
            _imshow(axes[row_index, column], display[column].numpy())
        axes[row_index, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
        axes[row_index, 0].tick_params(length=0)
    if class_names and n_per_class:
        for position, name in enumerate(class_names):
            column = position * n_per_class
            if column < n:
                axes[0, column].set_title(name, fontsize=9)
    fig.suptitle("Real images, CVAE reconstructions, and hybrid-CGAN samples")
    fig.tight_layout(rect=(0.06, 0, 1, 0.94))
    _save(fig, output)


def save_latent_projection(points, labels, class_names, output):
    """2-D projection of encoder latent vectors, coloured by class."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, class_name in enumerate(class_names):
        mask = labels == label
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=14, alpha=0.75, label=class_name)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("CVAE encoder latent-space projection")
    ax.legend(frameon=False, markerscale=1.4)
    fig.tight_layout()
    _save(fig, output)


def encoder_latents(enc, loader, device, limit=None):
    """Posterior means and labels over a loader, for latent-space visualisation."""
    enc.eval()
    mus, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            mu, _ = enc(x.to(device), y.to(device))
            mus.append(mu.cpu()); ys.append(y)
            if limit and sum(m.size(0) for m in mus) >= limit:
                break
    enc.train()
    points = torch.cat(mus).numpy()
    return points[:limit], torch.cat(ys).numpy()[:limit]


def reconstruction_triplet(enc, dec, x, y, latent_dim, device):
    """(real, reconstruction, generation from noise) for one batch."""
    dec.eval(); enc.eval()
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        mu, logvar = enc(x, y)
        recon = dec(reparameterize(mu, logvar), y)
        gen = dec(torch.randn(x.size(0), latent_dim, device=device), y)
    dec.train(); enc.train()
    return x.cpu(), recon.cpu(), gen.cpu()


def _save(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True, help="joint.pt from train_joint.py")
    ap.add_argument("--history", required=True, help="history.csv from train_joint.py")
    ap.add_argument("--sweep", required=True, help="sweep_metrics.csv from train_classifier.py")
    ap.add_argument("--cm_dir", required=True, help="directory holding cm_r*.npy")
    ap.add_argument("--subset", required=True,
                    help="the real subset the generator was trained on, e.g. 'pore=40,deposit=150,discontinuity=300,stain=600'")
    ap.add_argument("--cm_ratios", default="0.0,1.0",
                    help="which two filling rates to compare in the confusion-matrix figure")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--fft_denoise", action="store_true")
    ap.add_argument("--fft_cutoff", type=float, default=0.25)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_show", type=int, default=4,
                    help="images per class in the comparison figure (columns = classes x this)")
    ap.add_argument("--tsne_samples", type=int, default=800)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ClassFolderDataset(args.data_root, args.img_size, args.channels,
                            args.fft_denoise, args.fft_cutoff)
    classes = ds.classes
    train_pool, _ = make_splits(ds.samples, args.test_frac, args.seed)
    subset_idx = sample_named_subset(ds, parse_name_counts(args.subset), train_pool, args.seed)
    real_counts = count_by_class(ds, subset_idx)
    synthetic_counts = filling_rate_counts(real_counts, 1.0)

    # ---- class distribution (no model needed) ----
    save_class_distribution(classes,
                            build_distribution_series(classes, real_counts, synthetic_counts),
                            out / "class_distribution.png")
    print("wrote", out / "class_distribution.png")

    # ---- training curves ----
    save_training_curves(read_history_csv(args.history), out / "training_curves.png")
    print("wrote", out / "training_curves.png")

    # ---- filling-rate sensitivity ----
    sweep = read_sweep_csv(args.sweep)
    save_filling_rate_curve(sweep, classes, out / "filling_rate_curve.png")
    print("wrote", out / "filling_rate_curve.png")

    # ---- confusion matrices ----
    cm_ratios = [float(r) for r in args.cm_ratios.split(",")]
    if len(cm_ratios) != 2:
        raise SystemExit("--cm_ratios takes exactly two ratios, e.g. '0.0,1.0'")
    matrices = [np.load(Path(args.cm_dir) / f"cm_r{r:g}.npy") for r in cm_ratios]

    def ratio_label(r):
        if r == 0.0:
            return f"r = {r:g}: real only"
        if r == 1.0:
            return f"r = {r:g}: balance-to-max"
        return f"r = {r:g}: partial fill"

    save_confusion_matrices(classes, matrices[0], matrices[1],
                            out / "confusion_matrices.png",
                            labels=(ratio_label(cm_ratios[0]), ratio_label(cm_ratios[1])))
    print("wrote", out / "confusion_matrices.png")

    # ---- model-dependent figures ----
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    ckpt_classes = list(ckpt.get("classes") or [])
    if ckpt_classes != classes:
        raise SystemExit(
            f"checkpoint {args.ckpt} was trained on classes {ckpt_classes} but "
            f"{args.data_root} has {classes}; the figures would be labelled with "
            f"the wrong class names")
    for flag, value in (("--subset", args.subset), ("--seed", args.seed),
                        ("--test_frac", args.test_frac)):
        recorded = ckpt.get(flag.lstrip("-"))
        if recorded is not None and recorded != value:
            raise SystemExit(
                f"{flag} {value!r} but checkpoint {args.ckpt} was trained with "
                f"{recorded!r}; the figures would mix two different splits")
    enc = Encoder(ckpt["img_channels"], ckpt["num_classes"], ckpt["latent_dim"],
                  ckpt["base_ch"], ckpt["img_size"]).to(device)
    dec = Decoder(ckpt["img_channels"], ckpt["num_classes"], ckpt["latent_dim"],
                  ckpt["base_ch"], ckpt["img_size"]).to(device)
    enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"])
    print(f"loaded checkpoint from best epoch {ckpt.get('best_epoch')}")

    per_class_idx = []
    for label, name in enumerate(classes):
        available = [i for i in subset_idx if ds.samples[i][1] == label]
        if len(available) < args.n_show:
            raise SystemExit(
                f"class '{name}' has only {len(available)} images in the subset but "
                f"--n_show is {args.n_show}; the column labels would be misaligned")
        per_class_idx.extend(available[:args.n_show])
    show_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, per_class_idx), batch_size=len(per_class_idx),
        shuffle=False)
    x, y = next(iter(show_loader))
    real, recon, gen = reconstruction_triplet(enc, dec, x, y, ckpt["latent_dim"], device)
    save_reconstruction_comparison(real, recon, gen, out / "reconstruction_comparison.png",
                                   class_names=classes, n_per_class=args.n_show)
    print("wrote", out / "reconstruction_comparison.png")

    latent_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, subset_idx), batch_size=64, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    points, labels = encoder_latents(enc, latent_loader, device, limit=args.tsne_samples)
    from sklearn.manifold import TSNE
    projected = TSNE(n_components=2, init="pca", random_state=args.seed,
                     perplexity=min(30, max(5, len(points) // 4))).fit_transform(points)
    save_latent_projection(projected, labels, classes, out / "latent_tsne.png")
    print("wrote", out / "latent_tsne.png")


if __name__ == "__main__":
    main()
