# -*- coding: utf-8 -*-
"""FID evaluation: distributional distance between generated and real images
(lower is better).

Default backend is **pytorch-fid** (mseitzer/pytorch-fid, Apache-2.0): the
community PyTorch port of the official TensorFlow Inception weights. That is
an evaluation-infrastructure choice, not a change to the paper method. This
module still owns the weld-specific protocol around that library:

* positional ``class_<i>`` folders written by ``src/generate.py`` are mapped
  to real class names through that tree's ``classes.txt`` manifest before any
  folder-name comparison (the same contract ``augment.py`` uses);
* class-balanced sampling over classes present in both trees, so a
  balance-to-max pool that skips a majority class does not fold a class-ratio
  difference into the score;
* optional FFT denoising that must match how the generator was trained;
* grayscale expansion to three replicated channels.

``--fid_backend legacy`` keeps the older in-repo torchvision InceptionV3 path
so published numbers in this repository remain reproducible. The two backends
are **not** on the same scale; do not mix them, and do not compare either to
the papers' FID.

Two details that are easy to get wrong on the legacy path:

* InceptionV3 with ``transform_input=False`` expects input in **[-1, 1]**.
* InceptionV3 is RGB-only.
"""
import argparse
import random
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import LinAlgWarning, sqrtm
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3

from augment import CLASS_MANIFEST
from data import ListDataset, assert_channels_match_data, build_loader

FID_BACKENDS = ("pytorch_fid", "legacy")
DEFAULT_FID_BACKEND = "pytorch_fid"


def parse_fid_backend(name):
    """Accept only the two documented backends; unknown names raise."""
    if name not in FID_BACKENDS:
        raise ValueError(
            f"fid backend must be one of {FID_BACKENDS}, got {name!r}")
    return name


def inception_input(x):
    """Legacy path: pipeline images ([0, 1], possibly grayscale) -> torchvision Inception.

    Grayscale is replicated across three channels rather than zero-padded, and
    the range is rescaled to [-1, 1] because ``transform_input=False`` means the
    model will not normalise for us. pytorch-fid does this rescale internally
    and wants [0, 1] RGB instead.
    """
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    if x.shape[-1] != 299:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


def _as_rgb(x):
    """[0, 1] NCHW, 1 or 3 channels -> 3-channel [0, 1] for pytorch-fid."""
    if x.shape[1] == 1:
        return x.expand(-1, 3, -1, -1)
    return x


class FeatureExtractor:
    """Cached Inception feature extractor for one FID backend.

    Built once per run and reused for the real reference and every generated
    set, including the per-epoch training-loop measurement.
    """

    def __init__(self, backend, device):
        self.backend = parse_fid_backend(backend)
        self.device = device
        if self.backend == "legacy":
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT,
                                 transform_input=False).to(device)
            model.fc = torch.nn.Identity()
            self.model = model.eval()
            return
        from pytorch_fid.inception import InceptionV3
        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block]).to(device).eval()

    @torch.no_grad()
    def features(self, loader):
        """2048-d pooled Inception features for every image a loader yields."""
        feats = []
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(self.device)
            if self.backend == "legacy":
                feats.append(self.model(inception_input(x)).cpu().numpy())
                continue
            pred = self.model(_as_rgb(x))[0]
            if pred.dim() == 4:
                pred = F.adaptive_avg_pool2d(pred, (1, 1))
            feats.append(pred.flatten(1).cpu().numpy())
        return np.concatenate(feats)


def _stats(feats):
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def fid_from_stats(mu1, s1, mu2, s2, eps=1e-6):
    """Frechet distance between two Gaussians N(mu, sigma). Used by --fid_backend legacy.

    A rank-deficient feature covariance is *expected* whenever there are fewer
    images than InceptionV3's 2048 feature dimensions, which is the normal case
    in the small-dataset regime this method targets. ``sqrtm`` still returns a
    usable result there, so its LinAlgWarning is suppressed deliberately; if the
    result comes back non-finite both covariances are regularised by a small
    diagonal and recomputed.

    The result is clamped at zero because FID is a distance. Without this,
    identical distributions return small negative values (about -1e-14, and
    materially larger on rank-deficient inputs) purely from sqrtm's
    floating-point error.
    """
    diff = mu1 - mu2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        covmean = sqrtm(s1.dot(s2))
        if not np.isfinite(covmean).all():
            offset = np.eye(s1.shape[0]) * eps
            covmean = sqrtm((s1 + offset).dot(s2 + offset))
    if not np.isfinite(covmean).all():
        raise np.linalg.LinAlgError(
            "FID covariance square root is not finite even after regularisation; "
            "the feature statistics are degenerate (too few or too similar images)")
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = float(diff.dot(diff) + np.trace(s1 + s2 - 2 * covmean))
    return max(value, 0.0)


def frechet_distance(mu1, s1, mu2, s2, backend=DEFAULT_FID_BACKEND):
    """Frechet distance using the selected backend's numerics."""
    backend = parse_fid_backend(backend)
    if backend == "legacy":
        return fid_from_stats(mu1, s1, mu2, s2)
    from pytorch_fid.fid_score import calculate_frechet_distance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def balanced_subset(ds, per_class, class_names, seed):
    """Class-balanced (path, label) subset of a dataset, restricted to class_names.

    Balancing matters: if one side of the comparison keeps classes the other
    lacks, the measured FID absorbs the class-ratio difference instead of
    measuring image quality alone. Matching by class *name* (not integer
    label) keeps the two trees independent of each other's folder ordering.
    """
    by_class = {name: [] for name in class_names}
    for sample in ds.samples:
        name = ds.classes[sample[1]]
        if name in by_class:
            by_class[name].append(sample)
    rng = random.Random(seed)
    selected = []
    for name in class_names:
        if len(by_class[name]) < per_class:
            raise ValueError(
                f"class '{name}' holds {len(by_class[name])} images, fewer than "
                f"the {per_class} needed for a class-balanced FID comparison")
        selected.extend(rng.sample(by_class[name], per_class))
    return selected


def balanced_labels(num_classes, per_class, device):
    """Labels for the generated FID set, balanced to match the real reference."""
    return torch.arange(num_classes, device=device).repeat_interleave(per_class)


def remap_generated_classes(ds, root, real_classes):
    """Map a generate.py tree's positional class_<i> folders to real class names.

    ``src/generate.py`` writes ``class_0/``, ``class_1/``, ... plus a
    ``classes.txt`` manifest recording which real class each index stands for
    (``augment.generated_by_class`` consumes the same contract). Without this
    mapping the generated folder names never match the real tree's names and
    the class-balanced comparison finds no shared class. Trees that already
    use real class names are left untouched.
    """
    positional = [c for c in ds.classes if re.fullmatch(r"class_\d+", c)]
    if not positional:
        return
    manifest = Path(root) / CLASS_MANIFEST
    if not manifest.is_file():
        raise SystemExit(
            f"{manifest} is missing, so the positional class_i folders in {root} "
            f"cannot be mapped to class names. Regenerate the pool with "
            f"src/generate.py, or point --fake_root at a tree with real class "
            f"names.")
    names = [line.strip() for line in
             manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping = {}
    for folder in positional:
        i = int(folder.split("_", 1)[1])
        if i >= len(names):
            raise SystemExit(
                f"folder {folder} in {root} exceeds the {len(names)} entries "
                f"recorded in {CLASS_MANIFEST}; the pool and its manifest "
                f"disagree. Regenerate the pool with src/generate.py.")
        mapping[folder] = names[i]
    unknown = sorted(set(mapping.values()) - set(real_classes))
    if unknown:
        raise SystemExit(
            f"{root} was generated for classes {names} but the real tree has "
            f"{sorted(real_classes)}; unknown class(es) {unknown}. Regenerate "
            f"the pool against this dataset or fix --real_root.")
    name_of = {i: mapping.get(c, c) for i, c in enumerate(ds.classes)}
    ds.classes = sorted(set(name_of.values()))
    index = {name: i for i, name in enumerate(ds.classes)}
    ds.samples = [(path, index[name_of[label]]) for path, label in ds.samples]
    print(f"note: mapped positional class folders in {root} to real class names "
          f"via {CLASS_MANIFEST} ({', '.join(mapping[c] for c in positional)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", required=True, help="Real images (class subfolders)")
    ap.add_argument("--fake_root", required=True, help="Generated images (same layout)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--fft_denoise", action="store_true",
                    help="must match how the generator was trained, or FID measures a "
                         "preprocessing mismatch instead of generation quality")
    ap.add_argument("--fft_cutoff", type=float, default=0.25)
    ap.add_argument("--no_balance", action="store_true",
                    help="compare both trees as-is with their natural class mixes; by "
                         "default both sides are sampled class-balanced over the "
                         "classes present in both trees, so FID measures generation "
                         "quality rather than the class-ratio difference")
    ap.add_argument("--balanced_per_class", type=int, default=0,
                    help="images per class on BOTH sides for the balanced comparison "
                         "(default: the smallest count over the shared classes)")
    ap.add_argument("--fid_backend", choices=FID_BACKENDS, default=DEFAULT_FID_BACKEND,
                    help="pytorch_fid (default): mseitzer/pytorch-fid with official "
                         "TensorFlow Inception weights. legacy: the torchvision path "
                         "used by published in-repo numbers. The two scales differ.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert_channels_match_data(args.real_root, args.channels)
    assert_channels_match_data(args.fake_root, args.channels)
    load_kw = dict(num_workers=args.num_workers, channels=args.channels,
                   fft_denoise=args.fft_denoise, fft_cutoff=args.fft_cutoff,
                   shuffle=False, drop_last=False)
    ds_real, real_loader = build_loader(args.real_root, args.img_size, args.batch_size,
                                        **load_kw)
    ds_fake, fake_loader = build_loader(args.fake_root, args.img_size, args.batch_size, **load_kw)
    remap_generated_classes(ds_fake, args.fake_root, ds_real.classes)
    if not args.no_balance:
        def counts_by_name(ds):
            counts = {}
            for _, label in ds.samples:
                name = ds.classes[label]
                counts[name] = counts.get(name, 0) + 1
            return counts

        real_counts, fake_counts = counts_by_name(ds_real), counts_by_name(ds_fake)
        shared = sorted(set(real_counts) & set(fake_counts))
        for name in sorted(set(real_counts) ^ set(fake_counts)):
            print(f"note: class '{name}' is missing on one side "
                  f"(real {real_counts.get(name, 0)}, fake {fake_counts.get(name, 0)} "
                  f"images) - excluded from the balanced comparison")
        if not shared:
            raise SystemExit("no class is present in both trees; nothing to compare")
        n_per = args.balanced_per_class or min(min(real_counts[n], fake_counts[n])
                                               for n in shared)
        real_loader = DataLoader(
            ListDataset(balanced_subset(ds_real, n_per, shared, seed=0),
                        ds_real.tf, ds_real.channels),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        fake_loader = DataLoader(
            ListDataset(balanced_subset(ds_fake, n_per, shared, seed=0),
                        ds_fake.tf, ds_fake.channels),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"FID comparison: {n_per} per class x {len(shared)} shared classes on "
              f"both sides (class-balanced; pass --no_balance to compare both trees "
              f"as-is)")
    extractor = FeatureExtractor(args.fid_backend, device)
    mu_r, s_r = _stats(extractor.features(real_loader))
    mu_f, s_f = _stats(extractor.features(fake_loader))
    print(f"FID ({args.fid_backend}) =", frechet_distance(mu_r, s_r, mu_f, s_f, args.fid_backend))


if __name__ == "__main__":
    main()
