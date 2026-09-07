# -*- coding: utf-8 -*-
"""FID evaluation: distributional distance between generated and real images
(lower is better).

Note: FID relies on InceptionV3 features and is noisy on small sample counts.
Use at least ~2000 real and ~2000 generated images for meaningful absolute
values; on smaller sets it should only be read as a training trend.

Two details that are easy to get wrong and are handled here:

* InceptionV3 with ``transform_input=False`` expects input in **[-1, 1]**. This
  pipeline stores images in [0, 1], so the rescale happens in ``inception_input``
  and nowhere else. Feeding [0, 1] directly silently shifts every activation
  off-distribution and produces FID values that are not comparable to anything.
* InceptionV3 is RGB-only, so ``inception_input`` expands single-channel
  grayscale to three replicated channels before the forward pass.
* By default both sides of the comparison are class-balanced over the classes
  present in both trees. A generated pool that intentionally skips a class (a
  balance-to-max pool contains no majority-class images) would otherwise skew
  the score by a class-ratio difference instead of an image-quality one.
"""
import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import LinAlgWarning, sqrtm
from torch.utils.data import DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3

from data import ListDataset, build_loader


def inception_input(x):
    """Pipeline images ([0, 1], possibly grayscale) -> what InceptionV3 wants.

    Grayscale is replicated across three channels rather than zero-padded, and
    the range is rescaled to [-1, 1] because ``transform_input=False`` means the
    model will not normalise for us.
    """
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    if x.shape[-1] != 299:
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    return x * 2.0 - 1.0


@torch.no_grad()
def _features(loader, device, model=None):
    """2048-d pooled InceptionV3 features for every image a loader yields."""
    if model is None:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT,
                             transform_input=False).to(device)
        model.fc = torch.nn.Identity()
        model.eval()
    feats = []
    for x, _ in loader:
        feats.append(model(inception_input(x.to(device))).cpu().numpy())
    return np.concatenate(feats)


def _stats(feats):
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def fid_from_stats(mu1, s1, mu2, s2, eps=1e-6):
    """Frechet distance between two Gaussians N(mu, sigma).

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


def balanced_subset(ds, per_class, class_names, seed):
    """Class-balanced (path, label) subset of a dataset, restricted to class_names.

    Balancing matters: if one side of the comparison keeps classes the other
    lacks, the measured FID absorbs the class-ratio difference instead of
    measuring image quality alone. Matching by class *name* (not integer
    label) keeps the two trees independent of each other's folder ordering.
    """
    wanted = set(class_names)
    by_class = {name: [] for name in class_names}
    for sample in ds.samples:
        name = ds.classes[sample[1]]
        if name in wanted:
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
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_kw = dict(num_workers=args.num_workers, channels=args.channels,
                   fft_denoise=args.fft_denoise, fft_cutoff=args.fft_cutoff,
                   shuffle=False, drop_last=False)
    ds_real, real_loader = build_loader(args.real_root, args.img_size, args.batch_size,
                                        **load_kw)
    ds_fake, fake_loader = build_loader(args.fake_root, args.img_size, args.batch_size, **load_kw)
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
    # Build InceptionV3 once and share it across both sides.
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT,
                         transform_input=False).to(device)
    model.fc = torch.nn.Identity()
    model.eval()
    mu_r, s_r = _stats(_features(real_loader, device, model))
    mu_f, s_f = _stats(_features(fake_loader, device, model))
    print("FID =", fid_from_stats(mu_r, s_r, mu_f, s_f))


if __name__ == "__main__":
    main()
