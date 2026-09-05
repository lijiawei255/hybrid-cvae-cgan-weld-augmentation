# -*- coding: utf-8 -*-
"""FID evaluation: distributional distance between generated and real images
(lower is better).

Note: FID relies on InceptionV3 features and is noisy on small sample counts.
Use at least ~2000 real and ~2000 generated images for meaningful absolute
values; on smaller sets it should only be read as a training trend.
"""
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights

from data import build_loader


@torch.no_grad()
def _features(loader, device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # 2048-d pooled features
    model.eval()
    feats = []
    for x, _ in loader:
        x = x.to(device) * 0.5 + 0.5
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        feats.append(model(x).cpu().numpy())
    return np.concatenate(feats)


def _stats(feats):
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def fid_from_stats(mu1, s1, mu2, s2):
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1 + s2 - 2 * covmean))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", required=True, help="Real images (class subfolders)")
    ap.add_argument("--fake_root", required=True, help="Generated images (same layout)")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, real_loader = build_loader(args.real_root, args.img_size, args.batch_size)
    _, fake_loader = build_loader(args.fake_root, args.img_size, args.batch_size)
    mu_r, s_r = _stats(_features(real_loader, device))
    mu_f, s_f = _stats(_features(fake_loader, device))
    print("FID =", fid_from_stats(mu_r, s_r, mu_f, s_f))


if __name__ == "__main__":
    main()
