# -*- coding: utf-8 -*-
"""Smoke test: end-to-end sanity check on synthetic weld-like images.
Validates that data loading -> CVAE -> CGAN -> generation runs without errors.
It verifies code correctness only, NOT generation quality.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))


def make_synthetic_weld_data(root, n_good=60, n_defect=20, size=128, seed=0):
    """Two fake classes: 'good' = smooth bright horizontal band;
    'defect' = same band with random dark blobs on top."""
    rng = np.random.RandomState(seed)
    for cname, n in [("good", n_good), ("defect", n_defect)]:
        d = Path(root) / cname
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.zeros((size, size, 3), np.float32) + 0.15
            band = size // 2 + rng.randint(-8, 8)
            img[band - 12:band + 12, :, :] = 0.75
            img += rng.normal(0, 0.03, img.shape).astype(np.float32)
            if cname == "defect":
                for _ in range(rng.randint(1, 4)):
                    cy, cx = rng.randint(band - 10, band + 10), rng.randint(0, size)
                    r = rng.randint(3, 8)
                    yy, xx = np.ogrid[:size, :size]
                    mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
                    img[mask] *= 0.2
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(d / f"{i:04d}.png")


def run(cmd_args, module):
    print("=" * 20, module, "=" * 20)
    sys.argv = ["prog"] + cmd_args
    import importlib
    importlib.import_module(module).main()


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="cvae_cgan_smoke_"))  # OS-agnostic temp dir
    root = str(tmp / "smoke_data")
    make_synthetic_weld_data(root, size=64)

    run(["--data_root", root, "--img_size", "64", "--latent_dim", "32", "--base_ch", "16",
         "--epochs", "2", "--batch_size", "16", "--out_dir", str(tmp / "runs" / "cvae")], "train_cvae")

    run(["--data_root", root, "--cvae_ckpt", str(tmp / "runs" / "cvae" / "cvae.pt"),
         "--epochs", "2", "--batch_size", "16", "--out_dir", str(tmp / "runs" / "cgan")], "train_cgan")

    run(["--ckpt", str(tmp / "runs" / "cgan" / "cgan.pt"),
         "--out_root", str(tmp / "smoke_gen"), "--per_class", "8"], "generate")
    print("SMOKE TEST OK")
