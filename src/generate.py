# -*- coding: utf-8 -*-
"""Sample class-conditional images from a trained generator into class folders,
mirroring the real-data layout, for FID evaluation and downstream augmentation."""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models import Decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="cgan.pt (or cvae.pt)")
    ap.add_argument("--out_root", default="generated")
    ap.add_argument("--per_class", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("G", ckpt.get("dec"))
    G = Decoder(3, ckpt["num_classes"], ckpt["latent_dim"], ckpt["base_ch"], ckpt["img_size"]).to(device)
    G.load_state_dict(state); G.eval()

    out = Path(args.out_root)
    with torch.no_grad():
        for c in range(ckpt["num_classes"]):
            d = out / f"class_{c}"; d.mkdir(parents=True, exist_ok=True)
            for i in range(args.per_class):
                z = torch.randn(1, ckpt["latent_dim"], device=device)
                y = torch.tensor([c], device=device)
                save_image(G(z, y) * 0.5 + 0.5, d / f"gen_{c}_{i:05d}.png")
    print("Generation finished ->", out)


if __name__ == "__main__":
    main()
