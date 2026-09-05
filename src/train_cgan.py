# -*- coding: utf-8 -*-
"""Stage 2: CGAN adversarial fine-tuning.
G is initialized from the Stage-1 CVAE decoder weights (the core of the hybrid design)."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

from models import Decoder, Discriminator, weights_init
from data import build_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cvae_ckpt", required=True, help="Path to cvae.pt from Stage 1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=4e-4)
    ap.add_argument("--out_dir", default="runs/cgan")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.cvae_ckpt, map_location=device)
    num_classes, latent_dim = ckpt["num_classes"], ckpt["latent_dim"]
    base_ch, img_size = ckpt["base_ch"], ckpt["img_size"]

    _, loader = build_loader(args.data_root, img_size, args.batch_size,
                             num_workers=args.num_workers)

    G = Decoder(3, num_classes, latent_dim, base_ch, img_size).to(device)
    G.load_state_dict(ckpt["dec"])          # key step: generator = pre-trained CVAE decoder
    D = Discriminator(3, num_classes, base_ch, img_size).to(device)
    D.apply(weights_init)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            b = x.size(0)
            real_lbl = torch.full((b,), 0.9, device=device)  # label smoothing: real = 0.9
            fake_lbl = torch.zeros(b, device=device)

            # ---- update D ----
            z = torch.randn(b, latent_dim, device=device)
            fake = G(z, y).detach()
            loss_d = bce(D(x, y), real_lbl) + bce(D(fake, y), fake_lbl)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ---- update G every second epoch to avoid overpowering it ----
            if epoch % 2 == 0:
                z = torch.randn(b, latent_dim, device=device)
                fake = G(z, y)
                loss_g = bce(D(fake, y), torch.ones(b, device=device))
                opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch}: loss_d={loss_d.item():.4f}")
            with torch.no_grad():
                z = torch.randn(16, latent_dim, device=device)
                y = torch.arange(num_classes, device=device).repeat(16 // num_classes + 1)[:16]
                save_image(G(z, y) * 0.5 + 0.5, out / f"gen_ep{epoch}.png", nrow=4)
    torch.save({"G": G.state_dict(), "num_classes": num_classes, "latent_dim": latent_dim,
                "base_ch": base_ch, "img_size": img_size}, out / "cgan.pt")
    print("CGAN checkpoint saved to", out / "cgan.pt")


if __name__ == "__main__":
    main()
