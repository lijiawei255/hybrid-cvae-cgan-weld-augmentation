# -*- coding: utf-8 -*-
"""Stage 1: train the CVAE. Its decoder is later reused as the CGAN generator."""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from models import Encoder, Decoder, reparameterize, cvae_loss, weights_init
from data import build_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl_weight", type=float, default=1.0,
                    help="Lower to 0.1-0.5 if generated images look too blurry")
    ap.add_argument("--out_dir", default="runs/cvae")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, loader = build_loader(args.data_root, args.img_size, args.batch_size,
                              num_workers=args.num_workers)
    num_classes = len(ds.dataset.classes) if isinstance(ds, torch.utils.data.Subset) else len(ds.classes)
    print(f"classes: {num_classes}, samples: {len(ds)}")

    enc = Encoder(3, num_classes, args.latent_dim, args.base_ch, args.img_size).to(device)
    dec = Decoder(3, num_classes, args.latent_dim, args.base_ch, args.img_size).to(device)
    enc.apply(weights_init); dec.apply(weights_init)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=args.lr, betas=(0.5, 0.999))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mu, logvar = enc(x, y)
            z = reparameterize(mu, logvar)
            loss, recon, kl = cvae_loss(x, dec(z, y), mu, logvar, args.kl_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch}: loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f}")
            with torch.no_grad():
                z = torch.randn(16, args.latent_dim, device=device)
                y = torch.arange(num_classes, device=device).repeat(16 // num_classes + 1)[:16]
                save_image(dec(z, y) * 0.5 + 0.5, out / f"sample_ep{epoch}.png", nrow=4)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                "num_classes": num_classes, "latent_dim": args.latent_dim,
                "base_ch": args.base_ch, "img_size": args.img_size}, out / "cvae.pt")
    print("CVAE checkpoint saved to", out / "cvae.pt")


if __name__ == "__main__":
    main()
