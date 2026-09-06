# -*- coding: utf-8 -*-
"""Joint CVAE-CGAN training, following the conference paper's single-phase setup.

The paper trains the reconstruction, KL, VGG19 perceptual and adversarial
objectives **together**: the decoder is simultaneously the VAE decoder D(z, y)
and the GAN generator G(z, y), and the generator and discriminator are updated
alternately within each minibatch using separate backward passes. There is no
separate CVAE pre-training stage. The two-stage split in this repo's earlier
skeleton predates reading the paper and is recorded as superseded in
CHANGELOG.md.

Combined objective, with this repo's calibrated weights (the paper's beta=30 and
gamma=1.0 belong to a different loss normalisation; see docs/CALIBRATION.md):

    L_G = MSE_recon + beta * KL + 0.1 * VGG19_perceptual + 0.1 * hinge_G
    L_D = hinge_D(real, fake), discriminator spectral-normalised

`beta` defaults to a constant 0.015 and can instead follow the journal paper's
annealing schedule via --kl_warmup.

Four components come from the journal extension rather than the conference paper
and are switchable: --d_norm group (GroupNorm in the discriminator, its Table 3
choice), --weighted_sampler (class-balanced batches at P ~ 1/N_class),
--kl_warmup ZERO,RAMP (beta held at zero, then ramped linearly to --kl_weight)
and --monitor val_recon (the metric its ReduceLROnPlateau watched). All four are
off by default - the defaults are the conference-paper behaviour (BatchNorm
discriminator, unbalanced sampling, constant beta, val_loss monitoring), which is
also what the tagged v0.2.0 runs used, so a recorded v0.2.0 config still
reproduces a v0.2.0 run. The recommended GroupNorm + weighted-sampler
configuration and its evidence are in docs/CALIBRATION.md section 9.

Leakage-free by construction. Images are split into a train pool and a real-only
held-out test set *first*; the generator only ever sees subsets drawn from the
train pool, and its validation set is drawn from train-pool images the generator
does not train on. The downstream classifier can therefore measure augmentation
gain on images the generator has never seen.

Validation is deterministic: it decodes the posterior mean `mu` rather than a
reparameterized sample, so the early-stopping and LR-schedule signals are not
jittered by latent noise. Training still samples z, which is what makes
generation from z ~ N(0, I) work at inference time.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image

from data import (ClassFolderDataset, balanced_sample_weights, count_by_class,
                  make_splits, parse_name_counts, sample_named_subset)
from eval_fid import _features, _stats, balanced_labels, fid_from_stats
from models import (Decoder, Discriminator, Encoder, PerceptualLoss, cvae_loss,
                    hinge_d, hinge_g, kl_schedule, reparameterize, weights_init)


def parse_kl_warmup(text):
    """Parse '--kl_warmup ZERO,RAMP' into (zero_epochs, ramp_epochs).

    Empty means annealing off, which keeps the constant beta the tagged v0.2.0 runs
    used. A ramp shorter than one epoch is rejected rather than treated as "off",
    so a typo cannot silently disable the schedule being tested.
    """
    if not text:
        return 0, 0
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--kl_warmup takes 'ZERO,RAMP' epoch counts, got {text!r}")
    zero, ramp = int(parts[0]), int(parts[1])
    if ramp < 1:
        raise ValueError(f"--kl_warmup ramp must be at least 1 epoch, got {text!r}")
    return zero, ramp


def build_inception(device):
    """Shared InceptionV3 feature extractor for FID (built once, reused)."""
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT,
                         transform_input=False).to(device)
    model.fc = nn.Identity()
    return model.eval()


def save_sample_grid(enc, dec, x, y, latent_dim, path, nrow=8):
    """Three rows - real, reconstruction, generation from z ~ N(0, I).

    Runs in eval mode deliberately: in train mode BatchNorm would normalise with
    this fixed display batch's statistics and would also *update* its running
    statistics from it and from pure-noise generations, which then leaks into the
    validation loss and FID of every later epoch.
    """
    was_training = enc.training
    enc.eval(); dec.eval()
    with torch.no_grad():
        mu, _ = enc(x, y)
        recon = dec(mu, y)
        gen = dec(torch.randn(x.size(0), latent_dim, device=x.device), y)
    if was_training:
        enc.train(); dec.train()
    save_image(torch.cat([x, recon, gen], dim=0), path, nrow=nrow)


@torch.no_grad()
def validate(enc, dec, perc, loader, device, kl_weight, perc_weight):
    """Deterministic validation loss; returns (total, recon) averaged per image."""
    enc.eval(); dec.eval()
    total = recon_sum = kl_sum = perc_sum = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mu, logvar = enc(x, y)
        rec = dec(mu, y)
        _, l_recon, l_kl = cvae_loss(x, rec, mu, logvar, kl_weight)
        l_perc = perc(rec, x)
        b = x.size(0)
        # cvae_loss returns means; scale back up so the average is per-image.
        recon_sum += l_recon.item() * b
        kl_sum += l_kl.item() * b
        perc_sum += l_perc.item() * b
        total += (l_recon.item() + kl_weight * l_kl.item()
                  + perc_weight * l_perc.item()) * b
        n += b
    enc.train(); dec.train()
    return total / n, recon_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/joint")
    ap.add_argument("--subset", required=True,
                    help="generator training subset by class NAME, e.g. 'pore=40,deposit=150,discontinuity=300,stain=600'")
    ap.add_argument("--val_per_class", type=int, default=500,
                    help="real validation images per class; also the FID reference set")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--fft_denoise", action="store_true",
                    help="apply the paper's FFT circular low-pass denoising step")
    ap.add_argument("--fft_cutoff", type=float, default=0.25)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_d", type=float, default=1e-3,
                    help="discriminator lr; kept at the paper's 1e-3 because hinge loss plus "
                         "spectral normalisation keep D in equilibrium on small data (measured on "
                         "LoHi-WELD: FID declined 371 -> ~200). The lower values recorded in "
                         "docs/CALIBRATION.md were calibrated under the superseded BCE objective, "
                         "where an unbounded adversarial term let a saturated D overwhelm "
                         "reconstruction.")
    ap.add_argument("--kl_weight", type=float, default=0.015,
                    help="KL weight equivalent to the paper's beta=30 under this repo's "
                         "mean-normalised [0,1] losses. The paper reports recon converging to "
                         "~1000 and KL stabilising near 9.0, which is consistent with recon as a "
                         "mean over pixels on a [0,255] scale and KL summed over its 32 latent "
                         "dims; converting both gives 30*32/65025 = 0.0148. Measured on RIAWELC: "
                         "0.006 leaves KL ~6x the paper's, 0.015 matches its order without "
                         "collapse, >=0.05 collapses the posterior and degrades reconstruction.")
    ap.add_argument("--perc_weight", type=float, default=0.1, help="paper's lambda")
    ap.add_argument("--adv_weight", type=float, default=0.1,
                    help="adversarial weight (paper's gamma is 1.0). At 1.0 the bounded hinge "
                         "term is still ~20x the reconstruction term on a small training set and "
                         "reconstruction never converges (measured 0.09-0.12 vs a 0.038 "
                         "constant-mean baseline); at 0.1 it converges to 0.035 with the "
                         "discriminator in a healthy equilibrium. Dataset-scale calibration, "
                         "see docs/CALIBRATION.md.")
    ap.add_argument("--patience", type=int, default=10, help="early-stopping patience in epochs")
    ap.add_argument("--lr_factor", type=float, default=0.2)
    ap.add_argument("--lr_patience", type=int, default=5)
    ap.add_argument("--d_norm", choices=("batch", "group"), default="batch",
                    help="discriminator normalisation. 'batch' is what the tagged v0.2.0 runs "
                         "used; 'group' is the journal paper's Table 3 choice, made \"to ensure "
                         "stability with small batch sizes, avoiding the statistical instability "
                         "associated with Batch Normalisation\". At --batch_size 8 a BatchNorm D "
                         "scores an image using statistics from its seven batch neighbours; our "
                         "measured hinge_d sits at a median of 0.82 (dominant D, 97% of epochs "
                         "below 1.5) where the paper reports an equilibrium near 2.0.")
    ap.add_argument("--weighted_sampler", action="store_true",
                    help="draw training batches with WeightedRandomSampler at P ~ 1/N_class, the "
                         "journal paper's data-balancing strategy. Off by default so the tagged "
                         "v0.2.0 runs stay reproducible. --subset caps per-class counts but does "
                         "not balance batches: a 40-image class in a 1090-image subset is 3.7% of "
                         "draws, so most batch_size=8 minibatches contain none of it.")
    ap.add_argument("--kl_warmup", default="",
                    help="KL annealing as 'ZERO,RAMP' epoch counts; empty (default) holds beta "
                         "constant, which is what the tagged v0.2.0 runs used. The journal paper "
                         "holds beta at 0 for 10 epochs then ramps linearly to its 0.5 target "
                         "over 50, naming a constant high beta as what \"risks posterior collapse "
                         "where the encoder ignores inputs\". Its schedule needs the --epochs "
                         "200 protocol: inside paper 1's 70 epochs only 10 epochs would run at "
                         "full beta.")
    ap.add_argument("--monitor", choices=("val_loss", "val_recon"), default="val_loss",
                    help="metric fed to ReduceLROnPlateau and early stopping. 'val_loss' is the "
                         "tagged v0.2.0 behaviour; 'val_recon' is the journal paper's, whose "
                         "scheduler \"monitored validation reconstruction loss\". This matters "
                         "under --kl_warmup: the total includes beta*KL, so a rising ramp trips "
                         "the scheduler and early stopping on its own schedule rather than on "
                         "reconstruction quality.")
    ap.add_argument("--fid_every", type=int, default=1, help="paper computes FID every epoch")
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    kl_zero, kl_ramp = parse_kl_warmup(args.kl_warmup)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = ClassFolderDataset(args.data_root, args.img_size, args.channels,
                            args.fft_denoise, args.fft_cutoff)
    classes, num_classes = ds.classes, len(ds.classes)
    if not ds.samples:
        raise SystemExit(f"no images found under {args.data_root}")

    # ---- leakage-free splits -------------------------------------------------
    train_pool, test_idx = make_splits(ds.samples, args.test_frac, args.seed)
    subset_counts = parse_name_counts(args.subset)
    gen_train = sample_named_subset(ds, subset_counts, train_pool, args.seed)
    used = set(gen_train)
    remaining = [i for i in train_pool if i not in used]
    gen_val = sample_named_subset(ds, {c: args.val_per_class for c in classes},
                                 remaining, args.seed + 1)

    print(f"classes ({num_classes}): {classes}")
    print(f"device={device} img_size={args.img_size} channels={args.channels} "
          f"fft_denoise={args.fft_denoise}")
    print(f"d_norm={args.d_norm} weighted_sampler={args.weighted_sampler} "
          f"kl_warmup={kl_zero},{kl_ramp} monitor={args.monitor}")
    print(f"held-out real test (never seen by generator or classifier training): "
          f"{count_by_class(ds, test_idx)} -> {len(test_idx)}")
    print(f"generator train subset: {count_by_class(ds, gen_train)} -> {len(gen_train)}")
    print(f"generator validation / FID reference: {count_by_class(ds, gen_val)} -> {len(gen_val)}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dl_kw = dict(num_workers=args.num_workers, pin_memory=True,
                 persistent_workers=args.num_workers > 0)
    sampler = None
    if args.weighted_sampler:
        labels = [ds.samples[i][1] for i in gen_train]
        sampler = WeightedRandomSampler(balanced_sample_weights(labels),
                                        num_samples=len(gen_train), replacement=True)
    loader = DataLoader(Subset(ds, gen_train), batch_size=args.batch_size,
                        shuffle=sampler is None, sampler=sampler,
                        drop_last=True, **dl_kw)
    val_loader = DataLoader(Subset(ds, gen_val), batch_size=64, shuffle=False,
                            drop_last=False, **dl_kw)

    # ---- models --------------------------------------------------------------
    enc = Encoder(args.channels, num_classes, args.latent_dim, args.base_ch, args.img_size).to(device)
    dec = Decoder(args.channels, num_classes, args.latent_dim, args.base_ch, args.img_size).to(device)
    dis = Discriminator(args.channels, num_classes, args.base_ch, args.img_size,
                        norm=args.d_norm).to(device)
    dis.apply(weights_init)
    perc = PerceptualLoss(args.channels).to(device)

    opt_g = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    opt_d = torch.optim.Adam(dis.parameters(), lr=args.lr_d)
    sched_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_g, factor=args.lr_factor, patience=args.lr_patience)
    sched_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_d, factor=args.lr_factor, patience=args.lr_patience)

    # ---- fixed FID reference (real validation images, class-balanced) ---------
    inception = build_inception(device)
    mu_r, s_r = _stats(_features(val_loader, device, inception))
    fid_labels = balanced_labels(num_classes, args.val_per_class, device)

    def compute_fid():
        dec.eval()
        batches = []
        with torch.no_grad():
            for y in fid_labels.split(100):
                z = torch.randn(y.size(0), args.latent_dim, device=device)
                batches.append((dec(z, y).cpu(), None))
        mu_f, s_f = _stats(_features(batches, device, inception))
        dec.train()
        return fid_from_stats(mu_r, s_r, mu_f, s_f)

    # Fixed class-balanced batch with each image's TRUE label, so sample grids are
    # comparable across epochs and each column is one class in all three rows.
    per_class_grid = max(1, 16 // num_classes)
    grid_idx = []
    for label in range(num_classes):
        grid_idx.extend([i for i in gen_val if ds.samples[i][1] == label][:per_class_grid])
    grid_x, grid_y = next(iter(DataLoader(Subset(ds, grid_idx),
                                          batch_size=len(grid_idx), shuffle=False)))
    grid_x, grid_y = grid_x.to(device), grid_y.to(device)
    grid_nrow = grid_x.size(0)

    history_path = out / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "beta", "recon", "kl", "perc", "loss_g", "loss_d",
                                "val_loss", "val_recon", "fid", "lr", "seconds"])

    best = {"val": float("inf"), "monitored": float("inf"), "epoch": 0,
            "fid": float("nan"), "state": None}
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        beta = kl_schedule(epoch, args.kl_weight, kl_zero, kl_ramp)
        enc.train(); dec.train(); dis.train()
        agg = dict(recon=0.0, kl=0.0, perc=0.0, g=0.0, d=0.0)
        steps = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            b = x.size(0)

            # ---- generator side: all four objectives in one backward pass ----
            mu, logvar = enc(x, y)
            z = reparameterize(mu, logvar)
            recon = dec(z, y)
            loss_cvae, l_recon, l_kl = cvae_loss(x, recon, mu, logvar, beta)
            l_perc = perc(recon, x)
            fake_g = dec(torch.randn(b, args.latent_dim, device=device), y)
            l_adv = hinge_g(dis(fake_g, y))
            loss_g = loss_cvae + args.perc_weight * l_perc + args.adv_weight * l_adv
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            # ---- discriminator side: separate backward pass, same minibatch ---
            fake_d = dec(torch.randn(b, args.latent_dim, device=device), y).detach()
            loss_d = hinge_d(dis(x, y), dis(fake_d, y))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            agg["recon"] += l_recon.item(); agg["kl"] += l_kl.item()
            agg["perc"] += l_perc.item(); agg["g"] += loss_g.item()
            agg["d"] += loss_d.item()
            steps += 1
        for key in agg:
            agg[key] /= max(steps, 1)
        if steps == 0:
            raise SystemExit(
                "training loader produced no batches: the subset is smaller than "
                "--batch_size with drop_last enabled; lower --batch_size")

        # Validation uses the same beta as training, so val_loss stays on the same
        # scale as the objective actually being optimised at this epoch.
        val_loss, val_recon = validate(enc, dec, perc, val_loader, device,
                                       beta, args.perc_weight)
        monitored = val_recon if args.monitor == "val_recon" else val_loss
        sched_g.step(monitored); sched_d.step(monitored)
        lr = opt_g.param_groups[0]["lr"]

        fid = compute_fid() if (args.fid_every and epoch % args.fid_every == 0) else float("nan")
        seconds = time.time() - t0

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{beta:.6f}", f"{agg['recon']:.6f}",
                                    f"{agg['kl']:.6f}",
                                    f"{agg['perc']:.6f}", f"{agg['g']:.6f}",
                                    f"{agg['d']:.6f}", f"{val_loss:.6f}",
                                    f"{val_recon:.6f}", f"{fid:.4f}",
                                    f"{lr:.3e}", f"{seconds:.1f}"])
        print(f"epoch {epoch}/{args.epochs} beta={beta:.4f} recon={agg['recon']:.4f} "
              f"kl={agg['kl']:.4f} perc={agg['perc']:.4f} loss_d={agg['d']:.4f} "
              f"val={val_loss:.4f} val_recon={val_recon:.4f} fid={fid:.2f} "
              f"lr={lr:.1e} ({seconds:.0f}s)")

        if (args.sample_every and epoch % args.sample_every == 0) or epoch == args.epochs:
            save_sample_grid(enc, dec, grid_x, grid_y, args.latent_dim,
                             out / f"samples_ep{epoch}.png", nrow=grid_nrow)

        if monitored < best["monitored"] - 1e-6:
            best = {"val": val_loss, "monitored": monitored, "epoch": epoch, "fid": fid,
                    "state": cpu_state(enc, dec, dis)}
            bad_epochs = 0
            torch.save(best_state_payload(best, args, classes, num_classes), out / "joint.pt")
            print(f"  -> new best {args.monitor}={monitored:.4f}, checkpoint saved")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}: no {args.monitor} improvement for "
                      f"{bad_epochs} epochs (best epoch {best['epoch']})")
                break

    print(f"best: epoch {best['epoch']} {args.monitor}={best['monitored']:.4f} "
          f"val_loss={best['val']:.4f} fid={best['fid']:.2f}")
    print(f"checkpoint -> {out / 'joint.pt'}; history -> {history_path}")


def cpu_state(enc, dec, dis):
    """Snapshot the three modules onto CPU so the best-so-far copy costs no VRAM."""
    def to_cpu(module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    return {"enc": to_cpu(enc), "dec": to_cpu(dec), "D": to_cpu(dis)}


def best_state_payload(best, args, classes, num_classes):
    """Everything generate.py and the figures need, including class NAMES so a
    different dataset cannot silently map labels to the wrong folders."""
    payload = dict(best["state"])
    payload.update({
        "num_classes": num_classes, "classes": classes,
        "latent_dim": args.latent_dim, "base_ch": args.base_ch,
        "img_size": args.img_size, "img_channels": args.channels,
        "best_epoch": best["epoch"], "best_val_loss": best["val"], "best_fid": best["fid"],
        "best_monitored_metric": args.monitor, "best_monitored_value": best["monitored"],
        "kl_weight": args.kl_weight, "perc_weight": args.perc_weight,
        "adv_weight": args.adv_weight, "seed": args.seed,
        "d_norm": args.d_norm, "weighted_sampler": args.weighted_sampler,
        "kl_warmup": args.kl_warmup, "monitor": args.monitor,
        "fft_denoise": args.fft_denoise, "fft_cutoff": args.fft_cutoff,
        "subset": args.subset, "test_frac": args.test_frac,
    })
    return payload


if __name__ == "__main__":
    main()
