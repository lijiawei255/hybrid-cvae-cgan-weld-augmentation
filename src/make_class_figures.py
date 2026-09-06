# -*- coding: utf-8 -*-
"""Per-class real-vs-generated close-up figures for quick sharing.

For each class this script renders a small two-row figure (top: real images
sampled from --real_root, bottom: generated images from
--gen_root/class_<i>) with a stats strip underneath (sample counts and a
per-class FID computed with the same InceptionV3 features as
src/eval_fid.py), sized to stay readable when pasted directly into a chat
window. Output files land in --out_dir as class_<NAME>.png.
"""
import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from augment import generated_by_class
from data import ClassFolderDataset, ListDataset, make_splits
from eval_fid import _features, _stats, fid_from_stats

CLASS_DESCRIPTIONS = {
    "CR": "cracks",
    "LP": "lack of penetration",
    "ND": "no defect",
    "PO": "porosity",
}


def load_row(files, per_row, img_size, seed):
    """At most per_row images from an explicit file list; never padded."""
    rng = random.Random(seed)
    files = sorted(files)
    if len(files) > per_row:
        files = rng.sample(files, per_row)
    return [Image.open(f).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", default="data")
    ap.add_argument("--gen_root", default="generated")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--per_row", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    real_root, gen_root = Path(args.real_root), Path(args.gen_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ClassFolderDataset(real_root, args.img_size, args.channels)
    s, g = args.img_size, 8

    try:
        font = ImageFont.load_default(size=20)
        small = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
        small = font

    pool = generated_by_class(gen_root, ds.classes)
    _, test_idx = make_splits(ds.samples, args.test_frac, args.seed)
    test_set = set(test_idx)

    for i, cname in enumerate(ds.classes):
        # FID reference excludes the held-out test split, matching train_joint.py.
        real_idx = [j for j, (_, y) in enumerate(ds.samples)
                    if y == i and j not in test_set]
        gen_files = pool[cname]
        if not gen_files:
            raise SystemExit(
                f"no generated images for class '{cname}' under {gen_root}")
        gen_samples = [(f, i) for f in gen_files]

        dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True)
        real_loader = DataLoader(Subset(ds, real_idx), shuffle=False,
                                 drop_last=False, **dl_kw)
        gen_loader = DataLoader(ListDataset(gen_samples, ds.tf, ds.channels), shuffle=False,
                                drop_last=False, **dl_kw)
        mu_r, s_r = _stats(_features(real_loader, device))
        mu_f, s_f = _stats(_features(gen_loader, device))
        fid = fid_from_stats(mu_r, s_r, mu_f, s_f)

        real_imgs = load_row([p for p in (real_root / cname).iterdir()
                              if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
                             args.per_row, s, seed=0)
        gen_imgs = load_row(gen_files, args.per_row, s, seed=1)

        label_w, header_h, stats_h = 110, 30, 30
        W = label_w + args.per_row * s + (args.per_row - 1) * g
        H = header_h + 2 * s + g + stats_h
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        desc = CLASS_DESCRIPTIONS.get(cname, "")
        title = f"{cname} ({desc})" if desc else cname
        draw.text((8, 5), title, fill=(0, 0, 0), font=font)
        draw.text((label_w + (W - label_w) * 0.25 - 24, 5), "Real",
                  fill=(90, 90, 90), font=small)
        draw.text((label_w + (W - label_w) * 0.75 - 52, 5), "Generated",
                  fill=(90, 90, 90), font=small)
        y = header_h
        for row, label in ((real_imgs, "Real"), (gen_imgs, "Gen")):
            tag = "real" if label == "Real" else "gen"
            bbox = draw.textbbox((0, 0), tag, font=small)
            draw.text((8, y + s // 2 - (bbox[3] - bbox[1]) // 2), tag,
                      fill=(60, 60, 60), font=small)
            x = label_w
            for img in row:
                canvas.paste(img, (x, y))
                x += s + g
            y += s + g
        stats = (f"real {len(real_idx):,} | generated {len(gen_files):,} | "
                 f"class FID {fid:.2f} (InceptionV3, trend-only scale)")
        draw.text((8, H - stats_h + 6), stats, fill=(40, 40, 40), font=small)

        out = out_dir / f"class_{cname}.png"
        canvas.save(out)
        print(f"{cname}: real {len(real_idx):,}, generated {len(gen_files):,}, "
              f"FID {fid:.2f} -> {out}")


if __name__ == "__main__":
    main()
