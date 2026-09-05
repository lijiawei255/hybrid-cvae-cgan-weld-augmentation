# -*- coding: utf-8 -*-
"""Build a real-vs-generated comparison grid (the standard qualitative GAN
presentation): one row pair per class, the upper row showing randomly picked
real images from --real_root and the lower row showing images produced by
src/generate.py from --gen_root/class_<i>.

Class indices follow the same convention as the rest of the pipeline: classes
are the sorted subdirectory names of --real_root (e.g. CR=0, LP=1, ND=2,
PO=3) and generated images live in class_<i> folders under --gen_root.

Pure PIL, no deep-learning dependencies. Output is a single PNG suitable for
the README and the repository's results/ folder.
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_images(folder, per_class, img_size, seed):
    rng = random.Random(seed)
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(files) > per_class:
        files = rng.sample(files, per_class)
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        imgs.append(img)
    while len(imgs) < per_class:
        imgs.append(Image.new("RGB", (img_size, img_size), (220, 220, 220)))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", default="data")
    ap.add_argument("--gen_root", default="generated")
    ap.add_argument("--out", default="results/real_vs_generated.png")
    ap.add_argument("--per_class", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--label_width", type=int, default=150)
    ap.add_argument("--font_size", type=int, default=22)
    args = ap.parse_args()

    real_root, gen_root = Path(args.real_root), Path(args.gen_root)
    class_names = sorted([p.name for p in real_root.iterdir() if p.is_dir()])
    s = args.img_size
    g, lw = args.gap, args.label_width

    rows = []  # (label, [PIL images])
    for i, cname in enumerate(class_names):
        real_imgs = load_images(real_root / cname, args.per_class, s, seed=0)
        gen_imgs = load_images(gen_root / f"class_{i}", args.per_class, s, seed=1)
        print(f"class_{i} -> {cname}: {len(gen_imgs)} generated shown, "
              f"{len(real_imgs)} real shown")
        rows.append((f"{cname} real", real_imgs))
        rows.append((f"{cname} gen", gen_imgs))

    try:
        font = ImageFont.load_default(size=args.font_size)
    except TypeError:
        font = ImageFont.load_default()

    header_h = 44
    W = lw + args.per_class * s + (args.per_class - 1) * g
    H = header_h + len(rows) * s + (len(rows) - 1) * g
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((lw + (W - lw) * 0.25 - 20, 10), "Real", fill=(0, 0, 0), font=font)
    draw.text((lw + (W - lw) * 0.75 - 40, 10), "Generated", fill=(0, 0, 0), font=font)

    y = header_h
    for label, imgs in rows:
        bbox = draw.textbbox((0, 0), label, font=font)
        ty = y + s // 2 - (bbox[3] - bbox[1]) // 2
        draw.text((8, ty), label, fill=(60, 60, 60), font=font)
        x = lw
        for img in imgs:
            canvas.paste(img, (x, y))
            x += s + g
        y += s + g

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("saved comparison grid ->", out, f"({W}x{H})")


if __name__ == "__main__":
    main()
