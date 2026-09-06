# -*- coding: utf-8 -*-
"""Build a real-vs-generated comparison grid (the standard qualitative GAN
presentation): one row pair per class, the upper row showing randomly picked
real images from --real_root and the lower row showing images produced by
src/generate.py from --gen_root/class_<i>.

Class indices follow the same convention as the rest of the pipeline: classes
are the sorted subdirectory names of --real_root (e.g. deposit=0,
discontinuity=1, pore=2, stain=3) and generated images live in class_<i>
folders under --gen_root.

Pure PIL, no deep-learning dependencies. Output is a single PNG suitable for
the README and the repository's results/ folder.
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_images(files, per_class, img_size, seed):
    """At most per_class images from an explicit file list; never padded."""
    rng = random.Random(seed)
    files = sorted(files)
    if len(files) > per_class:
        files = rng.sample(files, per_class)
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img = img.resize((img_size, img_size), Image.BILINEAR)
        imgs.append(img)
    return imgs


def row_labels(class_names):
    labels = []
    for cname in class_names:
        labels.append(f"{cname} real")
        labels.append(f"{cname} gen")
    return labels


def grid_caption():
    return "each class: one row of real crops, one row of generated"


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

    from augment import generated_by_class

    real_root, gen_root = Path(args.real_root), Path(args.gen_root)
    class_names = sorted([p.name for p in real_root.iterdir() if p.is_dir()])
    pool = generated_by_class(gen_root, class_names)
    s = args.img_size
    g, lw = args.gap, args.label_width

    image_rows = []
    for cname in class_names:
        real_files = [p for p in (real_root / cname).iterdir()
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if not pool[cname]:
            raise SystemExit(
                f"no generated images for class '{cname}' under {gen_root}; "
                f"refusing to render a showcase figure with an empty row")
        real_imgs = load_images(real_files, args.per_class, s, seed=0)
        gen_imgs = load_images(pool[cname], args.per_class, s, seed=1)
        print(f"{cname}: {len(pool[cname])} generated available ({len(gen_imgs)} shown), "
              f"{len(real_files)} real available ({len(real_imgs)} shown)")
        image_rows.append(real_imgs)
        image_rows.append(gen_imgs)

    rows = list(zip(row_labels(class_names), image_rows))

    try:
        font = ImageFont.load_default(size=args.font_size)
    except TypeError:
        font = ImageFont.load_default()

    header_h = 44
    n_cols = max(len(imgs) for _, imgs in rows)
    W = lw + n_cols * s + (n_cols - 1) * g
    H = header_h + len(rows) * s + (len(rows) - 1) * g
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((8, 10), grid_caption(), fill=(0, 0, 0), font=font)

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
