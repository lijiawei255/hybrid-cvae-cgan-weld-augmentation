# -*- coding: utf-8 -*-
"""Sample class-conditional images from a trained generator into class folders.

Output layout mirrors the real dataset so FID evaluation and downstream
augmentation can consume it directly:

    out_root/class_0/gen_0_00000.png
    out_root/class_1/gen_1_00000.png
    ...

Images are written straight from the decoder's sigmoid output, i.e. already in
[0, 1]; there is no Tanh rescale anywhere in this pipeline.

Counts can be given uniformly (--per_class) or per class **by name**
(--counts "CR=560,PO=400"). Names are checked against the class list stored in
the checkpoint, so a dataset with a different class order cannot silently
generate into the wrong folders. Filenames are zero-padded and generated in a
fixed seed order, which makes the sorted file list a stable prefix ordering -
the filling-rate sweep relies on that to reuse one nested pool instead of
regenerating for every ratio.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from torchvision.utils import save_image

from augment import CLASS_MANIFEST
from models import Decoder


def parse_counts(text, classes):
    """'CR=560,PO=400' -> {label_index: count}, validated against `classes`."""
    from data import parse_name_counts

    name_to_label = {name: i for i, name in enumerate(classes)}
    parsed = parse_name_counts(text)
    unknown = sorted(name for name in parsed if name not in name_to_label)
    if unknown:
        raise KeyError(
            f"unknown class name(s) {unknown}; checkpoint classes are {classes}")
    return {name_to_label[name]: count for name, count in parsed.items()}


@torch.no_grad()
def generate(G, latent_dim, counts, out_root, device, batch_size=64, classes=None):
    """Write `counts[label]` images per label; returns {label: list of paths}."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if classes is not None:
        (out_root / CLASS_MANIFEST).write_text("\n".join(classes) + "\n", encoding="utf-8")
    written = {}
    for label, total in sorted(counts.items()):
        folder = out_root / f"class_{label}"
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        index = 0
        while index < total:
            n = min(batch_size, total - index)
            z = torch.randn(n, latent_dim, device=device)
            y = torch.full((n,), label, dtype=torch.long, device=device)
            images = G(z, y).clamp(0, 1)
            for row in range(n):
                # Six digits, not five: the sweep relies on sorted order matching
                # generation order, and gen_0_100000.png would sort before
                # gen_0_99999.png and silently break that nesting.
                path = folder / f"gen_{label}_{index + row:06d}.png"
                save_image(images[row], path)
                paths.append(path)
            index += n
        written[label] = paths
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="joint.pt (or any checkpoint holding generator weights)")
    ap.add_argument("--out_root", default="generated")
    ap.add_argument("--per_class", type=int, default=0,
                    help="uniform count for every class (ignored if --counts is given)")
    ap.add_argument("--counts", default="",
                    help="per-class counts by NAME, e.g. 'CR=560,PO=400,ND=300,LP=0'")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.counts and args.per_class <= 0:
        raise SystemExit("provide either --per_class N or --counts 'NAME=N,...'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)

    state = ckpt.get("G", ckpt.get("dec"))
    if state is None:
        raise KeyError(f"no generator weights ('G' or 'dec') in {args.ckpt}")
    channels = ckpt.get("img_channels", 1)
    classes = ckpt.get("classes")
    num_classes = ckpt["num_classes"]
    latent_dim = ckpt["latent_dim"]

    G = Decoder(channels, num_classes, latent_dim, ckpt["base_ch"], ckpt["img_size"]).to(device)
    G.load_state_dict(state)
    G.eval()

    if args.counts:
        if classes is None:
            raise KeyError(
                "--counts needs class names, but this checkpoint stores none; "
                "retrain it or use --per_class instead")
        counts = parse_counts(args.counts, classes)
    else:
        counts = {label: args.per_class for label in range(num_classes)}

    print(f"checkpoint: img_size={ckpt['img_size']} channels={channels} "
          f"latent_dim={latent_dim} classes={classes}")
    for label, n in sorted(counts.items()):
        name = classes[label] if classes else str(label)
        print(f"  class_{label} -> {name}: {n} images")

    out = Path(args.out_root)
    # A re-run with different counts must not leave orphan images from a previous
    # checkpoint: augment.generated_by_class would then mix two generators into
    # one sweep pool and silently break the nested-prefix invariant.
    for stale in list(out.glob("class_*")) + [out / CLASS_MANIFEST, out / "meta.json"]:
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.is_file():
            stale.unlink()

    written = generate(G, latent_dim, counts, args.out_root, device, args.batch_size,
                       classes=classes)
    meta = {"classes": classes, "counts": {classes[l] if classes else str(l): n
                                           for l, n in sorted(counts.items())},
            "subset": ckpt.get("subset"), "seed": ckpt.get("seed"),
            "test_frac": ckpt.get("test_frac"), "img_size": ckpt["img_size"],
            "channels": channels, "generator": str(args.ckpt)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Generation finished ->", args.out_root,
          f"({sum(len(v) for v in written.values())} images)")


if __name__ == "__main__":
    main()
