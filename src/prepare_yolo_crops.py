# -*- coding: utf-8 -*-
"""Convert an annotated object-detection dataset into class-folder crops.

Several public weld-defect datasets ship as detection data rather than
classification folders. Two layouts are common and both are supported:

* standard YOLO: an ``images/`` tree plus a parallel ``labels/`` tree with one
  ``<stem>.txt`` per image;
* flat pairs: image and label side by side, possibly nested (LoHi-WELD puts
  ``<stem>.jpg`` next to ``<stem>.yolo`` inside ``high_resolution_welds/`` and
  ``low_resolution_welds/``).

Each label line is ``class_id x_center y_center width height`` in normalised
coordinates. For generative augmentation we want tight defect patches, not whole
frames, so this script crops every bounding box (with a configurable margin) and
writes the crops into ``<out_root>/<ClassName>/``.

That crop step is also the reproduced paper's ROI-extraction preprocessing step:
the papers extract the region of interest before training, and here the
annotated boxes provide it exactly.

Class names come from ``--classes`` (comma separated, in class_id order), or from
a ``data.yaml`` / ``classes.txt`` found at the input root. Per-class counts and
the resulting imbalance ratio are printed so the imbalance - the property the
method exists to address - is visible before any training starts.
"""
import argparse
from pathlib import Path

from PIL import Image

try:
    import yaml
except ImportError:  # PyYAML is optional; --classes / classes.txt still work
    yaml = None

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
LABEL_EXTENSIONS = (".yolo", ".txt")


def read_class_names(input_root, classes_arg):
    """Class names in class_id order, from --classes / data.yaml / classes.txt."""
    if classes_arg:
        return [c.strip() for c in classes_arg.split(",") if c.strip()]
    for candidate in ("data.yaml", "data.yml", "classes.txt"):
        path = input_root / candidate
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            if yaml is None:
                continue  # cannot parse YAML without PyYAML; try the next candidate
            parsed = yaml.safe_load(text)
            names = parsed.get("names")
            if isinstance(names, dict):
                return [names[k] for k in sorted(names)]
            if isinstance(names, list):
                return [str(n) for n in names]
        else:
            return [line.strip() for line in text.splitlines() if line.strip()]
    raise SystemExit(
        f"no class names: pass them with --classes 'a,b,c' in class_id order, or put "
        f"a data.yaml / classes.txt at {input_root}")


def find_label(image_path):
    """The label file for an image: sibling .yolo/.txt first, else parallel labels/.

    Side-by-side pairs (LoHi-WELD) win at any nesting depth. Otherwise the
    closest ``images`` ancestor is taken as the images root and the image's
    path under it is mirrored into a sibling ``labels`` tree, covering both
    ``images/x.jpg -> labels/x.txt`` and the split-subfolder form
    ``images/<split>/x.jpg -> labels/<split>/x.txt``.
    """
    for ext in LABEL_EXTENSIONS:
        sibling = image_path.with_suffix(ext)
        if sibling.is_file():
            return sibling
    for images_root in (p for p in image_path.parents if p.name == "images"):
        rel = image_path.relative_to(images_root)
        for ext in LABEL_EXTENSIONS:
            parallel = (images_root.parent / "labels" / rel).with_suffix(ext)
            if parallel.is_file():
                return parallel
    return None


def iter_images(input_root):
    """Every image under input_root, skipping label trees and hidden dirs."""
    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        if "labels" in path.parts or any(p.startswith(".") for p in path.parts):
            continue
        yield path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True,
                    help="directory holding the detection data (any nesting)")
    ap.add_argument("--out_root", required=True, help="class-folder tree to write")
    ap.add_argument("--classes", default="",
                    help="comma-separated names in class_id order, e.g. "
                         "'pore,deposit,discontinuity,stain'")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--channels", type=int, default=3, choices=(1, 3),
                    help="3 keeps a colour dataset's native channels; 1 converts to "
                         "grayscale (only appropriate when the source is monochrome)")
    ap.add_argument("--margin", type=float, default=0.1,
                    help="fractional margin added around each box before cropping")
    ap.add_argument("--min_side", type=int, default=16,
                    help="skip crops whose longest box edge is below this many "
                         "pixels (the filter has always tested the longest edge; "
                         "the flag name is kept for compatibility)")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    if not input_root.is_dir():
        raise SystemExit(f"input_root {input_root} is not a directory")
    names = read_class_names(input_root, args.classes)
    print(f"classes ({len(names)}): {names}")

    out_root = Path(args.out_root)
    counts = {name: 0 for name in names}
    no_label = skipped = bad_lines = written = 0
    for image_path in iter_images(input_root):
        label_path = find_label(image_path)
        if label_path is None:
            no_label += 1
            continue
        with Image.open(image_path) as handle:
            width, height = handle.size
            lines = [ln.split() for ln in
                     label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for k, parts in enumerate(lines):
                if len(parts) < 5:
                    bad_lines += 1
                    print(f"warning: {label_path.name}:{k + 1} has {len(parts)} fields, "
                          f"expected 'class xc yc w h'; skipped")
                    continue
                try:
                    class_id = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    bad_lines += 1
                    print(f"warning: {label_path.name}:{k + 1} is not numeric; skipped")
                    continue
                if not 0 <= class_id < len(names):
                    raise SystemExit(
                        f"{label_path.name} references class_id {class_id}, outside "
                        f"0..{len(names) - 1} for names {names}")
                bw, bh = w * width, h * height
                if max(bw, bh) < args.min_side:
                    skipped += 1
                    continue
                cx, cy = xc * width, yc * height
                half_w = bw * (1 + args.margin) / 2
                half_h = bh * (1 + args.margin) / 2
                box = (int(max(0, cx - half_w)), int(max(0, cy - half_h)),
                       int(min(width, cx + half_w)), int(min(height, cy + half_h)))
                if box[0] >= box[2] or box[1] >= box[3]:
                    bad_lines += 1
                    print(f"warning: {label_path.name}:{k + 1} box {box} is empty after "
                          f"clamping to the frame; skipped")
                    continue
                crop = handle.crop(box).convert("L" if args.channels == 1 else "RGB")
                crop = crop.resize((args.img_size, args.img_size), Image.BILINEAR)
                folder = out_root / names[class_id]
                folder.mkdir(parents=True, exist_ok=True)
                # Include the relative path: same-stem images in different
                # subfolders must not overwrite each other and understate counts.
                stem = "__".join(image_path.relative_to(input_root).with_suffix("").parts)
                crop.save(folder / f"{stem}_{k:03d}.png")
                counts[names[class_id]] += 1
                written += 1

    print(f"wrote {written} crops to {out_root} "
          f"({no_label} images without labels, {skipped} boxes below --min_side "
          f"on their longest edge, "
          f"{bad_lines} malformed/inverted boxes skipped)")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {count}")
    if counts:
        biggest, smallest = max(counts.values()), min(counts.values())
        print(f"imbalance ratio (max/min): {biggest / max(smallest, 1):.2f}x")


if __name__ == "__main__":
    main()
