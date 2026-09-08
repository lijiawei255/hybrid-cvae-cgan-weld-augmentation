# -*- coding: utf-8 -*-
"""The paper's balance-to-max augmentation protocol.

Both reference papers fill each minority class with synthetic images until it
reaches the majority class count, and report a sensitivity analysis over that
"filling rate": 0.0 is the original imbalanced data, 1.0 is fully balanced. The
sweep matters because the synthetic-to-real ratio is itself a finding - too few
synthetic samples add noise without actually fixing the imbalance.

Class identity is expressed by NAME throughout, so this module works unchanged on
a dataset with different classes or a different class ordering.
"""
import json
from pathlib import Path

from data import parse_name_counts

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: Written by generate.py next to the class_i folders and checked by
#: generated_by_class, so a positional folder tree cannot be read with the wrong
#: class ordering.
CLASS_MANIFEST = "classes.txt"


def filling_rate_counts(real_counts, ratio):
    """{class_name: synthetic_count} needed to reach `ratio` of the gap to N_max.

    N_max is the majority class count, so the majority class itself always needs
    zero synthetic images and a class already at N_max is left alone.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"filling rate must be in [0, 1], got {ratio}")
    n_max = max(real_counts.values())
    return {name: int(round(ratio * (n_max - count)))
            for name, count in real_counts.items()}


def synthetic_prefix(paths, count):
    """First `count` paths in sorted order.

    Using a sorted prefix rather than a fresh random draw at each ratio is what
    makes a filling-rate sweep comparable: the images used at 0.25 are a subset of
    those used at 0.5, so the curve reflects the amount of augmentation and not a
    change of sample.
    """
    ordered = sorted(Path(p) for p in paths)
    if count > len(ordered):
        raise ValueError(
            f"requested {count} synthetic images but only {len(ordered)} exist; "
            f"regenerate with a larger count")
    return ordered[:count]


def generated_by_class(gen_root, classes):
    """{class_name: sorted [Path]} read from a generate.py output tree.

    The folders are positional (``class_0``, ``class_1``, ...) so that mapping is
    only valid if this pool was generated from the same class list. generate.py
    writes a ``classes.txt`` manifest next to the folders and it is checked here:
    a pool with no manifest, or one recorded in a different order, is rejected
    rather than silently assigning every synthetic image to the wrong class.

    A class with no folder or no files yields an empty list, which is legitimate
    - the majority class needs no augmentation. A tree with no images at all is
    almost always a wrong path, so that case raises instead of quietly producing
    a "no augmentation" experiment.
    """
    root = Path(gen_root)
    manifest = root / CLASS_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{manifest} is missing, so the class_i folders cannot be mapped to "
            f"class names safely. Regenerate this pool with src/generate.py, which "
            f"writes the manifest from the checkpoint's own class list.")
    recorded = [line.strip() for line in
                manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if recorded != list(classes):
        raise ValueError(
            f"{root} was generated for classes {recorded} but this dataset has "
            f"{list(classes)}; the positional class_i folders would map to the "
            f"wrong classes. Regenerate the pool from a checkpoint trained on "
            f"this dataset.")

    out = {}
    for label, name in enumerate(classes):
        folder = root / f"class_{label}"
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in IMG_EXTENSIONS) if folder.is_dir() else []
        out[name] = files
    if not any(out.values()):
        raise FileNotFoundError(
            f"no generated images under {root}; expected subfolders "
            f"class_0..class_{len(classes) - 1}")
    return out


def verify_generation_meta(gen_root, subset, seed, test_frac=None):
    """Refuse to sweep against a pool generated from a different split.

    The r=0.0 condition is only meaningful if it is exactly the real data the
    generator trained on. A retyped ``--subset``, a different ``--seed`` or a
    different ``--test_frac`` would silently compare augmentation against the
    wrong baseline, and the whole sweep would be measuring nothing.
    """
    meta_path = Path(gen_root) / "meta.json"
    if not meta_path.is_file():
        print(f"note: {meta_path} is missing; skipping the split-configuration "
              f"consistency check (pools written before meta.json existed cannot "
              f"be checked)")
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    problems = []
    # Subsets are compared as parsed mappings, so a reordered 'a=1,b=2' against
    # a recorded 'b=2,a=1' is not a false alarm; a different mapping still refuses.
    if (meta.get("subset") is not None
            and parse_name_counts(meta["subset"]) != parse_name_counts(subset)):
        problems.append(f"--subset {subset!r} but pool was generated with "
                        f"{meta['subset']!r}")
    if meta.get("seed") is not None and meta["seed"] != seed:
        problems.append(f"--seed {seed} but pool was generated with {meta['seed']}")
    if (test_frac is not None and meta.get("test_frac") is not None
            and abs(meta["test_frac"] - test_frac) > 1e-12):
        problems.append(f"--test_frac {test_frac} but pool was generated with "
                        f"{meta['test_frac']}")
    if problems:
        raise SystemExit(
            f"{gen_root} was generated from a different split than this run uses: "
            + "; ".join(problems) + ". Regenerate the pool or match the flags.")
