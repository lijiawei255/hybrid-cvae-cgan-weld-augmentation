# -*- coding: utf-8 -*-
"""Data loading, preprocessing and crop-level split helpers.

Two conventions here are dictated by the reproduced paper and are relied on by
every other module - keep them consistent if you change them:

* Images live in **[0, 1]**. The paper's decoder ends in a sigmoid producing
  [0, 1], so there is no Tanh-style rescaling to [-1, 1] anywhere in this
  pipeline. The channel count is a parameter (`channels`): the papers' camera
  produced grayscale, while the current primary dataset keeps native RGB.
* Class subsets are requested **by class name**, never by index. Hardcoding
  indices silently samples the wrong classes on any other dataset.
"""
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_RGB_MODES = {"RGB", "RGBA", "YCbCr", "P"}


def peek_image_mode(data_root):
    """Mode of the first image under data_root, or None if the tree is empty."""
    root = Path(data_root)
    if not root.is_dir():
        return None
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for fp in sorted(folder.glob("*")):
            if fp.suffix.lower() in IMG_EXTENSIONS:
                with Image.open(fp) as img:
                    return img.mode
    return None


def assert_channels_match_data(data_root, channels):
    """Refuse --channels 1 on colour files, which would silently convert to L.

    The papers' melt-pool camera was grayscale; LoHi-WELD is RGB. Converting
    colour crops to one channel hides defect signal and is almost always a
    leftover default rather than an intentional choice.
    """
    if channels != 1:
        return
    mode = peek_image_mode(data_root)
    if mode in _RGB_MODES:
        raise ValueError(
            f"{data_root} looks like {mode} imagery; --channels 1 would silently "
            f"convert it to grayscale. Pass --channels 3, or convert the files "
            f"offline if grayscale really is intended.")



def fft_lowpass(arr, cutoff=0.25):
    """Circular low-pass filter in the frequency domain.

    This is the paper's FFT denoising step: transform, mask with a circular
    filter, inverse transform. `cutoff` is the kept radius as a fraction of the
    distance from the spectrum centre to the nearest edge.
    """
    a = np.asarray(arr, dtype=np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(a))
    h, w = a.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    spectrum[radius > cutoff * min(cy, cx)] = 0.0
    out = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
    return np.clip(out, 0, 255).astype(np.uint8)


class FFTLowPass:
    """PIL-to-PIL wrapper so FFT denoising composes with torchvision transforms."""

    def __init__(self, cutoff=0.25):
        self.cutoff = cutoff

    def __call__(self, img):
        arr = np.asarray(img)
        if arr.ndim == 3:
            filtered = np.stack(
                [fft_lowpass(arr[:, :, c], self.cutoff) for c in range(arr.shape[2])], axis=-1)
        else:
            filtered = fft_lowpass(arr, self.cutoff)
        return Image.fromarray(filtered)


class ClassFolderDataset(Dataset):
    """Class-folder image dataset.

    Expected layout:
        data_root/
          <class_a>/  *.png|jpg|...
          <class_b>/  ...

    Labels are the indices of the sorted subdirectory names, and `self.classes`
    is that sorted name list - the single source of truth for the name <-> label
    mapping used everywhere else.
    """

    def __init__(self, data_root, img_size=224, channels=1, fft_denoise=False, fft_cutoff=0.25):
        self.root = Path(data_root)
        self.channels = channels
        # Hidden dirs are skipped: a .ipynb_checkpoints or editor leftover must
        # not become a phantom class that shifts every label index (prepare_yolo_
        # crops.py applies the same rule to its input tree).
        self.classes = sorted(p.name for p in self.root.iterdir()
                              if p.is_dir() and not p.name.startswith("."))
        self.samples = []
        for idx, cname in enumerate(self.classes):
            # Sort by name, not by Path object: Path comparison is
            # case-insensitive on Windows and case-sensitive on POSIX, so a
            # tree with any uppercase filename would otherwise be ordered
            # differently on the two platforms - and this order is what the
            # seeded split indexes into.
            for fp in sorted((self.root / cname).glob("*"), key=lambda q: q.name):
                if fp.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((fp, idx))

        steps = [transforms.Resize((img_size, img_size))]
        if fft_denoise:
            steps.append(FFTLowPass(fft_cutoff))
        steps.append(transforms.ToTensor())  # -> [0, 1]; no rescaling to [-1, 1]
        self.tf = transforms.Compose(steps)

    def __len__(self):
        return len(self.samples)

    def _open(self, fp):
        img = Image.open(fp)
        return img.convert("L" if self.channels == 1 else "RGB")

    def __getitem__(self, i):
        fp, label = self.samples[i]
        return self.tf(self._open(fp)), label


class ListDataset(Dataset):
    """Flat (path, label) list that reuses another dataset's preprocessing.

    Used to mix generated images into a real-image dataset without duplicating
    the transform pipeline, so real and generated samples are always processed
    identically.
    """

    def __init__(self, samples, tf, channels=1):
        self.samples = list(samples)
        self.tf = tf
        self.channels = channels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fp, label = self.samples[i]
        img = Image.open(fp)
        return self.tf(img.convert("L" if self.channels == 1 else "RGB")), label


#: prepare_yolo_crops.py names each crop "<encoded-source-path>_<k:03d>.png", so
#: stripping that trailing box index recovers the source frame it was cut from.
_CROP_INDEX_SUFFIX = re.compile(r"_\d{3}$")


def source_key(path):
    """Source-frame identity of a crop produced by prepare_yolo_crops.py.

    Crops are written as ``<encoded-source-path>_<k:03d>.png``, so removing the
    trailing three-digit box index recovers the frame. A filename that does not
    carry that suffix is its own group, which makes the grouping degrade to the
    crop-level behaviour rather than silently merging unrelated files.
    """
    return _CROP_INDEX_SUFFIX.sub("", Path(path).stem)


def make_splits(samples, test_frac=0.2, seed=42, split_by="crop"):
    """(train_pool, test) index split. The test half is real-only throughout.

    ``split_by="crop"`` is the published protocol and the default: a stratified
    split of individual crops. Crop indices never cross pools, but nothing groups
    crops by source image, so crops cut from the same source frame can sit on both
    sides - measured at 99.8-100% of test crops on the published seeds
    (docs/CALIBRATION.md section 9). Absolute downstream numbers under it are
    optimistic about generalisation to unseen source frames.

    ``split_by="source"`` groups by source frame first, so no source frame
    contributes to both halves. Classes are still handled independently and each
    one's frames are consumed until its test share reaches ``test_frac``, so the
    split stays stratified to within one frame's worth of crops. This is the
    experiment docs/USAGE.md names as the one a fork should run; it is not the
    protocol the published tables used.

    Deterministic for a given seed so separate scripts agree on the same split.
    """
    if split_by not in ("crop", "source"):
        raise ValueError(f"split_by must be 'crop' or 'source', got {split_by!r}")
    rng = random.Random(seed)
    by_label = {}
    for i, (_, label) in enumerate(samples):
        by_label.setdefault(label, []).append(i)

    if split_by == "crop":
        train_idx, test_idx = [], []
        for label, idxs in sorted(by_label.items()):
            idxs = idxs[:]
            rng.shuffle(idxs)
            k = round(len(idxs) * test_frac)
            test_idx.extend(idxs[:k])
            train_idx.extend(idxs[k:])
        return sorted(train_idx), sorted(test_idx)

    # Source-grouped: a frame is assigned as a whole, across every class it
    # contributes to, so no source frame can appear on both sides. One frame
    # often carries crops of several classes, so grouping per class instead
    # would still let a frame straddle the split.
    groups = {}
    for i, (path, label) in enumerate(samples):
        groups.setdefault(source_key(path), []).append(i)
    counts_of = {}
    for key, members in groups.items():
        c = {}
        for i in members:
            c[samples[i][1]] = c.get(samples[i][1], 0) + 1
        counts_of[key] = c
    total = {label: len(idxs) for label, idxs in by_label.items()}
    target = {label: round(n * test_frac) for label, n in total.items()}
    taken = {label: 0 for label in by_label}

    keys = sorted(groups)
    rng.shuffle(keys)
    in_test = set()
    # Pass 1. Take a frame when it does more good than harm: more of its crops
    # fill a class still short of its target than overshoot one already met.
    for key in keys:
        counts = counts_of[key]
        useful = sum(min(n, max(0, target[label] - taken[label]))
                     for label, n in counts.items())
        if useful * 2 >= len(groups[key]):
            in_test.add(key)
            for label, n in counts.items():
                taken[label] += n

    # Pass 2. A class whose crops sit in a few large frames can be shut out of
    # the test half entirely by pass 1: every one of its frames overshoots the
    # target on its own, so none of them ever looks worth taking. Give each such
    # class the frame that overshoots least, so no class silently ends up with
    # no test crops at all.
    for label in sorted(by_label):
        if target[label] < 1 or taken[label] > 0:
            continue
        candidates = [k for k in keys if k not in in_test and counts_of[k].get(label)]
        if not candidates:
            continue
        best = min(candidates, key=lambda k: (counts_of[k][label], k))
        in_test.add(best)
        for lab, n in counts_of[best].items():
            taken[lab] += n

    train_idx, test_idx = [], []
    for key in keys:
        (test_idx if key in in_test else train_idx).extend(groups[key])

    # A class whose crops all come from source frames that the other side needs
    # cannot be represented on both sides at all - no frame-level split of this
    # data exists at this test_frac. Say so instead of returning a split where
    # that class silently has no test images, or no training images.
    for label in sorted(by_label):
        if total[label] < 2:
            continue
        if taken[label] == 0 or taken[label] == total[label]:
            side = "test" if taken[label] == 0 else "training"
            frames = sorted({source_key(samples[i][0]) for i in by_label[label]})
            raise ValueError(
                f"class label {label} has {total[label]} crops from only "
                f"{len(frames)} source frame(s), so a source-grouped split at "
                f"test_frac={test_frac} leaves it with no {side} images. Its crops "
                f"are too concentrated in single frames for split_by='source'; use "
                f"split_by='crop', change test_frac, or add more source frames.")
    return sorted(train_idx), sorted(test_idx)


def sample_named_subset(dataset, per_class_counts, pool_idx, seed):
    """Sample {class_name: count} images out of `pool_idx`.

    Raises rather than silently doing something else: an unknown class name is a
    KeyError, and asking for more images than the pool holds is a ValueError.
    Both would otherwise quietly produce a wrong experiment.
    """
    name_to_label = {name: i for i, name in enumerate(dataset.classes)}
    unknown = sorted(name for name in per_class_counts if name not in name_to_label)
    if unknown:
        raise KeyError(
            f"unknown class name(s) {unknown}; dataset classes are {dataset.classes}")

    by_label = {}
    for i in pool_idx:
        by_label.setdefault(dataset.samples[i][1], []).append(i)

    rng = random.Random(seed)
    keep = []
    for name in sorted(per_class_counts):
        want = int(per_class_counts[name])
        available = by_label.get(name_to_label[name], [])
        if want > len(available):
            raise ValueError(
                f"class '{name}': requested {want} images but only {len(available)} "
                f"are available in the pool")
        if want:
            keep.extend(rng.sample(available, want))
    return sorted(keep)


def count_by_class(dataset, indices):
    """{class_name: count} for a list of dataset indices."""
    out = {name: 0 for name in dataset.classes}
    for i in indices:
        out[dataset.classes[dataset.samples[i][1]]] += 1
    return out


def balanced_sample_weights(labels):
    """One resampling weight per sample, proportional to 1/N_class.

    This is the journal paper's data-balancing strategy (its Table 3 lists
    WeightedRandomSampler), whose stated purpose is that "each training batch
    possesses a balanced class distribution", so rare defects contribute to every
    gradient update instead of to a few.

    Capping per-class counts with --subset does not achieve that. A 40-image
    minority class inside a 1090-image subset is still only 3.7% of draws, so at
    batch_size=8 most minibatches contain no minority sample at all.

    The shape is one weight per *sample* rather than per class so the result can be
    handed straight to torch.utils.data.WeightedRandomSampler.
    """
    counts = Counter(labels)
    return [1.0 / counts[label] for label in labels]


def parse_name_counts(text):
    """Parse 'pore=40,deposit=150' into {'pore': 40, 'deposit': 150}.

    Shared by every script that takes per-class counts, so class identity is
    always expressed by name and validated against the dataset in use.
    """
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"expected 'NAME=count', got {part!r}")
        name, value = part.split("=", 1)
        name = name.strip()
        if name in out:
            raise ValueError(f"class '{name}' appears more than once in {text!r}")
        out[name] = int(value)
    return out


def build_loader(data_root, img_size=224, batch_size=8, num_workers=4, channels=1,
                 fft_denoise=False, fft_cutoff=0.25, shuffle=True, drop_last=True,
                 indices=None):
    """Build (dataset, loader). `indices` restricts the loader to a subset."""
    ds = ClassFolderDataset(data_root, img_size, channels, fft_denoise, fft_cutoff)
    target = torch.utils.data.Subset(ds, indices) if indices is not None else ds
    # Workers must persist across epochs: the per-epoch worker teardown join
    # deadlocks on Windows (observed 2026-09-06: epoch teardown stalled for
    # 30+ minutes with spawn-based workers).
    loader = DataLoader(target, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=drop_last, pin_memory=True,
                        persistent_workers=num_workers > 0)
    return ds, loader
