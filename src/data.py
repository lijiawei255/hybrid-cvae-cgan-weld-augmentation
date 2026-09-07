# -*- coding: utf-8 -*-
"""Data loading, preprocessing and leakage-free split helpers.

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
                return Image.open(fp).mode
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


class _FFTLowPass:
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
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.samples = []
        for idx, cname in enumerate(self.classes):
            for fp in sorted((self.root / cname).glob("*")):
                if fp.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((fp, idx))

        steps = [transforms.Resize((img_size, img_size))]
        if fft_denoise:
            steps.append(_FFTLowPass(fft_cutoff))
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


def make_splits(samples, test_frac=0.2, seed=42):
    """Stratified (train_pool, test) index split.

    The test half is real-only and held out from *both* the generator and the
    classifier, which is what makes the downstream gain measurement honest.
    Deterministic for a given seed so separate scripts agree on the same split.
    """
    rng = random.Random(seed)
    by_label = {}
    for i, (_, label) in enumerate(samples):
        by_label.setdefault(label, []).append(i)
    train_idx, test_idx = [], []
    for label, idxs in sorted(by_label.items()):
        idxs = idxs[:]
        rng.shuffle(idxs)
        k = round(len(idxs) * test_frac)
        test_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])
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
