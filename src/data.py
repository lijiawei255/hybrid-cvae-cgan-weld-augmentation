# -*- coding: utf-8 -*-
"""Data loading: class-folder image dataset + class-imbalance simulation utility."""
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ClassFolderDataset(Dataset):
    """Expected layout:
        data_root/
          good/      *.jpg|png   (non-defective welds / melt pools)
          defect/    *.jpg|png   (defective)
    Number of classes is inferred from subdirectories; label = index of the
    sorted subdirectory names.
    """

    def __init__(self, data_root, img_size=128):
        self.root = Path(data_root)
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.samples = []
        for idx, cname in enumerate(self.classes):
            for fp in sorted((self.root / cname).glob("*")):
                if fp.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    self.samples.append((fp, idx))
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1], matches the Tanh output
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fp, label = self.samples[i]
        return self.tf(Image.open(fp).convert("RGB")), label


def make_imbalanced_subset(dataset, keep_ratios, seed=42):
    """Simulates the "small and imbalanced" setting studied in the paper.
    keep_ratios: dict {class_index: keep_fraction}, e.g. {0: 1.0, 1: 0.2}
    keeps only 20% of the defective class. Returns torch.utils.data.Subset.
    """
    import random
    rng = random.Random(seed)
    idx_by_class = {}
    for i, (_, y) in enumerate(dataset.samples):
        idx_by_class.setdefault(y, []).append(i)
    keep = []
    for y, idxs in idx_by_class.items():
        r = keep_ratios.get(y, 1.0)
        k = max(1, int(len(idxs) * r))
        keep.extend(rng.sample(idxs, k))
    return torch.utils.data.Subset(dataset, sorted(keep))


def build_loader(data_root, img_size=128, batch_size=64, num_workers=4, keep_ratios=None):
    ds = ClassFolderDataset(data_root, img_size)
    if keep_ratios is not None:
        ds = make_imbalanced_subset(ds, keep_ratios)
    # Workers must persist across epochs: the per-epoch worker teardown join
    # deadlocks on Windows (observed 2026-09-06: epoch teardown stalled for
    # 30+ minutes with spawn-based workers).
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=num_workers > 0)
    return ds, loader
