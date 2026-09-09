# -*- coding: utf-8 -*-
"""Measure how much source-frame identity is shared across a split.

This reproduces the audit reported in docs/CALIBRATION.md section 9. The crops
this repo trains on are cut from a smaller number of source frames
(prepare_yolo_crops.py writes ``<encoded-source-path>_<k:03d>.png``), so a
stratified split of individual *crops* can put two crops of the same frame on
opposite sides. The default ``--split_by crop`` protocol does exactly that, and
this script quantifies it rather than leaving the reader to take it on trust.

It re-derives the split with the same ``make_splits`` / ``sample_named_subset``
code the training scripts call, so the numbers describe the real experiment and
not a re-implementation of it. It reads only filenames - no image is opened, no
GPU or checkpoint is needed.

Reported per seed:
  * test crops whose source frame also feeds the train pool;
  * test crops whose source frame also feeds the --subset training set itself;
  * test source frames that also feed the train pool.

Under ``--split_by source`` every one of those is 0 by construction, which is
what the flag is for.
"""
import argparse

from data import (ClassFolderDataset, count_by_class, make_splits,
                  parse_name_counts, sample_named_subset, source_key)


def overlap_report(ds, test_frac, seed, subset_counts, split_by):
    """One row of the table: counts and shares for a single seed."""
    train_pool, test_idx = make_splits(ds.samples, test_frac, seed, split_by)
    subset_idx = sample_named_subset(ds, subset_counts, train_pool, seed)

    frames_of = lambda idxs: {source_key(ds.samples[i][0]) for i in idxs}
    pool_frames = frames_of(train_pool)
    subset_frames = frames_of(subset_idx)
    test_frames = frames_of(test_idx)

    shared_pool = sum(source_key(ds.samples[i][0]) in pool_frames for i in test_idx)
    shared_subset = sum(source_key(ds.samples[i][0]) in subset_frames for i in test_idx)
    return {
        "seed": seed,
        "n_test": len(test_idx),
        "n_train_pool": len(train_pool),
        "n_subset": len(subset_idx),
        "test_crops_sharing_pool_frame": shared_pool,
        "test_crops_sharing_subset_frame": shared_subset,
        "test_frames": len(test_frames),
        "test_frames_also_in_pool": len(test_frames & pool_frames),
        "test_counts": count_by_class(ds, test_idx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--subset", required=True,
                    help="the training subset the sweep uses, e.g. "
                         "'pore=40,deposit=150,discontinuity=300,stain=600'")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--split_by", choices=("crop", "source"), default="crop")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--channels", type=int, default=3)
    args = ap.parse_args()

    ds = ClassFolderDataset(args.data_root, args.img_size, args.channels)
    subset_counts = parse_name_counts(args.subset)
    frames = {source_key(path) for path, _ in ds.samples}
    per_frame = len(ds.samples) / len(frames) if frames else 0
    print(f"{len(ds.samples)} crops from {len(frames)} source frames "
          f"({per_frame:.1f} crops per frame on average); split_by={args.split_by}")
    print()
    header = (f"{'seed':>5} {'test':>6} {'sharing a train-pool frame':>28} "
              f"{'sharing a --subset frame':>26} {'test frames also in pool':>26}")
    print(header)
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        r = overlap_report(ds, args.test_frac, seed, subset_counts, args.split_by)
        pool_pct = 100 * r["test_crops_sharing_pool_frame"] / max(r["n_test"], 1)
        sub_pct = 100 * r["test_crops_sharing_subset_frame"] / max(r["n_test"], 1)
        frame_pct = 100 * r["test_frames_also_in_pool"] / max(r["test_frames"], 1)
        print(f"{r['seed']:>5} {r['n_test']:>6} "
              f"{r['test_crops_sharing_pool_frame']:>17} ({pool_pct:5.1f}%) "
              f"{r['test_crops_sharing_subset_frame']:>15} ({sub_pct:5.1f}%) "
              f"{r['test_frames_also_in_pool']:>15} ({frame_pct:5.1f}%)")
        print(f"      test per class: {r['test_counts']}; train pool {r['n_train_pool']}, "
              f"subset {r['n_subset']}")


if __name__ == "__main__":
    main()
