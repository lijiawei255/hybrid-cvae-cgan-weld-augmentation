# -*- coding: utf-8 -*-
"""Plot the seed-aggregated filling-rate curve against single-seed reference arms.

Reads only sweep_metrics.csv files, so it needs neither a GPU, a checkpoint nor
the dataset: --seed_sweep is repeated once per seed of the measured arm,
--reference takes "label=path" pairs for the curves it is compared against, and
--exclude drops an individual "path=ratio" arm whose classifier run is invalid.

    python src/make_multiseed_figure.py \
      --seed_sweep runs/sweep_paper2_s42/sweep_metrics.csv \
      --seed_sweep runs/sweep_paper2_s43/sweep_metrics.csv \
      --seed_sweep runs/sweep_paper2_s44/sweep_metrics.csv \
      --exclude "runs/sweep_paper2_s42/sweep_metrics.csv=0.25" \
      --reference "published v0.2.0 (seed 42)=runs/sweep/sweep_metrics.csv" \
      --reference "matched BatchNorm control (seed 42)=runs/sweep_ctrl_s42_clean/sweep_metrics.csv" \
      --minority pore --out results/filling_rate_multiseed.png
"""
import argparse
from pathlib import Path

from make_paper_figures import read_sweep_csv, save_multi_seed_filling_rate


def drop_arms(loaded, exclusions):
    """{path: sweep} minus the named 'path=ratio' arms, as a new dict.

    An arm whose classifier run is invalid has to be removed from the aggregate
    without removing that ratio from the other seeds, and an exclusion that
    matches nothing is an error rather than a no-op: silently keeping an invalid
    score would change the curve with no visible sign.
    """
    kept = {path: dict(sweep) for path, sweep in loaded.items()}
    for item in exclusions:
        path, sep, raw_ratio = item.partition("=")
        if not sep:
            raise SystemExit(f"--exclude {item!r} is not 'path=ratio'")
        if path not in kept:
            raise SystemExit(f"--exclude {item!r} names {path!r}, which is not one "
                             f"of the --seed_sweep files {sorted(kept)}")
        try:
            ratio = float(raw_ratio)
        except ValueError:
            raise SystemExit(f"--exclude {item!r} has a non-numeric ratio {raw_ratio!r}")
        if ratio not in kept[path]:
            raise SystemExit(f"--exclude {item!r}: {path} has no ratio {ratio}, only "
                             f"{sorted(kept[path])}")
        del kept[path][ratio]
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_sweep", action="append", required=True,
                    help="sweep_metrics.csv of one seed of the measured arm; repeat")
    ap.add_argument("--reference", action="append", default=[],
                    help="'label=path/to/sweep_metrics.csv'; repeat")
    ap.add_argument("--arm", action="append", default=[],
                    help="'label=path;path;...': a further seed-aggregated arm drawn "
                         "as its own mean and band; repeat")
    ap.add_argument("--label", default="GroupNorm + balanced sampler",
                    help="legend label of the --seed_sweep arm")
    ap.add_argument("--title", default=None, help="figure title override")
    ap.add_argument("--exclude", action="append", default=[],
                    help="'path=ratio' arm to drop as invalid; repeat")
    ap.add_argument("--minority", required=True,
                    help="class name whose per-class F1 gets its own panel")
    ap.add_argument("--out", default="results/filling_rate_multiseed.png")
    args = ap.parse_args()

    if len(args.seed_sweep) < 2:
        raise SystemExit("--seed_sweep given once; a seed-aggregated figure needs "
                         "at least two seeds, else use save_filling_rate_curve")

    loaded = {path: read_sweep_csv(path) for path in args.seed_sweep}
    if len(loaded) != len(args.seed_sweep):
        raise SystemExit("--seed_sweep lists the same file twice; that would weight "
                         "one seed double")
    loaded = drop_arms(loaded, args.exclude)
    seeds = list(loaded.values())

    references = {}
    for item in args.reference:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--reference {item!r} is not 'label=path'")
        references[label] = read_sweep_csv(path)

    extra_arms = {}
    for item in args.arm:
        label, _, paths = item.partition("=")
        files = [f for f in paths.split(";") if f]
        if not label or len(files) < 2:
            raise SystemExit(f"--arm {item!r} is not 'label=path;path[;path...]' "
                             f"with at least two seeds")
        extra_arms[label] = [read_sweep_csv(f) for f in files]

    metric = f"f1:{args.minority}"
    for path, sweep in loaded.items():
        missing = [r for r, m in sweep.items() if metric not in m]
        if missing:
            raise SystemExit(f"{path} has no {metric!r} at ratios {sorted(missing)}; "
                             f"--minority {args.minority!r} is probably not a class name")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_multi_seed_filling_rate(seeds, references, args.minority, out,
                                 arm_label=args.label, extra_arms=extra_arms,
                                 title=args.title)
    print(f"wrote {out} from {len(seeds)} seeds, {len(extra_arms)} further "
          f"seed-aggregated arms and {len(references)} reference arms")


if __name__ == "__main__":
    main()
