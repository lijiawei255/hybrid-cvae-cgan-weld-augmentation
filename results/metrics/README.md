# Published run metrics

Raw metric exports backing the published tables and figures. These are
verbatim copies (SHA-256 verified at publication time) of files produced by
the commands recorded in [`docs/CALIBRATION.md`](../../docs/CALIBRATION.md)
section 9; `runs/` itself stays git-ignored, so this directory is the
published record. Subdirectory names match the run names in those commands,
so replacing `runs/` with `results/metrics/` in them reproduces every figure
that needs neither the dataset nor a GPU - including
`src/make_multiseed_figure.py` with its `--exclude` and `--reference` inputs.

Two file kinds:

* `sweep_metrics.csv` - long-format `ratio,metric,value` exports written by
  `src/train_classifier.py`: per-class precision/recall/F1/support, accuracy,
  macro/weighted F1, and the real/synthetic training-set sizes per ratio.
  Every sweep was scored at the final epoch - the `--selection final`
  protocol behind the published tables (the v0.2.0-era `sweep` predates the
  flag and used the same behaviour).
* `history.csv` - per-epoch generator training logs written by
  `src/train_joint.py`. The v0.2.0-era `joint_lohi` log predates the `beta`
  column that v0.4.0 added for the KL-annealing schedule.

| File | Run | Backs |
| --- | --- | --- |
| `sweep_paper2_s42/sweep_metrics.csv` | classifier sweep, seed 42 | the multi-seed figure; the excluded r=0.25 arm (macro-F1 0.5868) discussed in CALIBRATION sections 9-10 |
| `sweep_paper2_s43/sweep_metrics.csv` | classifier sweep, seed 43 | the multi-seed figure; the three-seed aggregates in CALIBRATION section 9 |
| `sweep_paper2_s44/sweep_metrics.csv` | classifier sweep, seed 44 | the multi-seed figure; the three-seed aggregates in CALIBRATION section 9 |
| `sweep_ctrl_s42_clean/sweep_metrics.csv` | matched latent-128 BatchNorm control, seed 42 | the control curve in the multi-seed figure and the control column in CALIBRATION section 9 |
| `sweep/sweep_metrics.csv` | the published v0.2.0 single-seed sweep | the README single-seed table and the "v0.2.0 published" column in CALIBRATION section 9 |
| `paper2_gn_wrs/history.csv` | generator training, seed 42, GroupNorm + weighted sampler | the discriminator-equilibrium median (1.998) in CALIBRATION section 9; the section 9 reproduce command |
| `paper2_gn_wrs_s43/history.csv` | generator training, seed 43 | the three-seed protocol in CALIBRATION section 9 |
| `paper2_gn_wrs_s44/history.csv` | generator training, seed 44 | the three-seed protocol in CALIBRATION section 9 |
| `joint_lohi/history.csv` | the published v0.2.0 generator | `training_curves.png`; the FID comparisons in CALIBRATION sections 5 and 9; the "FID 371 -> ~200" observation quoted in the `--lr_d` help text |
| `probe_eq_g0.1/history.csv` | latent-128 BatchNorm generator probe | the BatchNorm discriminator-equilibrium median (0.819) and the minimum-FID comparison in CALIBRATION section 9 |

Nothing here was edited or retrained; the numbers are the historical
snapshots the docs already quote.
