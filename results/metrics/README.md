# Published run metrics

Raw metric exports backing the published tables and figures. These are
verbatim copies (SHA-256 verified at publication time) of files produced by
the commands recorded in [`docs/CALIBRATION.md`](../../docs/CALIBRATION.md)
section 9; `runs/` itself stays git-ignored, so this directory is the
published record. Subdirectory names match the run names in those commands,
so replacing `runs/` with `results/metrics/` in them reproduces every figure
that needs neither the dataset nor a GPU - including
`src/make_multiseed_figure.py` with its `--exclude` and `--reference` inputs.

Three file kinds:

* `sweep_metrics.csv` - long-format `ratio,metric,value` exports written by
  `src/train_classifier.py`: per-class precision/recall/F1/support, accuracy,
  macro/weighted F1, and the real/synthetic training-set sizes per ratio.
  Every sweep was scored at the final epoch - the `--selection final`
  protocol behind the published tables (the v0.2.0-era `sweep` predates the
  flag and used the same behaviour).
* `history.csv` - per-epoch generator training logs written by
  `src/train_joint.py`. The v0.2.0-era `joint_lohi` log predates the `beta`
  column that v0.4.0 added for the KL-annealing schedule. **Every `fid` column
  here is on the `--fid_backend legacy` (torchvision) scale**: all of these runs
  predate v0.4.0's switch to pytorch-fid as the default, and the two scales are
  not comparable. Do not plot them on one axis.
* `cm_r*.npy` - the per-ratio confusion matrices `src/train_classifier.py`
  writes, one 4x4 integer array per filling rate, rows = true class in the
  dataset's sorted class order (deposit, discontinuity, pore, stain). They are
  what `src/make_paper_figures.py --cm_dir` reads, so
  `results/confusion_matrices.png` redraws from this directory alone. Each one
  reproduces its own row of the neighbouring `sweep_metrics.csv`.

`SHA256SUMS` lists every file here. Verify with `sha256sum -c SHA256SUMS` from
this directory.

## Redrawing the four CSV-only figures

`src/make_paper_figures.py` writes the class-distribution, training-curve,
filling-rate and confusion-matrix figures without a checkpoint, a GPU or the
dataset. Omit `--ckpt` and `--data_root` and it skips the model-dependent
figures and uses `--subset` for the class names and counts:

```bash
python src/make_paper_figures.py \
  --history results/metrics/joint_lohi/history.csv \
  --sweep results/metrics/sweep/sweep_metrics.csv \
  --cm_dir results/metrics/sweep \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --out_dir results
```

Checked against the committed images: `class_distribution.png`,
`training_curves.png` and `filling_rate_curve.png` come back byte-identical.
`confusion_matrices.png` differs in 0.35% of its pixels, all of them inside one
band of title text; the matrices themselves are identical, and the difference is
font rendering drift between matplotlib versions.

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

Runs added in v0.5.2 (`docs/CALIBRATION.md` section 11). Nothing above was
edited or replaced; these are new directories. The generator histories keep
the legacy FID scale; the sweeps were scored with `--selection final` unless
the name says `_bestval`.

| File | Run | Backs |
| --- | --- | --- |
| `fid_cross_scale.csv` | `src/eval_fid.py` and `src/make_class_figures.py` on the six published pools, both backends | section 11.1, the two FID scales side by side |
| `bce_nosn_g0.1/history.csv`, `bce_nosn_g1.0/history.csv` | conference objective (`--adv_loss bce --no_d_spectral_norm`), 20 epochs, gamma 0.1 and 1.0 | section 11.2 |
| `paper2_gn_only/history.csv` | GroupNorm discriminator without the weighted sampler, seed 42 | section 11.3 generator table |
| `sweep_gn_only/` | its sweep | section 11.3 downstream table |
| `probe_eq_g0.1_s43/history.csv`, `probe_eq_g0.1_s44/history.csv` | matched BatchNorm control, seeds 43 and 44 | section 11.4 generator table |
| `sweep_ctrl_s43/`, `sweep_ctrl_s44/` | their sweeps | section 11.4; the control band in `filling_rate_multiseed_v052.png` |
| `paper2_src_s42/`, `_s43/`, `_s44/` (`history.csv`) | recommended arm under `--split_by source`, `--val_per_class 190` | section 11.5 generator paragraph |
| `sweep_src_s42/`, `_s43/`, `_s44/` | their sweeps, `--split_by source` | section 11.5; the source-split band in `filling_rate_multiseed_v052.png`; the README v0.5.2 blockquote |
| `sweep_paper2_s42_bestval/`, `_s43_bestval/`, `_s44_bestval/`, `sweep_ctrl_s42_bestval/`, `sweep_v020_bestval/` | the five published sweeps rescored with `--selection best_val` from their own pools | section 11.6 |

`results/filling_rate_multiseed_v052.png` is drawn from the `sweep_*`
directories above plus the section 9 ones; the command is in section 11.
