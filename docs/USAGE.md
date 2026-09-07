# Usage notes

The [README](../README.md) is the entry point. This file holds the longer
operating notes: journal-extension switches, porting to your own images,
repository layout, and the full limitations list. Calibration measurements
stay in [CALIBRATION.md](CALIBRATION.md). Contribution rules:
[CONTRIBUTING.md](../CONTRIBUTING.md).

Environment: Python 3.11 is the tested version (the one CI runs); other
versions are untested. First run downloads the VGG19 and InceptionV3 weights
(~640 MB total) and needs internet.

## Compute (order of magnitude)

On an RTX 4000 Ada Laptop (12 GB), a 70-epoch joint run on the paper-scale
subset (~1,100 images, 224x224, FID every epoch) finished in about 40
minutes. RGB LoHi-WELD at the same resolution is in the same ballpark, on
the order of **one hour** for the recommended generator. The five-ratio
classifier sweep (100 epochs each) is typically a few hours. Peak VRAM at
224 / batch 8 was about 1.5 GB in the grayscale timing table in
[CALIBRATION.md](CALIBRATION.md) section 4. The smoke test is minutes on
CPU or GPU.

## Journal-extension switches

The README Quickstart is the **recommended** LoHi-WELD configuration. Four
`train_joint.py` switches implement components that appear only in the
journal extension (MSSP 2026). All four default to the conference-paper
behaviour.

| switch | default | journal-extension value | status |
|---|---|---|---|
| `--d_norm` | `batch` | `group` | measured, recommended |
| `--weighted_sampler` | off | on | measured, recommended |
| `--kl_warmup` | off (constant beta) | `10,50` | implemented and unit-tested, **not training-validated** |
| `--monitor` | `val_loss` | - | only needed together with `--kl_warmup` |

- `--d_norm group` replaces the discriminator's BatchNorm with GroupNorm. At
  `--batch_size 8` BatchNorm makes a sample's score depend on its seven batch
  neighbours. Measured effect: the discriminator's hinge loss moves from a
  median of 0.819 to 1.998, matching the journal paper's report that it
  "settles to an approximately constant value around 2.0".
- `--weighted_sampler` draws batches with weight proportional to `1/N_class`.
  `--subset` caps counts but does not balance batches.
- `--kl_warmup ZERO,RAMP` needs the journal paper's 200-epoch protocol; it is
  deliberately not trained inside a 70-epoch run.
- `--monitor val_recon` is required under `--kl_warmup`.

Enabling `--d_norm group --weighted_sampler` is the configuration in
[CALIBRATION.md](CALIBRATION.md) section 9. Read the four things that
measurement does **not** establish before reusing the numbers.

## Public substitute data

Headline runs use LoHi-WELD, prepared with the command in the README Data
section. The 224x224 canvas overstates true resolution: the median source
box is about 47 px, so almost every crop is an interpolation upsample
(measured in [CALIBRATION.md](CALIBRATION.md) section 4). RIAWELC is
historical only; do not quote its v0.1.0 numbers.

## Using your own dataset

Point `--data_root` at a folder of class subfolders. Nothing else is
dataset-specific:

- Class identity is expressed **by name** everywhere (`--subset`, `--counts`).
  An unknown or misspelled class raises.
- Choose `--subset` counts from your own per-class counts. Pick a minority
  count low enough that the task is not already solved, and set the majority
  count to the `N_max` you want balance-to-max to fill up to.
- `--channels 3` for colour; `--channels 1` is refused on RGB files.
  `--img_size` must be divisible by 16.
- `--fft_denoise` is the paper's FFT low-pass. It is off by default and a
  near no-op on the datasets used here (see [CALIBRATION.md](CALIBRATION.md)
  section 3). If you turn it on, pass it to `eval_fid.py` as well.
- `--kl_weight` must be rescaled if you change the loss normalisation. See
  `models.cvae_loss` and `CHANGELOG.md`.

### Porting checklist

1. One subfolder per class under `--data_root`.
2. Class names everywhere (`--subset`, `--counts`, generated-pool metadata).
3. `--channels 3` for RGB, `--channels 1` for grayscale; `--img_size` divisible by 16.
4. Match `--subset` to your minority count and desired `N_max`.
5. If you enable `--fft_denoise`, pass it to both `train_joint.py` and `eval_fid.py`.
6. Retrain the classifier sweep with the same `--subset`, `--seed`, and `--test_frac`.

## Repository layout

```
src/models.py             encoder, decoder (= generator), discriminator, losses
src/data.py               class-folder loader, FFT denoising, leakage-free splits
src/augment.py            balance-to-max filling-rate protocol
src/prepare_yolo_crops.py detection-format datasets -> class-folder crops
src/train_joint.py        joint CVAE-CGAN training (the paper's single phase)
src/generate.py           class-conditional sampling, counts by class name
src/eval_fid.py           FID via pytorch-fid (default) or --fid_backend legacy
src/train_classifier.py   downstream filling-rate sweep (from-scratch ResNet-18)
src/make_paper_figures.py training curves, sensitivity curve, t-SNE, confusions
src/make_multiseed_figure.py seed-aggregated sensitivity curve, from CSVs alone
src/make_comparison.py    real-vs-generated comparison grid
src/make_class_figures.py per-class close-ups with per-class FID
src/smoke_test.py         unit checks + end-to-end run on synthetic data
```

## Limitations

Read this before quoting any number from this repo. The README "Scope at a
glance" is the short form.

- **Different data, same modality.** Proprietary melt-pool imagery versus
  public LoHi-WELD weld beads. No result here reproduces the papers' numbers.
- **The "small and imbalanced" condition is simulated** via `--subset`.
- **The downstream classifier is not the papers' classifier.** ResNet-18 on
  single frames, not LSTM/GRU on 21-frame sequences. Published tables used
  `--selection final`; the current default is `--selection best_val`.
- **FID is a trend indicator.** Default backend is pytorch-fid; published
  in-repo numbers used `--fid_backend legacy`. Do not mix scales or compare
  to the papers.
- **Four hyperparameters were calibrated, not copied.** See
  [CALIBRATION.md](CALIBRATION.md).
- **Mostly single seed**, except the three-seed GroupNorm arm.
- **Leakage-free by construction.** The generator never sees the classifier's
  held-out test images.

### What this repo deliberately does not reproduce

- Journal physics-guided losses (no thermal signal on static weld images).
- 21-frame temporal augmentation and sequence classifiers.
- The paper's FFT step in the reported runs (implemented, off by default).
- Conference BCE adversarial objective (hinge + spectral norm instead).
- Paper residual encoder conv body (plain 4x4 stride-2; drop-in replaceable).
- DR/MDR (undefined on LoHi-WELD: no non-defect class).
