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
| `--g_norm` | `batch` | `group` | expressible, **not measured** |
| `--kl_warmup` | off (constant beta) | `10,50` | implemented and unit-tested, **not training-validated** |
| `--monitor` | `val_loss` | - | only needed together with `--kl_warmup` |

Two further switches exist for the **conference** paper's original objective,
which the default configuration replaces:

| switch | default | conference-paper value | status |
|---|---|---|---|
| `--adv_loss` | `hinge` | `bce` | expressible, **measured to fail here** |
| `--no_d_spectral_norm` | off (SN on) | on (no SN) | expressible, **measured to fail here** |

And one that is neither paper's, because neither states a split protocol:

| switch | default | status |
|---|---|---|
| `--split_by` | `crop` (the published protocol) | `source` isolates source frames; see the limitations below |

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
- `--g_norm group` puts GroupNorm in the encoder and decoder too. The journal
  paper's Table 3 specifies GroupNorm for every network, so full fidelity to it
  needs this as well as `--d_norm group`. Unlike `--d_norm`, no run in this
  repository has measured it, and every published checkpoint was trained with
  BatchNorm generators. A checkpoint records which it used, so `generate.py`
  rebuilds the right one automatically.
- `--split_by source` needs each class to have crops from enough distinct source
  frames. A class whose crops all come from one frame cannot appear on both
  sides of a frame-level split at all, and one concentrated in a few large
  frames can only be stratified to within a whole frame. The split raises with
  the class named rather than returning a split where that class has no test or
  no training images. On LoHi-WELD (8,012 crops from 1,022 frames) every class
  lands within 19.4-21.7% at `--test_frac 0.2`.
- `--adv_loss bce` with `--no_d_spectral_norm` is the conference paper's own
  adversarial objective. It is expressible so that the argument in
  [CALIBRATION.md](CALIBRATION.md) section 6 can be checked rather than taken on
  trust: BCE's generator term is unbounded above, and at this data scale a
  confident discriminator drove it to roughly 50x the reconstruction term while
  FID rose. Expect it to fail; that is the point of being able to run it.

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
6. Retrain the classifier sweep with the same `--subset`, `--seed`, and
   `--test_frac`, against a pool generated from the same **unchanged** crop
   data. `verify_generation_meta` checks the split configuration, not data
   identity - regenerate the pool after re-cropping, adding or renaming
   images, because those change the actual split without changing the
   recorded flags.

## Repository layout

```
src/models.py             encoder, decoder (= generator), discriminator, losses
src/data.py               class-folder loader, FFT denoising, crop-level splits
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
- **Crop-level split, not source-isolated.** The generator and the classifier
  share one stratified random split of individual crops: crop indices never
  cross pools and the test set stays real-only, but crops cut from the same
  source frame can sit on both sides. Measured on the published
  seed-42/43/44 splits of the 8,012 crops (1,022 source frames, median 8
  crops per frame), 99.8-100% of test crops come from source frames that
  also contribute crops to the train pool, and 63-67% share a source frame
  with the 1,090-crop training subset itself (method and table in
  [CALIBRATION.md](CALIBRATION.md) section 9). Absolute downstream numbers
  are therefore optimistic about generalisation to unseen sources; the
  filling-rate arms share the same test set and baseline subset, so
  within-run comparisons are less affected. Nothing here evidences
  weld-level generalisation. `--split_by source` now implements the
  source-frame-grouped split on both `train_joint.py` and
  `train_classifier.py`, and `src/measure_split_overlap.py` verifies that it
  brings every overlap column to 0%; CALIBRATION section 11 reports what the
  downstream numbers do under it. "Source frame" is one LoHi-WELD original
  image; grouping by weld would be coarser still. A generated pool records its
  own `split_by`, and a sweep refuses a pool whose value disagrees with its own.
- **`--selection best_val` selects on a pool the generator also saw.** The
  classifier's leftover validation pool is every train-pool image outside
  `--subset`, and the generator's own validation and FID reference set (200 per
  class) is drawn from that same leftover. So under the current default the
  classifier picks its reported epoch using images the generator was early
  stopped on. The held-out test set is untouched, so this is not test leakage,
  but it is a soft optimistic bias that the published `--selection final`
  tables do not carry.
- **Diagnostic frequency is part of the training configuration.** FID checks
  and sample grids draw from the same RNG stream as training, so changing
  `--fid_every` or `--sample_every` changes the subsequent training random
  stream. Exact reproduction keeps the diagnostic settings identical too.

### What this repo deliberately does not reproduce

- Journal physics-guided losses (no thermal signal on static weld images).
- 21-frame temporal augmentation and sequence classifiers.
- The paper's FFT step in the reported runs (implemented, off by default).
- Conference BCE adversarial objective (hinge + spectral norm instead).
- Journal projection discriminator, Eq. 4 (the discriminator keeps the
  conference paper's spatial-map label conditioning, with hinge +
  spectral normalisation on top).
- Journal free-bits KL form, Eq. 9 `max(0, KL - delta)` (plain KL with an
  optional beta ramp instead; the paper does not give `delta`).
- Journal Charbonnier reconstruction loss (Eq. 6, weight 3.0) and its Tanh
  decoder over [-1, 1] inputs; this repo follows the conference paper's MSE and
  sigmoid over [0, 1].
- Journal encoder channel widths (32-64-128-256); this repo uses 64-128-256-512
  via `--base_ch 64`.
- Paper residual encoder conv body (plain 4x4 stride-2; drop-in replaceable),
  and the encoder's flatten into the latent heads, replaced by global average
  pooling as a measured stability fix (CALIBRATION section 7, bug 1).
- Spectral normalisation on the first three conv blocks only; this repo
  normalises all four plus the dense head.
- The perceptual loss as a weighted sum over several VGG19 layers (Eq. 7); this
  repo distances the single final `features` output. Neither paper names the
  layers individually.
- The papers' grayscale preprocessing, on the primary runs (`--channels 3`;
  `--channels 1` reproduces it).
- DR/MDR (undefined on LoHi-WELD: no non-defect class).

Choices this repo had to make because the papers do not specify them, listed so
they are not mistaken for reproduction: the classifier's hyperparameters (batch
32, 100 epochs, Adam 1e-3, no input normalisation and no classifier-side
augmentation); the train/validation/test proportions; the filling-rate grid
`0.0,0.25,0.5,0.75,1.0`, which does **not** sample the 0.1-0.2 region the journal
extension reports as degrading; the GroupNorm group count (32); and that the
adversarial term is taken on fresh prior samples only, never on reconstructions.
The journal's "traditional augmentation" baseline arm is also not implemented,
though unlike the physics losses nothing prevents it.
