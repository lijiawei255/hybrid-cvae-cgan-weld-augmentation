# Changelog

Deliberate deviations from the reference skeleton shipped in the reproduction
guide (its Appendix A), and from the papers themselves, are recorded here with
the reason for each.

Measurements behind the four calibrated hyperparameters - the KL weight, the
discriminator learning rate, the FFT denoising step and the resolution - live in
[`docs/CALIBRATION.md`](docs/CALIBRATION.md), with the commands that reproduce
them. This file keeps only the conclusions.

## [Unreleased]

### Added

- **Bilingual README.** `README_zh-CN.md` mirrors the English README for
  Chinese-speaking readers; both files carry a navigation link at the top and a
  sync note stating the English version is normative and the Chinese translation
  is maintained in sync with it.
- **Explicit out-of-scope statement** in `README.md` ("Scope: what this repo
  deliberately does NOT reproduce"): the journal extension's physics-guided
  generation and its temporal/sequence augmentation are not reproduced (static
  weld images carry no thermal or temporal signal to condition on); the reported
  runs do not include the paper's FFT denoising step; the papers' DR/MDR metrics
  are undefined on LoHi-WELD because it has no non-defect class.
- **Structure-vs-objective clarification**: training structure follows the
  conference paper, the adversarial objective follows the journal extension.
- **Encoder conv-body deviation stated explicitly** (plain 4x4 stride-2 blocks vs
  the papers' 3x3 residual blocks; discriminator differs only by omitted
  dropout), with the note that topology is not a claimed contribution of either
  paper and the module signatures make swapping drop-in.
- **Implementation-component citations** in `CITATION.cff`: Heusel et al. (FID),
  van der Maaten & Hinton (t-SNE), Szegedy et al. (InceptionV3), He et al.
  (ResNet-18), Simonyan & Zisserman (VGG19), grouped separately from the
  reproduced papers and datasets.
- **LoHi-WELD provenance note** in `DATA_SOURCES.md`: its repository is built on
  WongKinYiu's YOLOv7; only the dataset and annotations are used here.
- **Four switches for components that appear only in the journal extension**,
  all defaulting to the conference-paper behaviour so the tagged v0.2.0 runs stay
  reproducible: `--d_norm {batch,group}` (discriminator normalisation),
  `--weighted_sampler` (`WeightedRandomSampler` at P ~ 1/N_class),
  `--kl_warmup ZERO,RAMP` (KL annealing) and `--monitor {val_loss,val_recon}`
  (best-checkpoint metric). The first two are measured and recommended; KL
  annealing is implemented and unit-tested but deliberately **not
  training-validated**, because it needs the journal paper's 200-epoch protocol
  and would leave only 10 epochs at full beta inside a 70-epoch run. Documented
  in both READMEs.
- **`docs/CALIBRATION.md` sections 8-10.** Section 8 records why the journal
  paper's headline 96.79% is not a target for this repo and must not be quoted as
  one. Section 9 records the measured journal-extension arm, including that its
  discriminator hinge loss moves to a median of 1.998 - matching the paper's
  "approximately constant value around 2.0" - and including two explicit
  qualifications: that arm changed two variables at once, and its FID got worse
  while its reconstruction improved. Section 10 records that the downstream
  classifier is not bit-reproducible on GPU and scores the final epoch.
- **Section 1 addendum withdrawing an earlier explanation.** The claim that a
  low-information latent is why our generations lack fidelity is withdrawn as
  unsupported: `latent_dim x kl` is already at the paper's reported magnitude
  under either reading of its normalisation.
- **Downstream sweep for the journal-extension configuration** (seed 42,
  `runs/sweep_paper2_s42`), which **reverses the published v0.2.0 finding that
  balance-to-max does not transfer**. At r=1.0 macro-F1 is 0.7168 against this
  arm's own 0.6848 real-only baseline (+0.032) where the published curve had
  0.6418 against 0.6936 (-0.052, its worst ratio); pore F1 is 0.5053 against the
  published 0.3871. Across the four ratios that completed cleanly the curve is
  monotone increasing, the shape the journal paper reports. Recorded with four
  caveats in `docs/CALIBRATION.md` section 9 and behind a superseded banner in
  both READMEs rather than by editing the published table: the new configuration
  differs from v0.2.0 in five respects at once, the matched latent-128 BatchNorm
  control has not been swept, its r=0.25 arm was invalidated by a final-epoch
  loss spike, and it is a single seed. Best FID is effectively unchanged against
  v0.2.0 (215.99 vs 216.87) even though best `val_recon` improved 2.8x (0.0177 vs
  0.0494), so "better generator" holds on the reconstruction axis only.

### Fixed

- **Confusion-matrix figure overlap.** The shared colorbar was created before
  `subplots_adjust`, so its position was computed from the pre-adjustment layout
  and then the axes moved underneath it, covering the right matrix's third column
  and title. The right-hand gutter is now reserved first (`right=0.80`) and the
  colorbar created afterwards; `results/confusion_matrices.png` was redrawn.
- **`.gitignore` only covered the exact name `generated/`.** An alternative
  generated-image pool (`generated_paper2/`, 1,312 synthetic images) showed up as
  untracked and could have been staged by accident. Widened to `generated*/`.
  `results/` remains tracked - committed result figures are intentional.
- **Off-by-one held-out test-set count.** Both READMEs said 1,602 images; the
  logged per-class supports (239 + 595 + 61 + 708) sum to 1,603.
- **Stale `--fft_denoise` claim.** The README said it was "left untested on
  LoHi-WELD". It has since been measured there: 99.2-99.9% of the crops' spectral
  energy sits inside normalised radius 0.1, so at `cutoff = 0.25` the filter
  retains 0.0% of the mid-band and changes the image by an MSE of 3-14 out of a
  possible 65,025. It mostly smooths the *target*, so any apparent gain must be
  read as "the task got easier", not "the method got stronger"
  (`docs/CALIBRATION.md` section 3).
- **The README's noise caveat was a guess.** "Treat differences under ~0.02
  macro-F1 as noise" is replaced with the measured spread from repeating an
  unchanged arm at the same seed: pore F1 moved 0.034, macro-F1 0.009.

## v0.2.0 - 2026-09-06 (commit 1b68788)

### Why this release re-did the model

The project brief required the skeleton's layer counts, `base_ch`, `latent_dim`,
epochs and learning rates to be aligned against the original papers' text once
the PDFs were available. The owner confirmed on 2026-09-06 that
the conference paper's configuration should be followed as closely as the
substitute dataset and hardware allow. Everything below follows from that; the
v0.1.0 results at the foot of this file are superseded.

### Aligned with the papers

- **Training structure.** The two-stage pipeline (train a CVAE, then re-use its
  decoder to initialise a CGAN) was our own reading of the title, written before
  the PDFs were available. Both papers describe a **single-phase joint** model:
  one decoder is the VAE decoder `D(z, y)` and the GAN generator `G(z, y)` at the
  same time, trained under one combined objective, with the generator and
  discriminator updated alternately within each minibatch using separate backward
  passes. `src/train_cvae.py` and `src/train_cgan.py` are deleted;
  `src/train_joint.py` replaces them.
- **Combined loss.** `MSE_recon + beta*KL + lambda*VGG19_perceptual +
  gamma*adversarial`, with `lambda = 0.1` as in the paper, `beta = 0.015` (the
  paper's 30 rescaled to this repo's mean-normalised losses) and `gamma = 0.1`
  under a hinge adversarial term. The VGG19 perceptual term was previously
  unimplemented and was recorded as a known gap pending owner decision; it is now
  a frozen ImageNet feature extractor.
- **Reconstruction loss** L1 -> MSE. **Latent dim** 128 -> 32. **Batch size**
  64 -> 8. **Epochs** 100 + 200 -> 70 joint. **Learning rate** 2e-4 / 1e-4 / 4e-4
  -> 1e-3. **Adam betas** (0.5, 0.999) -> defaults.
- **Channels and range.** RGB in [-1, 1] with a Tanh decoder -> **[0, 1] with a
  sigmoid decoder**, 1 or 3 channels. The primary LoHi-WELD experiment runs at 3
  channels because colour carries defect signal there; the papers' grayscale was
  a property of their melt-pool camera, not of the method.
- **Schedule.** Added early stopping (patience 10) and `ReduceLROnPlateau`
  (factor 0.2, patience 5), both described by the paper and previously absent.
- **GAN objective.** BCE minimax with label smoothing (real = 0.9) replaced by the
  journal extension's **hinge loss with spectral normalisation** on the
  discriminator. Under BCE the unbounded `-log(D(G(z)))` term let a saturated
  discriminator overwhelm reconstruction on small training sets; hinge plus
  spectral normalisation bounds it by construction. The `--real_label` flag is
  gone with the BCE objective.
- **Feature aggregation.** Encoder: flatten + linear -> **1x1 channel reduction +
  global average pooling**. Discriminator: flatten + linear -> **global max
  pooling** + dense. Both are what the paper describes, and both turned out to be
  stability requirements rather than stylistic choices (bug 1 below).
- **FID cadence.** every 20 epochs -> **every epoch**, against a real validation
  set.
- **Augmentation protocol.** Arbitrary `--keep_ratios` downsampling plus a flat
  1000 generated images per class is replaced by the papers' actual rule:
  **balance-to-max** with a **filling-rate** sweep. New module `src/augment.py`.
- **Test-set leakage.** The generator previously trained on all 24,407 images,
  including the classifier's held-out 20% test split. Both papers state that
  evaluation is on "real, unseen images" with synthetic samples "strictly limited
  to the training phase". Splits are now made first and every generator subset is
  drawn from the train pool only, with a separate validation set the generator
  does not train on.

### The four calibrated deviations

Each differs from the paper's stated value, and each was set by measurement
against the training dynamics the paper itself reports. Details, sweep tables and
reproduction commands: [`docs/CALIBRATION.md`](docs/CALIBRATION.md).

| Item | Paper | Here | Why |
|---|---|---|---|
| KL weight | `beta = 30` | `0.015` | `beta` is a ratio between the KL and reconstruction terms, so it is not scale-free. The paper's reported recon ~1000 and KL ~9.0 imply pixel-mean loss on a [0, 255] scale with KL summed over latent dims; converted to this repo's mean-normalised [0, 1] losses that is `30 * 32 / 65025 = 0.0148`. Measured: 0.006 leaves KL ~6x the paper's, 0.015 matches its order without collapse, >= 0.05 collapses the posterior. |
| Discriminator lr | single `1e-3` for both nets | same 2e-4 | `1e-3` for both | **aligned with the conference paper**: hinge + spectral normalisation keep D in equilibrium on small data, so no reduction is needed; the 4e-4 value in docs/CALIBRATION.md was calibrated under the superseded BCE objective |
| FFT denoising | in the preprocessing pipeline | implemented, off by default | RIAWELC radiographs put 99.98% of their spectral energy below r = 0.1 and essentially none above r = 0.3, so there is no high-frequency noise for the paper's circular low-pass to remove; at cutoff 0.25 it retains only ~4% of the mid-frequency structure that carries defect edges. Kept behind `--fft_denoise` for users with noisy visible-light data. |
| Resolution | 400x400 | 224x224 | RIAWELC is natively 227x227. Reaching 400 needs upsampling, which invents detail while costing ~3x the compute (measured 12.4 h vs 3.8 h for 70 epochs on the full train pool). 224 loses 1.3% and divides by the architecture's x16 factor. |

Also owner-approved and forced by the substitute dataset rather than calibrated:
the paper-scale imbalanced subset (CR 40 / PO 200 / ND 300 / LP 600 = 1,140
images, 15x imbalance against the paper's 1,898 images at 14.7x), and keeping an
image classifier downstream because static radiographs have no temporal axis for
the papers' LSTM/GRU sequence models. Both are covered in `README.md`.

### Dataset pivot: RIAWELC -> LoHi-WELD

RIAWELC was abandoned as the primary experiment dataset after the calibrated
generator hit a reconstruction ceiling on it: crack and no-defect columns stayed
near-flat in the sample grids even once reconstruction beat the constant-mean
baseline (arm G1, above). The cause is a property of the data, not of the
optimiser - X-ray radiographs put 99.98% of their spectral energy below r=0.1 of
the Nyquist radius and have only 1.71x natural imbalance, so they match neither
the papers' visible-light modality nor their "small and imbalanced" premise.

[LoHi-WELD](https://github.com/SylvioBlock/LoHi-Weld) (Block et al., IEEE Access
2024) replaces it as primary: visible-light weld beads like the papers' melt-pool
imagery, whose high-resolution frames crop to 8,012 defect patches
(stain 3,540 / discontinuity 2,975 / deposit 1,193 / pore 304) at **11.6x**
imbalance, close to the conference paper's 14.7x. It ships as detection data, so
`src/prepare_yolo_crops.py` (new) crops each annotated box into class folders -
which is also the papers' ROI-extraction preprocessing step - using only the
high-resolution subset, since low-resolution beads crop to ~16px boxes that carry
no usable signal at 224. Crops keep their native **RGB** channels: the papers'
grayscale is a property of their camera, not of the method.

```bash
python src/prepare_yolo_crops.py \
  --input_root <archive>/weld-dataset/high_resolution_welds \
  --out_root data/lohi --img_size 224 --channels 3 --min_side 16 \
  --classes pore,deposit,discontinuity,stain
```

RIAWELC is demoted to a historical footnote in `DATA_SOURCES.md`: no current
figure, table or number derives from it, but its two required citations are
**retained**, because v0.1.0 and the generator calibration record were produced
with it and are already public. Removing a citation for data that was actually
used would be a compliance violation, not a simplification.

### Results (v0.2.0, LoHi-WELD)

Five-point filling-rate sweep, one from-scratch ResNet-18 per ratio, common
held-out real-only test set (full table in `README.md`):

- **The method's core claim holds on a public dataset**: the scarce class (pore,
  304 real crops) gains +0.095 F1 at r=0.25 and +0.112 at r=0.75; macro-F1 peaks
  at r=0.25 with +0.033 over the real-only baseline (0.6936 -> 0.7264).
- **The papers' balance-to-max recommendation does not transfer**: r=1.0 is the
  worst ratio here (macro-F1 0.6418, below the 0.6936 baseline), against the
  journal extension's reported monotone improvement peaking at 1.0. Recorded as
  measured, with the likely cause (generator fidelity lower than the papers', so
  flooding dilutes the real signal) stated as a hypothesis, not a conclusion.
- Generator diagnostics: FID 371 -> 216 by epoch 15 then plateau ~200;
  reconstruction MSE 0.049-0.060 vs a 0.062 constant-mean baseline; early
  stopping at epoch 25 of 70.

Single seed; differences under ~0.02 macro-F1 are within noise.

### Bugs fixed

Each of these would otherwise have corrupted a result without raising. The first
six are detailed in [`docs/CALIBRATION.md`](docs/CALIBRATION.md#7-bugs-found-while-calibrating).

1. **Encoder/discriminator fan-in made the paper's lr=1e-3 diverge.** Flattening
   the convolutional map into the latent heads gives `fc_logvar` a fan-in of
   100,352 at 224x224; one Adam step shifts `logvar` by tens and `exp(logvar)`
   overflows. Measured KL of **1.97e25** in epoch 1, all losses NaN at lr=1e-3.
   Fixed by the pooling the paper describes.
2. **FID was fed [0, 1] images to InceptionV3 with `transform_input=False`**,
   which expects [-1, 1]. This silently invalidated every FID number published in
   v0.1.0. Conversion now happens in one tested place, `eval_fid.inception_input`.
3. **FID returned negative values** (-2.5e-14 for identical distributions, larger
   on rank-deficient input). Clamped at zero, with the standard epsilon-regularised
   `sqrtm` fallback and a hard error if the result is still non-finite.
4. **`--sample_every 0` crashed** with `ZeroDivisionError` after the first epoch's
   log line. 0 now disables intermediate grids; the final epoch always writes
   one, so a run always leaves at least one sample grid.
5. **Sample grids used fabricated class labels**, so the reconstruction row was
   conditioned on the wrong classes and columns did not correspond across rows.
6. **`save_sample_grid` ran in train mode**, letting BatchNorm update its running
   statistics from a fixed display batch and from pure-noise generations, which
   leaked into the validation loss and FID of every later epoch.
7. **Hardcoded RIAWELC class indices.** `train_classifier.py` defaulted to
   `--keep_ratios "0:0.2,1:0.2,2:1.0,3:0.2"` and `--gen_classes "0,1,3"`, and
   `make_paper_figures.py` to `--gen_classes "0,1,3"`. On a dataset with a
   different class order these silently select the wrong classes. All per-class
   configuration is now keyed **by name** and validated against the dataset or
   checkpoint.
8. **Generated pools could be read with the wrong class order.** The `class_i`
   folders are positional, so pointing the sweep at a pool generated from a
   differently ordered dataset would silently assign every synthetic image to the
   wrong class. `generate.py` now writes a `classes.txt` manifest and
   `augment.generated_by_class` rejects a missing or mismatched one.
9. **A final training batch of one image** would make BatchNorm raise in train
   mode, potentially hours into a sweep. Training loaders now drop the last batch.
10. **Citation and data-fact errors** (compliance):
    - Conference paper authors were wrong: `Mu, Haonan` -> **Haochen Mu**;
      `He, Feng` -> **Fengyang He**; **Huijun Li** was omitted entirely.
    - The journal extension (MSSP 250, 2026, 114138) was not cited anywhere,
      despite being used to fill in architecture details. It is open access under
      CC BY 4.0, so a direct DOI link is now provided.
    - RIAWELC's **second mandatory citation** (Perri et al., Manufacturing
      Letters) was documented in `DATA_SOURCES.md` but missing from
      `CITATION.cff` and the README BibTeX. It is recorded as "in press" with no
      invented year, volume or pages.
    - `DATA_SOURCES.md` stated 224x224; all 24,407 files were verified to be
      **227x227 mode `L`**.

### Added

- `src/train_joint.py`: single-phase joint trainer with the combined four-term
  loss, per-epoch FID, early stopping, LR scheduling, and a `history.csv` that is
  the single machine-readable source of truth for the training-curve figures.
- `src/augment.py`: the balance-to-max filling-rate protocol.
  `filling_rate_counts` derives per-class synthetic counts from the majority
  class count; `synthetic_prefix` takes a sorted prefix so one generated pool
  serves a whole sweep and the ratios stay nested rather than comparing different
  random draws; `generated_by_class` validates the class manifest.
- `src/data.py`: `fft_lowpass`, `make_splits` (stratified, leakage-free),
  `sample_named_subset` (name-keyed, raises on unknown names or insufficient
  pool), `count_by_class`, `parse_name_counts`, and a shared `ListDataset`.
- `src/models.py`: `PerceptualLoss` (frozen ImageNet VGG19).
- `src/make_paper_figures.py`: training-history curves, filling-rate sensitivity
  curve, real/reconstruction/generated comparison, encoder latent t-SNE,
  row-normalised confusion matrices, and class distribution. Reads the CSV
  artifacts rather than parsing stdout.
- `docs/CALIBRATION.md`: the measurements behind the four calibrated deviations.
- `src/smoke_test.py`: unit checks for the [0, 1] grayscale contract, the FFT
  filter, the no-leakage split invariant, the decoder's sigmoid range, the MSE
  loss, the fan-in fix, the perceptual loss, FID input normalisation and
  numerics, name-keyed count parsing, the filling-rate arithmetic, prefix
  nesting, the class manifest, and every figure function. The end-to-end section
  runs joint training -> generation -> filling-rate sweep.

### Changed

- `src/generate.py`: reads channel count and **class names** from the checkpoint,
  writes the `classes.txt` manifest, accepts name-keyed `--counts`, generates in
  batches, and takes a `--seed`. Filenames are zero-padded to six digits so the
  sorted-prefix nesting holds above 99,999 images per class.
- `src/eval_fid.py`: `inception_input` extracted and tested; InceptionV3 built
  once and shared; class-balanced FID sampling helpers moved here from the
  deleted `train_cgan.py`; `--fft_denoise` added so evaluation can match
  training preprocessing.
- `src/train_classifier.py`: rewritten as the filling-rate sweep, exporting
  `sweep_metrics.csv` and a confusion matrix per ratio. ResNet-18 is trained
  **from scratch** with a single-channel `conv1`.
- `src/make_class_figures.py`: uses the shared `ListDataset` instead of a local
  duplicate that converted to RGB and disagreed with the pipeline's channel
  convention.
- `requirements.txt`: added `matplotlib` and `scikit-learn` (t-SNE).
- `torch.load` calls use `weights_only=True`; the checkpoints hold only tensors
  plus plain strings, numbers and bools, so object deserialisation is unnecessary
  and should not be enabled.

## Results - v0.1.0 (superseded, do not quote)

Produced by the two-stage skeleton on the full dataset before the papers were
read. Retained because v0.1.0 was published with these numbers and the FID values
need an explicit correction rather than silent removal.

- CVAE (100 epochs): final L1 reconstruction 0.102, KL ~7.0. CGAN (200 epochs,
  decoder-initialised G): final D loss 0.38, no mode collapse.
- FID 37.96, and per-class CR 71.87 / LP 63.50 / ND 14.71 / PO 83.49.
  **All invalid** - computed with bug 2 above, and against a real reference set
  whose class proportions differed from the generated set. Not comparable to
  literature FIDs, to each other across the fix, or to current results.
- Downstream, pretrained ResNet-18: condition A 99.36% accuracy / macro-F1 0.9931
  vs condition B 99.36% / 0.9932. Honest negative result - a pretrained backbone
  saturates RIAWELC, leaving no headroom for augmentation to help.
- Downstream, from-scratch ResNet-18 (calibration run, leaky protocol, flat 1000
  generated per class for CR/LP/PO): accuracy 0.6611 -> 0.6884, macro-F1 0.6471
  -> 0.6792; CR +0.0843 recall, LP +0.0897, ND +0.0017, **PO -0.0641**. Converted
  to the papers' filling rate this configuration sits at ~0.25 for every
  augmented class, which is precisely the range the journal extension reports as
  *degrading* performance - a likely explanation for the PO regression and the
  small overall gain. Superseded by the sweep.

## Paper-vs-code discrepancy list

Factual hyperparameter and architecture statements read from the two referenced
publications (IEEE CYBER 2025 conference paper; MSSP journal extension by the
same first author). Nothing was copied verbatim from either paper. Status
reflects the state after the 2026-09-06 owner decisions.

| Item | Conference paper (2025) | Journal extension (2026) | This repo | Status |
|---|---|---|---|---|
| Framework | TensorFlow 2.18 / Keras 3.9 | same lineage | PyTorch | deliberate (guide's design) |
| Training structure | single phase, combined loss, G and D alternated per minibatch | same lineage | single phase, combined loss, alternated per minibatch | **aligned** |
| Image size | 400x400 grayscale | 400x400 grayscale | 224x224 grayscale | calibrated (native 227x227) |
| Channels | 1 (grayscale melt-pool camera) | 1 | 1 or 3; primary runs at 3 | dataset-driven: colour carries defect signal in LoHi-WELD; the papers' grayscale is their sensor's property, not a method requirement |
| Decoder output | sigmoid, [0, 1] | n/a | sigmoid, [0, 1] | **aligned** |
| Preprocessing | ROI extraction, grayscale, FFT circular low-pass | same | grayscale; ROI already cropped by RIAWELC; FFT implemented, off | calibrated (measured band-limited) |
| Encoder body | 3x3 convs, residual connections, max pooling, 1x1 channel reduction | residual blocks, Group Normalization | 4 DCGAN-style 4x4 stride-2 convs + BatchNorm, then 1x1 reduction + global average pooling | partial: aggregation aligned, conv body deviation kept |
| Discriminator head | global max pooling + dense + dropout | GAP -> 512-d + projection | global max pooling + single dense | partial: pooling aligned, dropout omitted |
| Decoder upsampling | Sub-Pixel Convolution (2 stages) | progressive | 4 PixelShuffle stages | **aligned with the conference paper** |
| Latent dim | 32 | 128 | 32 | **aligned with the conference paper** |
| Reconstruction loss | MSE | weighted (3.0) | MSE | **aligned** |
| KL weight | beta = 30 | annealed 0 -> 0.5 | 0.015 | calibrated; equivalent under this repo's normalisation |
| Perceptual loss | VGG19, weight 0.1 | VGG19, weight 0.02 | frozen VGG19, weight 0.1 | **aligned** |
| Adversarial weight | gamma = 1 | n/a | 0.1 | calibrated: at 1.0 under this repo's mean-normalised losses the adversarial term outweighs reconstruction ~20:1 and reconstruction never converges (docs/CALIBRATION.md) |
| GAN objective | BCE minimax | hinge + projection discriminator + spectral norm | hinge + spectral norm | **aligned with the journal extension** (see below) |
| Label conditioning | label embedded as a spatial map concatenated with the image (E and D); decoder concatenates a class embedding with z | same design | same design | matches |
| G:D update ratio | alternated within each minibatch | 1:1 | alternated within each minibatch, separate backward passes | **aligned** |
| Generator lr | Adam 1e-3 | Adam beta1=0, lr 2e-4 | Adam 1e-3, default betas | **aligned with the conference paper** |
| Discriminator lr | same 1e-3 | same 2e-4 | 1e-3 | **aligned**: hinge + spectral normalisation keep D in equilibrium at the paper's own rate; the 4e-4 probe value in docs/CALIBRATION.md belongs to the superseded BCE objective |
| Batch size | 8 | 16 | 8 | **aligned** |
| Epochs | 70 | 200 | 70 | **aligned** |
| Early stopping / LR schedule | patience 10; ReduceLROnPlateau factor 0.2 patience 5 | n/a | same | **aligned** |
| FID | every epoch, vs real validation samples, InceptionV3 average pooling | same | every epoch, vs a real validation set, InceptionV3 pooled features | **aligned** |
| Augmentation strategy | balance all classes to 600 | balance-to-max to N_max, filling-rate sensitivity sweep | balance-to-max to N_max, filling-rate sweep | **aligned** |
| Training data | proprietary WAAM molten-pool, 1,898 images, 9 classes, 14.7x imbalance | own WA-DED data | RIAWELC subset, 1,140 images, 4 classes, 15x imbalance (public) | deliberate substitution (guide sec. 3) |
| Downstream classifier | LSTM and GRU over 21-frame sequences | GRU and LSTM over sequences | ResNet-18 from scratch over single images | forced deviation (no temporal axis); disclosed |
| Generator sees test images | evaluation on "real, unseen images" | same | no - splits made first, generator restricted to the train pool | **aligned** |
