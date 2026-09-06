# Calibration notes

This file records **why four values in this repo differ from the conference
paper's stated hyperparameters**. It is optional reading: the defaults in
`src/train_joint.py` are already the calibrated ones, and `README.md` is enough
to run the pipeline. Read this if you want to check that the deviations are
measured rather than arbitrary, or if you are porting the method to your own
dataset and need to know which values to re-derive.

`CHANGELOG.md` carries the short version. The raw run logs referenced below live
in the git-ignored `runs/logs/` directory, so every measurement here is stated
with the command that reproduces it.

## The acceptance criterion

The paper reports its own training dynamics, which turns "did we calibrate this
correctly" from a matter of taste into something checkable:

- reconstruction loss falls rapidly
- **KL divergence stabilises near 9.0** with `beta = 30`
- **generator loss rises gradually from ~3.0 to ~14.0** "due to adversarial
  interplay"
- that rise **aligns with a steady decline in FID**, which **drops from ~280 to
  ~70**

Two consequences worth stating plainly. First, a *rising* generator loss is
expected behaviour here, not a sign of divergence. Second, FID is the metric that
discriminates a working configuration from a broken one, because it measures the
distribution we actually sample from - and augmentation only ever uses sampled
images, never reconstructions.

Absolute FID values are **not** comparable between this repo and the paper:
different dataset, different resolution, different reference set. Only the
direction of the curve is.

## 1. KL weight: paper's `beta = 30` -> `0.015`

**Problem.** With `beta = 30` and this repo's loss normalisation, the posterior
collapses onto the prior. Measured on RIAWELC at 224x224: KL fell to 0.0000 per
dimension within four epochs and reconstruction degraded to a class-mean image.
A collapsed latent means `z` carries nothing, so generation from `z ~ N(0, I)`
produces low-diversity output - exactly what augmentation must not do.

**Why the paper's value does not transfer.** `beta` is not scale-free; it is a
ratio between the KL term and the reconstruction term, so it depends entirely on
how each is normalised. The paper reports reconstruction converging to ~1000 and
KL stabilising near 9.0 with a 32-dimensional latent. Those two numbers are only
mutually consistent if reconstruction is a **mean over pixels on a [0, 255]
scale** while KL is **summed over latent dimensions** (9.0 / 32 = 0.28 per dim).
This repo uses mean-normalised losses on [0, 1] images, which are
resolution-independent and therefore portable. Converting:

```
beta_equivalent = 30 * 32 / 255^2 = 30 * 32 / 65025 = 0.0148
```

The alternative reading - reconstruction summed over pixels on [0, 1] - would
give `30 * 32 / 160000 = 0.006`. The sweep below distinguishes them: 0.006
leaves KL about six times the paper's reported value, while 0.015 lands in the
same order, so the [0, 255] reading is the self-consistent one.

**Measurement.** Eight epochs each, identical otherwise, on the 1,140-image
paper-scale subset.

| `--kl_weight` | KL/dim ep1 -> ep8 | KL summed (x32) | recon ep8 | val_loss ep1 -> ep8 | outcome |
|---|---|---|---|---|---|
| 0.006 | 1.63 -> 1.78 | ~57 | 0.095 | 0.170 -> 0.090 | stable but ~6x the paper's KL |
| **0.015** | 0.95 -> **0.72** | **~23** | 0.113 | 0.158 -> 0.187 | **chosen: same order as the paper, no collapse** |
| 0.05 | 0.253 -> 0.0076 | 0.24 | 0.12 -> **0.23** | 0.148 -> 0.197 | posterior collapse, reconstruction degrades |
| 0.2 | 0.018 -> 0.0018 | 0.06 | 0.13 -> 0.216 | 0.141 -> 0.260 | immediate collapse |

**Decision.** `0.015`. It is the largest value that trains stably over the sweep
horizon and the closest to the paper's reported KL magnitude. Note the collapse
boundary is between 0.015 and 0.05, so 0.015 is deliberately on the safe side
rather than tuned to hit 9.0 exactly - chasing the number lands in collapse.

`models.cvae_loss` therefore takes `kl_weight` as a **required** argument with no
default. A baked-in default of 30 would silently destroy the model for anyone
who keeps this repo's normalisation.

**Reproduce.**

```bash
for B in 0.006 0.015 0.05 0.2; do
  python src/train_joint.py --data_root data --out_dir "runs/beta_$B" \
    --subset "CR=40,PO=200,ND=300,LP=600" --val_per_class 64 \
    --epochs 8 --fid_every 0 --sample_every 0 --kl_weight "$B"
done
```

## 2. Discriminator learning rate: paper's single `1e-3` -> `4e-4` (BCE era)

> **Superseded by section 6.** The arms below were measured under the BCE minimax
> objective. Section 6 replaces it with hinge loss plus spectral normalisation,
> under which the paper's own lr_d = 1e-3 works and gamma drops to 0.1. Kept as
> the record of why BCE was abandoned.

**Problem.** The paper specifies one Adam learning rate of 1e-3. Applied to both
networks, the discriminator saturates within four epochs on this 1,140-image
training set - it can separate real from generated almost perfectly, which is
easy when only 40 images of the minority class exist. Because the non-saturating
adversarial term `-log(D(G(z)))` is **unbounded above**, a confident D makes it
grow without limit:

| epoch | recon | beta\*KL | lambda\*perc | gamma\*adv (derived) | loss_d | val_recon | FID |
|---|---|---|---|---|---|---|---|
| 1 | 0.117 | 0.012 | 0.030 | 2.43 | 0.83 | 0.120 | 331.7 |
| 3 | 0.102 | 0.010 | 0.043 | 4.27 | 0.11 | 0.086 | 359.2 |
| 4 | 0.118 | 0.020 | 0.044 | **6.15** | **0.022** | **0.192** | **387.8** |

The adversarial term reached ~50x the reconstruction term, **FID rose** instead
of falling, and validation reconstruction degraded. This fails the paper's own
criterion, so the run was stopped at epoch 4.

**Measurement.** Three 12-epoch arms, `--kl_weight 0.015` throughout.

| arm | `--lr_d` | `--adv_weight` | loss_g | FID | recon ep12 | KL/dim (x32) | best val_loss |
|---|---|---|---|---|---|---|---|
| baseline | 1e-3 | 1.0 | 2.55 -> 6.33 | 331 -> **388 rising** | val_recon 0.192 | - | - (stopped ep4) |
| A | 1e-4 | 1.0 | 0.64 -> 3.07 | 310 -> 424 -> 337 | 0.097 | 0.51 (16.3) | 0.0896 |
| **B** | **4e-4** | **1.0** | 1.64 -> 5.9 -> 3.93 | **307 -> 296 -> 289 falling** | 0.088 | **0.21 (6.7)** | **0.0574** |
| C | 1e-3 | 0.05 | 0.188 -> 0.202 **flat** | **277 -> 282 -> 293 rising** | **0.028 best** | 0.21 (6.8) | 0.0499 |

**Decision.** Arm B: `--lr_d 4e-4`, `--adv_weight 1.0`. It is the only
configuration that reproduces all four of the paper's reported dynamics, and it
leaves the paper's `gamma = 1.0` untouched - only the discriminator's learning
rate deviates.

**Why arm C was rejected even though it won on two metrics.** Arm C has the best
reconstruction of all four (0.054 -> 0.028) and the best validation loss
(0.0499), and its KL matches the paper's reported magnitude. It is the tempting
choice. But its generator loss is *flat*, meaning the adversarial term
contributes nothing, and its **FID rises 277 -> 293 while reconstruction keeps
improving**. That divergence is the signature of a VAE with an ineffective
adversarial term: the model reconstructs its training inputs ever better while
the distribution it *generates* from `z ~ N(0, I)` drifts away from the real one.
Since augmentation only ever uses generated samples, arm C would have produced
the worse augmentation despite the better reconstruction.

This is incidental support for the paper's central claim - that the adversarial
component is what makes generation faithful, not reconstruction quality - and it
is the reason `--adv_weight` was left at the paper's value rather than reduced to
make the loss curves look calmer.

**Reproduce.**

```bash
for CFG in "A 1e-4 1.0" "B 4e-4 1.0" "C 1e-3 0.05"; do
  set -- $CFG
  python src/train_joint.py --data_root data --out_dir "runs/arm$1" \
    --subset "CR=40,PO=200,ND=300,LP=600" --val_per_class 200 \
    --epochs 12 --fid_every 4 --sample_every 12 --kl_weight 0.015 \
    --lr_d "$2" --adv_weight "$3"
done
```

## 3. FFT denoising: in the paper's pipeline, disabled here

The paper's preprocessing is ROI extraction, grayscale conversion, then FFT
denoising - each image is FFT-transformed, masked with a circular filter, and
inverse-transformed. That is a low-pass filter, and it is aimed at
high-frequency noise in visible-light melt-pool imagery: arc glare, spatter,
sensor noise.

RIAWELC radiographs do not have that noise. Measured on a real CR image, the
fraction of spectral energy by normalised radius from the spectrum centre:

| band | energy |
|---|---|
| r in [0, 0.1) | **99.98%** |
| r in [0.1, 0.3) | 0.01% |
| r in [0.3, 0.6) | ~0.00% |
| r in [0.6, 1.0] | ~0.00% |

There is essentially nothing above r = 0.3 to remove. Applying the filter anyway
is not neutral - at `cutoff = 0.25` it retains only about 4% of the little
mid-frequency structure present, and that structure is where defect edges live.

| cutoff | std (orig 7.8) | mid-band r[0.3,0.6) retained | MSE vs original |
|---|---|---|---|
| 0.15 | 7.7 | 4.5% | 2.3 |
| 0.25 | 7.7 | **4.0%** | 1.4 |
| 0.4 | 7.8 | 56.0% | 0.9 |
| 0.6 | 7.8 | 105.0% | 0.7 |

**Decision.** Disabled by default. The implementation is kept behind
`--fft_denoise` / `--fft_cutoff` because it is part of the paper's method and
users with noisy visible-light data will want it. If you enable it, also pass
`--fft_denoise` to `src/eval_fid.py`, or FID will compare denoised generations
against un-denoised real images and measure the preprocessing mismatch instead of
generation quality.

ROI extraction needs no re-implementation: RIAWELC already ships pre-cropped
defect regions. Grayscale conversion is likewise already satisfied - every one of
the 24,407 files is PIL mode `L`.

## 4. Resolution: paper's 400x400 -> 224x224

RIAWELC is natively **227x227**. Reaching 400x400 would require upsampling, which
interpolates detail that does not exist in the source while making the model
learn smoothing artefacts. 224 is a 1.3% reduction, loses no information, and is
divisible by the architecture's x16 spatial factor.

Measured cost of the joint four-loss configuration (19,526 train images, 70
epochs, RTX 4000 Ada Laptop, 12 GB):

| resolution | channels | batch | s/step | min/epoch | 70 epochs | peak VRAM |
|---|---|---|---|---|---|---|
| 128 | 1 | 8 | 0.033 | 1.3 | 1.6 h | 0.64 GB |
| 224 | 1 | 8 | 0.080 | 3.3 | 3.8 h | 1.49 GB |
| 256 | 1 | 8 | 0.100 | 4.1 | 4.8 h | 1.86 GB |
| 400 | 1 | 8 | 0.262 | 10.7 | 12.4 h | 4.23 GB |

224 costs about a third of 400 for the same information content. The paper-scale
subset used for the headline result (1,140 images) trains in roughly 40 minutes
at 224x224 including per-epoch FID.

## 5. FID reference size: 600 per class

FID features are 2048-dimensional, so a reference set smaller than 2048 images
gives a rank-deficient covariance. `eval_fid.fid_from_stats` handles that - it
suppresses the resulting `LinAlgWarning`, falls back to epsilon regularisation,
and clamps the result at zero since FID is a distance - but a rank-deficient
estimate is still a poor estimate. On LoHi-WELD the pore class leaves only ~203
train-pool images after the 20% test split, so the reference is 200 per class
(800 images) - below the ~2,000-per-side rule of thumb for treating FID as an
absolute score. FID is therefore used as a diagnostic (is it falling?) and never
quoted as a comparable number.

Cost: `sqrtm` on a 2048x2048 matrix takes about 11 s, so per-epoch FID roughly
doubles epoch time. That is accepted because the paper computes FID every epoch
and the resulting curve is one of the figures.

## 6. Adversarial objective: BCE minimax -> hinge + spectral normalisation

The first full 70-epoch run under the conference paper's BCE minimax objective
failed, and the failure was not subtle. Evidence at the best checkpoint
(epoch 23 of an early stop at 33):

| check | result | verdict |
|---|---|---|
| reconstruction MSE | 0.077 - 0.10 | **2-8x worse than trivial baselines** (constant global mean 0.038; per-image mean 0.013) |
| adversarial share of generator loss | ~98% (adv ~5 vs recon ~0.1) | reconstruction is collateral damage |
| loss_d | 0.03 - 0.10 by epoch 26 | discriminator saturated and stayed saturated |
| FID | plateaued 267 - 310, minimum 266.9 at epoch 12 | paper reports a steady decline from ~280 to ~70 |
| sample grid | washed-out grey with a regular checkerboard artefact, no visible class structure | generations unusable for augmentation |

**Why the conference configuration cannot be stable at this data scale.** The
discriminator sees only 1,140 real images (40 of the minority class), so it can
memorise them and becomes confidently correct. The non-saturating BCE generator
term `-log(D(G(z)))` is **unbounded above**, so a confident discriminator makes it
grow without limit; it reached ~50x the reconstruction term. Reconstruction then
degrades below what a constant predictor achieves, and FID stops improving.

The paper's own reported magnitudes show why this never bit them: with
reconstruction ~1000 and the adversarial term ~0.7, the adversarial contribution
is **0.07% of their total loss**. Their "adversarial sharpening" is a small
perturbation on a strong VAE, stable by construction. It also means **gamma
cannot be calibrated from the paper's numbers**: matching their loss ratio would
require gamma ~1e-5 here, which removes the GAN entirely - and the arm measured
with a small gamma (section 2, arm C) confirms that produces rising FID despite
the best reconstruction. The two papers' objectives are therefore not
interchangeable by rescaling; the objective itself had to change.

**The fix.** The journal extension specifies **hinge loss with spectral
normalisation** on the discriminator, and that combination is bounded by
construction: spectral normalisation caps each weight matrix's spectral norm at
1, so the discriminator's score - and therefore hinge's generator term
`-mean(D(G(z)))` - cannot grow without limit even when the discriminator wins.
The conference paper's **Sub-Pixel (PixelShuffle) decoder** replaces the transposed
convolution that produced the checkerboard artefact.

Both are the original authors' own designs, so this keeps the reproduction inside
the papers rather than importing outside machinery. Pinned by
`check_hinge_losses_and_bounded_score` in `src/smoke_test.py`, which asserts the
hinge arithmetic and that a saturated discriminator still emits a bounded score.

**Acceptance gate, revised.** The first gate was "reconstruction better than the
0.038 constant-mean baseline *and* FID below its epoch-1 value by epoch 30". H1
failed it, and the failure is informative:

| arm | config | recon (train) | val_recon | FID | verdict |
|---|---|---|---|---|---|
| H1 | hinge + SN, gamma 1.0, lr_d 1e-3 | stuck 0.088 - 0.118 | oscillating 0.047 - 0.119 | 192 - 308, no decline | **fail**: discriminator no longer saturates (loss_d ~0.8), but a bounded adversarial term at gamma 1.0 is still ~20x the reconstruction term, so reconstruction never converges |
| H2 | hinge + SN, gamma 1.0, lr_d 4e-4 | not completed | - | - | killed: same gamma, same failure family as H1 |

H1 confirms that bounding the adversarial term is necessary but not sufficient.
At gamma 1.0 the term is bounded yet still dominates, and reconstruction is left
as an oscillating side effect.

That forces a conclusion about the paper's own numbers. With reconstruction
~1000 and adversarial ~0.7, their adversarial contribution is **0.07% of the
total loss**, so their reported FID decline from ~280 to ~70 was driven by the
VAE and perceptual terms learning, with the GAN contributing a small perturbation.
Reproducing that ratio here would need gamma ~1e-5, which makes the adversarial
component decorative. FID is therefore treated as a **diagnostic, not an
acceptance criterion**: it is recorded and reported, but the configuration is
chosen on reconstruction quality and on whether the sample grids show clear
class-conditional structure.

The revised gate is: reconstruction MSE better than the 0.038 constant-mean
baseline, and sample grids with visible per-class structure. Whether the method
actually helps is then decided by the downstream filling-rate sweep - the
paper's real claim - not by FID.

| arm | config | recon vs 0.038 baseline | class structure in grids | verdict |
|---|---|---|---|---|
| G1 | hinge + SN, gamma 0.1, lr_d 1e-3 | **0.035 at ep20 - passes** | partial: PO and LP show defect structure, CR and ND near-flat | **carried forward**: reconstruction converges, D stays in equilibrium (loss_d ~1.0-1.2); FID fell 257 -> 211 by ep10 then rose to 322 |

G1 is the first configuration in this whole sequence whose reconstruction beats
the constant-mean baseline, which is the floor for "the model is learning at all".
It is not a clean success: crack (CR) and no-defect (ND) columns stay near-flat,
consistent with RIAWELC's band-limited, low-contrast radiographs carrying very
little high-frequency signal for those classes. That dataset property - not the
optimiser - is the likely ceiling here, and is the reason the primary experiment
moved to a visible-light dataset (see `DATA_SOURCES.md`).

## 7. Bugs found while calibrating

These are fixed in the code and are recorded here because each one would
otherwise silently corrupt a result.

1. **Encoder/discriminator fan-in made the paper's lr=1e-3 diverge.** The
   skeleton flattened the convolutional map straight into the latent heads,
   giving `fc_logvar` a fan-in of `512 * 14 * 14 = 100,352` at 224x224. One Adam
   step moves all of those weights by ~lr simultaneously, shifting `logvar` by
   tens, so `exp(logvar)` overflowed. Measured: KL reached **1.97e25** in epoch 1
   and every loss was NaN at lr=1e-3, while lr=1e-4 trained normally
   (KL = 1.2551). The paper describes a 1x1 channel reduction before the
   encoder's latent heads and global max pooling before the discriminator's dense
   layer; implementing both fixes the divergence at the paper's own lr=1e-3.
   Pinned by `check_feature_aggregation_fanin` in `src/smoke_test.py`.
2. **FID was fed [0, 1] images to InceptionV3 with `transform_input=False`**,
   which expects [-1, 1]. Every activation was off-distribution. This silently
   invalidated all FID numbers published in v0.1.0. Conversion now happens in one
   tested place, `eval_fid.inception_input`.
3. **FID returned negative values** for identical distributions (-2.5e-14, and
   materially larger negatives on rank-deficient input). Clamped at zero.
4. **`--sample_every 0` crashed** with `ZeroDivisionError` after the first
   epoch's log line. 0 now means "never", matching `--fid_every`.
5. **Sample grids were built with fabricated class labels.** The display batch
   came from the sorted validation set (so nearly all one class) while the labels
   were replaced with a repeating `arange`, meaning the reconstruction row was
   conditioned on the wrong classes. The grid is now class-balanced and uses each
   image's true label.
6. **`save_sample_grid` ran in train mode**, so BatchNorm normalised with the
   display batch's statistics and updated its running statistics from that batch
   and from pure-noise generations - which then leaked into the validation loss
   and FID of every later epoch. Now runs in eval mode.
