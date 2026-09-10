# Calibration notes

This file records **why this repo's configuration differs from the two papers it
reproduces** - which values were re-derived, which components were taken from the
journal extension rather than the conference paper, and which of the papers'
results cannot be reproduced here at all. It is optional reading: the defaults in
`src/train_joint.py` are already the calibrated ones, and `README.md` is enough
to run the pipeline. Longer operating notes (journal-extension switches,
own-data porting, limitations) live in [`USAGE.md`](USAGE.md). Read this if
you want to check that the deviations are measured rather than arbitrary, or
if you are porting the method to your own dataset and need to know which
values to re-derive.

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

From v0.4.0 the default computation is [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
(official TensorFlow Inception weights). Every FID number already published in
this repository was computed with `--fid_backend legacy` (torchvision
InceptionV3). Those two backends are not on the same scale; reproduce a
published number with `legacy`, and do not mix them in one curve.

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

**Addendum: the logged KL is already at the paper's magnitude, and beta is not
the fidelity lever.** The sweep above was eight epochs on RIAWELC. Re-measured on
LoHi-WELD over full 70-epoch runs, `cvae_loss` logs `kl` as a mean over the batch
*and* over latent dimensions, so the quantity comparable to the paper's is
`latent_dim x kl` - the total latent information budget in nats:

Each range below is over that run's **final three epochs**; the window was not
stated when these were first published, and it matters, because the same runs
span 5.4 - 21.4 (v0.2.0) and 2.6 - 23.1 (equal budget) if taken over the whole
run instead. Recomputed from the committed histories:

| run | run directory | latent dim | beta | `latent_dim x kl`, last 3 epochs |
|---|---|---|---|---|
| v0.2.0 published | `runs/joint_lohi` (published) | 32 | 0.015 | 6.78 - 7.23 |
| probe, latent 128 | `runs/probe_lat128_g0.1` (**unpublished probe**) | 128 | 0.059 | 8.35 - 8.74 |
| probe, latent 128, equal budget | `runs/probe_eq_g0.1` (published) | 128 | 0.059 | 12.14 - 15.85 |

The middle row's history is not in `results/metrics/`: it comes from a probe run
that was never published, so that row cannot be checked from this repository and
is marked accordingly. The third row was published as "11.7 - 15.9"; recomputing
its last three epochs from the committed CSV gives 12.14 - 15.85, so its original
window must have been slightly wider. Neither difference changes the conclusion
below.

Paper 1 reports "KL divergence stabilizes near 9.0 despite a high weighting
(beta = 30.0)" at a 32-dimensional latent. That sentence admits two readings, and
neither one motivates lowering beta further:

- If 9.0 is the **raw summed KL**, then 6.8 - 15.9 straddles it and this repo is
  already reproducing the paper's operating point.
- If 9.0 is **beta x KL**, the paper's raw KL is 0.3 nats over 32 dimensions -
  0.009 per dimension, i.e. a posterior within 1% of the prior. On that reading
  the paper reaches FID 15 - 20 and 96.79% downstream accuracy *with a
  near-collapsed latent*, which would mean latent information is not what drives
  their fidelity at all.

Either way the conclusion is the same, so the ambiguity does not need resolving:
**the explanation offered earlier in this file - that a low-information latent is
why our generations lack fidelity - is withdrawn as unsupported.** The measured
ceiling is dataset provenance: LoHi-WELD's crops are upsamples of small source
boxes and 99.2 - 99.9% of their spectral energy sits inside r < 0.10
(sections 3 and 4). No latent width, no beta and no adversarial weight can
generate detail the source pixels never contained. See the Data section in `README.md`.

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

LoHi-WELD, now the primary dataset, is visible-light rather than radiographic, so
the filter deserves a second look there. It does not change the answer. Measured
on one image per class (all 224x224, PIL mode `RGB`, converted to `L` for the
spectrum):

| class | r in [0, 0.1) | r in [0.1, 0.3) | r in [0.3, 0.6) | r >= 0.6 | mid-band retained at cutoff 0.25 | MSE vs original |
|---|---|---|---|---|---|---|
| deposit | 99.32% | 0.57% | 0.07% | 0.03% | **0.0%** | 12.1 |
| discontinuity | 99.22% | 0.68% | 0.07% | 0.03% | **0.0%** | 14.1 |
| pore | 99.86% | 0.10% | 0.02% | 0.01% | **0.0%** | 3.4 |
| stain | 99.46% | 0.50% | 0.03% | 0.01% | **0.0%** | 7.4 |

MSE is on the [0, 255] scale, so 3-14 out of a possible 65,025 is a near no-op:
the filter removes almost nothing because there is almost nothing above
r = 0.1 to remove. That is a consequence of how this dataset is built - the crops
are upsamples of small source bounding boxes (section 4 measures the
distribution), and upsampling cannot create high-frequency content. The step is
therefore inert on both datasets for opposite reasons: on RIAWELC because
radiographs are band-limited, on LoHi-WELD because the crops were already
low-pass filtered by interpolation.

**Decision.** Disabled by default. The implementation is kept behind
`--fft_denoise` / `--fft_cutoff` because it is part of the paper's method and
users with noisy visible-light data will want it. If you enable it, also pass
`--fft_denoise` to `src/eval_fid.py`, or FID will compare denoised generations
against un-denoised real images and measure the preprocessing mismatch instead of
generation quality.

ROI extraction needs no re-implementation: both datasets already ship pre-cropped
defect regions.

Grayscale conversion is a **declared step of the paper's method**, not an
artefact of its camera: the conference paper states that its original RGB frames
"were converted to grayscale to reduce redundancy and suppress irrelevant color
variations". This repository does not follow it on LoHi-WELD, and that is a
deviation rather than a neutral dataset detail. The reason is that LoHi-WELD's
defect classes include `stain` and `discontinuity`, whose visible-light
appearance carries colour information that a grayscale collapse discards, so
`--channels 3` keeps it and lets the first convolution learn the mixing.
`--channels 1` remains available and reproduces the paper's step. RIAWELC
needed no such choice: every one of its 24,407 files is already PIL mode `L`.
Earlier releases justified this here and in `CHANGELOG.md` by calling the papers'
grayscale "a property of their camera"; that reading is withdrawn as wrong.

## 4. Resolution: paper's 400x400 -> 224x224

RIAWELC is natively **227x227**. Reaching 400x400 would require upsampling, which
interpolates detail that does not exist in the source while making the model
learn smoothing artefacts. 224 is a 1.3% reduction, loses no information, and is
divisible by the architecture's x16 spatial factor.

LoHi-WELD, now the primary dataset, ships at **exactly 224x224** after
`src/prepare_yolo_crops.py`, so on it the choice involves no resizing at all. The
caveat is what those 224 pixels contain. Measured over all 8,012 crops actually
produced from the 1,022 high-resolution images, the **native** size of the source
bounding box before upscaling:

| statistic | native box max side (px) |
|---|---|
| p25 | 27 |
| median | **47** |
| mean | 56.7 |
| p75 | 70 |
| p95 | 134 |
| max | 423 |
| share >= 224 px | **1.40%** |

So 98.6% of the crops are upsamples, and half of them come from a box smaller
than 47 px stretched to 224 - a 4.8x interpolation. The canvas is 224; the
information in it is not. Raising `--img_size` would interpolate an
already-interpolated image a second time, which is why the resolution and
`--min_side` levers were measured and then abandoned rather than tuned. This is
the fidelity ceiling referred to in sections 1 and 3, and it is a property of the
dataset's provenance rather than of anything in this repo's configuration.

Reproduce:

```bash
python - <<'PY'
import pathlib, numpy as np
from PIL import Image
root = pathlib.Path("<archive>/weld-dataset/high_resolution_welds")
sides = []
for p in [q for q in root.rglob("*") if q.suffix.lower() in (".jpg", ".jpeg", ".png")]:
    lab = p.with_suffix(".yolo")
    if not lab.exists():
        continue
    with Image.open(p) as h:
        W, H = h.size
    for line in lab.read_text().splitlines():
        f = line.split()
        if len(f) >= 5:
            bw, bh = float(f[3]) * W, float(f[4]) * H
            if max(bw, bh) >= 16:
                sides.append(max(bw, bh))
s = np.array(sides)
print(len(s), np.median(s), np.percentile(s, [25, 75, 95]), (s >= 224).mean())
PY
```

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

## 5. FID reference size: 200 per class

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
| reconstruction MSE | 0.077 - 0.10 | **2-8x worse than trivial baselines** (constant global mean 0.038; per-image mean 0.013, both on RIAWELC - on LoHi-WELD's 800-image validation split the equivalents are **0.0556** and **0.0339**) |
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

Both are components the papers themselves specify, so this keeps the
reproduction inside their design space rather than importing outside machinery.
Neither technique originates with them: spectral normalisation is Miyato et al.
(2018) and Sub-Pixel convolution is Shi et al. (2016), both cited in
`CITATION.cff`.

**Spectral-norm scope is a deviation.** The journal extension applies spectral
normalisation to the discriminator's first three convolutional blocks; this repo
applies it to all four plus the dense head. Bounding more layers can only tighten
the Lipschitz bound the argument above relies on, so it is conservative in the
direction that matters, but it is not what the paper specifies and the difference
has not been measured. `--no_d_spectral_norm` removes it entirely, which together
with `--adv_loss bce` is the conference paper's original objective. Pinned by
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

**Caveat, added later: the H1/H2 verdict is confounded with early stopping.** Both
arms ran under `patience=10`, `lr_patience=5`. A subsequent controlled
measurement at gamma 0.1 - identical in every other respect, only
`patience`/`lr_patience` raised to 70/70 so the learning rate stays constant -
moved best `val_recon` from **0.038010 to 0.033214**, which is what first put this
repo below the per-image-mean anchor. The gain came from the budget, not from
gamma. Since *both* papers use gamma = 1.0 and our gamma = 1.0 failure was only
ever observed under `patience=10`, it is possible that gamma 1.0 was rejected
partly because the run was stopped before it settled. That has not been
re-tested at equal budget (roughly 30 minutes per arm), so gamma 0.1 stands as a
measured choice rather than a proven necessity, and "gamma 1.0 fails" should be
read as "gamma 1.0 failed under this schedule".

That forces a conclusion about the paper's own numbers. With reconstruction
~1000 and adversarial ~0.7, their adversarial contribution is **0.07% of the
total loss**, so their reported FID decline from ~280 to ~70 was driven by the
VAE and perceptual terms learning, with the GAN contributing a small perturbation.
Reproducing that ratio here would need gamma ~1e-5, which makes the adversarial
component decorative. FID is therefore treated as a **diagnostic, not an
acceptance criterion**: it is recorded and reported, but the configuration is
chosen on reconstruction quality and on whether the sample grids show clear
class-conditional structure.

The revised gate is: reconstruction MSE better than the constant-mean baseline
(0.038 on RIAWELC, 0.0556 on LoHi-WELD), and sample grids with visible per-class
structure. The per-image-mean anchor (0.013 RIAWELC, 0.0339 LoHi-WELD) is the
stricter of the two and the one worth clearing, since a model that only beats the
constant mean is still worse than predicting each input's own average. Whether
the method actually helps is then decided by the downstream filling-rate sweep -
the paper's real claim - not by FID.

| arm | config | recon vs 0.038 baseline | class structure in grids | verdict |
|---|---|---|---|---|
| G1 | hinge + SN, gamma 0.1, lr_d 1e-3 | **0.035 at ep20 - passes** | partial: PO and LP show defect structure, CR and ND near-flat | **carried forward**: reconstruction converges, D stays in equilibrium (loss_d ~1.0-1.2); FID fell 257 -> 211 by ep10 then rose to 322 |

G1 is the first configuration in this whole sequence whose reconstruction beats
the constant-mean baseline, which is the floor for "the model is learning at all".
It is not a clean success: crack (CR) and no-defect (ND) columns stay near-flat,
consistent with RIAWELC's band-limited, low-contrast radiographs carrying very
little high-frequency signal for those classes. That dataset property - not the
optimiser - is the likely ceiling here, and is the reason the primary experiment
moved to a visible-light dataset (see the Data section in `README.md`).

> **Partly superseded by section 9.** Two claims in this section were later
> measured to be wrong or incomplete. First, G1's `loss_d ~1.0-1.2` is described
> above as an equilibrium; for hinge loss the equilibrium is **2.0**, and 1.0-1.2
> is still a dominant discriminator. Second, this section attributes the
> dominant-D regime to the small data scale. Section 9 measures an arm with the
> discriminator's **BatchNorm replaced by GroupNorm** (plus a class-balanced
> sampler) that reaches a `loss_d` median of 1.998 with no epoch below 1.5, on
> the same small dataset - so data scale was not the binding constraint.
> GroupNorm is the change with a mechanism behind it, but the two were switched
> together and have not been separated. The BCE analysis above stands - BCE's
> generator term really is unbounded - but the "our data is too small for a
> stable discriminator" reading of the *hinge* arms does not.

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

## 8. The journal paper's headline number is not reproducible here, and why

The journal extension reports **96.79%** downstream accuracy against a **89.46%**
vanilla CVAE-GAN baseline and calls it "+7.33% over the Vanilla baseline". That
gain is not produced by the CVAE-CGAN architecture this repo reproduces. It is
produced by two physics-informed losses, `L_bound` and `L_therm`, and the paper's
own ablation (its Fig. 15(a)) shows this directly.

> The six accuracy values in the table below are taken from Fig. 15(a) of Yang
> et al., "Physics-guided generative data augmentation for vision-based signal
> processing under class-imbalanced conditions in directed energy deposition
> monitoring system", *Mechanical Systems and Signal Processing* **250** (2026)
> 114138, [doi:10.1016/j.ymssp.2026.114138](https://doi.org/10.1016/j.ymssp.2026.114138),
> which is open access under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are reproduced
> here under that licence, with attribution, because the argument of this section
> cannot be made without them. They are the **authors' own results on their own
> proprietary data**, not results of this repository.

| variant | accuracy |
|---|---|
| real images only | 83.04% |
| traditional augmentation | 88.93% |
| V2 - vanilla CVAE-GAN | 89.46% |
| V3 - vanilla + `L_bound` only | **83.93%** |
| V4 - vanilla + `L_therm` only | **86.61%** |
| V5 - full model, both losses | **96.79%** |

Each physics loss applied *alone* is worse than the vanilla baseline it is added
to. The paper attributes this to the fact that "individual physical constraints
tend to over-regularize the generator when applied alone". The entire headline
therefore comes from the two acting together, and there is no partial credit
available for implementing one of them.

This repo implements **neither**, and declares them out of scope rather than
substituting something else: both need WAAM thermal-field ground truth (melt-pool
boundary and temperature evolution) that neither public substitute dataset
carries. Inventing a stand-in loss would produce a number that looks like a
reproduction and is not one. The consequence, stated plainly so that nobody
mistakes it for a failure to reproduce: **96.79% is not a target for this repo and
must not be quoted as one.** What this repo can and does measure is the
downstream minority-class gain from the components both papers share - the joint
CVAE-CGAN generator and the filling-rate augmentation protocol.

Two further numbers from the journal paper must not be used as health anchors for
a run of this repo:

- Its reconstruction values (~0.057 for the full model, ~0.068 for vanilla) are
  **Charbonnier loss on [-1, 1]**; this repo logs MSE on [0, 1]. The scales are
  not comparable without conversion.
- Its FID of 15 - 20 is computed against a large reference set. This repo's
  reference is 200 images per class and rank-deficient (section 5), so its FID is
  a diagnostic - is the curve falling? - and never a comparable score.

## 9. Components taken from the journal extension, not the conference paper

Four things in `src/train_joint.py` come from the journal extension's Table 3
rather than from the conference paper. Each is a switch that **defaults to the
configuration the tagged v0.2.0 runs were produced with**, so a recorded v0.2.0
command line still reproduces a v0.2.0 run. Nothing here silently changes a
published result.

| switch | default (v0.2.0) | journal paper | status |
|---|---|---|---|
| `--d_norm` | `batch` | `group` | **measured, recommended** |
| `--weighted_sampler` | off | on | **measured, recommended** |
| `--kl_warmup ZERO,RAMP` | off | `10,50` | implemented and tested, **not training-validated** |
| `--monitor` | `val_loss` | n/a | only needed together with `--kl_warmup` |

One journal KL detail is **not** implemented: its Eq. 9 free-bits form
`max(0, KL - delta)` (the paper gives no value for `delta`). `--kl_warmup`
implements the annealing schedule only; the logged term is plain KL.

### The discriminator equilibrium: a normalisation change, not a data-scale one

Until this arm, every run in this repo showed a **dominant discriminator**: over
the 70-epoch BatchNorm baseline `loss_d` had a median of 0.82 and 97% of epochs
came in below 1.5. Section 6 explained that as a data-scale effect - with only
1,140 real images, 40 of them minority class, D can memorise the training set and
become confidently correct.

That explanation was wrong, or at least incomplete. For hinge loss, a
discriminator that outputs approximately 0 for both real and fake gives
`loss_d = 2.0` exactly; that is the equilibrium spectral normalisation is
supposed to produce. Replacing the discriminator's BatchNorm with GroupNorm moved
`loss_d` onto that value and kept it there for the whole run:

| arm | discriminator norm | `loss_d` median | `loss_d` range | epochs below 1.5 |
|---|---|---|---|---|
| `runs/probe_eq_g0.1` | batch | 0.819 | 0.400 - 2.410 | **97%** (dominant D) |
| `runs/paper2_gn_wrs` | group | **1.998** | 1.958 - 2.278 | **0%** (equilibrium) |

The journal paper reports precisely this: "the discriminator loss settles to an
approximately constant value around 2.0". The mechanism is concrete rather than
mysterious. At `batch_size = 8`, a BatchNorm discriminator scores an image using
statistics computed from the other seven images in its minibatch, so its output
depends on batch composition; GroupNorm normalises within each sample and does
not. The result is therefore consistent with **batch-statistics leakage rather
than our real and generated images being trivially separable** - though the arm
also enabled the balanced sampler, so the two have not been separated
(qualification 1 below).

### The measured arm

`runs/paper2_gn_wrs` turns on `--d_norm group` and `--weighted_sampler` together
and is otherwise **identical** to `runs/probe_eq_g0.1`: latent 128, base_ch 64,
`kl_weight` 0.059, `perc_weight` 0.1, `adv_weight` 0.1, 224px, 3 channels,
subset `pore=40,deposit=150,discontinuity=300,stain=600`, seed 42, `test_frac`
0.2, 70 epochs, constant lr = lr_d = 1e-3, patience 70/70. Verified by reading
both checkpoints' stored configs, not by recalling the launch commands.

| metric | BatchNorm baseline | GroupNorm + balanced sampler |
|---|---|---|
| best `val_recon` | 0.033214 (ep69) | **0.017675** (ep60) |
| train recon, mean of last 10 epochs | 0.0398 | **0.0234** |
| `loss_d`, mean of last 10 epochs | 0.777 | **1.993** |
| `latent_dim x kl` range | 2.59 - 23.06 | 5.50 - 11.32 |
| FID, first -> last | 324.0 -> 186.9 | 251.5 -> 216.5 |
| FID, minimum | **142.3** | 191.9 |

Against the trivial-predictor anchors on this exact 800-image validation split
(constant-mean 0.0556, per-image-mean 0.0339), 0.017675 is the first result in
this repo that clears the **stricter** anchor by a clear margin rather than
scraping under it.

**Two honest qualifications.**

1. *The reconstruction gain is not attributable to either switch alone.* The arm
   changed two variables at once. Separating them needs a third arm (GroupNorm
   without the sampler), which has not been run. The `loss_d` shift is the part
   with a clear mechanism behind it, and the reconstruction gain is reported as
   the joint effect of the pair.
2. *FID got worse while reconstruction got better* (minimum 191.9 vs 142.3).
   This is the same divergence section 2 measured with arm C, and it is why FID
   is a diagnostic here rather than an acceptance criterion. It is recorded
   rather than explained away: on the evidence in this file, better
   reconstruction and better FID are not the same claim, and augmentation only
   ever uses sampled images. The downstream filling-rate sweep is what decides
   which arm actually helps.

`--weighted_sampler` is the journal paper's data-balancing strategy, whose stated
purpose is that "each training batch possesses a balanced class distribution".
Capping per-class counts with `--subset` does not achieve that: a 40-image
minority class inside a 1,090-image subset is 3.7% of draws, so at `batch_size=8`
most minibatches contain no minority sample at all and it contributes to few
gradient updates.

### The downstream verdict: r=1.0 keeps a positive mean gain, but is not uniquely optimal

Qualification 2 deferred to the filling-rate sweep. It has now been run for the
GroupNorm + balanced-sampler arm at three independently trained generator/pool/
classifier seeds (`runs/sweep_paper2_s42`, `s43`, and `s44`) and for its matched
latent-128 BatchNorm, unweighted-sampler control at seed 42
(`runs/sweep_ctrl_s42_clean`, pool `generated_ctrl`). Each pool has 1,310 synthetic
images at r=1.0: pore 560 / deposit 450 / discontinuity 300 / stain 0. The
published v0.2.0 curve remains `runs/sweep`.

macro-F1; gains in brackets are over the same column's r=0 baseline:

| ratio | v0.2.0 published | GroupNorm + balanced sampler (3 seeds) | latent-128 BatchNorm control |
|---|---|---|---|
| 0.00 (real only) | 0.6936 | 0.6785 ± 0.0447 | 0.6332 |
| 0.25 | **0.7264** (+0.033) | 0.7419 ± 0.0181 (n=2) | 0.7123 |
| 0.50 | 0.6767 (-0.017) | **0.7210 ± 0.0246** (+0.0425) | 0.6341 (+0.0010) |
| 0.75 | 0.7029 (+0.009) | 0.6972 ± 0.0239 (+0.0187) | **0.7124** (+0.0793) |
| 1.00 (balance-to-max) | 0.6418 (**-0.052**, worst) | 0.7185 ± 0.0019 (+0.0399) | 0.7009 (+0.0677) |

pore F1, the minority class the whole experiment exists for:

| ratio | v0.2.0 published | GroupNorm + balanced sampler (3 seeds) | latent-128 BatchNorm control |
|---|---|---|---|
| 0.00 | 0.3778 | 0.3527 ± 0.0611 | 0.4000 |
| 0.25 | 0.4731 | **0.5380 ± 0.0695** (n=2) | 0.4554 |
| 0.50 | 0.3590 | 0.4554 ± 0.0480 (+0.1028) | 0.2192 (-0.1808) |
| 0.75 | 0.4902 | 0.4458 ± 0.0541 (+0.0931) | **0.4632** (+0.0632) |
| 1.00 | 0.3871 | 0.4658 ± 0.0475 (+0.1132) | 0.4222 (+0.0222) |

**The supported structural result is narrower than the seed-42 curve.** The
GroupNorm + balanced-sampler arm has mean r=1.0 macro-F1 0.7185, +0.0399 over
its r=0 baseline, and mean pore F1 0.4658, +0.1132. The individual r=1.0
macro-F1 deltas are +0.0321, +0.0895, and -0.0018; it is a positive mean effect,
not an improvement guaranteed on every seed. The three-seed data do **not**
establish a monotone curve or a unique optimum at r=1.0: r=0.50 has the slightly
higher complete-seed macro-F1 mean (0.7210), and r=0.25 is 0.7419 on the two
retained seeds.

The matched BatchNorm control preserves the central sign flip: r=1.0 is 0.7009,
+0.0677 over its own 0.6332 real-only baseline, and its pore F1 rises from 0.4000
to 0.4222. Its curve is also non-monotone and peaks at r=0.75 (0.7124), not r=1.0.
Together these measurements rule out the claim that GroupNorm or the weighted
sampler is *required* for the positive r=1.0 result. They do not estimate either
switch's separate benefit.

> **Withdrawn in v0.5.2 (section 11.4).** The paragraph above rests on the
> control's seed 42 alone. At seeds 43 and 44 the same control's r=1.0 deltas
> are -0.0223 and -0.0248 macro-F1, for a three-seed mean of +0.0069, and its
> pore F1 falls at r=1.0 on two seeds of three. The control does not
> reproduce the sign flip, so the sentence "rule out the claim that GroupNorm
> or the weighted sampler is required" is withdrawn; whether the pair is
> required is open again. The paragraph is kept as the published record.
> Section 11.5 further reports that the recommended arm's own r=1.0 gain does
> not survive a source-frame-grouped split.

**The seed-42 r=0.25 arm is excluded from the aggregate, not declared invalid.**
That classifier's loss sat at 0.0001 by epoch 90, then reported 0.2617 at epoch
100, and `train_classifier.py` in that sweep scored the final epoch (section 10;
the default became `--selection best_val` in v0.4.0, and `--selection final`
still reproduces this behaviour). Under that protocol a late spike is itself a
behaviour of the protocol, and the exclusion was applied after seeing the
results - the n=2 aggregate is post-hoc and exploratory, not a pre-registered
validity criterion. The r=0.25 GroupNorm statistics therefore cover only seeds
43 and 44 and are marked n=2; the excluded score itself (macro-F1 0.5868) is
retained in the published CSV under `results/metrics/sweep_paper2_s42/`.

**Four things this does not establish.**

1. *It does not isolate the remaining three configuration changes.* The matched
   control holds BatchNorm and the unweighted sampler while retaining the
   latent-128, `kl_weight=0.059`, full-70-epoch configuration. Because its r=1.0
   gain is positive, the GroupNorm/sampler pair is not necessary for the sign
   flip. But comparison with published v0.2.0 still changes `latent_dim` 32 ->
   128, `kl_weight` 0.015 -> 0.059, and early stopping at 25 epochs -> the full
   70 together; none can be individually credited. The control is also one seed,
   so it does not quantify the two switches' effect or their interaction.
2. *"Better generator" is true only on the reconstruction axis.* Against v0.2.0
   this arm's best `val_recon` is 0.0177 vs 0.0494, a 2.8x improvement, but its
   best FID is 215.99 vs 216.87 - effectively unchanged. So the README's earlier
   conjecture that flooding at r=1.0 failed because "our samples are blurrier than
   the papers'" is *consistent* with this result, not demonstrated by it: the axis
   that moved is reconstruction, and FID did not follow.

   FID's direction here depends on which pair and which statistic is taken, and
   qualification 2 above reports the opposite sign, so all three measured
   comparisons are stated rather than one being picked:

   | comparison | statistic | this arm | reference | direction |
   |---|---|---|---|---|
   | vs v0.2.0 (`runs/joint_lohi`) | FID at each arm's own best-val epoch | 215.99 | 216.87 | unchanged |
   | vs v0.2.0 (`runs/joint_lohi`) | minimum FID over the run | 191.93 | 184.84 | worse |
   | vs the matched latent-128 BatchNorm control (`runs/probe_eq_g0.1`) | minimum FID over the run | 191.93 | 142.29 | worse |

   The at-best-epoch comparison against v0.2.0 is the only one in which FID does
   not move. Note also that the two 70-epoch arms computed FID on only 14 of 70
   epochs, so their minima are sampled, not exhaustive; v0.2.0 computed it every
   epoch over 25. None of this changes the conclusion - FID is a diagnostic here,
   not an acceptance criterion - but it does mean "FID got worse" and "FID did not
   move" are both defensible statements about the same two checkpoints, and any
   reuse of them has to say which comparison is meant.
3. *The GroupNorm + balanced-sampler aggregate has three seeds, but the control
   has one.* Seeds 43 and 44 retrained the generator and regenerated their pools,
   so the GroupNorm values include generator as well as classifier variation. The
   r=0.25 aggregate has only two retained scores. Section 10's fixed-seed classifier
   repeatability warning remains relevant, and a three-seed control would still be
   required to estimate a reliable control mean or an incremental switch effect.
4. *The absolute numbers remain non-comparable to the papers'*, for the reasons in
   section 8: a from-scratch ResNet-18 over single images, not an LSTM/GRU over
   21-frame sequences.

### The published v0.2.0 runs

`runs/joint_lohi` and `runs/sweep` back the README single-seed table and four of
its six figure rows, and their command lines were never written down. Both are
recovered below from the configuration stored inside `runs/joint_lohi/joint.pt`
(latent 32, `base_ch` 64, `kl_weight` 0.015, `perc_weight` 0.1, `adv_weight` 0.1,
224 px, 3 channels, seed 42, best epoch 15) plus the flags this run own
`history.csv` implies - 25 epochs of a 70-epoch budget under the default
patience 10, and an FID value on every epoch.

```bash
# The v0.2.0 generator. Everything not written out is a default of that release:
# --patience 10 --lr_patience 5 --fid_every 1 are what stop it at epoch 25.
python src/train_joint.py --data_root data/lohi \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
  --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 32 \
  --kl_weight 0.015 --perc_weight 0.1 --adv_weight 0.1 \
  --fid_backend legacy --out_dir runs/joint_lohi

python src/generate.py --ckpt runs/joint_lohi/joint.pt --seed 42 \
  --out_root generated \
  --counts "deposit=450,discontinuity=300,pore=560,stain=300"

# The v0.2.0 sweep. --selection did not exist yet; its behaviour is what
# --selection final now reproduces.
python src/train_classifier.py --data_root data/lohi --gen_root generated \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --channels 3 --selection final --out_dir runs/sweep
```

Two caveats on these two commands specifically. The generator predates the
`test_frac` and `monitor` keys being stored in the checkpoint, so those are
inferred from that release defaults rather than read back. And the `generated`
pool holds `stain=300`, unlike every later balance-to-max pool, which is why its
`meta.json` records no `test_frac` and why it is the one pool whose split
configuration cannot be fully checked.

### Split-overlap measurement: crop-level, not source-isolated

Read-only audit of the published splits (added in v0.5.1). Source identity is
recovered exactly from the crop filenames - `prepare_yolo_crops.py` writes
`<encoded-source-path>_<k:03d>.png`, so stripping the trailing three-digit box
index recovers the source image - and the seed-42/43/44 splits were recomputed
with the exact `make_splits` / `sample_named_subset` logic, verified
index-for-index identical to the shipped implementation via SHA-256 of the
index lists.

The 8,012 crops come from 1,022 source frames (2-18 crops each, median 8), and
nothing in the split groups by source frame:

| seed | test crops sharing a source frame with the train pool | ... with the 1,090-crop training subset | test source frames that also feed the train pool |
|---|---|---|---|
| 42 | 1,599 / 1,603 (99.8%) | 1,014 (63.3%) | 812 / 813 (99.9%) |
| 43 | 1,603 / 1,603 (100%) | 1,069 (66.7%) | 802 / 802 (100%) |
| 44 | 1,603 / 1,603 (100%) | 1,043 (65.1%) | 820 / 820 (100%) |

The split therefore guarantees crop-index disjointness only. Absolute
downstream numbers are optimistic about unseen sources; the filling-rate arms
share the test set and the baseline subset, so within-run comparisons are less
affected.

**Reproduce.** `src/measure_split_overlap.py` re-derives the splits with the same
`make_splits` / `sample_named_subset` calls the training scripts make, and prints
the table above. It opens no images and needs no GPU or checkpoint.

```bash
python src/measure_split_overlap.py --data_root data/lohi \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --seeds 42,43,44
```

A source-frame-grouped split is no longer only an experiment a fork should run:
`--split_by source` implements it, the same script verifies that it reports 0% on
every column above, and section 11 reports what the downstream numbers do under
it.

### KL annealing: implemented, tested, deliberately not trained

`--kl_warmup ZERO,RAMP` implements the journal paper's schedule - beta held at
zero for `ZERO` epochs, then raised linearly to its target over `RAMP` - because
the paper names a constant high beta from the start as the thing that "risks
posterior collapse where the encoder ignores inputs". It is covered by a unit
test on the schedule arithmetic and by an end-to-end training run in
`src/smoke_test.py`, and `history.csv` gained a `beta` column so the ramp is
visible in the logged record.

It has **not** been training-validated on this dataset, for two reasons:

- The paper's own `10,50` schedule needs its 200-epoch protocol. Inside the
  conference paper's 70 epochs it consumes 60 of them, leaving only 10 at full
  beta, and during the beta = 0 phase the encoder is entirely unconstrained in
  `|mu|`, so a later ramp can destroy reconstruction that has already formed.
- Section 1's addendum removed the motivation. The logged KL is already at the
  paper's reported magnitude under either reading of it, so there is no measured
  collapse for annealing to fix.

`--monitor val_recon` exists only to support it: the total loss includes
`beta x KL`, so a rising ramp would trip `ReduceLROnPlateau` and early stopping
on its own schedule rather than on model quality. Pass it whenever
`--kl_warmup` is non-empty.

### Reproduce

```bash
# The arm above.
python src/train_joint.py --data_root data/lohi \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
  --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 128 \
  --lr 1e-3 --lr_d 1e-3 --kl_weight 0.059 --perc_weight 0.1 --adv_weight 0.1 \
  --patience 70 --lr_patience 70 \
  --d_norm group --weighted_sampler \
  --fid_every 5 --sample_every 10 --out_dir runs/paper2_gn_wrs

# The BatchNorm baseline it is compared against: identical, minus the two flags.
python src/train_joint.py --data_root data/lohi \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
  --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 128 \
  --lr 1e-3 --lr_d 1e-3 --kl_weight 0.059 --perc_weight 0.1 --adv_weight 0.1 \
  --patience 70 --lr_patience 70 \
  --fid_every 5 --sample_every 10 --out_dir runs/probe_eq_g0.1
```

The downstream sweep in the subsection above needs a generated pool first. The
counts are exactly what balance-to-max requires at `N_max = 600` - each class is
filled from its real subset count up to 600, which is why `stain` needs none -
and lower ratios draw a prefix of the same pool rather than a second generation.

```bash
# Pool for the measured arm.
python src/generate.py --ckpt runs/paper2_gn_wrs/joint.pt --seed 42 \
  --out_root generated_paper2 \
  --counts "deposit=450,discontinuity=300,pore=560,stain=0"

# Its 5-ratio sweep. --ratios, --epochs 100, --img_size 224, --test_frac 0.2 and
# --seed 42 are all defaults and are left implicit; --channels 3 is written out
# because it was not the default when this section was measured (it has been
# since v0.4.0).
python src/train_classifier.py --data_root data/lohi --gen_root generated_paper2 \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --channels 3 --out_dir runs/sweep_paper2_s42
```

Seeds 43 and 44 repeat all three steps, because the pool's seed is inherited from
the checkpoint and the sweep refuses a pool whose seed disagrees with its own. So
the three-seed aggregate varies the generator as well as the classifier:

```bash
for S in 43 44; do
  python src/train_joint.py --data_root data/lohi \
    --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
    --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 128 \
    --lr 1e-3 --lr_d 1e-3 --kl_weight 0.059 --perc_weight 0.1 --adv_weight 0.1 \
    --patience 70 --lr_patience 70 --seed "$S" \
    --d_norm group --weighted_sampler \
    --fid_every 5 --sample_every 10 --out_dir "runs/paper2_gn_wrs_s$S"

  python src/generate.py --ckpt "runs/paper2_gn_wrs_s$S/joint.pt" --seed "$S" \
    --out_root "generated_paper2_s$S" \
    --counts "deposit=450,discontinuity=300,pore=560,stain=0"

  python src/train_classifier.py --data_root data/lohi \
    --gen_root "generated_paper2_s$S" \
    --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
    --channels 3 --seed "$S" --out_dir "runs/sweep_paper2_s$S"
done
```

The matched latent-128 BatchNorm control is the same three commands pointed at
`runs/probe_eq_g0.1`, into its own pool directory - `verify_generation_meta`
refuses to sweep a pool whose `subset`, `seed` or `test_frac` disagree with the
sweep's, and `generate.py` writes the seed from the checkpoint, so pool and sweep
have to be produced as a matched pair rather than reused across arms. That check
compares the split *configuration*, not data identity: re-cropping, adding or
renaming images changes the actual split without changing the recorded flags,
so the pool must be regenerated whenever the underlying crop tree changes.

```bash
python src/generate.py --ckpt runs/probe_eq_g0.1/joint.pt --seed 42 \
  --out_root generated_ctrl \
  --counts "deposit=450,discontinuity=300,pore=560,stain=0"

# --num_workers 0 because the worker processes were the source of a mid-sweep
# crash on this machine; the numbers in section 9 come from this run directory.
python src/train_classifier.py --data_root data/lohi --gen_root generated_ctrl \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --channels 3 --num_workers 0 --out_dir runs/sweep_ctrl_s42_clean
```

`results/filling_rate_multiseed.png` is drawn from the `sweep_metrics.csv` files of
the three runs above, so it needs neither a GPU nor the dataset. `--exclude` drops
the seed-42 `r=0.25` arm - the post-hoc exclusion discussed above - from the
aggregate without removing that ratio
from seeds 43 and 44, which is why that ratio is annotated `n=2` in the figure;
the band is the sample standard deviation, the same convention as the table above.

```bash
python src/make_multiseed_figure.py \
  --seed_sweep runs/sweep_paper2_s42/sweep_metrics.csv \
  --seed_sweep runs/sweep_paper2_s43/sweep_metrics.csv \
  --seed_sweep runs/sweep_paper2_s44/sweep_metrics.csv \
  --exclude "runs/sweep_paper2_s42/sweep_metrics.csv=0.25" \
  --reference "published v0.2.0, latent-32 BatchNorm (seed 42)=runs/sweep/sweep_metrics.csv" \
  --reference "matched latent-128 BatchNorm control (seed 42)=runs/sweep_ctrl_s42_clean/sweep_metrics.csv" \
  --minority pore --out results/filling_rate_multiseed.png
```

The `sweep_metrics.csv` inputs (and the generator `history.csv` logs of the
arms above) are published under `results/metrics/`, one subdirectory per run
name, as byte-identical copies; replacing `runs/` with `results/metrics/` in
the command above reproduces the figure without any original run directory.
Provenance for each file: `results/metrics/README.md`.

The committed showcase grids come from the same recommended checkpoint but from a
*separate* pool. `generated_paper2` holds `stain=0` because balance-to-max needs no
stain images, and adding some would break the `meta.json` contract that
`verify_generation_meta` checks against the sweep; a showcase figure, however, has
to show all four classes. So the grids are generated into their own directory,
which no sweep ever reads:

```bash
python src/generate.py --ckpt runs/paper2_gn_wrs/joint.pt --seed 42 \
  --out_root generated_showcase \
  --counts "deposit=300,discontinuity=300,pore=300,stain=300"

# --per_class 6 and --label_width 210 keep the geometry of the committed figure;
# the default 150 truncates the "discontinuity real" row label.
python src/make_comparison.py --real_root data/lohi --gen_root generated_showcase \
  --out results/real_vs_generated.png --per_class 6 --label_width 210

python src/make_class_figures.py --real_root data/lohi \
  --gen_root generated_showcase --out_dir results --channels 3
```

## 10. The downstream classifier is not bit-reproducible, and it scores the final epoch

> **Partly superseded by v0.4.0.** The second property no longer describes the
> default: `train_classifier.py` now checkpoints and restores the
> best-validation epoch (`--selection best_val`), so a late loss spike no
> longer defines the reported numbers. `--selection final` keeps this
> section's behaviour and reproduces the published tables, which were scored
> that way. The bit-reproducibility analysis (Cause 1) is unaffected. Kept as
> the record of how the published numbers were produced.

Two properties of `train_classifier.py` bound how every filling-rate number in
this repo - published or new - may be read. Both surfaced when the `r=0` control
arm of the section 9 sweep failed to reproduce the published v0.2.0 baseline.

`r=0` adds zero synthetic images, so it exercises the real-data path only and
should be identical across the two runs. The published arm reported
accuracy 0.8222 / macro-F1 0.6936 / pore F1 0.3778; the repeat reported
0.8253 / 0.6848 / 0.3441.

**The split is not what differs.** Both runs logged the same training subset
(`deposit 150, discontinuity 300, pore 40, stain 600` -> 1,090) and the same
held-out test set (`deposit 239, discontinuity 595, pore 61, stain 708` ->
1,603), and the per-class supports in both classification reports match exactly.
`src/train_classifier.py` has an empty diff against the tagged release, and
`src/data.py`'s diff is purely additive - a new `balanced_sample_weights` helper
that the classifier never calls.

### Cause 1: no deterministic algorithms

`train_classifier.py` seeds `random`, `numpy` and `torch` at the top of each
ratio's loop. When the published runs were made it did nothing further, so on
CUDA the convolution backward passes used atomic-add kernels whose accumulation
order is not fixed, and identical seed, data and code still diverged from the
first step. (Since v0.4.0 the script has a `--deterministic` flag that sets
`cudnn.deterministic` and disables `cudnn.benchmark`. It was not used for any
published run, and it is not a guarantee either: `torch.use_deterministic_
algorithms(True)` is still called nowhere in this repository. `train_joint.py`
has no equivalent flag at all, so generator runs cannot opt in.) The loss curves confirm it: epoch 10 was
0.4266 in the published run and 0.4008 in the repeat.

**The measured spread is the noise floor for this benchmark.** On 61 pore test
images, one repeat of an unchanged configuration moved pore F1 by **0.034** and
macro-F1 by **0.009**. Any single-seed difference smaller than that is not
evidence, in either direction. (**Superseded in v0.5.2**: with eight repeats
of the real-only arm across three seeds, section 11.7 measures the spread at
up to 0.127 macro-F1 and 0.255 pore F1; the pair quoted here was a lucky
one. The README now quotes the section 11.7 figures.) That is granular because the minority class has
only 61 test images: one image is worth ~0.016 of pore recall on its own. These
two figures are now quoted in `README.md` beside the filling-rate table, which
is where a reader meets the per-ratio differences they bound. (Earlier releases
of this file attributed a "treat differences under ~0.02 macro-F1 as noise"
caveat to the README; no such sentence was ever in it.)

The generator side has its own reproduction caveat: `compute_fid` and
`save_sample_grid` draw from the same default RNG stream as training, so
changing `--fid_every` or `--sample_every` changes the number of draws before
each step and shifts every subsequent training random stream. Exact
reproduction therefore keeps the diagnostic settings identical too.

### Cause 2: the final epoch is what gets scored

When the published tables were produced, `train_one` ran a constant Adam
learning rate for the full `--epochs` with no schedule, no early stopping and no
checkpointing, and `evaluate` was then called on whatever weights remained;
there was no best-epoch selection anywhere in the file. `--selection final`
still reproduces exactly that. Since v0.4.0 the default is `--selection
best_val`, which keeps and restores the lowest-loss epoch measured on the
real-only leftover pool.

This matters because the loss can spike late. In the section 9 sweep the `r=0.25`
arm sat at 0.0001 by epoch 90 and then reported **0.2617 at epoch 100** - three
orders of magnitude - and those spiked weights are exactly what was scored,
giving macro-F1 0.5868 against the published arm's 0.7264.

The published v0.2.0 curve is exposed to the same failure mode; none of its five
arms happened to spike. That is luck, not a property of the method.

**What was not done.** Neither cause was fixed, because both fixes would break
comparability with the tagged v0.2.0 numbers: selecting a best epoch, or forcing
deterministic algorithms, changes what every previously published row means. The
honest position is that this repo's downstream numbers carry a run-to-run
uncertainty of roughly +/-0.03 pore F1 at fixed seed, and that a single ratio
arm can be invalidated outright by a late spike. Averaging over seeds reduces the
first problem; only a protocol change would reduce the second, and section 11
reports what happened when the published sweeps were rescored under
`--selection best_val`.

**Reproduce the noise floor.** Run the `r=0` arm twice and compare; no code
change is needed to see it.

```bash
python src/train_classifier.py --data_root data/lohi --gen_root generated \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --ratios "0.0" --channels 3 --epochs 100 --seed 42 --out_dir runs/noise_a
```

## 11. Post-v0.5.1 additional measurements

Everything in this section was measured after v0.5.1, on the code at tag
`v0.5.2`, with the recorded commands below. It is **additive**: sections 9 and
10 and both README tables are kept exactly as published, and nothing here
replaces a number in them. Six questions those sections left open are answered
here, each by a run that did not previously exist:

| arm | question | runs |
|---|---|---|
| D0 | what the published pools score under the current default `pytorch_fid` backend, so the two FID scales can be placed side by side once | `results/metrics/fid_cross_scale.csv` |
| D5 | whether the conference paper's BCE objective still fails on today's code and this dataset, rather than only in the section 6 RIAWELC record | `runs/bce_nosn_g0.1`, `runs/bce_nosn_g1.0` |
| D2 | GroupNorm **without** the weighted sampler, separating the two switches section 9 changed together | `runs/paper2_gn_only`, `runs/sweep_gn_only` |
| D3 | the matched BatchNorm control at seeds 43 and 44, so it has a three-seed mean like the GroupNorm arm | `runs/probe_eq_g0.1_s43`, `_s44`, `runs/sweep_ctrl_s43`, `_s44` |
| D4 | the recommended arm under `--split_by source`, three seeds: how much of the downstream result survives when no test crop shares a source frame with training | `runs/paper2_src_s42`, `_s43`, `_s44`, `runs/sweep_src_s4x` |
| D1 | the five published sweeps rescored under `--selection best_val`, the current default, from their own pools | `runs/sweep_*_bestval` |

Every metric export is under `results/metrics/<run>/` (`SHA256SUMS` covers
them), so each table below recomputes from the committed CSVs without a GPU.
Section 10's noise floor still bounds every single-seed difference here: pore
F1 moved 0.034 and macro-F1 0.009 on one repeat of an unchanged arm.


### 11.1 The two FID scales, side by side (D0)

Every `fid` column in `results/metrics/` and every FID quoted in sections 2,
5, 6 and 9 is on the `--fid_backend legacy` (torchvision InceptionV3) scale,
because all of those runs predate v0.4.0's switch to pytorch-fid as the default.
The two scales were never placed next to each other on the same images. This
does that once, with `src/eval_fid.py` on the six pools behind the published
tables and figures: both backends see the same class-balanced sample (300 per
class over the classes present on both sides, drawn with the script's fixed
seed), so the pair differs only in the Inception weights.

| pool | generator | classes compared | `legacy` | `pytorch_fid` |
|---|---|---|---|---|
| `generated` (v0.2.0) | `runs/joint_lohi` | 4 | 220.70 | 278.62 |
| `generated_ctrl` | `runs/probe_eq_g0.1` | 3 | 160.69 | 274.15 |
| `generated_paper2` | `runs/paper2_gn_wrs` | 3 | 213.23 | 284.27 |
| `generated_paper2_s43` | `runs/paper2_gn_wrs_s43` | 3 | 185.36 | 232.67 |
| `generated_paper2_s44` | `runs/paper2_gn_wrs_s44` | 3 | 243.73 | 360.61 |
| `generated_showcase` | `runs/paper2_gn_wrs` | 4 | 216.38 | 283.65 |

Per-class, on the showcase pool against every seed-42 train-pool crop of the
class (the `make_class_figures.py` protocol):

| class | real crops | `legacy` | `pytorch_fid` |
|---|---|---|---|
| deposit | 954 | 198.82 | 268.48 |
| discontinuity | 2,380 | 224.86 | 290.16 |
| pore | 243 | 185.87 | 295.86 |
| stain | 2,832 | 222.45 | 306.36 |

Three things follow. The `pytorch_fid` numbers are higher throughout, by a
factor that is not constant (1.26x to 1.71x on the pool rows), so there is no
conversion between the scales and none is offered. The *ordering* of the pools
is not preserved either: on the legacy scale the BatchNorm control is the best
pool by a wide margin (160.69), on pytorch-fid it sits between the three
GroupNorm seeds. And the per-class ranking flips - pore is the best class on
one backend and the second-worst on the other. So "FID is a diagnostic, not an
acceptance criterion" (section 6) is not only about the papers' numbers: two
Inception weight sets disagree about which of this repo's own pools is better.
Any FID comparison in this repository has to name its backend, and the
committed showcase figures are on the legacy scale: regenerating them with
`--fid_backend legacy` reproduces all four `results/class_*.png` pixel for
pixel, which also pins the pool and the figure pipeline to the published
checkpoint.

Source: `results/metrics/fid_cross_scale.csv`, one row per pool, backend and
scope, with the real reference each row used.

### 11.2 The conference paper's objective on today's code (D5)

Section 6 rejected the conference paper's BCE minimax objective on the
evidence of one RIAWELC run made before `--adv_loss` existed, and `docs/USAGE.md`
has since told readers to "expect it to fail". With the switch in place the
claim can be checked on this dataset and this code: `--adv_loss bce
--no_d_spectral_norm` is the conference configuration, run for 20 epochs at
seed 42 with every other flag equal to the section 9 BatchNorm control
(`runs/probe_eq_g0.1`), once at the papers' own gamma = 1.0 and once at this
repository's 0.1. Twenty epochs is a diagnostic budget, not the 70-epoch
protocol; the comparison rows are the first 20 epochs of the two published
hinge runs. `loss_g` is the adversarial generator term alone.

| arm (20 epochs) | objective | D norm | gamma | best `val_recon` | epochs below 0.0556 / 0.0339 | `loss_g` median | `loss_d` median | `kl x 128`, last 5 | FID ep5 -> ep20 |
|---|---|---|---|---|---|---|---|---|---|
| `bce_nosn_g1.0` (new) | BCE, no SN | batch | **1.0** | 0.0774 (ep6) | **0 / 0** | **2.97** | 0.41 | **2.5** | 300.3 -> 321.0 |
| `bce_nosn_g0.1` (new) | BCE, no SN | batch | 0.1 | 0.0464 (ep17) | 3 / 0 | 0.38 | 0.51 | 18.6 | 290.3 -> 282.0 |
| `probe_eq_g0.1`, first 20 | hinge + SN | batch | 0.1 | 0.0560 (ep13) | 0 / 0 | 0.30 | 0.70 | 15.0 | 324.0 -> 289.2 |
| `paper2_gn_wrs`, first 20 | hinge + SN | group | 0.1 | **0.0207** (ep19) | **19 / 14** | 0.05 | 2.00 | 9.8 | 251.5 -> 213.9 |

At the papers' gamma the section 6 failure reproduces on today's code and on
LoHi-WELD, mechanism included: the unbounded generator term sits at a median
of 2.97 against a reconstruction term of about 0.13, some 20x, the
discriminator saturates (`loss_d` falls to 0.16 at epoch 9), the KL term
collapses to `kl x 128` = 2.5 - the posterior-collapse signature section 1
looked for and did not find in the hinge runs - and validation reconstruction
never gets under the constant-mean anchor; the best checkpoint is epoch 1.
That is what "BCE fails at this data scale" means, and it is now a
reproducible statement rather than a historical one.

At gamma = 0.1 it is a different picture, and section 6 should not be read as
covering it. The adversarial term is bounded in practice by the small weight
(median 0.38, about 6x reconstruction), the discriminator wins but does not
saturate, and reconstruction reaches 0.0464 - under the constant-mean anchor on
three epochs of twenty. Over the same first 20 epochs the hinge BatchNorm
control is no better (0.0560, never under the anchor); at this budget the two
objectives are indistinguishable on reconstruction, and the arm that is
different is the GroupNorm one (0.0207, under the stricter anchor on 14
epochs). So at this data scale the objective matters at the papers' gamma and
the discriminator normalisation matters at this repository's gamma. Whether
BCE at gamma 0.1 would reach the hinge control's 70-epoch 0.0332 was not run,
and neither BCE arm was taken to a pool or a sweep; both would be needed
before saying anything about BCE downstream.

### 11.3 GroupNorm alone: the equilibrium and most of the reconstruction gain are its (D2)

Section 9 changed `--d_norm group` and `--weighted_sampler` together and said so
(qualification 1). `runs/paper2_gn_only` is the missing arm: identical to
`runs/paper2_gn_wrs` in every flag except that the sampler is off, seed 42,
70 epochs, so the three runs form a ladder BatchNorm -> GroupNorm -> GroupNorm +
sampler. Recomputed from the three committed histories:

| metric | BatchNorm control (`probe_eq_g0.1`) | GroupNorm only (`paper2_gn_only`, new) | GroupNorm + sampler (`paper2_gn_wrs`) |
|---|---|---|---|
| `loss_d` median | 0.819 | **2.001** | 1.998 |
| `loss_d` range | 0.400 - 2.410 | 1.978 - 2.259 | 1.958 - 2.278 |
| epochs with `loss_d` below 1.5 | 97% | **0%** | 0% |
| best `val_recon` | 0.033214 (ep69) | **0.018456** (ep44) | 0.017675 (ep60) |
| `val_recon` at the checkpointed epoch | 0.033214 (ep69) | 0.018487 (ep66) | 0.017675 (ep60) |
| train recon, mean of last 10 epochs | 0.0398 | 0.0240 | 0.0234 |
| FID first -> last (legacy scale) | 324.0 -> 186.9 | 228.8 -> 167.5 | 251.5 -> 216.5 |
| FID minimum | 142.3 | **141.8** | 191.9 |

Two of section 9's open readings close. The discriminator equilibrium at 2.0
**is** the GroupNorm effect: with the sampler off the hinge loss still sits at
a median of 2.001 with no epoch below 1.5, so the batch-statistics mechanism
section 9 proposed needs nothing from the sampler. And the reconstruction gain
is almost entirely GroupNorm's as well: 0.0332 -> 0.0185 from the
normalisation change alone, then 0.0185 -> 0.0177 from adding the sampler,
the second step being of the order of the epoch-to-epoch wobble of one run and
not a measured benefit. What the sampler does buy on this axis is nothing;
what it costs is FID: the GroupNorm-only run reaches a minimum of 141.8, level
with the BatchNorm control and 50 points under the recommended arm's 191.9.
So the "FID got worse while reconstruction got better" divergence of section 9
belongs to the sampler, not to GroupNorm. One seed, legacy scale (as every
in-training FID here is), and FID remains a diagnostic rather than a criterion;
but the direction is not small.

Downstream, the picture inverts. `runs/sweep_gn_only` is the same sweep as
`runs/sweep_paper2_s42` pointed at the new pool (`generated_gn_only`, seed 42,
`--selection final`), and every synthetic ratio scores **below** its own
real-only baseline:

macro-F1, gains in brackets over the same column's r=0:

| ratio | BatchNorm control (`sweep_ctrl_s42_clean`) | GroupNorm only (`sweep_gn_only`, new) | GroupNorm + sampler, seed 42 (`sweep_paper2_s42`) | GroupNorm + sampler, 3 seeds |
|---|---|---|---|---|
| 0.00 (real only) | 0.6332 | 0.7106 | 0.6848 | 0.6785 ± 0.0447 |
| 0.25 | 0.7123 (+0.0791) | 0.6949 (-0.0156) | excluded | 0.7419 ± 0.0181 (+0.0665) (n=2) |
| 0.50 | 0.6341 (+0.0010) | 0.6803 (-0.0302) | 0.6932 (+0.0085) | 0.7210 ± 0.0246 (+0.0425) |
| 0.75 | 0.7124 (+0.0793) | 0.6877 (-0.0228) | 0.7102 (+0.0254) | 0.6972 ± 0.0239 (+0.0187) |
| 1.00 (balance-to-max) | 0.7009 (+0.0677) | **0.6516 (-0.0590)** | 0.7168 (+0.0321) | 0.7185 ± 0.0019 (+0.0399) |

pore F1:

| ratio | BatchNorm control | GroupNorm only (new) | GroupNorm + sampler, seed 42 | GroupNorm + sampler, 3 seeds |
|---|---|---|---|---|
| 0.00 | 0.4000 | 0.4400 | 0.3441 | 0.3527 ± 0.0611 |
| 0.25 | 0.4554 (+0.0554) | 0.4086 (-0.0314) | excluded | 0.5380 ± 0.0695 (+0.1811) (n=2) |
| 0.50 | 0.2192 (-0.1808) | 0.3908 (-0.0492) | 0.4000 (+0.0559) | 0.4554 ± 0.0480 (+0.1028) |
| 0.75 | 0.4632 (+0.0632) | 0.4000 (-0.0400) | 0.5055 (+0.1614) | 0.4458 ± 0.0541 (+0.0931) |
| 1.00 | 0.4222 (+0.0222) | 0.4259 (-0.0141) | 0.5053 (+0.1612) | 0.4658 ± 0.0475 (+0.1132) |

Read with two cautions before drawing the obvious conclusion. First, the
GroupNorm-only column has the **highest real-only baseline of any seed-42 arm
in this repository** (0.7106 against 0.6332, 0.6848 and the published
0.6936), and the r=0 classifier never touches a pool: those four values are
four runs of the *same* configuration on the same real images at the same
seed. Their spread, 0.077 macro-F1 and 0.096 pore F1, is the true fixed-seed
noise floor of this benchmark, and it is far wider than the single repeat
section 10 measured (0.009 / 0.034). A gain measured against an unusually good
baseline shrinks by construction, so the negative brackets here are partly
that. Second, it is one seed. What survives both cautions is this: the pool
with the best FID of the three (141.8) has the worst r=1.0 score of the three
(0.6516 against the control's 0.7009 and the recommended arm's 0.7168), and
the only published arm below it is v0.2.0's latent-32 generator (0.6418). So
in this repository FID does not predict downstream usefulness even in sign,
which is the strongest form of section 6's "diagnostic, not a criterion". On the evidence here the weighted sampler is what makes the
GroupNorm generator's samples *useful* for augmentation, even though it
costs FID and adds nothing to reconstruction; that reading rests on one seed
and is stated as such.

### 11.4 The BatchNorm control at three seeds: its r=1.0 gain does not survive (D3)

Section 9 compared a three-seed GroupNorm + sampler arm with a one-seed
BatchNorm control and listed that asymmetry as the third thing it did not
establish. `runs/probe_eq_g0.1_s43` and `_s44` are the missing seeds: the
section 9 control command with `--seed 43` and `--seed 44`, each followed by
its own pool (`generated_ctrl_s43`, `_s44`) and sweep (`runs/sweep_ctrl_s43`,
`_s44`, `--selection final`, `--num_workers 4`). The control now has the same
protocol as the recommended arm - generator, pool and classifier all vary with
the seed.

Generator side, recomputed from the six committed histories:

| seed | arm | best `val_recon` | `loss_d` median | epochs `loss_d` < 1.5 | FID min |
|---|---|---|---|---|---|
| 42 | BatchNorm control | 0.033214 | 0.819 | 97% | 142.3 |
| 43 | BatchNorm control (new) | 0.041503 | 0.707 | 99% | 171.6 |
| 44 | BatchNorm control (new) | 0.047160 | 0.527 | 99% | 209.5 |
| 42 | GroupNorm + sampler | 0.017675 | 1.998 | 0% | 191.9 |
| 43 | GroupNorm + sampler | 0.020810 | 1.999 | 0% | 155.4 |
| 44 | GroupNorm + sampler | 0.016896 | 1.998 | 0% | 191.6 |

The section 9 generator claims hold on every seed, not only the first: the
BatchNorm discriminator is dominant on all three (median `loss_d` 0.53-0.82,
97-99% of epochs below 1.5) and the GroupNorm one sits at 2.0 on all three,
and the recommended arm's reconstruction is roughly twice as good on every
seed (0.0169-0.0208 against 0.0332-0.0472). FID again does not follow: the
control's minimum is better on seed 42, worse on 43 and 44.

Downstream, macro-F1, gains over the same column's r=0:

| ratio | control s42 | control s43 (new) | control s44 (new) | control, 3 seeds | GroupNorm + sampler, 3 seeds |
|---|---|---|---|---|---|
| 0.00 (real only) | 0.6332 | 0.7584 | 0.7150 | 0.7022 ± 0.0636 | 0.6785 ± 0.0447 |
| 0.25 | 0.7123 (+0.0791) | 0.7259 (-0.0325) | 0.7347 (+0.0197) | 0.7243 ± 0.0113 (+0.0221) | 0.7419 ± 0.0181 (+0.0665) (n=2) |
| 0.50 | 0.6341 (+0.0010) | 0.7407 (-0.0177) | 0.7283 (+0.0133) | 0.7011 ± 0.0583 (-0.0011) | 0.7210 ± 0.0246 (+0.0425) |
| 0.75 | 0.7124 (+0.0793) | 0.7277 (-0.0307) | 0.7211 (+0.0061) | 0.7204 ± 0.0077 (+0.0182) | 0.6972 ± 0.0239 (+0.0187) |
| 1.00 (balance-to-max) | 0.7009 (+0.0677) | 0.7361 (-0.0223) | 0.6902 (-0.0248) | **0.7091 ± 0.0240 (+0.0069)** | **0.7185 ± 0.0019 (+0.0399)** |

pore F1:

| ratio | control s42 | control s43 (new) | control s44 (new) | control, 3 seeds | GroupNorm + sampler, 3 seeds |
|---|---|---|---|---|---|
| 0.00 | 0.4000 | 0.5510 | 0.4176 | 0.4562 ± 0.0826 | 0.3527 ± 0.0611 |
| 0.25 | 0.4554 (+0.0554) | 0.5053 (-0.0458) | 0.4583 (+0.0408) | 0.4730 ± 0.0280 (+0.0168) | 0.5380 ± 0.0695 (+0.1811) (n=2) |
| 0.50 | 0.2192 (-0.1808) | 0.5490 (-0.0020) | 0.4255 (+0.0079) | 0.3979 ± 0.1666 (-0.0583) | 0.4554 ± 0.0480 (+0.1028) |
| 0.75 | 0.4632 (+0.0632) | 0.4902 (-0.0608) | 0.3913 (-0.0263) | 0.4482 ± 0.0511 (-0.0080) | 0.4458 ± 0.0541 (+0.0931) |
| 1.00 | 0.4222 (+0.0222) | 0.4865 (-0.0645) | 0.3585 (-0.0591) | **0.4224 ± 0.0640 (-0.0338)** | **0.4658 ± 0.0475 (+0.1132)** |

**Section 9's control sentence is withdrawn.** It read: "The matched BatchNorm
control preserves the central sign flip ... Together these measurements rule
out the claim that GroupNorm or the weighted sampler is *required* for the
positive r=1.0 result." That rested on seed 42 alone, and seed 42 turns out to
be the control's one good seed: its per-seed r=1.0 macro-F1 deltas are +0.0677,
-0.0223 and -0.0248, for a mean of +0.0069, and its pore F1 falls at r=1.0
on two seeds of three for a mean of -0.0338. The recommended arm's +0.0399 /
+0.1132 at r=1.0 therefore stands *without* a BatchNorm counterpart, and
whether the GroupNorm + sampler pair is required for the r=1.0 gain is back
to being open - what the three seeds show is that the control does not
reproduce it, not that the pair causes it (section 11.3 shows GroupNorm alone
does not either, on one seed). Two things do survive for the control: r=0.25
is positive on two seeds of three and in the mean (+0.0221), and its
three-seed real-only baseline (0.7022) is *higher* than the recommended arm's
(0.6785) even though the r=0 classifier never sees a generator, which is the
fixed-seed noise floor of section 11.7 again and a reminder that the two
arms' absolute rows are not on a common footing; only the within-column gains
are.

### 11.5 The source-grouped split: the downstream gain does not survive (D4)

This is the measurement section 9's split-overlap audit asked for. The
recommended arm was rerun end to end - generator, pool, sweep - at seeds 42,
43 and 44 with `--split_by source` on every step, so that no LoHi-WELD source
frame contributes crops to both the train pool and the held-out test set
(`src/measure_split_overlap.py --split_by source` reports 0 shared crops and 0
shared frames on all three seeds). Everything else is the section 9 command
line, with one forced change: `--val_per_class 190` instead of 200, because
under frame grouping the seed-43 pore train pool holds 198 crops after the
40-image subset is drawn, so a 200-image validation set does not fit; the
validation and FID reference is therefore 760 images here rather than 800.
The test sets are 1,623 / 1,635 / 1,633 crops (pore 62 / 66 / 59) - the same
size as the crop-level 1,603 to within one frame's worth per class - but they
are *different images* from the crop-level test sets, so absolute rows are
not comparable across the two protocols; within-column gains are.

The generator does not notice the protocol. All three source-split runs sit at
the GroupNorm equilibrium (`loss_d` median 1.998-2.002, 0% of epochs below
1.5) and reach best `val_recon` 0.0158 / 0.0166 / 0.0184, the same band as the
crop-split seeds (0.0169-0.0208; the validation sets differ, so only the band
is comparable), with minimum FID 186.5 / 195.3 / 186.1 (legacy scale).

Downstream, macro-F1, gains over the same column's r=0:

| ratio | source s42 | source s43 | source s44 | **source split, 3 seeds** | crop split, 3 seeds (section 9) |
|---|---|---|---|---|---|
| 0.00 (real only) | 0.7239 | 0.6937 | 0.7465 | 0.7214 ± 0.0265 | 0.6785 ± 0.0447 |
| 0.25 | 0.7286 (+0.0046) | 0.7153 (+0.0215) | 0.7032 (-0.0432) | 0.7157 ± 0.0127 (-0.0057) | 0.7419 ± 0.0181 (+0.0665) (n=2) |
| 0.50 | 0.6946 (-0.0293) | 0.6696 (-0.0241) | 0.6896 (-0.0569) | 0.6846 ± 0.0132 (-0.0368) | 0.7210 ± 0.0246 (+0.0425) |
| 0.75 | 0.7157 (-0.0083) | 0.6677 (-0.0261) | 0.6834 (-0.0631) | 0.6889 ± 0.0245 (-0.0325) | 0.6972 ± 0.0239 (+0.0187) |
| 1.00 (balance-to-max) | 0.7305 (+0.0065) | 0.6751 (-0.0186) | 0.7026 (-0.0439) | **0.7027 ± 0.0277 (-0.0187)** | **0.7185 ± 0.0019 (+0.0399)** |

pore F1:

| ratio | source s42 | source s43 | source s44 | **source split, 3 seeds** | crop split, 3 seeds (section 9) |
|---|---|---|---|---|---|
| 0.00 | 0.5600 | 0.4040 | 0.5983 | 0.5208 ± 0.1029 | 0.3527 ± 0.0611 |
| 0.25 | 0.4848 (-0.0752) | 0.5200 (+0.1160) | 0.3951 (-0.2032) | 0.4666 ± 0.0644 (-0.0541) | 0.5380 ± 0.0695 (+0.1811) (n=2) |
| 0.50 | 0.3913 (-0.1687) | 0.3409 (-0.0631) | 0.3902 (-0.2080) | 0.3742 ± 0.0288 (-0.1466) | 0.4554 ± 0.0480 (+0.1028) |
| 0.75 | 0.4471 (-0.1129) | 0.4176 (+0.0135) | 0.3448 (-0.2535) | 0.4032 ± 0.0526 (-0.1176) | 0.4458 ± 0.0541 (+0.0931) |
| 1.00 | 0.4944 (-0.0656) | 0.3371 (-0.0670) | 0.4186 (-0.1797) | **0.4167 ± 0.0787 (-0.1041)** | **0.4658 ± 0.0475 (+0.1132)** |

**The two questions this arm was run to answer.**

*How much do the absolute numbers drop?* They do not. The source-split
real-only baseline is 0.7214 ± 0.0265 macro-F1 and 0.5208 ± 0.1029 pore F1,
against 0.6785 ± 0.0447 and 0.3527 ± 0.0611 on the crop split. So the
expectation section 9 and `docs/USAGE.md` stated - that crop-level numbers
are "optimistic about generalisation to unseen sources" - is not what the
measurement shows for the *real-only* classifier: a from-scratch ResNet-18
trained on 1,090 real crops scores as well on unseen frames as on seen ones
(the two test sets differ, so this is a comparison of bands, not of rows,
and the pore column has 59-66 test images). What the crop split was
optimistic about is the next question.

*Does the r=1.0 gain survive?* **No.** With source frames isolated, the
recommended arm's r=1.0 macro-F1 delta is +0.0065, -0.0186 and -0.0439 on the
three seeds (mean -0.0187), and its pore F1 falls at r=1.0 on **every** seed
(-0.0656, -0.0670, -0.1797; mean -0.1041). Every ratio's three-seed mean is at
or below the real-only baseline on both metrics; the only individual positive
cells are seed 42's +0.0046 / +0.0065 macro-F1 at r=0.25 / r=1.0 and seed 43's
r=0.25 (+0.0215 macro-F1, +0.1160 pore F1), none of which recurs on another
seed. The section 9 result - "+0.0399 macro-F1 and +0.1132 pore F1 at r=1.0,
a positive mean effect" - is therefore a property of the crop-level split,
under which a generator trained on crops of the test frames produces samples
that help a classifier on those same frames. Isolating the frames removes the
help and leaves a cost to the minority class.

This is the single most consequential measurement in this file, and it is
stated with its limits: three seeds, one dataset, one classifier, one
generator configuration (the recommended one; the BatchNorm control and the
GroupNorm-only arm were not rerun under the source split), the fixed-seed
noise floor of section 11.7 (which is wide enough to absorb any one of these
cells but not the sign of eight of the nine pore deltas at r >= 0.5, the ninth
being seed 43's +0.0135 at r=0.75), and the
760-image validation set the generator was checkpointed on. Within those
limits the reading is: **on LoHi-WELD, the measured augmentation benefit of
this re-implementation does not generalise to unseen source frames.** The
README's three-seed blockquote and the section 9 tables remain correct
statements about the crop-level protocol, and are kept; they should no longer
be read as evidence that the method helps on new welds.

### 11.6 The published sweeps rescored under `--selection best_val` (D1)

Section 10 asked what happens when the five published sweeps are scored under
the current default, `--selection best_val`, instead of the final-epoch
protocol behind every published table. The five were rerun from their own
pools with only that flag changed (`runs/sweep_paper2_s42_bestval`, `_s43_`,
`_s44_`, `runs/sweep_ctrl_s42_bestval`, `runs/sweep_v020_bestval`; the last
uses the v0.2.0 `generated` pool and the v0.2.0 configuration). Because the
classifier is not bit-reproducible (section 10, cause 1), each of these is a
fresh training run as well as a different scoring rule, so the comparison
below is protocol *plus* one repeat's noise; section 11.7 says how large that
noise is.

macro-F1, gains over the same column's r=0:

| ratio | v0.2.0, `final` (published) | v0.2.0, `best_val` | control s42, `final` | control s42, `best_val` | GroupNorm + sampler, 3 seeds, `final` | GroupNorm + sampler, 3 seeds, `best_val` |
|---|---|---|---|---|---|---|
| 0.00 (real only) | 0.6936 | 0.6760 | 0.6332 | 0.6056 | 0.6785 ± 0.0447 | 0.6495 ± 0.0591 |
| 0.25 | 0.7264 (+0.0327) | 0.6914 (+0.0154) | 0.7123 (+0.0791) | 0.6497 (+0.0440) | 0.7419 ± 0.0181 (+0.0665) (n=2) | 0.6415 ± 0.0596 (-0.0080) |
| 0.50 | 0.6767 (-0.0169) | 0.6171 (-0.0588) | 0.6341 (+0.0010) | 0.6779 (+0.0723) | 0.7210 ± 0.0246 (+0.0425) | 0.6648 ± 0.0372 (+0.0152) |
| 0.75 | 0.7029 (+0.0093) | 0.6130 (-0.0630) | 0.7124 (+0.0793) | 0.6351 (+0.0295) | 0.6972 ± 0.0239 (+0.0187) | 0.6835 ± 0.0260 (+0.0339) |
| 1.00 (balance-to-max) | 0.6418 (-0.0518) | **0.7086 (+0.0326)** | 0.7009 (+0.0677) | 0.6747 (+0.0691) | 0.7185 ± 0.0019 (+0.0399) | **0.6537 ± 0.0583 (+0.0041)** |

pore F1:

| ratio | v0.2.0, `final` | v0.2.0, `best_val` | control s42, `final` | control s42, `best_val` | GroupNorm + sampler, 3 seeds, `final` | GroupNorm + sampler, 3 seeds, `best_val` |
|---|---|---|---|---|---|---|
| 0.00 | 0.3778 | 0.3878 | 0.4000 | 0.2254 | 0.3527 ± 0.0611 | 0.3107 ± 0.1680 |
| 0.25 | 0.4731 (+0.0953) | 0.4082 (+0.0204) | 0.4554 (+0.0554) | 0.3333 (+0.1080) | 0.5380 ± 0.0695 (+0.1811) (n=2) | 0.3287 ± 0.1970 (+0.0180) |
| 0.50 | 0.3590 (-0.0188) | 0.2637 (-0.1240) | 0.2192 (-0.1808) | 0.4211 (+0.1957) | 0.4554 ± 0.0480 (+0.1028) | 0.3448 ± 0.1099 (+0.0341) |
| 0.75 | 0.4902 (+0.1124) | 0.1905 (-0.1973) | 0.4632 (+0.0632) | 0.2857 (+0.0604) | 0.4458 ± 0.0541 (+0.0931) | 0.4446 ± 0.0572 (+0.1339) |
| 1.00 | 0.3871 (+0.0093) | 0.4719 (+0.0842) | 0.4222 (+0.0222) | 0.3373 (+0.1120) | 0.4658 ± 0.0475 (+0.1132) | 0.3341 ± 0.1100 (+0.0234) |

Per seed, the recommended arm under `best_val` (its `final` value in
brackets): r=1.0 macro-F1 deltas +0.0036 (+0.0321), +0.0072 (+0.0895),
+0.0016 (-0.0018); pore F1 deltas +0.1127 (+0.1612), -0.0203 (+0.1829),
-0.0223 (-0.0045). The seed-42 r=0.25 arm, excluded in section 9 for its
final-epoch loss spike, scores 0.6043 macro-F1 under `best_val` against a
0.5885 real-only baseline for that run - no longer an outlier relative to its
own column, so the exclusion was a property of the final-epoch protocol and
does not carry over.

**Three things this establishes.**

1. *The conclusions depend on the scoring protocol.* Under `best_val` the
   recommended arm's r=1.0 gain shrinks from +0.0399 to +0.0041 macro-F1 (a
   value the noise floor swallows, though its sign is the same on all three
   seeds) and from +0.1132 to +0.0234 pore F1 (positive on one seed of
   three); the published v0.2.0 curve's "r=1.0 is the worst ratio" becomes
   "r=1.0 is the best ratio" (+0.0326); and the control's r=1.0 gain is
   unchanged (+0.0691 against +0.0677). No reading that depends on which
   ratio is best survives a change of protocol.
2. *`best_val` is not the better protocol here; it is a different one.* In
   all 25 rescored arms it restored an epoch between 6 and 24 of 100. The
   leftover-pool validation loss bottoms out early and then climbs past 1.0
   as the classifier fits its 1,090 real images, while the held-out test
   score keeps improving, so `best_val` scores an under-trained classifier:
   every `best_val` real-only baseline is below its `final` counterpart
   (0.6495 vs 0.6785, 0.6056 vs 0.6332, 0.6760 vs 0.6936). It also selects
   on a pool the generator was early-stopped on (`docs/USAGE.md`, limitations),
   so it trades the final-epoch protocol's exposure to late loss spikes for
   a soft optimistic bias and a systematic under-training. Neither protocol
   is a clean estimate; a selection pool disjoint from the generator's
   validation set, or a fixed schedule with more seeds, would be, and neither
   was run.
3. *The current default should be read as a change of protocol, not a fix.*
   v0.4.0 made `best_val` the default on the argument that a late spike should
   not define a reported number. That argument stands, but this rescoring
   shows the default also moves every absolute number down and every gain
   toward zero. Tables produced with the default are not comparable to the
   published ones, and `--selection final` remains the flag that reproduces
   them.

### 11.7 The fixed-seed noise floor, remeasured: 0.08-0.13 macro-F1, not 0.009

Section 10 quoted a noise floor of 0.009 macro-F1 and 0.034 pore F1 from one
repeat of the r=0 arm. The r=0 arm never touches a generated pool, so every
sweep in this repository at the same seed and split protocol is a further
repeat of exactly that configuration - same 1,090 real images, same test set,
same classifier, same seed, `--selection final` - and there are now enough
of them to measure the spread properly:

| seed, split | run | accuracy | macro-F1 | pore F1 |
|---|---|---|---|---|
| 42, crop | `sweep` (v0.2.0, published) | 0.8222 | 0.6936 | 0.3778 |
| 42, crop | `sweep_paper2_s42` | 0.8253 | 0.6848 | 0.3441 |
| 42, crop | `sweep_ctrl_s42_clean` (`--num_workers 0`) | 0.7867 | 0.6332 | 0.4000 |
| 42, crop | `sweep_gn_only` (new) | 0.8235 | 0.7106 | 0.4400 |
| 43, crop | `sweep_paper2_s43` | 0.7860 | 0.6311 | 0.2963 |
| 43, crop | `sweep_ctrl_s43` (new) | 0.8497 | 0.7584 | 0.5510 |
| 44, crop | `sweep_paper2_s44` | 0.8571 | 0.7198 | 0.4176 |
| 44, crop | `sweep_ctrl_s44` (new) | 0.8497 | 0.7150 | 0.4176 |

Spread between repeats of the identical configuration: **0.0774 macro-F1 and
0.0959 pore F1 at seed 42 (four runs), 0.1273 and 0.2547 at seed 43 (two
runs), 0.0048 and 0.0000 at seed 44 (two runs).** Section 10's pair was the
lucky kind. The mechanism is the one section 10 named - non-deterministic CUDA
kernels, 100 epochs at a constant learning rate with no schedule, and a
61-image minority class in the test set - but its magnitude is an order of
magnitude larger than was published, and it is the bound every single-seed
difference in sections 9, 10 and 11 has to clear. In particular the section 9
seed-42 control's +0.0677 at r=1.0, the section 11.3 GroupNorm-only -0.0590,
and every individual cell of section 11.6 are inside it. What is *not* inside
it is a sign that repeats across seeds and ratios, which is why section 11.5's
pore result (eight of nine negative) and section 11.4's three-seed control
mean are reported as findings and the single-seed cells are not. The
`docs/USAGE.md` limitation and the README sentence that quoted the 0.009 /
0.034 figures now quote these instead.

### 11.8 What section 11 does and does not establish

Established, within the stated limits:

- The discriminator equilibrium at `loss_d` = 2.0 is caused by the GroupNorm
  discriminator alone; the weighted sampler is not involved (11.3, one seed).
- Most of the reconstruction gain of the recommended arm is GroupNorm's too;
  the sampler adds about 0.0008 `val_recon`, below the run-to-run wobble
  (11.3).
- The recommended arm's r=1.0 gain on the crop-level split is a property of
  that split: under source-frame isolation it is -0.0187 macro-F1 and -0.1041
  pore F1 at three seeds, negative for pore on every seed (11.5).
- The BatchNorm control does not show an r=1.0 gain at three seeds (+0.0069 /
  -0.0338); section 9's control sentence is withdrawn (11.4).
- The downstream conclusions change with the scoring protocol, and
  `best_val` under-trains (11.6); the fixed-seed noise floor is 0.08-0.13
  macro-F1, not 0.009 (11.7).
- The two FID backends disagree about the ranking of this repository's own
  pools, and the pool with the best legacy FID is the worst downstream
  (11.1, 11.3).
- The conference paper's BCE objective on today's code and this data (11.2).

Not established, and not to be read into the above:

- Why the crop-level gain exists. The natural reading - a generator trained
  on crops of the test frames emits frame-specific detail that helps on those
  frames - is consistent with 11.5 but was not tested directly (for example
  by measuring nearest-neighbour distances from generated to test crops).
- Whether any configuration helps on unseen frames. Only the recommended arm
  was run under `--split_by source`; the control, the GroupNorm-only arm, the
  v0.2.0 configuration and filling rates in the 0.1-0.2 region were not.
- Anything about the papers' data or numbers (section 8).
- Anything at one seed. Every single-seed cell here sits inside the 11.7
  noise floor.

### Reproduce

All commands run from the repository root with `data/lohi` prepared as in the
README. The generator commands are the section 9 ones with the flags shown
changed; the sweeps add nothing but `--seed`, `--selection` and, for 11.5,
`--split_by source`. `--num_workers 4` throughout except where noted.

```bash
# 11.1 cross-scale FID: both backends on the same balanced sample of each pool.
for POOL in generated generated_ctrl generated_paper2 generated_paper2_s43 \
            generated_paper2_s44 generated_showcase; do
  for BE in pytorch_fid legacy; do
    python src/eval_fid.py --real_root data/lohi --fake_root "$POOL" \
      --channels 3 --fid_backend "$BE" --batch_size 16
  done
done
# Per-class FID of the showcase pool, into a scratch directory so the committed
# figures are untouched; --fid_backend legacy reproduces results/class_*.png
# pixel for pixel.
python src/make_class_figures.py --real_root data/lohi --gen_root generated_showcase \
  --out_dir /tmp/classfig_legacy --channels 3 --fid_backend legacy

# 11.2 the conference objective, 20 epochs, gamma 0.1 and 1.0.
for G in 0.1 1.0; do
  python src/train_joint.py --data_root data/lohi \
    --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
    --epochs 20 --batch_size 8 --img_size 224 --channels 3 --latent_dim 128 \
    --lr 1e-3 --lr_d 1e-3 --kl_weight 0.059 --perc_weight 0.1 --adv_weight "$G" \
    --patience 70 --lr_patience 70 --seed 42 --fid_backend legacy \
    --fid_every 5 --sample_every 10 --adv_loss bce --no_d_spectral_norm \
    --out_dir "runs/bce_nosn_g$G"
done

# 11.3 GroupNorm without the sampler: the section 9 recommended command minus
# --weighted_sampler, then its pool and sweep.
python src/train_joint.py --data_root data/lohi \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
  --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 128 \
  --lr 1e-3 --lr_d 1e-3 --kl_weight 0.059 --perc_weight 0.1 --adv_weight 0.1 \
  --patience 70 --lr_patience 70 --seed 42 --d_norm group --fid_backend legacy \
  --fid_every 5 --sample_every 10 --out_dir runs/paper2_gn_only
python src/generate.py --ckpt runs/paper2_gn_only/joint.pt --seed 42 \
  --out_root generated_gn_only --counts "deposit=450,discontinuity=300,pore=560,stain=0"
python src/train_classifier.py --data_root data/lohi --gen_root generated_gn_only \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --channels 3 --seed 42 --selection final --out_dir runs/sweep_gn_only

# 11.4 the BatchNorm control at seeds 43 and 44: the section 9 control command
# with --seed, then pool and sweep, exactly as for the recommended arm.
for S in 43 44; do
  python src/train_joint.py --data_root data/lohi \
    --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 200 \
    --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 128 \
    --lr 1e-3 --lr_d 1e-3 --kl_weight 0.059 --perc_weight 0.1 --adv_weight 0.1 \
    --patience 70 --lr_patience 70 --seed "$S" --fid_backend legacy \
    --fid_every 5 --sample_every 10 --out_dir "runs/probe_eq_g0.1_s$S"
  python src/generate.py --ckpt "runs/probe_eq_g0.1_s$S/joint.pt" --seed "$S" \
    --out_root "generated_ctrl_s$S" --counts "deposit=450,discontinuity=300,pore=560,stain=0"
  python src/train_classifier.py --data_root data/lohi --gen_root "generated_ctrl_s$S" \
    --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
    --channels 3 --seed "$S" --selection final --out_dir "runs/sweep_ctrl_s$S"
done

# 11.5 the recommended arm under the source-grouped split, three seeds.
# --val_per_class 190: the seed-43 pore train pool has 198 crops left after the
# subset under frame grouping, so the published 200 does not fit.
for S in 42 43 44; do
  python src/train_joint.py --data_root data/lohi \
    --subset "pore=40,deposit=150,discontinuity=300,stain=600" --val_per_class 190 \
    --epochs 70 --batch_size 8 --img_size 224 --channels 3 --latent_dim 128 \
    --lr 1e-3 --lr_d 1e-3 --kl_weight 0.059 --perc_weight 0.1 --adv_weight 0.1 \
    --patience 70 --lr_patience 70 --seed "$S" --d_norm group --weighted_sampler \
    --split_by source --fid_backend legacy --fid_every 5 --sample_every 10 \
    --out_dir "runs/paper2_src_s$S"
  python src/generate.py --ckpt "runs/paper2_src_s$S/joint.pt" --seed "$S" \
    --out_root "generated_src_s$S" --counts "deposit=450,discontinuity=300,pore=560,stain=0"
  python src/train_classifier.py --data_root data/lohi --gen_root "generated_src_s$S" \
    --subset "pore=40,deposit=150,discontinuity=300,stain=600" --channels 3 \
    --seed "$S" --selection final --split_by source --out_dir "runs/sweep_src_s$S"
done
python src/measure_split_overlap.py --data_root data/lohi \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --seeds 42,43,44 --split_by source

# 11.6 the five published sweeps rescored: same pool, same seed, --selection best_val.
python src/train_classifier.py --data_root data/lohi --gen_root generated_paper2 \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" --channels 3 \
  --seed 42 --selection best_val --out_dir runs/sweep_paper2_s42_bestval
#   ... then generated_paper2_s43 / _s44 with --seed 43 / 44, generated_ctrl
#   with --seed 42 (runs/sweep_ctrl_s42_bestval), and generated with --seed 42
#   (runs/sweep_v020_bestval).

# The v0.5.2 figure, from the committed CSVs alone.
python src/make_multiseed_figure.py \
  --seed_sweep results/metrics/sweep_paper2_s42/sweep_metrics.csv \
  --seed_sweep results/metrics/sweep_paper2_s43/sweep_metrics.csv \
  --seed_sweep results/metrics/sweep_paper2_s44/sweep_metrics.csv \
  --exclude "results/metrics/sweep_paper2_s42/sweep_metrics.csv=0.25" \
  --label "GroupNorm + balanced sampler, crop-level split (section 9)" \
  --arm "same arm, source-grouped split (section 11.5)=results/metrics/sweep_src_s42/sweep_metrics.csv;results/metrics/sweep_src_s43/sweep_metrics.csv;results/metrics/sweep_src_s44/sweep_metrics.csv" \
  --arm "matched BatchNorm control, crop-level split (section 11.4)=results/metrics/sweep_ctrl_s42_clean/sweep_metrics.csv;results/metrics/sweep_ctrl_s43/sweep_metrics.csv;results/metrics/sweep_ctrl_s44/sweep_metrics.csv" \
  --reference "published v0.2.0, latent-32 BatchNorm (seed 42)=results/metrics/sweep/sweep_metrics.csv" \
  --title "Filling-rate sensitivity across seeds: crop-level versus source-grouped split (v0.5.2)" \
  --minority pore --out results/filling_rate_multiseed_v052.png
```

The generator checkpoints and pools behind every run above are attached to
the Zenodo record (`release_assets/MANIFEST.md` lists each file and its
SHA-256), so the sweeps can be repeated without retraining a generator.
