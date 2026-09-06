# Calibration notes

This file records **why this repo's configuration differs from the two papers it
reproduces** - which values were re-derived, which components were taken from the
journal extension rather than the conference paper, and which of the papers'
results cannot be reproduced here at all. It is optional reading: the defaults in
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

**Addendum: the logged KL is already at the paper's magnitude, and beta is not
the fidelity lever.** The sweep above was eight epochs on RIAWELC. Re-measured on
LoHi-WELD over full 70-epoch runs, `cvae_loss` logs `kl` as a mean over the batch
*and* over latent dimensions, so the quantity comparable to the paper's is
`latent_dim x kl` - the total latent information budget in nats:

| run | latent dim | beta | `latent_dim x kl` |
|---|---|---|---|
| v0.2.0 published | 32 | 0.015 | 6.8 - 7.2 |
| probe, latent 128 | 128 | 0.059 | 8.35 - 8.74 |
| probe, latent 128, equal budget | 128 | 0.059 | 11.7 - 15.9 |

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
defect regions. Grayscale conversion is dataset-dependent. RIAWELC already
satisfies it - every one of its 24,407 files is PIL mode `L`. LoHi-WELD does not:
its files are mode `RGB`, so this repo trains on it with `--channels 3` and lets
the first convolution learn the channel mixing rather than collapsing it
upstream. Both are faithful to the paper's intent, which is a fixed input
representation, not specifically one channel.

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
own ablation (its Fig. 15(a)) shows this directly:

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

### The downstream verdict: filling rate 1.0 flips sign

Qualification 2 deferred to the filling-rate sweep. It has now been run for this
arm (`runs/sweep_paper2_s42`, seed 42, pool `generated_paper2`, 1,310 synthetic
images: pore 560 / deposit 450 / discontinuity 300 / stain 0), against the
published v0.2.0 curve (`runs/sweep`).

macro-F1, and in brackets the gain over **that arm's own** r=0 baseline:

| ratio | v0.2.0 published | GroupNorm + balanced sampler |
|---|---|---|
| 0.00 (real only) | 0.6936 | 0.6848 |
| 0.25 | **0.7264** (+0.033) | 0.5868 - *invalid, see below* |
| 0.50 | 0.6767 (-0.017) | 0.6932 (+0.008) |
| 0.75 | 0.7029 (+0.009) | 0.7102 (+0.025) |
| 1.00 (balance-to-max) | 0.6418 (**-0.052**, worst) | **0.7168** (**+0.032**, best) |

pore F1, the minority class the whole experiment exists for:

| ratio | v0.2.0 published | GroupNorm + balanced sampler |
|---|---|---|
| 0.00 | 0.3778 | 0.3441 |
| 0.25 | 0.4731 | 0.3039 - *invalid* |
| 0.50 | 0.3590 | 0.4000 |
| 0.75 | 0.4902 | **0.5055** |
| 1.00 | 0.3871 | **0.5053** |

**The structural result is the sign flip at r=1.0.** In the published curve,
filling every class to `N_max` was the *worst* ratio, 0.052 macro-F1 below its own
baseline, and the README concluded that the papers' balance-to-max recommendation
"does not transfer here". Under this arm r=1.0 is the *best* ratio, 0.032 above
its own baseline, and pore F1 there is 0.5053 against the published 0.3871 - a
difference of 0.118, several times the section 10 noise floor. Across the four
arms that completed cleanly the new curve is **monotone increasing** in filling
rate (0.6848 -> 0.6932 -> 0.7102 -> 0.7168), which is the shape the journal paper
reports and the shape the v0.2.0 curve did not have.

**r=0.25 is not a measurement.** Its training loss sat at 0.0001 by epoch 90, then
reported 0.2617 at epoch 100, and `train_classifier.py` scores the final epoch
(section 10). The spiked weights are what was evaluated. It is reported rather
than dropped, and excluded from the monotonicity claim.

**Four things this does not establish.**

1. *It is not an isolated test of the two switches.* The comparison is against the
   published v0.2.0 curve, and those two configurations differ in five respects at
   once: `latent_dim` 32 -> 128, `kl_weight` 0.015 -> 0.059, discriminator
   BatchNorm -> GroupNorm, balanced sampler off -> on, and early-stopped at 25
   epochs -> the full 70. The matched control - sweeping the pool from
   `runs/probe_eq_g0.1`, the latent-128 BatchNorm arm this one was trained against
   - **has not been run**. Until it is, the r=1.0 flip cannot be attributed to
   GroupNorm or the sampler rather than to the larger latent or the longer
   schedule.
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
3. *Single seed.* Section 10 measures run-to-run spread of 0.034 pore F1 and 0.009
   macro-F1 at fixed seed. The r=1.0 deltas clear that comfortably; the r=0.5 and
   r=0.75 deltas (+0.041 and +0.015 pore F1) do not clear it decisively and should
   not be quoted on their own.
4. *The absolute numbers remain non-comparable to the papers'*, for the reasons in
   section 8: a from-scratch ResNet-18 over single images, not an LSTM/GRU over
   21-frame sequences.

Seeds 43 and 44 are being run to replace caveat 3, each retraining the generator
and regenerating the pool, because `verify_generation_meta` refuses to sweep a
pool drawn from a different split.

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
# --seed 42 are all defaults and are left implicit; --channels is not (default 1).
python src/train_classifier.py --data_root data/lohi --gen_root generated_paper2 \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --channels 3 --out_dir runs/sweep_paper2_s42
```

The matched control that caveat 1 asks for is the same two commands pointed at
`runs/probe_eq_g0.1`, into its own pool directory - `verify_generation_meta`
refuses to sweep a pool whose `subset`, `seed` or `test_frac` disagree with the
sweep's, and `generate.py` writes the seed from the checkpoint, so pool and sweep
have to be produced as a matched pair rather than reused across arms.

```bash
python src/generate.py --ckpt runs/probe_eq_g0.1/joint.pt --seed 42 \
  --out_root generated_ctrl \
  --counts "deposit=450,discontinuity=300,pore=560,stain=0"

python src/train_classifier.py --data_root data/lohi --gen_root generated_ctrl \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --channels 3 --out_dir runs/sweep_ctrl_s42
```

## 10. The downstream classifier is not bit-reproducible, and it scores the final epoch

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
ratio's loop, but never calls `torch.use_deterministic_algorithms(True)` or sets
`cudnn.deterministic`. On CUDA the convolution backward passes use atomic-add
kernels whose accumulation order is not fixed, so identical seed, data and code
still diverge from the first step. The loss curves confirm it: epoch 10 was
0.4266 in the published run and 0.4008 in the repeat.

**The measured spread is the noise floor for this benchmark.** On 61 pore test
images, one repeat of an unchanged configuration moved pore F1 by **0.034** and
macro-F1 by **0.009**. Any single-seed difference smaller than that is not
evidence, in either direction. The README's caveat ("treat differences under
~0.02 macro-F1 as noise") is in the right ballpark for macro-F1 but too
optimistic for the minority class, where 61 test images make F1 granular: one
image is worth ~0.016 of pore recall on its own.

### Cause 2: the final epoch is what gets scored

`train_one` runs a constant Adam learning rate for the full `--epochs` with no
schedule, no early stopping and no checkpointing; `evaluate` is then called on
whatever weights remain. There is no best-epoch selection anywhere in the file.

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
arm can be invalidated outright by a late spike. Averaging over seeds (task 28)
reduces the first problem; only a protocol change would reduce the second.

**Reproduce the noise floor.** Run the `r=0` arm twice and compare; no code
change is needed to see it.

```bash
python src/train_classifier.py --data_root data/lohi --gen_root generated \
  --subset "pore=40,deposit=150,discontinuity=300,stain=600" \
  --ratios "0.0" --channels 3 --epochs 100 --seed 42 --out_dir runs/noise_a
```
