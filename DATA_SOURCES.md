# Data Sources

This repository bundles **no data**. Every dataset used is fetched by the user
from its official source, and each dataset's own terms apply. Never commit
dataset files to this repository or attach them to releases.

## Used in this project

### RIAWELC (primary)

- **What**: 24,407 radiographic (X-ray) weld defect images, 224x224, 8-bit PNG.
  Four classes: LP (lack of penetration), PO (porosity), CR (cracks),
  ND (no defect).
- **Source**: https://github.com/stefyste/RIAWELC (multi-part RAR archives in-repo)
- **Terms**: the authors state the dataset "is released freely"; no formal
  license file. Using it requires citing the two papers below. We do not
  re-upload or redistribute the dataset; the only exception is the small
  qualitative figure described below.
- **Qualitative figure exception**: `results/real_vs_generated.png` in this
  repository shows 32 randomly picked real RIAWELC thumbnails (downsampled to
  128x128) side by side with generated samples, strictly to illustrate what
  this re-implementation's output looks like. The dataset itself is never
  bundled, attached to releases, or re-hosted; fetch it from the official
  repository above and cite the two papers below if you use it.
- **Required citations**:
  1. Benito Totino, Fanny Spagnolo, Stefania Perri, "RIAWELC: A Novel Dataset
     of Radiographic Images for Automatic Weld Defects Classification",
     Proc. Interdisciplinary Conference on Mechanics, Computers and Electrics
     (ICMECE 2022), Barcelona, Spain, 6-7 October 2022.
  2. Stefania Perri, Fanny Spagnolo, Fabio Frustaci, Pasquale Corsonello,
     "Welding Defects Classification Through a Convolutional Neural Network",
     Manufacturing Letters, Elsevier.
- **Modality note**: X-ray imagery; the reproduced paper used visible-light
  melt-pool images. The substitution is disclosed in `README.md`.

## Alternatives (not used by default)

- **Roboflow Universe "weld" dataset (Kaggle-sourced)**: ~4756 visible-light
  weld images, CC BY 4.0 (attribution required). Closer to the paper's
  imaging modality. Requires a free Roboflow account and API key.
- **GDXray welds series** (https://grima.cl/datasets/): X-ray weld images,
  free for research use.

## Reproduced paper (for attribution, not a data source)

J. Yang, L. Yuan, H. Mu, F. He, D. Ding, ..., Z. Pan, "Generation of WAAM
Defect Images Using a Hybrid CVAE-CGAN: A Data Augmentation Strategy for Small
and Imbalanced Datasets", IEEE conference paper, 2025 (IEEE Xplore doc. no.
11168313). The original WAAM dataset is proprietary and is NOT used here.
