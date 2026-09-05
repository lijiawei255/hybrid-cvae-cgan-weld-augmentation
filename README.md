# Hybrid CVAE-CGAN for WAAM Defect Image Augmentation (Unofficial Re-implementation)

An **unofficial, method-level re-implementation attempt** of the hybrid CVAE-CGAN framework proposed in:

> J. Yang, L. Yuan, H. Mu, F. He, D. Ding, ..., Z. Pan, *"Generation of WAAM Defect Images Using a Hybrid CVAE-CGAN: A Data Augmentation Strategy for Small and Imbalanced Datasets"*, IEEE conference paper, 2025 (IEEE Xplore doc. no. 11168313).

This repo exists as a **public reference for anyone attempting a similar
reproduction**: it documents what was re-implemented from the paper, which
public substitute dataset was used, and where the implementation diverges
from the original.

The method generates class-conditional weld defect images to augment small and
imbalanced datasets. Training follows two stages: a Conditional VAE learns the
class-conditional image distribution, then its decoder is reused as the
generator of a conditional GAN for adversarial sharpening.

## Disclaimer

- This repository is an **independent re-implementation based solely on the published paper**. It is **not affiliated with, endorsed by, or connected to the original authors or their institutions**.
- The original authors' WAAM dataset is proprietary and **was not used, accessed, or requested** for this project.
- All experiments here run on **publicly available, openly licensed datasets** (see `DATA_SOURCES.md`). Each dataset's own license applies; attribution notices are kept in the data download instructions.
- No figures, tables, or text from the paper are reproduced in this repository.
- This project re-implements a *method*, and does **not** claim to reproduce the paper's experimental results or reported numbers.

## Quickstart

```bash
pip install -r requirements.txt

# Stage 1: train the CVAE
python src/train_cvae.py --data_root data --img_size 128 --epochs 100

# Stage 2: adversarial fine-tuning (G initialized from the CVAE decoder)
python src/train_cgan.py --data_root data --cvae_ckpt runs/cvae/cvae.pt --epochs 200

# Generate class-conditional images
python src/generate.py --ckpt runs/cgan/cgan.pt --per_class 1000

# FID evaluation
python src/eval_fid.py --real_root data --fake_root generated

# 1-minute end-to-end sanity check on synthetic data
python src/smoke_test.py
```

## Results (this repository's own runs)

Single-seed runs on the full RIAWELC dataset (24,407 images, 128x128, batch
64). These numbers describe *this* re-implementation on a substitute dataset
and are **not** comparable to the original paper's results.

| Stage | Setting | Result |
|---|---|---|
| Stage 1 CVAE | 100 epochs, latent dim 128 | final L1 recon 0.102, KL ~7.0 |
| Stage 2 CGAN | 200 epochs, G initialized from the CVAE decoder | final D loss 0.38, no mode collapse |
| FID (InceptionV3) | 24,407 real vs 4,000 generated | 37.96 |

Downstream augmentation-gain experiment (pretrained ResNet-18, stratified
80/20 real-only test split, seed 42): condition A = imbalanced real subset
(20% of each defect class kept, ~7.7k images), condition B = A + 1,000
generated images per defect class (~10.7k). Both conditions reach ~99.4%
test accuracy (macro-F1 0.9931 vs 0.9932) — **no measurable augmentation
gain on this dataset**: the ImageNet-pretrained classifier solves RIAWELC
almost perfectly even in the imbalanced baseline, so the task saturates and
leaves no headroom for augmentation to help. The paper's setting (1,898
proprietary melt-pool images, as few as 36 samples per class) is far harder;
demonstrating a gain would need a harder substitute task. This negative
result is reported as-is. Protocol notes: `src/train_classifier.py`; the
generator was trained on all real images, so generated data may carry
information about test images (same protocol limitation as the paper).

Real vs. generated samples per class (upper row real, lower row generated):

![Real vs generated samples](results/real_vs_generated.png)

## Repository layout

```
src/models.py           CVAE encoder/decoder + CGAN discriminator (class-conditional)
src/data.py             class-folder loader + imbalance simulation
src/train_cvae.py       Stage 1 training
src/train_cgan.py       Stage 2 adversarial fine-tuning
src/generate.py         class-conditional sampling
src/eval_fid.py         FID evaluation (InceptionV3 features)
src/train_classifier.py downstream augmentation-gain experiment (ResNet-18)
src/make_comparison.py  real-vs-generated comparison grid into results/
src/smoke_test.py       end-to-end sanity check on synthetic data
```

## Data

Experiments use **[RIAWELC](https://github.com/stefyste/RIAWELC)**: 24,407
radiographic weld defect images (four classes: LP / PO / CR / ND), released
freely by its authors with a citation requirement — if you use it, cite the
two papers listed in `DATA_SOURCES.md`.

Note the modality difference: RIAWELC contains X-ray images, whereas the
original paper used visible-light melt-pool images. The *method* transfers
unchanged, but this repo makes no claim about the paper's original imagery.

No dataset is bundled here, and none should ever be committed — `data/`,
`generated/`, `runs/` and `*.pt` are git-ignored by design. See
`DATA_SOURCES.md` for download links, usage terms, and license-clean
alternatives.

## Reusing this work

Want to generate your own training data? Prepare class-folder images (or pull
RIAWELC per `DATA_SOURCES.md`), train Stage 1 then Stage 2, and sample with
`src/generate.py`. Images generated by models you train are yours to use; the
code is MIT-licensed. A link back to this repo is appreciated but not required.

## License

Code: [MIT](LICENSE). Third-party datasets remain under their own licenses.

## How to cite

If this re-implementation helps your work, cite the **original paper** and the
dataset (GitHub's "Cite this repository" widget shows both). Citing this
repository itself is optional.

```bibtex
@inproceedings{yang2025hybridcvaecgan,
  title  = {Generation of WAAM Defect Images Using a Hybrid CVAE-CGAN:
            A Data Augmentation Strategy for Small and Imbalanced Datasets},
  author = {Yang, Junle and Yuan, Lei and Mu, Haonan and He, Feng and
            Ding, Donghong and Pan, Zengxi and others},
  year   = {2025},
  note   = {IEEE conference paper, IEEE Xplore doc. no. 11168313}
}

@inproceedings{totino2022riawelc,
  title     = {RIAWELC: A Novel Dataset of Radiographic Images for
               Automatic Weld Defects Classification},
  author    = {Totino, Benito and Spagnolo, Fanny and Perri, Stefania},
  booktitle = {Proc. Interdisciplinary Conference on Mechanics, Computers
               and Electrics (ICMECE 2022)},
  year      = {2022}
}
```
