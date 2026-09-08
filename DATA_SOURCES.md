# Data Sources

This repository bundles **no data**. Every dataset used is fetched by the user
from its official source, and each dataset's own terms apply. Never commit
dataset files to this repository or attach them to releases.

## Used in this project

### LoHi-WELD (primary)

- **What**: 3,022 **visible-light** weld-bead images (1,022 high-resolution
  ~780x190, 2,000 low-resolution ~100-200x40) with four defect classes
  (pore, deposit, discontinuity, stain). The visible-light modality matches the
  reproduced papers' melt-pool imagery.
- **Source**: Google Drive archive linked from
  https://github.com/SylvioBlock/LoHi-Weld (download requires a Google account,
  so the owner fetches it manually; it is never re-hosted here).
- **Format**: ships as YOLO-style detection data - each ``<stem>.jpg`` paired
  with a ``<stem>.yolo`` label file (a labelme-style ``.json`` is also present
  and is redundant). `src/prepare_yolo_crops.py` crops every annotated box into
  `<out_root>/<ClassName>/`, which simultaneously performs the reproduced
  papers' ROI-extraction preprocessing step.
- **Preparation actually used** (high-resolution subset only; the low-resolution
  beads crop to ~16px boxes, which upscaled to 224 carry no usable signal):

  ```bash
  python src/prepare_yolo_crops.py \
    --input_root <archive>/weld-dataset/high_resolution_welds \
    --out_root data/lohi --img_size 224 --channels 3 --min_side 16 \
    --classes pore,deposit,discontinuity,stain
  ```

  yielding 8,012 crops: stain 3,540 / discontinuity 2,975 / deposit 1,193 /
  pore 304 - an **11.6x** imbalance, close to the conference paper's 14.7x.
  (`--min_side 16` filters on each box's **longest** edge - the behaviour the
  historical 8,012-crop protocol has always used.)
  Crops keep their native **RGB** channels: the papers' grayscale is a property
  of their camera, not of the method, and colour carries defect signal here.
- **Provenance**: the LoHi-WELD repository is built on the YOLOv7 implementation
  by WongKinYiu. This repo uses only the dataset and its annotations, not that
  code, so no citation of it is required; the provenance is recorded here for
  transparency.
- **Terms**, as stated by the authors: the dataset and code "can be used for
  research, non-comercial or comercial purposes for free with proper citation".
  Citing the paper below is therefore a usage requirement.
- **Required citation**: Sylvio Biasuz Block, Ricardo Dutra da Silva,
  Andre Eugenio Lazzaretti, Rodrigo Minetto, "LoHi-WELD: A Novel Industrial
  Dataset for Weld Defect Detection and Classification, a Deep Learning Study,
  and Future Perspectives", IEEE Access, vol. 12, pp. 77442-77453, 2024.
  DOI: [10.1109/ACCESS.2024.3407019](https://doi.org/10.1109/ACCESS.2024.3407019)
  (IEEE Access is gold open access). The committed figures in `results/` are
  derived from this dataset's imagery, so they stay subject to this citation
  requirement as well.

### RIAWELC (previously used; superseded by LoHi-WELD)

Retained here as a historical footnote, not as a current experiment dataset.
No figure, table or number in the current results derives from it.

- **What**: 24,407 radiographic (X-ray) weld defect images, 227x227, PIL mode
  `L`, four classes (CR 4,452 / LP 7,635 / ND 6,000 / PO 6,320).
  Source: https://github.com/stefyste/RIAWELC.
- **Why it was superseded**: X-ray radiographs are band-limited and low
  contrast (99.98% of spectral energy below r=0.1 of the Nyquist radius), and
  the dataset's natural imbalance is only 1.71x, so it matches neither the
  papers' visible-light modality nor their "small and imbalanced" premise. The
  calibrated generator reached a reconstruction ceiling on it - crack and
  no-defect columns stayed near-flat - recorded in `docs/CALIBRATION.md`.
- **Why the citation stays**: v0.1.0 and the generator calibration record were
  produced with RIAWELC and are already public (see the v0.1.0 tag and
  `CHANGELOG.md`). The citation obligation attaches to having used it, so both
  required citations are retained even though no current result uses it:
  1. Benito Totino, Fanny Spagnolo, Stefania Perri, "RIAWELC: A Novel Dataset
     of Radiographic Images for Automatic Weld Defects Classification",
     Proc. Interdisciplinary Conference on Mechanics, Computers and Electrics
     (ICMECE 2022), Barcelona, Spain, 6-7 October 2022.
  2. Stefania Perri, Fanny Spagnolo, Fabio Frustaci, Pasquale Corsonello,
     "Welding Defects Classification Through a Convolutional Neural Network",
     Manufacturing Letters, vol. 35, pp. 29-32, 2023.
     DOI: [10.1016/j.mfglet.2022.11.006](https://doi.org/10.1016/j.mfglet.2022.11.006)

## Alternatives (not used by default)

- **Roboflow Universe "weld" dataset (Kaggle-sourced)**: ~4756 visible-light
  weld images, CC BY 4.0 (attribution required). Closer to the paper's
  imaging modality. Requires a free Roboflow account and API key.
- **GDXray welds series** (https://grima.cl/datasets/): X-ray weld images,
  free for research use.

## Reproduced papers (for attribution, not data sources)

Neither paper's dataset is used here. Both original datasets are proprietary
and are NOT distributed, re-hosted or requested by this repository. Only the
published *method* is re-implemented. Bibliographic details below are factual
citation data; no figures, tables or passages are copied from either paper.

1. **Primary method reference (conference).** Junle Yang, Lei Yuan, Haochen Mu,
   Fengyang He, Donghong Ding, Zengxi Pan, Huijun Li, "Generation of WAAM
   Defect Images Using a Hybrid CVAE-CGAN: A Data Augmentation Strategy for
   Small and Imbalanced Datasets", Proc. 15th IEEE International Conference on
   CYBER Technology in Automation, Control, and Intelligent Systems (CYBER
   2025), Shanghai, China, 15-18 July 2025, pp. 1-6.
   DOI: [10.1109/CYBER67662.2025.11168313](https://doi.org/10.1109/CYBER67662.2025.11168313)
   (IEEE Xplore doc. no. 11168313; IEEE copyright, not open access).

2. **Secondary method reference (journal extension by the same first author).**
   Junle Yang, Lei Yuan, Fengyang He, Zening Wu, Donghong Ding, Zengxi Pan,
   Huijun Li, "Physics-guided generative data augmentation for vision-based
   signal processing under class-imbalanced conditions in directed energy
   deposition monitoring system", Mechanical Systems and Signal Processing,
   vol. 250, article 114138, 2026.
   DOI: [10.1016/j.ymssp.2026.114138](https://doi.org/10.1016/j.ymssp.2026.114138)
   **Open access under CC BY 4.0** - freely readable at the DOI link above.
   This repo uses it to fill in architecture and hyperparameter details the
   conference paper omits.
