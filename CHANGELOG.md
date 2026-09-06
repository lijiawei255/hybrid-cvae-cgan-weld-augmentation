# Changelog

All notable deliberate deviations from the reference skeleton shipped in the
reproduction guide (its Appendix A) are documented here.

## [Unreleased]

### Added

- Initial repository rebuild from the reproduction guide's Appendix A:
  verbatim recreation of `README.md`, `CITATION.cff`, `DATA_SOURCES.md`,
  `requirements.txt` and the seven `src/` scripts.
- `src/train_classifier.py`: downstream utility experiment required by the
  reproduction guide (sections 5-6) but not shipped in its Appendix A. It
  fine-tunes a pretrained ResNet-18 on (A) an imbalanced real subset vs (B)
  the same subset plus generated augmentation, and compares per-class
  recall/F1 on a held-out real-only test set.
- `src/make_comparison.py`: builds a real-vs-generated comparison grid (one
  row pair per class, the standard qualitative GAN presentation) into
  `results/`, for the README and for anyone pulling the repository to see
  what the pipeline's output looks like.
- `src/make_class_figures.py`: renders compact per-class real-vs-generated
  close-ups with the per-class FID in the caption (computed with the same
  InceptionV3 features as `src/eval_fid.py`), sized to stay readable when
  pasted directly into a chat window.

### Changed

- `CITATION.cff`: the owner-name placeholders were filled with the owner's
  GitHub username (`lijiawei255`), as explicitly sanctioned by the
  reproduction guide.
- `src/data.py`, `src/train_cvae.py`, `src/train_cgan.py` (pre-registered
  deviation, applied after a measured failure on Windows): `build_loader` now
  sets `persistent_workers=True` whenever `num_workers > 0`, and the two
  training scripts accept `--num_workers`. Evidence: with the original
  per-epoch worker respawn, CVAE+CGAN stages ran normally at ~37-44 s/epoch,
  but starting at CGAN epoch ~91 every epoch-end worker teardown blocked the
  main thread in `DataLoader._shutdown_workers -> multiprocessing.Process.join`
  for 30+ minutes per epoch (py-spy stack captured), stalling training to a
  halt while the GPU sat idle. Persistent workers eliminate the per-epoch
  teardown; `src/smoke_test.py` passes with the change.

## Results (first full-data run, 2026-09-06)

- CVAE (100 epochs, ~37 s/epoch): final reconstruction 0.102 (L1), KL ~7.0.
- CGAN (200 epochs, ~24 s/epoch, decoder-initialized G): final D loss 0.38,
  no mode collapse; class-conditional structure visible in samples
  (smooth ND column vs. defect-bearing CR/LP/PO).
- FID: 37.96 (24,407 real vs 4,000 generated, InceptionV3 features). Note:
  this implementation feeds [0,1] images without ImageNet normalization, so
  absolute values are not comparable to literature FIDs; treat as a trend
  indicator only.
- Per-class FID: CR 71.87, LP 63.50, ND 14.71, PO 83.49. The no-defect class
  is by far the easiest (smooth, uniform texture); porosity is the hardest
  (small blob-shaped indications).
- Downstream augmentation-gain experiment (pretrained ResNet-18, stratified
  real-only test split, seed 42): condition A (imbalanced real, ~7.7k)
  99.36% accuracy / macro-F1 0.9931; condition B (A + 3,000 generated)
  99.36% / 0.9932. Honest negative result: the pretrained classifier
  saturates RIAWELC (per-class recall >= 0.98 even in the imbalanced
  baseline), leaving no headroom for augmentation to help. The paper's
  harder setting (as few as 36 images per class) is not reproduced by this
  substitute dataset.

## Paper-vs-code discrepancy list (recorded, pending owner confirmation)

Values below are factual hyperparameter/architecture statements read from the
two referenced publications (IEEE CYBER 2025 conference paper; MSSP journal
extension by the same first author). Nothing was copied verbatim from either
paper. Per the reproduction guide, the shipped skeleton was left untouched
pending owner confirmation; items marked "runtime" are already aligned through
command-line arguments only.

| Item | Conference paper (2025) | Journal extension (2026) | This repo (skeleton) | Status |
|---|---|---|---|---|
| Framework | TensorFlow 2.18 / Keras 3.9 | same lineage | PyTorch | deliberate (guide's design) |
| Image size | 400x400 grayscale | 400x400 grayscale | 128x128, gray converted to RGB | deliberate (guide sec. 2/6; 256 documented feasible) |
| Encoder body | 3x3 convs, residual connections, max pooling, 1x1 channel reduction | residual blocks, Group Normalization | 4 DCGAN-style 4x4 stride-2 convs + BatchNorm | deviation, kept |
| Latent dim | 32 | 128 | 128 (default) | runtime default matches journal; conference used 32 |
| Decoder upsampling | Sub-Pixel Convolution (2 stages) | progressive | 4 ConvTranspose stages | deviation, kept |
| Decoder output act. | sigmoid, [0,1] | n/a | Tanh, [-1,1] with matched input normalization | internally consistent, kept |
| Reconstruction loss | MSE | weighted (3.0) | L1 | deviation, kept |
| KL weight | beta = 30 | annealed 0 -> 0.5 (50-epoch warm-up) | 1.0 constant | deviation, kept (guide sec. 7 documents lowering on blur) |
| Perceptual loss | VGG19, weight 0.1 | VGG19, weight 0.02 | not implemented | known gap, pending owner decision |
| GAN objective | BCE minimax | hinge loss + projection discriminator with spectral normalization | BCE + label smoothing (real=0.9), no SN | deviation, kept |
| Discriminator head | global max pooling + dense + dropout | GAP -> 512-d + projection | flatten + linear | deviation, kept |
| Label conditioning | label embedded as spatial map, concatenated with image (E and D); decoder concatenates class embedding with z | same design | same design | matches |
| G:D update ratio | alternate within each minibatch | 1:1 | D every batch, G every 2nd epoch | kept (guide sec. 7: 1:1 also works) |
| Optimizer / schedule | Adam lr 1e-3, batch 8, 70 epochs, early stopping, ReduceLROnPlateau | Adam beta1=0, lr 2e-4, batch 16, 200 epochs | Adam betas (0.5, 0.999); lr 2e-4 (CVAE), 1e-4 G / 4e-4 D; batch 64; 100/200 epochs | guide sec. 6 commands kept |
| Training data | proprietary WAAM molten-pool, 1,898 images, 9 classes | own WA-DED data | RIAWELC, 24,407 X-ray images, 4 classes (public) | deliberate (guide sec. 3, disclosed substitution) |
| Downstream classifier | LSTM and GRU sequence models | GRU and LSTM sequence models | ResNet-18 fine-tune (guide sec. 5 "e.g. ResNet-18") | deliberate, recorded |
