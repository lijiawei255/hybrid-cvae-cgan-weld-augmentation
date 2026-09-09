# Changelog

Deliberate deviations from the two papers this repository re-implements are
recorded here with the reason for each. Entries up to v0.2.0 also refer to an
earlier in-house skeleton that predates the papers being read; it was never
published, and every one of its choices that survived is stated here in its own
right rather than by reference to it.

Measurements behind the four calibrated hyperparameters - the KL weight, the
discriminator learning rate, the FFT denoising step and the resolution - live in
[`docs/CALIBRATION.md`](docs/CALIBRATION.md), with the commands that reproduce
them. This file keeps only the conclusions.

## Unreleased

Documentation-only presentation pass; nothing was retrained and no code,
defaults, or published numbers change.

### Added

- README (English and Chinese): a shields badge row (license, CI, Python,
  PyTorch, both paper DOIs, Zenodo DOI, maintenance status), a Quick facts
  summary table, a table of contents, and a Zenodo archive link in the
  citation section.

## v0.5.1 - 2026-09-08

Interface fixes and disclosure pass prompted by an external audit and a
paper-vs-code method review. No method code, training defaults, or published
numbers change; nothing was retrained.

### Fixed

- **Standalone FID could not read a `generate.py` output tree.** The generated
  side uses positional `class_0/class_1/...` folders while real trees use real
  class names, and `eval_fid.py` compared folder names directly, so the
  default balanced mode died with "no class is present in both trees" on the
  README quickstart command (a failure mode introduced by the v0.3.1
  name-based balancing fix). `eval_fid.py` now maps positional folders through
  the pool's `classes.txt` manifest - the same contract `augment.py` uses -
  and refuses explicitly on a missing manifest, an out-of-range `class_i`
  index, or a manifest naming classes the real tree lacks. Real-named fake
  trees are untouched; FID backends and scales are unchanged, and in-training
  FID never used this path. Pinned by a new smoke-test check plus an
  end-to-end `generate -> eval_fid` stage.
- **The v0.5.0 `find_label` fix for split-subfolder YOLO layouts never
  worked.** Its parallel-label candidates were anchored at the image's
  grandparent, probing `dataset/images/labels/...` instead of
  `dataset/labels/...`, so `images/<split>/x.jpg` still fell through as
  "no label". `find_label` now anchors at the nearest `images` ancestor and
  mirrors the image's path under it into the sibling `labels` tree, covering
  both `images/x.jpg -> labels/x.txt` and `images/<split>/x.jpg ->
  labels/<split>/x.txt`; side-by-side labels keep priority. Pinned by a new
  smoke-test check.
- **The non-finite loss guard ran after the generator's `backward()/step()`**
  in `train_joint.py`, so a non-finite generator loss had already written NaN
  gradients into the weights before the abort its comment promised to
  prevent. Generator and discriminator losses are now each checked before
  their own network's backward; update order and all finite-loss behaviour
  are unchanged.

### Changed

- `--min_side`'s help text now states the longest-edge semantics the filter
  has always implemented (`max(bw, bh) < min_side`); filtering behaviour and
  the historical 8,012-crop protocol are unchanged.
- `verify_generation_meta` prints a note when `meta.json` is missing instead
  of silently skipping the check.

### Added

- **`results/metrics/`: the raw metric exports behind the published figures
  and tables** - the three seed sweeps plus the matched control and the
  v0.2.0 sweep (`sweep_metrics.csv`), and the generator training histories
  the docs quote (`history.csv`). Byte-identical copies, SHA-256 verified,
  per-file provenance in `results/metrics/README.md`. The documented
  `make_multiseed_figure.py` command now runs from these copies alone.

### Docs

- **Split source-overlap disclosure.** The split is crop-level, not
  source-isolated: measured on the published seed-42/43/44 splits (8,012
  crops from 1,022 source frames, median 8 crops per frame), 99.8-100% of
  test crops share a source frame with the train pool and 63-67% with the
  1,090-crop training subset. The "Leakage-free by construction" wording is
  replaced by this measured statement (README both languages, USAGE
  limitations, CALIBRATION section 9 with method and table, and the affected
  docstrings). Absolute downstream numbers are optimistic about unseen
  sources; the filling-rate arms share the leakage, so within-run
  comparisons are less affected.
- The seed-42 r=0.25 exclusion is reworded from "invalid" to a post-hoc,
  exploratory aggregate exclusion (CALIBRATION sections 9-10, smoke test);
  all numbers and n counts are unchanged.
- New method disclosures from a full paper-vs-code review: the journal's
  projection discriminator (Eq. 4) and free-bits KL form (Eq. 9) are not
  implemented (USAGE, CALIBRATION section 9); the perceptual loss notes that
  neither paper names VGG19 layers. CALIBRATION section 5's stale
  "600 per class" heading corrected to the 200 the body and code use.
- `verify_generation_meta` documented as a split-*configuration* check, not
  data identity; pools must be regenerated after any change to the crop
  tree (CALIBRATION section 9, USAGE porting checklist).
- Reproduction note: `compute_fid`/`save_sample_grid` draw from the training
  RNG stream, so exact reproduction keeps `--fid_every`/`--sample_every`
  identical (CALIBRATION section 10, USAGE limitations).
- README (both languages): the legacy-FID quickstart command is written out
  in full instead of `...`.

## v0.5.0 - 2026-09-07

Correction and hardening release. Published tables are unchanged; every
recorded run passes its flags explicitly, so the behaviour changes below apply
to future runs only.

### Fixed

- **In-training FID under `--fft_denoise` compared differently filtered
  images.** `train_joint.py` built the real reference through the dataset's
  FFT low-pass but fed raw sigmoid output on the generated side, so
  `history.csv`'s fid column measured a preprocessing mismatch instead of
  generation quality. Generated images now pass through the identical
  PIL-domain filter (uint8 quantisation included), matching `eval_fid.py`,
  which applies one load path to both trees. Pinned by a new smoke-test
  check.
- **`make_class_figures.py` had no `--fft_denoise`/`--fft_cutoff`**, so its
  per-class FID strip silently compared unfiltered images against an
  FFT-trained generator. It now takes both flags (defaults unchanged).
- `ClassFolderDataset` no longer treats hidden directories (e.g.
  `.ipynb_checkpoints`) as phantom classes that shift every label index;
  `prepare_yolo_crops.py` already applied the same rule to its input tree.
- `peek_image_mode` leaked an image file handle on Windows.
- `prepare_yolo_crops.py`: `find_label` now also resolves the standard
  split-subfolder layout (`images/<split>/x.jpg -> labels/<split>/x.txt`),
  previously silently dropped as "no label"; malformed annotation lines and
  boxes that invert when clamped to the frame are skipped with counted
  warnings instead of crashing. An out-of-range `class_id` still aborts.
- `make_paper_figures.py` cross-checks `--channels`/`--img_size` against the
  checkpoint and fails with a targeted message instead of a tensor-shape
  error further in.
- Malformed table row in the v0.3.0 entry (discriminator lr had a leftover
  fifth cell).
- `docs/CALIBRATION.md` section 10 now carries a "Partly superseded by
  v0.4.0" banner (`--selection best_val` is the default since v0.4.0;
  `--selection final` still reproduces the published behaviour). Two stale
  cross-references - the section 9 note and the `--channels` default remark -
  are annotated the same way; measured numbers are untouched.

### Changed

- **`--val_per_class` default 500 -> 200** in `train_joint.py`. Every
  documented command passes 200 explicitly; the old default could exceed a
  small class's train-pool remainder and abort with a ValueError.
- **`--channels 1` on colour files now raises everywhere** (`eval_fid.py`,
  `make_paper_figures.py`, `make_class_figures.py`), matching what
  `docs/USAGE.md` already stated, instead of silently converting to grayscale
  in the evaluation scripts.
- `--subset` agreement checks (`augment.py`, `make_paper_figures.py`) compare
  parsed name-to-count mappings, so a reordered but identical subset no
  longer reads as a split mismatch; a different mapping still refuses.
- DCGAN init is documented as deliberately discriminator-only (comment at its
  single call site); the encoder/decoder keep PyTorch default init, as in
  every tagged run.
- Removed the dead `_features` helper from `eval_fid.py`: it had no callers
  and its "kept for smoke tests" comment was inaccurate.

### Added

- Divergence guards: `logvar` is clamped to +/-10 in `reparameterize` and
  `cvae_loss` (unreachable in healthy training; stops one exploding step from
  poisoning a run), and `train_joint.py` aborts with the epoch and step on a
  non-finite loss instead of silently writing NaN history rows.
- Smoke test: an FFT parity check (generated-side post-processing must equal
  the dataset transform on uint8 input) and temp-dir cleanup on success
  (kept for inspection on failure).
- CI: pip and torch-weight caching, a 60-minute timeout, and cancellation of
  superseded runs.
- Documented the Python version (3.11, what CI runs) in both READMEs,
  `docs/USAGE.md` and `requirements.txt`.
- Both READMEs state the data trade-off explicitly: the papers' reported
  numbers cannot be externally verified because their dataset is
  proprietary, while every number in this repo is reproducible end-to-end
  from public data.
- `CITATION.cff` now carries `version`, `repository-url` and `date-released`,
  and the author field matches the LICENSE spelling.
- `.gitattributes` (`* text=auto`) and `.gitignore` coverage for editor/agent
  tooling (`.cursor/`, `.mimosa/`, `AGENTS.md`).

## v0.4.1 - 2026-09-07

Positioning and documentation pass. No training code, published tables, or
defaults change.

### Added

- **`CONTRIBUTING.md`**: accept documentation typos and "cannot reproduce"
  bugs; reject new methods, backbones, private data, and feature work. The
  maintainer may merge nothing.
- **`docs/USAGE.md`**: journal-extension switches, own-data porting,
  repository layout, compute order of magnitude, and the full limitations
  list (moved out of the READMEs).

### Changed

- Both READMEs are now a short entry: identity, maintenance status
  (complete / as-is, not actively developed), Quickstart, LoHi-WELD prep,
  current results and showcase figures, and BibTeX.
- [`CITATION.cff`](CITATION.cff) title and message describe a completed
  unofficial reference implementation.

## v0.4.0 - 2026-09-07

> **Never tagged separately.** The repository's tags go v0.3.1 -> v0.4.1, and
> the commit tagged `v0.4.1` is the one that carries the changes below, so
> `git checkout v0.4.0` fails. Documents that say a default "became X in
> v0.4.0" mean the release described here; the code state is at `v0.4.1`.
> (Noted in v0.5.2; no tag was created after the fact, because there is no
> commit that held only these changes.)

Credibility and evaluation-infrastructure release. Published tables are
unchanged; new defaults apply to *future* runs.

### Added

- **Scope-at-a-glance** in both READMEs: paper method, this repo's
  adaptations, data inequivalence, non-comparable results, and what is not
  fully reproduced.
- **`--fid_backend {pytorch_fid,legacy}`**. Default `pytorch_fid` wraps
  [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
  (Apache-2.0; official TensorFlow Inception weights). `legacy` keeps the
  torchvision path used by published in-repo numbers. The two scales differ.
  Class-balanced sampling and FFT matching stay in this repo. Attribution:
  `NOTICE`.
- **`--selection {best_val,final}`** on `train_classifier.py`. Default
  `best_val` restores the lowest-loss epoch on a real-only leftover
  validation pool. `--selection final` reproduces the published tables.
- **`--deterministic`** on `train_classifier.py` (CuDNN deterministic;
  still not bit-identical on GPU).
- **`assert_channels_match_data`**: `--channels 1` on colour files raises.
- GitHub Actions workflow running `src/smoke_test.py`.

### Changed

- **`--channels` default is 3** in `train_joint.py` and
  `train_classifier.py` (LoHi-WELD is RGB). Grayscale remains available as
  `--channels 1`.
- Quickstart now shows the recommended LoHi-WELD configuration.
- Superseded v0.1.0 RIAWELC table removed from the READMEs; it stays here.
- Paper-vs-code table below updated for LoHi-WELD as the primary experiment.

## v0.3.1 - 2026-09-07

Documentation and evaluation-hygiene release. The pipeline and all published
numbers are unchanged; what ships is a standalone-FID measurement fix (no
published number used that path), disclosure and wording tightening prompted
by an external review, and README version-labelling sync.

### Fixed

- **Standalone FID class-prior mismatch in `src/eval_fid.py`.** The real side
  was sampled class-balanced but the generated side was not, so the Quickstart
  path compared a balanced real reference against a class-skewed fake tree (a
  balance-to-max pool contains no majority-class images) and the score mixed a
  class-ratio difference into the image-quality distance. Both sides are now
  class-balanced over the classes present in both trees (default: the smallest
  count over the shared classes); classes missing on one side are excluded
  with a printed note, and `--no_balance` keeps the compare-as-is behaviour.
  No published number changes: training-time FID already generated balanced
  fake sets, and no reported result used the standalone path.

### Changed

- **`--kl_weight` help states the full conversion**
  (`beta_eq = 30 * latent_dim / 65025`: 0.015 at latent dim 32, 0.059 at
  latent dim 128, the recommended LoHi-WELD configuration) instead of only the
  latent-32 instance, and labels the sweep's RIAWELC provenance.
- **README (en/zh) discloses the 224px fidelity ceiling in the Data section**:
  the median source box is ~47 px and 98.6% of crops are interpolation
  upsamples - the canvas is 224, the information in it is not (measured in
  `docs/CALIBRATION.md` section 4).
- **Two positioning claims tightened (en/zh).** "faithful, from-scratch
  re-implementation" is now "from-scratch re-implementation ... with every
  deviation from the papers measured and documented", and the recommended
  configuration is described as combining selected journal-extension
  components, not as the journal paper's full configuration (its
  physics-guided losses stay out of scope).
- **Top citation sentence neutralised (en/zh).** The intro no longer tells
  readers what not to cite or frames the repo as existing to direct readers to
  the papers; it asks users who found the repo useful to cite the original
  papers as the source of the re-implemented method.
- **Results section header no longer pinned to v0.2.0 (en/zh).** "Results -
  v0.2.0 (LoHi-WELD, current)" became "Results - LoHi-WELD (current)", with an
  opening line stating that the v0.2.0 single-seed table is kept as published
  and the v0.3.0 three-seed note is the current reading. All other v0.2.0
  mentions are figure/run provenance and stay as they are.

## v0.3.0 - 2026-09-07

Documentation, multi-seed verification and audit release. The pipeline is
unchanged from v0.2.0; what moves here is the *reading* of its results (the
three-seed sweep supersedes v0.2.0's single-seed "non-transfer" conclusion),
the journal-extension switches that make the recommended configuration
expressible, a bilingual README, regenerated showcase figures, and citation /
doc accuracy fixes from a post-release audit.

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
  "approximately constant value around 2.0" - and listing four things the
  measurement does not establish: it does not isolate the remaining three
  configuration changes, "better generator" holds only on the reconstruction axis
  because FID did not follow, the GroupNorm aggregate has three seeds while the
  control has one, and the absolute scores stay non-comparable to the papers'.
  Section 10 records that the downstream classifier is not bit-reproducible on GPU
  and scores the final epoch.
- **Section 1 addendum withdrawing an earlier explanation.** The claim that a
  low-information latent is why our generations lack fidelity is withdrawn as
  unsupported: `latent_dim x kl` is already at the paper's reported magnitude
  under either reading of its normalisation.
- **Downstream sweep for the journal-extension configuration** at three seeds
  (`runs/sweep_paper2_s42`, `_s43`, `_s44`), which **reverses the published v0.2.0
  finding that balance-to-max does not transfer**. At r=1.0 mean macro-F1 is
  0.7185 +/- 0.0019 against this arm's own 0.6785 +/- 0.0447 real-only mean
  (+0.0399) where the published curve had 0.6418 against 0.6936 (-0.052, its worst
  ratio); mean pore F1 is 0.4658 +/- 0.0475 against 0.3527 +/- 0.0611 (+0.1132).
  The per-seed r=1.0 macro-F1 deltas are +0.0321, +0.0895 and -0.0018, so it is a
  positive mean effect rather than a gain on every seed. The **matched latent-128
  BatchNorm, unweighted-sampler control** (`runs/sweep_ctrl_s42_clean`) preserves
  that sign flip - r=1.0 macro-F1 0.7009 against its own 0.6332 baseline (+0.0677),
  pore F1 0.4222 against 0.4000 - so GroupNorm and the weighted sampler are **not
  required** for it. Neither curve is monotone and neither peaks uniquely at r=1.0:
  the GroupNorm arm's highest complete-seed mean is r=0.50 (0.7210 +/- 0.0246) and
  the control peaks at r=0.75 (0.7124). Recorded in `docs/CALIBRATION.md` section 9
  and behind a superseded banner in both READMEs rather than by editing the
  published table, with four things it does not establish: it does not isolate the
  remaining three configuration changes (`latent_dim` 32 -> 128, `kl_weight` 0.015
  -> 0.059, early stopping at 25 epochs -> the full 70), "better generator" holds
  only on the reconstruction axis, the control has one seed against the GroupNorm
  arm's three, and the absolute scores stay non-comparable to the papers'. The
  seed-42 r=0.25 arm was invalidated by a final-epoch loss spike, so that ratio
  aggregates two seeds. FID at each arm's own best-validation epoch
  is effectively unchanged against v0.2.0 (215.99 vs 216.87) even though best
  `val_recon` improved 2.8x (0.0177 vs 0.0494), so "better generator" holds on the
  reconstruction axis only. Taking the minimum FID over each run instead reverses
  that reading (191.93 vs 184.84 against v0.2.0, and 191.93 vs 142.29 against the
  matched latent-128 BatchNorm control); all three comparisons are tabulated in
  `docs/CALIBRATION.md` section 9, because FID is a diagnostic here rather than an
  acceptance criterion and its direction depends on which pair and statistic is
  used.
- **Seed-aggregated filling-rate figure** `results/filling_rate_multiseed.png`,
  produced by the new `src/make_multiseed_figure.py`. It reads only the three
  `sweep_metrics.csv` files, so it needs neither a GPU nor the dataset, and its
  band is the sample standard deviation (ddof=1) so the figure cannot disagree
  with the `docs/CALIBRATION.md` section 9 table. `--exclude` drops the invalid
  seed-42 r=0.25 arm from the aggregate without removing that ratio from the other
  two seeds, which the figure annotates `n=2`; an `--exclude` that matches nothing
  is an error rather than a silent no-op.
- **Per-figure provenance table in both READMEs.** Every file in `results/` now
  states which run produced it, because the sweep the READMEs report is the
  three-seed re-run while most committed figures still come from the published
  v0.2.0 runs. In particular `filling_rate_curve.png` was labelled "the sweep
  above", which after the re-run pointed at the wrong sweep.

### Changed

- **Committed showcase grids regenerated** from the recommended
  `runs/paper2_gn_wrs` generator instead of the v0.2.0 `runs/joint_lohi` one, so
  the images a clone user sees match the configuration the READMEs recommend. The
  v0.2.0 versions are kept as `results/v0.2.0_*.png` rather than deleted; their
  generated rows are near-identical within each class, which is the mode collapse
  the recommended config reduces. The grids are sampled into a showcase-only pool
  with all four classes, separate from the `generated_paper2` sweep pool, whose
  `stain=0` is required by balance-to-max and checked by `verify_generation_meta`.

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
- **The noise caveat is now measured rather than guessed.** Repeating an
  unchanged arm at the same seed moved pore F1 by 0.034 and macro-F1 by 0.009.
  (This entry originally described the replaced text as a README caveat reading
  "treat differences under ~0.02 macro-F1 as noise"; no such sentence was ever
  in either README. Corrected in v0.5.2.)
- **README prep command could not produce the advertised crops.** Both READMEs'
  Data sections pointed `--input_root` at the archive root, which also contains
  the 2,000 low-resolution beads; the documented 8,012 crops come from
  `<archive>/weld-dataset/high_resolution_welds` with `--min_side 16`, as
  `DATA_SOURCES.md` and the preparation record already showed. Fixed in both
  languages, with the reason for taking only that subset stated.
- **Stale "in press" citation completed.** The RIAWELC-required Perri et al.,
  *Manufacturing Letters* paper has been published since January 2023
  (vol. 35, pp. 29-32, DOI 10.1016/j.mfglet.2022.11.006; verified via Crossref),
  so `CITATION.cff`, both READMEs' BibTeX and `DATA_SOURCES.md` now carry the
  real details instead of the deliberate "assert nothing" placeholder. The
  conference paper's pages (1-6) and LoHi-WELD's volume and pages (12,
  77442-77453) are filled in at the same places.
- **`DATA_SOURCES.md` was unreachable from the published docs.** No committed
  `.md` linked it once the provenance moved into the READMEs, so its (correct)
  prep command sat next to a diverging (wrong) README copy. Both READMEs now
  link it from the Data section, and their License sections state that the
  committed `results/` figures are LoHi-WELD derivatives and stay subject to
  that dataset's citation requirement alongside the repo's MIT license.
- **Misleading "Real"/"Generated" column headers on the showcase figures.** The
  grids alternate real/generated by row, but both figure scripts drew headers at
  25%/75% of the width implying "left half real, right half generated" - the
  opposite of the actual layout. Replaced with a row-order caption
  (`src/make_comparison.py`, `src/make_class_figures.py`), pinned by the new
  `check_comparison_grid_labels` smoke check, and the five committed figures
  regenerated from the same showcase pool (per-class FIDs unchanged: 198.82 /
  224.86 / 185.87 / 222.45).
- **Stale RIAWELC class names in `--help` and docstrings.** Seven
  argparse/docstring examples used `CR=..,PO=..` names that raise `KeyError`
  against the documented `data/lohi` root; they now use the LoHi-WELD subset and
  sweep pools (`train_joint.py`, `train_classifier.py`, `make_paper_figures.py`,
  `generate.py`, `data.py`). `make_class_figures.py` also lost its dead
  RIAWELC-only `CLASS_DESCRIPTIONS` dict, whose `.get` fallback silently
  stripped the description from every current figure title.
- **Grayscale invariant and related doc drift.** `data.py`, `models.py` and
  `smoke_test.py` claimed a single-channel grayscale invariant the code does not
  have (the channel count is a parameter and the tests assert the RGB path);
  `train_classifier.py` said "static radiographs" where the primary experiment
  is visible-light weld beads; `models.py`'s header now also credits the
  journal-extension components it hosts; `train_joint.py` no longer reads as if
  `--d_norm group`/`--monitor val_recon` were defaults; `generate.py`'s layout
  example matches its six-digit filenames; `make_paper_figures.py --channels`
  default changed 1 -> 3 to match the other scripts and the documented commands.
- **Duplicate `.gitignore` entries** (`papers/` and `REPRODUCTION_GUIDE.md`
  each listed twice) collapsed to one block.

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
  sigmoid decoder**, 1 or 3 channels, following the conference paper. (The
  journal extension specifies Tanh and [-1, 1] instead; this repo follows the
  conference paper on both.) The primary LoHi-WELD experiment runs at 3 channels
  because colour carries defect signal there. Both papers convert to grayscale
  as a declared preprocessing step, so running at 3 channels is a deviation
  rather than a neutral dataset detail; `--channels 1` reproduces the step.
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
| Discriminator lr | single `1e-3` for both nets | `1e-3` for both | **aligned with the conference paper**: hinge + spectral normalisation keep D in equilibrium on small data, so no reduction is needed; the 4e-4 value in docs/CALIBRATION.md was calibrated under the superseded BCE objective |
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
  reconstruction MSE 0.049-0.060; early stopping at epoch 25 of 70.
  (**Corrected in v0.5.2**: the constant-mean baseline was published here as
  0.062. Remeasured on this run's own 800-image validation split it is 0.0556,
  and the stricter per-image-mean anchor is 0.0339. The run's best validation
  reconstruction is 0.0498, so it clears the first and not the second. The
  reported reconstruction figures themselves are unchanged.)

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
| Image size | 400x400 grayscale | 400x400 grayscale | 224x224; primary LoHi-WELD runs are RGB | calibrated (source boxes are much smaller; see CALIBRATION.md §4) |
| Channels | 1; the paper states RGB frames "were converted to grayscale to reduce redundancy and suppress irrelevant color variations" | 1 | 1 or 3; primary runs at 3 | **deviates**: grayscale is a declared preprocessing step of the method, not a camera property. LoHi-WELD's stain and discontinuity classes carry colour signal, so the primary runs keep RGB; `--channels 1` reproduces the paper's step |
| Decoder output | sigmoid, [0, 1] | **Tanh, [-1, 1]** (inputs normalised to match) | sigmoid, [0, 1] | aligned with the conference paper; **deviates** from the journal extension |
| Preprocessing | ROI extraction, grayscale, FFT circular low-pass | same | LoHi-WELD ROI via prepare_yolo_crops.py; FFT implemented, off | calibrated (measured band-limited) |
| Encoder body | 3x3 convs, residual connections, max pooling, 1x1 channel reduction, then **flattening** into the dense heads | residual blocks, Group Normalization, widths 32-64-128-256 | 4 DCGAN-style 4x4 stride-2 convs + BatchNorm (`--g_norm group` available), then 1x1 reduction + **global average pooling**; widths 64-128-256-512 | **deviates**: the 1x1 reduction is the paper's, the pooling replaces its flatten (a measured stability fix, docs/CALIBRATION.md section 7 bug 1), the conv body and the channel widths are ours |
| Discriminator head | global max pooling + dense layers + dropout | GAP -> 512-d + projection discriminator (Eq. 4) | global max pooling + single dense, no dropout | partial: pooling aligned; dropout omitted, projection not implemented |
| Decoder upsampling | Sub-Pixel Convolution (2 stages) | progressive | 4 PixelShuffle stages | **aligned with the conference paper** |
| Normalisation | BatchNorm | **GroupNorm in every network** (Table 3) | selectable per side: `--d_norm` (measured, recommended `group`) and `--g_norm` (added v0.5.2, **not measured**); both default to `batch` | expressible everywhere; only the discriminator's setting has evidence behind it |
| Latent dim | 32 | 128 | 32 default (conference); recommended LoHi-WELD uses 128 | both values are expressible |
| Reconstruction loss | MSE | **Charbonnier**, weight 3.0 | MSE, weight 1.0 | aligned with the conference paper; **deviates** from the journal extension, whose Charbonnier term behaves like L1 on bright outliers |
| KL weight | beta = 30 | annealed 0 -> 0.5 | 0.015 at latent 32; 0.059 at latent 128 | calibrated; equivalent under this repo's normalisation |
| Perceptual loss | VGG19, weight 0.1; Eq. 7 is a **weighted sum over several layers** | VGG19, weight 0.02, over **intermediate** activations | frozen VGG19, weight 0.1, MSE on the **single final** `features` output | weight aligned with the conference paper; **deviates** on layer selection, which neither paper names individually |
| Adversarial weight | gamma = 1 | n/a | 0.1 | calibrated: at 1.0 under this repo's mean-normalised losses the adversarial term outweighs reconstruction ~20:1 and reconstruction never converges (docs/CALIBRATION.md) |
| GAN objective | BCE minimax | hinge + projection discriminator + spectral norm on the **first three** conv blocks | hinge + spectral norm on **all four** conv blocks and the dense head; `--adv_loss bce` and `--no_d_spectral_norm` express the conference configuration | aligned with the journal extension on the objective; **deviates** on SN scope (docs/CALIBRATION.md section 6) |
| Label conditioning | label embedded as a spatial map concatenated with the image (E and D); decoder concatenates a class embedding with z | same design | same design | matches |
| G:D update ratio | alternated within each minibatch | 1:1 | alternated within each minibatch, separate backward passes | **aligned** |
| Generator lr | Adam 1e-3 | Adam beta1=0, lr 2e-4 | Adam 1e-3, default betas | **aligned with the conference paper** |
| Discriminator lr | same 1e-3 | same 2e-4 | 1e-3 | **aligned**: hinge + spectral normalisation keep D in equilibrium at the paper's own rate; the 4e-4 probe value in docs/CALIBRATION.md belongs to the superseded BCE objective |
| Batch size | 8 | 16 | 8 | **aligned** |
| Epochs | 70 | 200 | 70 | **aligned** |
| Early stopping / LR schedule | patience 10; ReduceLROnPlateau factor 0.2 patience 5 | n/a | same defaults | **aligned** as defaults, but the recommended arm and every seed of the three-seed table ran `--patience 70 --lr_patience 70`, i.e. both disabled for the full 70 epochs (docs/CALIBRATION.md section 9) |
| FID | every epoch, vs real validation samples, InceptionV3 average pooling | same | every epoch, vs a real validation set; default backend pytorch-fid (TF Inception weights); `--fid_backend legacy` for published in-repo numbers | protocol aligned; feature weights are an evaluation-infrastructure choice |
| Augmentation strategy | balance all classes to 600 | balance-to-max to N_max, filling-rate sensitivity sweep | balance-to-max to N_max, filling-rate sweep | **aligned** |
| Training data | proprietary WAAM molten-pool, 1,898 images, 9 classes, 14.7x imbalance | own WA-DED data | LoHi-WELD (primary); RIAWELC historical only | deliberate public substitution |
| Downstream classifier | LSTM and GRU over 21-frame sequences | GRU and LSTM over sequences | ResNet-18 from scratch over single images | forced deviation (no temporal axis); disclosed |
| Generator sees test images | evaluation on "real, unseen images" | same | no - splits made first, generator restricted to the train pool | **aligned** |
