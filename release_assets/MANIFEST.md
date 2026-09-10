# Artefact manifest for v0.5.2

The generator checkpoints and generated pools behind every committed figure
and every sweep in `docs/CALIBRATION.md` sections 9 and 11 are too large for
git and are **not redistributed**. This file records their fingerprints -
SHA-256, size, the run each belongs to, and the configuration read back from
each checkpoint - so that a checkpoint or pool produced by rerunning the
recorded commands can be told apart from the originals, and so that every
number in those sections is traceable to a specific file. Retraining is the
only way to obtain equivalent files; because GPU training is not
bit-reproducible (section 10), a retrained checkpoint will not match these
digests, and its downstream numbers will differ within the noise floor of
section 11.7.

The digest of a checkpoint is of `runs/<run>/joint.pt` as written by
`src/train_joint.py`; the digest of a pool is of a deterministic zip of the
`generate.py` output tree (paths relative to the repository root, fixed
timestamps, deflate).

## Generator checkpoints

Each `joint.pt` holds the encoder, decoder and discriminator weights of that
run's best epoch, plus the configuration it was trained with. `src/generate.py`
reads the class names and the normalisation choice straight out of it; a
checkpoint without a `g_norm`, `adv_loss` or `split_by` key predates those
switches and used their defaults (`batch`, `hinge`, `crop`).

| file | size | run | sha256 |
|---|---|---|---|
| `paper2_gn_wrs/joint.pt` | 148 MB | recommended arm, seed 42 (GroupNorm D + weighted sampler). Backs results/real_vs_generated.png and results/class_*.png. | `61d908bf534ec84cfcd9aa4d835820e14e815f78e01a8399471f2918212cc689` |
| `paper2_gn_wrs_s43/joint.pt` | 148 MB | recommended arm, seed 43. | `c1b27eacfbb62f839971ce0473d7c0c263f55b9fdf52fe2cbc6db674578bab78` |
| `paper2_gn_wrs_s44/joint.pt` | 148 MB | recommended arm, seed 44. | `281a9c30d6d369020cdc241c8dcd3a442a57c319971f7473c4ba8bea734944af` |
| `probe_eq_g0.1/joint.pt` | 148 MB | matched latent-128 BatchNorm control, seed 42. | `42bf3f2ffcd88f1136a3a0e694a2c170d0956f5a398859ecaafeadf9adb963be` |
| `probe_eq_g0.1_s43/joint.pt` | 148 MB | matched BatchNorm control, seed 43 (added v0.5.2, CALIBRATION section 11). | `328d172abedd5bcaa7a8ad5982379d325913f7a544592a396b8678f6089f4dda` |
| `probe_eq_g0.1_s44/joint.pt` | 148 MB | matched BatchNorm control, seed 44 (added v0.5.2, CALIBRATION section 11). | `5459c691c372e5a104665a694ba91fca252a1ee5c2d477feefa90f162ee02f05` |
| `paper2_gn_only/joint.pt` | 148 MB | GroupNorm D without the weighted sampler, seed 42 (added v0.5.2, section 11). | `8a79a69a15b18c2bab9a21a4844ea4cd96fc9eae38d95d07921abf110bb036c0` |
| `paper2_src_s42/joint.pt` | 148 MB | recommended arm under --split_by source, seed 42 (added v0.5.2, section 11). | `92282081c86a3dd715e8507548c4d6cf347733abe89d61f2081a6e70e06c38ba` |
| `paper2_src_s43/joint.pt` | 148 MB | recommended arm under --split_by source, seed 43 (added v0.5.2, section 11). | `5cd05b51aa5955b48ec484ee6f6502586a49e38b388a611be6ae49a8338523b1` |
| `paper2_src_s44/joint.pt` | 148 MB | recommended arm under --split_by source, seed 44 (added v0.5.2, section 11). | `fdd9d52a426bb27b979c37308ee36581624765ab9c7c45eafa50ef1120a8005e` |
| `joint_lohi/joint.pt` | 75 MB | the published v0.2.0 generator. Backs results/training_curves.png, reconstruction_comparison.png and latent_tsne.png. | `4ab33752ee2e2edeb72ebf3f8c0c5560bf21579abe79386509cd8e60beb4e6e0` |

Total checkpoint size: 1.52 GB.

## Stored configuration of each checkpoint

Read back from the files themselves, not from the launch commands. `null`
where the checkpoint holds NaN (no FID was computed at the best epoch) or
predates a key.

```json
{
 "paper2_gn_wrs": {
  "seed": 42,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "group",
  "weighted_sampler": true,
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 60,
  "best_val_loss": 0.03646824544519186,
  "best_fid": 215.98504521733946
 },
 "paper2_gn_wrs_s43": {
  "seed": 43,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "group",
  "weighted_sampler": true,
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 32,
  "best_val_loss": 0.03984709329783917,
  "best_fid": null
 },
 "paper2_gn_wrs_s44": {
  "seed": 44,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "group",
  "weighted_sampler": true,
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 40,
  "best_val_loss": 0.03670795654505492,
  "best_fid": 232.68462497283878
 },
 "probe_eq_g0.1": {
  "seed": 42,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "best_epoch": 69,
  "best_val_loss": 0.0593763070115447,
  "best_fid": null
 },
 "probe_eq_g0.1_s43": {
  "seed": 43,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "batch",
  "g_norm": "batch",
  "adv_loss": "hinge",
  "d_spectral_norm": true,
  "weighted_sampler": false,
  "split_by": "crop",
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 69,
  "best_val_loss": 0.06552255595445634,
  "best_fid": null
 },
 "probe_eq_g0.1_s44": {
  "seed": 44,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "batch",
  "g_norm": "batch",
  "adv_loss": "hinge",
  "d_spectral_norm": true,
  "weighted_sampler": false,
  "split_by": "crop",
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 64,
  "best_val_loss": 0.07495695364415646,
  "best_fid": null
 },
 "paper2_gn_only": {
  "seed": 42,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "group",
  "g_norm": "batch",
  "adv_loss": "hinge",
  "d_spectral_norm": true,
  "weighted_sampler": false,
  "split_by": "crop",
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 66,
  "best_val_loss": 0.03664822277367115,
  "best_fid": null
 },
 "paper2_src_s42": {
  "seed": 42,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "group",
  "g_norm": "batch",
  "adv_loss": "hinge",
  "d_spectral_norm": true,
  "weighted_sampler": true,
  "split_by": "source",
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 67,
  "best_val_loss": 0.03445251258988129,
  "best_fid": null
 },
 "paper2_src_s43": {
  "seed": 43,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "group",
  "g_norm": "batch",
  "adv_loss": "hinge",
  "d_spectral_norm": true,
  "weighted_sampler": true,
  "split_by": "source",
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 69,
  "best_val_loss": 0.03425035820524944,
  "best_fid": null
 },
 "paper2_src_s44": {
  "seed": 44,
  "latent_dim": 128,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.059,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "d_norm": "group",
  "g_norm": "batch",
  "adv_loss": "hinge",
  "d_spectral_norm": true,
  "weighted_sampler": true,
  "split_by": "source",
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "test_frac": 0.2,
  "monitor": "val_loss",
  "best_epoch": 59,
  "best_val_loss": 0.03805534361310696,
  "best_fid": null
 },
 "joint_lohi": {
  "seed": 42,
  "latent_dim": 32,
  "base_ch": 64,
  "img_size": 224,
  "img_channels": 3,
  "kl_weight": 0.015,
  "perc_weight": 0.1,
  "adv_weight": 0.1,
  "subset": "pore=40,deposit=150,discontinuity=300,stain=600",
  "best_epoch": 15,
  "best_val_loss": 0.0721696286857128,
  "best_fid": 216.86784488545948
 }
}
```

## Generated image pools

Each zip unpacks to one `src/generate.py` output tree (`class_i/` folders plus
`classes.txt` and `meta.json`) at the repository root, under the directory name
the sweeps expect. A sweep refuses a pool whose recorded split configuration
disagrees with its own, so pool and sweep travel together.

| zip | images | generator | used by | size | sha256 of the zip |
|---|---|---|---|---|---|
| `generated.zip` | 1610 | `runs/joint_lohi/joint.pt` | runs/sweep (published v0.2.0 table) and runs/sweep_v020_bestval | 58 MB | `b52d56e3557b39d924ae99a2c5220d1f6687e960b136793cdb8e09ea6ca90555` |
| `generated_ctrl.zip` | 1310 | `runs/probe_eq_g0.1/joint.pt` | runs/sweep_ctrl_s42_clean and runs/sweep_ctrl_s42_bestval | 58 MB | `bc96aaaa4455056bcc729ef8ba87742b94e14076041ac615be4eba426d7cd1a1` |
| `generated_ctrl_s43.zip` | 1310 | `runs/probe_eq_g0.1_s43/joint.pt` | runs/sweep_ctrl_s43 | 55 MB | `50ce5163733167405b65e93deceda1f0acf3902a8e58c4e8ccf2750beba012b7` |
| `generated_ctrl_s44.zip` | 1310 | `runs/probe_eq_g0.1_s44/joint.pt` | runs/sweep_ctrl_s44 | 62 MB | `c65096f1a1c8a400d04f6d92093ca886b56ff68f15ef1f51b6a212bf049ea716` |
| `generated_paper2.zip` | 1310 | `runs/paper2_gn_wrs/joint.pt` | runs/sweep_paper2_s42 and runs/sweep_paper2_s42_bestval | 86 MB | `837b60244653bf42df0d3c00e7f75ff483c0c0da02ea9b42a973cecfe15a7b6f` |
| `generated_paper2_s43.zip` | 1310 | `runs/paper2_gn_wrs_s43/joint.pt` | runs/sweep_paper2_s43 and runs/sweep_paper2_s43_bestval | 74 MB | `f7a57655d13013cbf1dfc46f4e558cbfa5d5800dd88ab250c6e3577053a9e733` |
| `generated_paper2_s44.zip` | 1310 | `runs/paper2_gn_wrs_s44/joint.pt` | runs/sweep_paper2_s44 and runs/sweep_paper2_s44_bestval | 88 MB | `294e21c4470e2faad6a4c322c4eceb019dd6ca2fc2c1d61b948f221c35d30ea9` |
| `generated_gn_only.zip` | 1310 | `runs/paper2_gn_only/joint.pt` | runs/sweep_gn_only | 72 MB | `d8743e591d8b253903e8a115679d02b2ce85b45b0ddc68ebe124f57bfda6b690` |
| `generated_src_s42.zip` | 1310 | `runs/paper2_src_s42/joint.pt` | runs/sweep_src_s42 | 80 MB | `1207cd53498d8e27bd7bfdb4dc2a5a9a51d98fb3ba90eb94300ac60213429110` |
| `generated_src_s43.zip` | 1310 | `runs/paper2_src_s43/joint.pt` | runs/sweep_src_s43 | 78 MB | `b1e73c1044f9bee299652b2de70321daa7ff5c0c1770fa7476485948cb0debc9` |
| `generated_src_s44.zip` | 1310 | `runs/paper2_src_s44/joint.pt` | runs/sweep_src_s44 | 77 MB | `b2c20607d438c28450372c25a78816966465d1594322d824c371a9a04b41dff3` |
| `generated_showcase.zip` | 1200 | `runs/paper2_gn_wrs/joint.pt` | results/real_vs_generated.png and results/class_*.png (no sweep) | 79 MB | `afe8664a86a2f84059c36c21ee7e6007bb4405001cc91702a241301f14202648` |

`generated` is the v0.2.0 pool with `stain=300`; every other sweep pool is the
balance-to-max set (`deposit=450,discontinuity=300,pore=560,stain=0`).

## Not attached

LoHi-WELD itself is not redistributed here: download it from its own
repository and run `src/prepare_yolo_crops.py` (see `DATA_SOURCES.md`).
The per-ratio confusion matrices and every metric CSV are committed to the
repository under `results/metrics/` with their own `SHA256SUMS`, so they are
not duplicated here.
