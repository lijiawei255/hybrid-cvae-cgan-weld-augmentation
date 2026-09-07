# -*- coding: utf-8 -*-
"""Smoke test: unit checks plus an end-to-end run on synthetic weld-like images.

The unit checks pin down the contracts that are easy to break silently - the
[0, 1] pixel range with selectable 1- or 3-channel input, the leakage-free
splits, this repo's calibrated loss weights, the FID input normalisation and
numerics, and name-keyed class configuration. The end-to-end part then runs
data loading -> joint CVAE-CGAN training -> generation.

It verifies code correctness only, NOT generation quality.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))


def make_synthetic_weld_data(root, n_good=60, n_defect=20, size=128, seed=0):
    """Two fake classes: 'good' = smooth bright horizontal band;
    'defect' = same band with random dark blobs on top."""
    rng = np.random.RandomState(seed)
    for cname, n in [("good", n_good), ("defect", n_defect)]:
        d = Path(root) / cname
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.zeros((size, size, 3), np.float32) + 0.15
            band = size // 2 + rng.randint(-8, 8)
            img[band - 12:band + 12, :, :] = 0.75
            img += rng.normal(0, 0.03, img.shape).astype(np.float32)
            if cname == "defect":
                for _ in range(rng.randint(1, 4)):
                    cy, cx = rng.randint(band - 10, band + 10), rng.randint(0, size)
                    r = rng.randint(3, 8)
                    yy, xx = np.ogrid[:size, :size]
                    mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
                    img[mask] *= 0.2
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(d / f"{i:04d}.png")


def run(cmd_args, module):
    print("=" * 20, module, "=" * 20)
    sys.argv = ["prog"] + cmd_args
    import importlib
    importlib.import_module(module).main()


def check_balanced_fid_sampling():
    import torch
    from eval_fid import balanced_labels, balanced_subset

    class _Ds:
        def __init__(self, classes, counts):
            self.classes = list(classes)
            self.samples = [(Path(f"{name}_{i}.png"), label)
                            for label, name in enumerate(classes)
                            for i in range(counts[label])]

    ds = _Ds(["pore", "deposit", "stain"], [5, 5, 5])
    selected = balanced_subset(ds, 2, ["pore", "deposit", "stain"], seed=7)
    assert selected == balanced_subset(ds, 2, ["pore", "deposit", "stain"], seed=7)
    names = [ds.classes[label] for _, label in selected]
    assert names.count("pore") == 2
    assert names.count("deposit") == 2
    assert names.count("stain") == 2
    # Restricting to a subset of classes must drop the others, not dilute them.
    restricted = balanced_subset(ds, 2, ["pore", "deposit"], seed=7)
    assert {ds.classes[label] for _, label in restricted} == {"pore", "deposit"}
    # Requesting more than a class holds raises instead of silently shrinking.
    try:
        balanced_subset(ds, 6, ["pore", "deposit", "stain"], seed=7)
        raise AssertionError("balanced_subset accepted per_class above availability")
    except ValueError:
        pass
    assert torch.bincount(balanced_labels(3, 2, "cpu"), minlength=3).tolist() == [2, 2, 2]


def check_class_distribution_figure(tmp):
    from make_paper_figures import save_class_distribution

    output = tmp / "class_distribution.png"
    save_class_distribution(["good", "defect"],
                            {"all real": [60, 20], "sparse real": [12, 4]}, output)
    assert output.is_file() and output.stat().st_size > 0


def check_distribution_protocol():
    from make_paper_figures import build_distribution_series

    series = build_distribution_series(["CR", "LP"], {"CR": 40, "LP": 600},
                                       {"CR": 560, "LP": 0})
    assert series == {"real (imbalanced)": [40, 600],
                      "real + synthetic (balance-to-max)": [600, 600]}, series


def check_reconstruction_figure(tmp):
    import torch
    from make_paper_figures import save_reconstruction_comparison

    gray = tmp / "reconstruction_gray.png"
    images = torch.rand(4, 1, 8, 8)
    save_reconstruction_comparison(images, images, images, gray)
    assert gray.is_file() and gray.stat().st_size > 0

    annotated = tmp / "reconstruction_annotated.png"
    grouped = torch.rand(6, 1, 8, 8)
    save_reconstruction_comparison(grouped, grouped, grouped, annotated,
                                   class_names=["CR", "LP", "ND"], n_per_class=2)
    assert annotated.is_file() and annotated.stat().st_size > 0

    rgb = tmp / "reconstruction_rgb.png"
    colour = torch.rand(3, 3, 8, 8)
    save_reconstruction_comparison(colour, colour, colour, rgb)
    assert rgb.is_file() and rgb.stat().st_size > 0


def check_latent_projection_figure(tmp):
    from make_paper_figures import save_latent_projection

    output = tmp / "latent_projection.png"
    save_latent_projection(np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
                           np.array([0, 0, 1, 1]), ["good", "defect"], output)
    assert output.is_file() and output.stat().st_size > 0


def check_history_csv_reading(tmp):
    from make_paper_figures import read_history_csv

    path = tmp / "history.csv"
    path.write_text(
        "epoch,beta,recon,kl,perc,loss_g,loss_d,val_loss,val_recon,fid,lr,seconds\n"
        "1,0.000000,0.150000,0.001400,0.307000,1.200000,0.870000,0.166700,0.140000,384.7700,1.000e-03,39.0\n"
        "2,0.015000,0.120000,0.000000,0.250000,1.100000,0.500000,0.150000,0.120000,nan,1.000e-03,22.0\n",
        encoding="utf-8")
    history = read_history_csv(path)
    assert history["epoch"] == [1, 2]
    assert history["beta"] == [0.0, 0.015]
    assert history["recon"] == [0.15, 0.12]
    assert history["lr"] == [1e-3, 1e-3]
    assert history["fid"][0] == 384.77
    # A skipped FID evaluation must stay NaN; turning it into 0 would plot a
    # fake perfect score.
    assert np.isnan(history["fid"][1])

    # Run directories from the tagged v0.2.0 work predate the beta column and
    # must still be plottable, so the reader has to key on the header rather
    # than assume a fixed column layout.
    legacy = tmp / "history_legacy.csv"
    legacy.write_text(
        "epoch,recon,kl,perc,loss_g,loss_d,val_loss,val_recon,fid,lr,seconds\n"
        "1,0.150000,0.001400,0.307000,1.200000,0.870000,0.166700,0.140000,384.7700,1.000e-03,39.0\n",
        encoding="utf-8")
    old = read_history_csv(legacy)
    assert old["epoch"] == [1] and old["recon"] == [0.15]
    assert "beta" not in old


def check_sweep_csv_reading(tmp):
    from make_paper_figures import read_sweep_csv

    path = tmp / "sweep.csv"
    path.write_text(
        "ratio,metric,value\n"
        "0.0,accuracy,0.4200\n0.0,f1:CR,0.2100\n"
        "1.0,accuracy,0.6100\n1.0,f1:CR,0.4400\n", encoding="utf-8")
    assert read_sweep_csv(path) == {0.0: {"accuracy": 0.42, "f1:CR": 0.21},
                                    1.0: {"accuracy": 0.61, "f1:CR": 0.44}}


def check_confusion_matrix_figure(tmp):
    from make_paper_figures import normalize_confusion_matrix, save_confusion_matrices

    matrix = np.array([[3, 1], [0, 0]])
    assert np.allclose(normalize_confusion_matrix(matrix), [[0.75, 0.25], [0.0, 0.0]])
    output = tmp / "confusion_matrices.png"
    save_confusion_matrices(["good", "defect"], matrix, matrix, output)
    assert output.is_file() and output.stat().st_size > 0


def check_grayscale_dataset(tmp):
    """Sigmoid decoder means tensors are in [0, 1]; channels stay selectable (1 or 3)."""
    from data import ClassFolderDataset

    ds = ClassFolderDataset(str(tmp / "smoke_data"), img_size=64, channels=1)
    x, y = ds[0]
    assert x.shape == (1, 64, 64), x.shape
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0, (float(x.min()), float(x.max()))
    assert isinstance(y, int)

    rgb = ClassFolderDataset(str(tmp / "smoke_data"), img_size=64, channels=3)[0][0]
    assert rgb.shape == (3, 64, 64), rgb.shape


def check_fft_denoise():
    """Paper's preprocessing FFT-transforms, applies a circular mask, inverse-FFTs."""
    from data import fft_lowpass

    rng = np.random.RandomState(0)
    noisy = (128 + rng.normal(0, 40, (32, 32))).clip(0, 255).astype(np.uint8)
    out = fft_lowpass(noisy, cutoff=0.25)
    assert out.shape == noisy.shape and out.dtype == np.uint8

    def high_freq_energy(a):
        f = np.abs(np.fft.fftshift(np.fft.fft2(a.astype(np.float32))))
        c = min(a.shape) // 2
        yy, xx = np.ogrid[:a.shape[0], :a.shape[1]]
        r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
        return float(f[r > 0.5 * c].sum())

    assert high_freq_energy(out) < 0.5 * high_freq_energy(noisy)
    # deterministic: same input and cutoff must give the same output
    assert np.array_equal(out, fft_lowpass(noisy, cutoff=0.25))


def check_no_leakage_splits(tmp):
    """The generator must never see the classifier's held-out test images.

    Subset selection is keyed by class NAME, not index, so a dataset with a
    different class order or count behaves correctly instead of silently
    sampling the wrong classes.
    """
    from data import ClassFolderDataset, make_splits, sample_named_subset

    ds = ClassFolderDataset(str(tmp / "smoke_data"), img_size=64, channels=1)
    train_pool, test_idx = make_splits(ds.samples, test_frac=0.2, seed=42)
    assert not (set(train_pool) & set(test_idx))
    assert len(train_pool) + len(test_idx) == len(ds)
    assert train_pool == make_splits(ds.samples, test_frac=0.2, seed=42)[0]

    def counts(idxs):
        out = {}
        for i in idxs:
            name = ds.classes[ds.samples[i][1]]
            out[name] = out.get(name, 0) + 1
        return out

    gen_train = sample_named_subset(ds, {"good": 10, "defect": 4}, train_pool, seed=42)
    assert counts(gen_train) == {"good": 10, "defect": 4}
    assert set(gen_train) <= set(train_pool)
    assert gen_train == sample_named_subset(ds, {"good": 10, "defect": 4}, train_pool, seed=42)

    remaining = [i for i in train_pool if i not in set(gen_train)]
    gen_val = sample_named_subset(ds, {"good": 5, "defect": 2}, remaining, seed=7)
    assert counts(gen_val) == {"good": 5, "defect": 2}
    assert not (set(gen_val) & set(gen_train))
    assert not (set(gen_val) & set(test_idx))
    assert not (set(gen_train) & set(test_idx))

    try:
        sample_named_subset(ds, {"good": 10, "nope": 1}, train_pool, seed=42)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown class name must raise, not be ignored")


def check_paper_decoder_and_loss():
    """Paper: sigmoid decoder output in [0, 1] and MSE reconstruction."""
    import torch
    from models import Decoder, cvae_loss

    torch.manual_seed(0)
    dec = Decoder(1, 4, 32, base_ch=8, img_size=64)
    out = dec(torch.randn(2, 32), torch.tensor([0, 1])).detach()
    assert out.shape == (2, 1, 64, 64), out.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, (float(out.min()), float(out.max()))

    x = torch.zeros(2, 1, 8, 8)
    recon_img = torch.full((2, 1, 8, 8), 0.5)
    zero_logvar = torch.zeros(2, 4)
    total, recon, kl = cvae_loss(x, recon_img, torch.zeros(2, 4), zero_logvar, kl_weight=0.015)
    # MSE gives 0.25 here; L1 would give 0.5, so this pins the reconstruction loss.
    assert abs(float(recon) - 0.25) < 1e-6, float(recon)
    assert abs(float(kl)) < 1e-6
    assert abs(float(total) - 0.25) < 1e-6

    total2, _, kl2 = cvae_loss(x, recon_img, torch.ones(2, 4), zero_logvar, kl_weight=0.015)
    assert abs(float(kl2) - 0.5) < 1e-6, float(kl2)
    assert abs(float(total2) - (0.25 + 0.015 * 0.5)) < 1e-6, float(total2)

    # kl_weight must scale the KL term linearly: callers rescale it when they
    # change the loss normalisation.
    _, _, _ = cvae_loss(x, recon_img, torch.ones(2, 4), zero_logvar, kl_weight=1.0)
    scaled, _, _ = cvae_loss(x, recon_img, torch.ones(2, 4), zero_logvar, kl_weight=2.0)
    assert abs(float(scaled) - (0.25 + 2 * 0.5)) < 1e-6, float(scaled)


def check_perceptual_loss():
    """Paper: frozen ImageNet VGG19 used purely as a feature extractor."""
    import torch
    from models import PerceptualLoss

    perc = PerceptualLoss(channels=1)
    assert not any(p.requires_grad for p in perc.parameters()), "VGG19 must stay frozen"

    a = torch.rand(2, 1, 32, 32)
    assert float(perc(a, a).detach()) < 1e-6, "identical inputs must give zero perceptual loss"

    generated = torch.rand(2, 1, 32, 32, requires_grad=True)
    value = perc(generated, torch.rand(2, 1, 32, 32))
    assert value.dim() == 0 and float(value.detach()) > 0
    assert value.requires_grad, "must be differentiable wrt the generated image"


def check_fid_backend_names():
    """Only the two documented backends are accepted."""
    from eval_fid import parse_fid_backend

    assert parse_fid_backend("pytorch_fid") == "pytorch_fid"
    assert parse_fid_backend("legacy") == "legacy"
    try:
        parse_fid_backend("clean_fid")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown FID backend must raise")


def check_channels_refuse_colour():
    """--channels 1 on RGB files must raise rather than silently convert."""
    from data import assert_channels_match_data

    with tempfile.TemporaryDirectory(prefix="cvae_channels_") as td:
        tmp = Path(td)
        folder = tmp / "pore"
        folder.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(folder / "0.png")
        try:
            assert_channels_match_data(tmp, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("RGB tree + --channels 1 must raise")
        assert_channels_match_data(tmp, 3)


def check_generated_fft_parity():
    """The in-training FID must filter generated images exactly like real ones.

    With --fft_denoise the real reference passes through the dataset transform
    (uint8 PIL -> FFTLowPass -> ToTensor). train_joint.py routes generated
    tensors through ToPILImage -> FFTLowPass -> ToTensor; ToPILImage quantises
    [0, 1] floats to uint8, the same quantisation save_image applies when
    eval_fid.py reads generated images from disk. On uint8 input the two paths
    must therefore agree exactly.
    """
    import torch
    from data import FFTLowPass
    from torchvision import transforms as T

    rng = np.random.RandomState(0)
    pil_img = Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    real_path = T.ToTensor()(FFTLowPass(0.25)(pil_img))
    generated_side = T.Compose([T.ToPILImage(), FFTLowPass(0.25), T.ToTensor()])
    gen_path = generated_side(T.ToTensor()(pil_img))
    assert torch.allclose(real_path, gen_path, atol=1e-6), (
        "generated-image FFT post-processing must match the dataset transform")


def check_fid_input_preprocessing():
    """InceptionV3 with transform_input=False wants [-1, 1] RGB at 299x299.

    The pipeline stores [0, 1] images (1- or 3-channel), so both conversions
    have to happen. Getting the range wrong does not raise - it just silently
    shifts every activation off-distribution and makes FID incomparable.
    """
    import torch
    from eval_fid import inception_input

    black = inception_input(torch.zeros(1, 1, 8, 8))
    assert black.shape == (1, 3, 299, 299), black.shape
    assert torch.allclose(black, torch.full_like(black, -1.0)), float(black.min())

    white = inception_input(torch.ones(1, 1, 8, 8))
    assert torch.allclose(white, torch.full_like(white, 1.0)), float(white.max())

    mid = inception_input(torch.full((1, 1, 8, 8), 0.5))
    assert torch.allclose(mid, torch.zeros_like(mid), atol=1e-6)

    # Grayscale must be replicated across channels, not zero-padded.
    gray = torch.rand(2, 1, 16, 16)
    expanded = inception_input(gray)
    assert expanded.shape == (2, 3, 299, 299)
    assert torch.equal(expanded[:, 0], expanded[:, 1])
    assert torch.equal(expanded[:, 1], expanded[:, 2])

    rgb = inception_input(torch.rand(2, 3, 299, 299))
    assert rgb.shape == (2, 3, 299, 299)


def check_generate_counts_by_name():
    """Counts are requested by class NAME so another dataset's ordering can't
    silently generate into the wrong folders."""
    from generate import parse_counts

    classes = ["CR", "LP", "ND", "PO"]
    assert parse_counts("CR=560,PO=400,LP=0", classes) == {0: 560, 3: 400, 1: 0}
    # A different class order must map to different indices, not to fixed ones.
    assert parse_counts("CR=5", ["PO", "CR"]) == {1: 5}
    assert parse_counts("", classes) == {}

    for bad in ("NOPE=1", "CR=1,NOPE=2"):
        try:
            parse_counts(bad, classes)
        except KeyError:
            pass
        else:
            raise AssertionError(f"{bad!r} must raise KeyError, not be ignored")


def check_fid_stats_numerics():
    """FID is a distance: never negative, large for separated distributions,
    and still finite when the feature covariance is rank-deficient.

    Rank deficiency is the normal case whenever a validation or generated set has
    fewer images than InceptionV3's 2048 feature dimensions, which is exactly the
    small-dataset regime this method targets.
    """
    import warnings

    from eval_fid import fid_from_stats

    rng = np.random.RandomState(0)

    def stats(feats):
        return feats.mean(0), np.cov(feats, rowvar=False)

    well_conditioned = rng.randn(64, 16)
    mu, cov = stats(well_conditioned)
    same = fid_from_stats(mu, cov, mu, cov)
    assert same >= 0.0, f"FID is a distance, got {same}"
    assert same < 1e-6, same

    # A mean shift of 3.0 across 16 dimensions is a squared distance of 144.
    shifted = rng.randn(64, 16) + 3.0
    mu_b, cov_b = stats(shifted)
    separated = fid_from_stats(mu, cov, mu_b, cov_b)
    assert separated > 100.0, f"separated distributions must give a large FID, got {separated}"

    # 4 samples in 64 dimensions -> covariance has rank 3 and is singular.
    deficient = rng.randn(4, 64)
    mu2, cov2 = stats(deficient)
    assert np.linalg.matrix_rank(cov2) < cov2.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = fid_from_stats(mu2, cov2, mu2, cov2)
    assert np.isfinite(value), value
    assert value >= 0.0, value


def check_filling_rate_protocol():
    """Paper's augmentation rule: fill each minority class toward N_max.

    ratio 0.0 is the original imbalanced data, ratio 1.0 is "balance-to-max"
    where every class reaches the majority class count.
    """
    from augment import filling_rate_counts

    counts = {"CR": 40, "PO": 200, "ND": 300, "LP": 600}
    assert filling_rate_counts(counts, 0.0) == {"CR": 0, "PO": 0, "ND": 0, "LP": 0}
    assert filling_rate_counts(counts, 1.0) == {"CR": 560, "PO": 400, "ND": 300, "LP": 0}
    assert filling_rate_counts(counts, 0.5) == {"CR": 280, "PO": 200, "ND": 150, "LP": 0}
    assert filling_rate_counts(counts, 0.25) == {"CR": 140, "PO": 100, "ND": 75, "LP": 0}

    # Already-balanced data has no gap to fill, so the protocol is a no-op.
    assert filling_rate_counts({"A": 10, "B": 10}, 1.0) == {"A": 0, "B": 0}

    for bad_ratio in (-0.1, 1.5):
        try:
            filling_rate_counts(counts, bad_ratio)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ratio {bad_ratio} must be rejected, not clamped")


def check_synthetic_prefix_nesting():
    """A filling-rate sweep must reuse one generated pool.

    Taking a sorted prefix guarantees the images used at ratio 0.25 are a subset
    of those at 0.5 and so on, which keeps the sensitivity curve monotonic in the
    treatment rather than comparing different random draws at each point.
    """
    from augment import synthetic_prefix

    paths = [Path(f"gen_{i:03d}.png") for i in range(10)]
    assert synthetic_prefix(paths, 0) == []
    assert synthetic_prefix(paths, 3) == paths[:3]
    assert synthetic_prefix(paths, 10) == paths
    assert set(synthetic_prefix(paths, 3)) <= set(synthetic_prefix(paths, 7))

    try:
        synthetic_prefix(paths, 11)
    except ValueError:
        pass
    else:
        raise AssertionError("requesting more images than exist must raise")


def check_feature_aggregation_fanin():
    """Encoder and discriminator must pool before their dense heads.

    Flattening a 512x14x14 map straight into fc_logvar gives a fan-in of 100,352.
    At the paper's lr=1e-3 one Adam step then moves every weight by ~lr at once,
    changing logvar by tens and overflowing exp(logvar) - observed on real data as
    KL reaching 1e25 and then NaN within the first epoch. The paper describes a 1x1
    channel reduction before the latent heads and global max pooling before the
    discriminator's dense layer, which is what keeps the step size sane.
    """
    import torch
    from models import Discriminator, Encoder

    for img_size in (64, 224):
        enc = Encoder(1, 4, 32, base_ch=64, img_size=img_size)
        assert enc.fc_mu.in_features <= 512, (img_size, enc.fc_mu.in_features)
        assert enc.fc_logvar.in_features == enc.fc_mu.in_features
        mu, logvar = enc(torch.rand(2, 1, img_size, img_size), torch.tensor([0, 1]))
        assert mu.shape == (2, 32) and logvar.shape == (2, 32)

        dis = Discriminator(1, 4, base_ch=64, img_size=img_size)
        assert dis.fc.in_features <= 512, (img_size, dis.fc.in_features)
        assert dis(torch.rand(2, 1, img_size, img_size), torch.tensor([0, 1])).shape == (2,)


def check_training_curves_figure(tmp):
    from make_paper_figures import save_training_curves

    history = {"epoch": [1, 2, 3], "recon": [0.30, 0.20, 0.15], "kl": [0.10, 0.05, 0.02],
               "perc": [0.50, 0.40, 0.35], "loss_g": [2.0, 1.5, 1.2],
               "loss_d": [0.90, 0.70, 0.60], "val_loss": [0.40, 0.30, 0.25],
               "val_recon": [0.31, 0.22, 0.16], "fid": [400.0, float("nan"), 350.0],
               "lr": [1e-3] * 3, "seconds": [30.0] * 3}
    output = tmp / "training_curves.png"
    save_training_curves(history, output)
    assert output.is_file() and output.stat().st_size > 0


def check_filling_rate_curve_figure(tmp):
    from make_paper_figures import save_filling_rate_curve

    sweep = {0.0: {"accuracy": 0.42, "macro_f1": 0.40, "weighted_f1": 0.41,
                   "f1:CR": 0.21, "f1:LP": 0.55},
             0.5: {"accuracy": 0.50, "macro_f1": 0.48, "weighted_f1": 0.49,
                   "f1:CR": 0.30, "f1:LP": 0.62},
             1.0: {"accuracy": 0.61, "macro_f1": 0.59, "weighted_f1": 0.60,
                   "f1:CR": 0.44, "f1:LP": 0.70}}
    output = tmp / "filling_rate_curve.png"
    save_filling_rate_curve(sweep, ["CR", "LP"], output)
    assert output.is_file() and output.stat().st_size > 0


def check_multi_seed_filling_rate_figure(tmp):
    """The multi-seed figure must aggregate over seeds, not over ratios.

    Its whole purpose is to show that the r=1.0 gain is a positive *mean* effect
    with visible seed spread, so a bug that averaged the wrong axis, or silently
    dropped a ratio that one seed is missing, would invert the published reading.

    The spread is the *sample* standard deviation (ddof=1), matching the table in
    docs/CALIBRATION.md section 9; a figure drawn with population std would show
    a narrower band than the numbers it illustrates.
    """
    from make_paper_figures import aggregate_seed_metrics, save_multi_seed_filling_rate

    seeds = [{0.0: {"macro_f1": 0.60, "f1:pore": 0.30},
              1.0: {"macro_f1": 0.70, "f1:pore": 0.40}},
             {0.0: {"macro_f1": 0.70, "f1:pore": 0.50},
              0.5: {"macro_f1": 0.65, "f1:pore": 0.35},
              1.0: {"macro_f1": 0.80, "f1:pore": 0.60}}]

    agg = aggregate_seed_metrics(seeds, "macro_f1")
    assert sorted(agg) == [0.0, 0.5, 1.0], sorted(agg)

    def close(got, mean, std, n):
        return abs(got[0] - mean) < 1e-9 and abs(got[1] - std) < 1e-9 and got[2] == n

    sample_std = 0.1 / 2 ** 0.5  # two values 0.1 apart
    assert close(agg[0.0], 0.65, sample_std, 2), agg[0.0]
    assert close(agg[1.0], 0.75, sample_std, 2), agg[1.0]
    # 0.5 exists in one seed only: it aggregates from that seed, with no spread.
    assert close(agg[0.5], 0.65, 0.0, 1), agg[0.5]

    pore = aggregate_seed_metrics(seeds, "f1:pore")
    assert close(pore[1.0], 0.50, 0.2 / 2 ** 0.5, 2), pore[1.0]

    output = tmp / "filling_rate_multiseed.png"
    save_multi_seed_filling_rate(
        seeds,
        references={"published v0.2.0 (seed 42)": {0.0: {"macro_f1": 0.6936,
                                                         "f1:pore": 0.3778},
                                                   1.0: {"macro_f1": 0.6418,
                                                         "f1:pore": 0.3871}}},
        minority="pore",
        output=output,
    )
    assert output.is_file() and output.stat().st_size > 0


def check_multi_seed_arm_exclusion():
    """Excluding one invalid arm must drop that arm only, and never silently.

    The seed-42 r=0.25 score in the section 9 sweep is invalid (final-epoch loss
    spike), so the figure has to drop one (file, ratio) pair while keeping the
    same ratio from the other seeds. Dropping it everywhere, or silently ignoring
    a mistyped path or ratio, would reshape the curve with no warning.
    """
    from make_multiseed_figure import drop_arms

    loaded = {"s42.csv": {0.0: {"macro_f1": 0.68}, 0.25: {"macro_f1": 0.73},
                          1.0: {"macro_f1": 0.72}},
              "s43.csv": {0.0: {"macro_f1": 0.63}, 0.25: {"macro_f1": 0.75},
                          1.0: {"macro_f1": 0.72}}}

    kept = drop_arms(loaded, ["s42.csv=0.25"])
    assert sorted(kept["s42.csv"]) == [0.0, 1.0], sorted(kept["s42.csv"])
    assert sorted(kept["s43.csv"]) == [0.0, 0.25, 1.0], sorted(kept["s43.csv"])
    # The input must not be mutated: the caller still holds the unfiltered sweeps.
    assert 0.25 in loaded["s42.csv"]

    for bad in ["s99.csv=0.25", "s42.csv=0.6", "s42.csv"]:
        try:
            drop_arms(loaded, [bad])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"--exclude {bad!r} matched nothing but was accepted")


def check_comparison_grid_labels():
    """The showcase grid alternates real/generated by row, so its caption must too.

    The published figures drew "Real" and "Generated" as column headers at 25% and
    75% of the width, which reads as "left half real, right half generated" - the
    opposite of the actual layout, where every class contributes a real row and a
    generated row.
    """
    from make_comparison import grid_caption, row_labels

    assert row_labels(["pore", "stain"]) == [
        "pore real", "pore gen", "stain real", "stain gen",
    ], row_labels(["pore", "stain"])

    caption = grid_caption().lower()
    assert "row" in caption, grid_caption()
    assert "column" not in caption, grid_caption()


def check_generated_class_manifest(tmp):
    """A generated pool must record the class order it was produced with.

    Folders are positional (class_0, class_1, ...), so a pool generated from a
    differently ordered dataset would silently assign every synthetic image to
    the wrong class - corrupting the filling-rate result with no error anywhere.
    """
    from augment import generated_by_class

    root = tmp / "gen_pool"
    (root / "class_0").mkdir(parents=True)
    (root / "class_1").mkdir()
    (root / "class_0" / "gen_0_00000.png").write_bytes(b"not really a png")
    (root / "class_1" / "gen_1_00000.png").write_bytes(b"not really a png")

    try:
        generated_by_class(root, ["CR", "LP"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a pool with no class manifest must be rejected")

    (root / "classes.txt").write_text("LP\nCR\n", encoding="utf-8")
    try:
        generated_by_class(root, ["CR", "LP"])
    except ValueError:
        pass
    else:
        raise AssertionError("a manifest in a different class order must be rejected")

    (root / "classes.txt").write_text("CR\nLP\n", encoding="utf-8")
    pooled = generated_by_class(root, ["CR", "LP"])
    assert [p.name for p in pooled["CR"]] == ["gen_0_00000.png"]
    assert [p.name for p in pooled["LP"]] == ["gen_1_00000.png"]


def check_hinge_losses_and_bounded_score():
    """Hinge objective plus spectral normalisation keep the adversarial term bounded.

    The unbounded BCE term -log(D(G(z))) was what let the adversarial loss reach
    ~50x the reconstruction term once D saturated. Hinge's generator term is the
    negated mean score, and spectral normalisation bounds that score, so the term
    can no longer dominate.
    """
    import torch
    from models import Discriminator, hinge_d, hinge_g

    assert float(hinge_d(torch.tensor([2.0]), torch.tensor([-2.0]))) == 0.0
    assert float(hinge_d(torch.zeros(1), torch.zeros(1))) == 2.0
    assert float(hinge_g(torch.tensor([3.0]))) == -3.0

    torch.manual_seed(0)
    dis = Discriminator(1, 2, base_ch=16, img_size=64)
    # A saturated discriminator must still emit a bounded score, otherwise the
    # generator's adversarial term can grow without limit.
    score = dis(torch.full((4, 1, 64, 64), 50.0), torch.zeros(4, dtype=torch.long)).detach()
    assert float(score.abs().max()) < 100.0, float(score.abs().max())


def check_discriminator_normalisation_choice():
    """The journal paper normalises with GroupNorm (Table 3), not BatchNorm.

    At this repo's batch_size=8 a BatchNorm discriminator scores an image using
    statistics from the seven other images sharing its batch, so the same image
    gets a different score depending on its neighbours. Measured consequence: our
    hinge discriminator loss sits at a median of 0.82 over a range of 0.40-2.41,
    with 97% of epochs below 1.5 (a dominant D), while the paper reports an
    equilibrium near 2.0 (D(x,c) ~ 0).

    ``norm="batch"`` must stay the default so the tagged v0.2.0 runs remain
    reproducible from their recorded config.
    """
    import torch
    from models import Discriminator

    torch.manual_seed(0)
    x = torch.rand(4, 1, 64, 64)
    y = torch.tensor([0, 1, 0, 1])

    batch_norm = Discriminator(1, 2, base_ch=16, img_size=64)
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in batch_norm.modules())
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in batch_norm.modules())

    group_norm = Discriminator(1, 2, base_ch=16, img_size=64, norm="group")
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in group_norm.modules())
    assert any(isinstance(m, torch.nn.GroupNorm) for m in group_norm.modules())

    # Each forward goes through a freshly seeded module, because spectral_norm
    # updates its power-iteration buffers on every training forward - reusing one
    # module would measure that drift instead of batch leakage.
    def score_first(kind, batch):
        torch.manual_seed(0)
        dis = Discriminator(1, 2, base_ch=16, img_size=64, norm=kind)
        dis.train()
        with torch.no_grad():
            return dis(batch, y[:batch.size(0)])[0]

    small_bn, full_bn = score_first("batch", x[:2]), score_first("batch", x)
    small_gn, full_gn = score_first("group", x[:2]), score_first("group", x)
    assert not torch.allclose(small_bn, full_bn, atol=1e-6), "BatchNorm must leak batch context"
    assert torch.allclose(small_gn, full_gn, atol=1e-9), (float(small_gn), float(full_gn))

    for bad in ("instance", "", None):
        try:
            Discriminator(1, 2, base_ch=16, img_size=64, norm=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"norm={bad!r} must be rejected, not fall back silently")


def check_kl_annealing_schedule():
    """The journal paper anneals beta: zero for the first 10 epochs, then linear
    to its target over the next 50 (Table 3, "KL Annealing Target 0.0 -> 0.5
    warm-up over 50 epochs"; Sec. 4, "beta was kept at zero during the first 10
    epochs"). Its stated reason is that a constant high beta from the start risks
    posterior collapse.

    ramp_epochs=0 must reproduce the constant-beta behaviour the tagged v0.2.0
    runs were produced with.
    """
    from models import kl_schedule

    assert kl_schedule(1, 0.5, zero_epochs=10, ramp_epochs=50) == 0.0
    assert kl_schedule(10, 0.5, zero_epochs=10, ramp_epochs=50) == 0.0
    assert kl_schedule(11, 0.5, zero_epochs=10, ramp_epochs=50) == 0.5 * (1 / 50)
    assert abs(kl_schedule(35, 0.5, zero_epochs=10, ramp_epochs=50) - 0.25) < 1e-12
    assert kl_schedule(60, 0.5, zero_epochs=10, ramp_epochs=50) == 0.5
    assert kl_schedule(200, 0.5, zero_epochs=10, ramp_epochs=50) == 0.5

    for epoch in (1, 10, 60, 200):
        assert kl_schedule(epoch, 0.059) == 0.059, epoch

    for bad in ((0, 0.5, 10, 50), (-1, 0.5, 10, 50),
                (5, 0.5, -1, 50), (5, 0.5, 10, -5)):
        try:
            kl_schedule(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"kl_schedule{bad} must be rejected, not clamped")


def check_class_balanced_sampling_weights():
    """The journal paper's data-balancing strategy is a WeightedRandomSampler with
    P proportional to 1/N_class, so every minibatch is class balanced (Table 3).

    Capping counts with --subset does not achieve this on its own: pore at 40 of
    1090 training images appears in roughly one batch in three at batch_size=8, so
    most gradient updates carry no minority-class signal at all.
    """
    import torch
    from data import balanced_sample_weights

    labels = [0] * 90 + [1] * 10
    weights = balanced_sample_weights(labels)
    assert len(weights) == len(labels)
    assert abs(weights[0] - 1 / 90) < 1e-12, weights[0]
    assert abs(weights[90] - 1 / 10) < 1e-12, weights[90]
    assert balanced_sample_weights([]) == []

    torch.manual_seed(0)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=40000,
                                                     replacement=True)
    drawn = [labels[i] for i in sampler]
    minority_freq = drawn.count(1) / len(drawn)
    assert abs(minority_freq - 0.5) < 0.02, minority_freq


if __name__ == "__main__":
    import torch

    check_balanced_fid_sampling()
    tmp = Path(tempfile.mkdtemp(prefix="cvae_cgan_smoke_"))  # OS-agnostic temp dir
    try:
        check_class_distribution_figure(tmp)
        check_distribution_protocol()
        check_reconstruction_figure(tmp)
        check_latent_projection_figure(tmp)
        check_history_csv_reading(tmp)
        check_sweep_csv_reading(tmp)
        check_training_curves_figure(tmp)
        check_filling_rate_curve_figure(tmp)
        check_multi_seed_filling_rate_figure(tmp)
        check_multi_seed_arm_exclusion()
        check_comparison_grid_labels()
        check_confusion_matrix_figure(tmp)
        root = str(tmp / "smoke_data")
        make_synthetic_weld_data(root, size=64)

        check_grayscale_dataset(tmp)
        check_fft_denoise()
        check_generated_fft_parity()
        check_no_leakage_splits(tmp)
        check_paper_decoder_and_loss()
        check_hinge_losses_and_bounded_score()
        check_discriminator_normalisation_choice()
        check_kl_annealing_schedule()
        check_class_balanced_sampling_weights()
        check_feature_aggregation_fanin()
        check_perceptual_loss()
        check_fid_backend_names()
        check_channels_refuse_colour()
        check_fid_input_preprocessing()
        check_fid_stats_numerics()
        check_generate_counts_by_name()
        check_filling_rate_protocol()
        check_synthetic_prefix_nesting()
        check_generated_class_manifest(tmp)

        # Three epochs with --kl_warmup 1,1 so the run passes through both the
        # zero-beta phase (epoch 1) and the ramp (epochs 2-3).
        run(["--data_root", root, "--img_size", "64", "--latent_dim", "8", "--base_ch", "16",
             "--epochs", "3", "--batch_size", "8", "--subset", "good=8,defect=4",
             "--val_per_class", "4", "--fid_every", "1", "--sample_every", "2",
             "--num_workers", "0", "--d_norm", "group", "--weighted_sampler",
             "--kl_warmup", "1,1", "--monitor", "val_recon",
             "--out_dir", str(tmp / "runs" / "joint")], "train_joint")

        run(["--ckpt", str(tmp / "runs" / "joint" / "joint.pt"),
             "--out_root", str(tmp / "smoke_gen"), "--counts", "good=4,defect=4"], "generate")

        # Same --subset and seed as train_joint above, so the r=0.0 condition is
        # exactly the real data the generator trained on.
        run(["--data_root", root, "--gen_root", str(tmp / "smoke_gen"),
             "--subset", "good=8,defect=4", "--ratios", "0.0,1.0",
             "--img_size", "64", "--channels", "3", "--epochs", "1", "--batch_size", "4",
             "--num_workers", "0", "--out_dir", str(tmp / "runs" / "sweep")], "train_classifier")

        sweep_dir = tmp / "runs" / "sweep"
        for expected in ("results.txt", "sweep_metrics.csv", "cm_r0.npy", "cm_r1.npy"):
            assert (sweep_dir / expected).is_file(), f"missing {expected}"
        print("SMOKE TEST OK")
    except BaseException:
        print(f"SMOKE TEST FAILED - keeping temp dir for inspection: {tmp}")
        raise
    shutil.rmtree(tmp, ignore_errors=True)
