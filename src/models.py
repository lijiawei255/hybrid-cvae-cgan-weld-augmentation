# -*- coding: utf-8 -*-
"""
Model definitions for the hybrid CVAE-CGAN pipeline
(method-level re-implementation of Yang et al., CYBER 2025, with selected
components from its journal extension in MSSP 2026: the spectral-normalised
hinge discriminator, the GroupNorm option and the KL annealing schedule; the
inner docstrings attribute each piece).

The "hybrid" is that one decoder plays both roles at once: it is the CVAE
decoder D(z, y) that reconstructs the input, and simultaneously the CGAN
generator G(z, y) that the discriminator tries to distinguish from real images.
Both papers train these objectives **jointly** under a single combined loss
(reconstruction + KL + VGG19 perceptual + adversarial), alternating the
generator and discriminator updates within each minibatch.

Conventions every module here shares, both dictated by the paper:
  * images live in [0, 1]; the channel count is a parameter (`channels`) - the
    papers' camera produced grayscale, the current primary dataset keeps RGB;
  * the decoder ends in a sigmoid, so its output is already in that range and
    nothing downstream needs a Tanh-style rescale.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision.models import VGG19_Weights, vgg19


def weights_init(m):
    """DCGAN-style initialization; noticeably improves GAN training stability."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _final_size(img_size):
    """Feature-map size after four stride-2 stages; refuses sizes that truncate."""
    if img_size % 16:
        raise ValueError(
            f"img_size must be divisible by 16 (four stride-2 stages), got {img_size}")
    return img_size // 16


GROUP_NORM_GROUPS = 32


def _group_count(channels):
    """Groups for a GroupNorm over `channels`, at most GROUP_NORM_GROUPS.

    Neither paper states a group count, so 32 is this repo's choice - it is what
    the GroupNorm paper uses and it divides every channel width the default
    base_ch=64 produces. A narrow stage (a small --base_ch, whose decoder ends at
    base_ch//2 channels) cannot be split 32 ways, so the count falls back to the
    largest divisor of the width that does not exceed 32. At the published widths
    this always returns 32, so it changes no tagged run.
    """
    if channels < 1:
        raise ValueError(f"GroupNorm needs at least one channel, got {channels}")
    for groups in range(min(GROUP_NORM_GROUPS, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _norm_layer(norm):
    """Normalisation factory; returns a callable channels -> layer.

    ``batch`` is what the tagged v0.2.0 runs were produced with. ``group`` is the
    journal paper's choice - its Table 3 lists GroupNorm as the normalisation layer
    for **every** network, chosen "to ensure stability with small batch sizes,
    avoiding the statistical instability associated with Batch Normalisation". That
    matters concretely here because this repo trains at batch_size=8, where a
    BatchNorm discriminator scores an image using statistics from the other seven
    images in its batch. The group count is 32 because the paper does not state one.

    The discriminator selects this with ``--d_norm`` and the encoder/decoder with
    ``--g_norm``; both default to ``batch``, which is what every tagged run used.
    Only the discriminator's setting has been measured (docs/CALIBRATION.md
    section 9).
    """
    if norm == "batch":
        return nn.BatchNorm2d
    if norm != "group":
        raise ValueError(f"norm must be 'batch' or 'group', got {norm!r}")

    def group_norm(channels):
        return nn.GroupNorm(_group_count(channels), channels)

    return group_norm


class Encoder(nn.Module):
    """CVAE encoder E(x, y) -> (mu, logvar) of the latent distribution.

    A 1x1 channel reduction followed by global average pooling feeds the latent
    heads, as the paper describes. This is not cosmetic: flattening the full
    convolutional map instead gives ``fc_logvar`` a fan-in of
    ``c*8*(img_size//16)**2`` - 100,352 at 224x224 - and a single Adam step at the
    paper's lr=1e-3 then moves every one of those weights by ~lr at once, shifting
    logvar by tens so that ``exp(logvar)`` overflows. Pooling first keeps the
    latent-head fan-in fixed regardless of resolution. (The label embedding is
    still sized ``img_size * img_size``, so the module expects the ``img_size``
    it was constructed with.)
    """

    def __init__(self, img_channels=1, num_classes=4, latent_dim=32, base_ch=64, img_size=224,
                 norm="batch"):
        super().__init__()
        # Class label is embedded as an extra spatial map and concatenated with
        # the image, which is how both papers condition the encoder.
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        c = base_ch
        normalise = _norm_layer(norm)
        # Four stride-2 convolutions: 224 -> 112 -> 56 -> 28 -> 14
        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1), normalise(c * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), normalise(c * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), normalise(c * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, c * 2, 1), normalise(c * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_mu = nn.Linear(c * 2, latent_dim)
        self.fc_logvar = nn.Linear(c * 2, latent_dim)

    def forward(self, x, y):
        y_map = self.label_emb(y).view(x.size(0), 1, x.size(2), x.size(3))
        h = self.net(torch.cat([x, y_map], dim=1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """CVAE decoder D(z, y), which is also the CGAN generator G(z, y).

    Upsampling uses Sub-Pixel (PixelShuffle) convolution rather than transposed
    convolution, as the conference paper describes. Transposed convolution left a
    visible checkerboard artefact in both reconstructions and generated samples;
    PixelShuffle rearranges channels into spatial positions instead of inserting
    zeros, which avoids it. Four shuffle stages take the map from img_size/16 up
    to img_size.
    """

    def __init__(self, img_channels=1, num_classes=4, latent_dim=32, base_ch=64, img_size=224,
                 norm="batch"):
        super().__init__()
        c = base_ch
        self.final_size = _final_size(img_size)
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim * 2, c * 8 * self.final_size ** 2)
        normalise = _norm_layer(norm)
        widths = [c * 8, c * 4, c * 2, c, c // 2]
        stages = []
        for w_in, w_out in zip(widths, widths[1:]):
            stages += [
                nn.Conv2d(w_in, w_out * 4, 3, 1, 1), normalise(w_out * 4),
                nn.ReLU(inplace=True), nn.PixelShuffle(2),
                nn.Conv2d(w_out, w_out, 3, 1, 1), normalise(w_out),
                nn.ReLU(inplace=True),
            ]
        stages += [nn.Conv2d(widths[-1], img_channels, 3, 1, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*stages)

    def forward(self, z, y):
        y_vec = self.label_emb(y)
        h = self.fc(torch.cat([z, y_vec], dim=1))
        h = h.view(z.size(0), -1, self.final_size, self.final_size)
        return self.net(h)


class Discriminator(nn.Module):
    """Conditional discriminator: is this (image, class label) pair real or generated?

    Global max pooling precedes the dense layer, as the paper describes. Like the
    encoder this is a stability requirement, not a stylistic one: flattening the
    convolutional map into the dense head gives it a 100,352 fan-in at 224x224,
    which diverged at the paper's lr=1e-3.

    Every weight matrix is spectral-normalised by default, as in the journal
    extension. That bounds the score's Lipschitz constant, which is what keeps the
    hinge generator term bounded - without it the adversarial loss overwhelmed the
    reconstruction term on the small training set (docs/CALIBRATION.md sections 2
    and 6). The journal paper applies SN to the first three convolutional blocks
    only; this repo applies it to all of them plus the dense head, which is the
    deviation recorded in docs/CALIBRATION.md section 6.

    ``spectral=False`` removes it entirely, which together with ``--adv_loss bce``
    is the conference paper's original objective. That combination is expressible
    on purpose but is not recommended: it is the configuration measured to fail in
    docs/CALIBRATION.md section 6.

    ``norm`` selects the normalisation layer (see _norm_layer). ``weights_init``
    matches on the substring "BatchNorm", so GroupNorm layers keep PyTorch's own
    defaults (weight 1, bias 0), which is what they should start at.
    """

    def __init__(self, img_channels=1, num_classes=4, base_ch=64, img_size=224, norm="batch",
                 spectral=True):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        c = base_ch
        normalise = _norm_layer(norm)
        layers = [
            nn.Conv2d(img_channels + 1, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1), normalise(c * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), normalise(c * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), normalise(c * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveMaxPool2d(1),
        ]
        if spectral:
            layers = [spectral_norm(m) if isinstance(m, nn.Conv2d) else m for m in layers]
        self.net = nn.Sequential(*layers)
        head = nn.Linear(c * 8, 1)
        self.fc = spectral_norm(head) if spectral else head

    def forward(self, x, y):
        y_map = self.label_emb(y).view(x.size(0), 1, x.size(2), x.size(3))
        return self.fc(self.net(torch.cat([x, y_map], dim=1)).flatten(1)).view(-1)


def hinge_d(real_score, fake_score):
    """Discriminator hinge loss over raw (unbounded-by-sigmoid) scores."""
    return (nn.functional.relu(1.0 - real_score).mean()
            + nn.functional.relu(1.0 + fake_score).mean())


def hinge_g(fake_score):
    """Generator's adversarial term under hinge.

    Bounded once the discriminator is spectral-normalised, unlike the BCE
    minimax term -log(D(G(z))) which grows without limit as D becomes confident.
    """
    return -fake_score.mean()


def bce_d(real_score, fake_score):
    """Discriminator loss under the conference paper's BCE minimax objective.

    Scores are raw logits, so this uses the numerically stable
    ``binary_cross_entropy_with_logits`` rather than a sigmoid followed by BCE.
    No label smoothing: the tagged runs that used smoothing predate the paper
    being read, and the paper specifies none.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    return (bce(real_score, torch.ones_like(real_score))
            + bce(fake_score, torch.zeros_like(fake_score)))


def bce_g(fake_score):
    """Generator's adversarial term under BCE, in its non-saturating form.

    This is ``-log(D(G(z)))``, which is **unbounded above**: as the discriminator
    becomes confident the term grows without limit. On this repo's data scale it
    reached ~50x the reconstruction term and FID rose instead of falling, which is
    why the default objective is hinge plus spectral normalisation instead
    (measured in docs/CALIBRATION.md sections 2 and 6). It is kept expressible so
    the conference paper's own configuration can be run and its failure observed.
    """
    return nn.functional.binary_cross_entropy_with_logits(
        fake_score, torch.ones_like(fake_score))


#: Adversarial objectives, by the name their --adv_loss flag uses.
ADV_LOSSES = {"hinge": (hinge_d, hinge_g), "bce": (bce_d, bce_g)}
DEFAULT_ADV_LOSS = "hinge"


def reparameterize(mu, logvar):
    """Reparameterization trick: makes sampling differentiable so the VAE can be trained."""
    # The clamp is a divergence guard, not part of the method: healthy training
    # keeps logvar far inside +/-10, but an exploding run would otherwise turn
    # exp(0.5*logvar) into inf and silently poison every later epoch.
    logvar = logvar.clamp(-10.0, 10.0)
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def cvae_loss(x, x_recon, mu, logvar, kl_weight):
    """Reconstruction + KL divergence toward N(0, I).

    Reconstruction is mean squared error over pixels, and KL is a mean over both
    the batch and the latent dimensions. Both are deliberately normalised so the
    balance between them does not depend on image resolution or latent size.

    `kl_weight` has no default on purpose. The paper reports beta = 30, but that
    value belongs to *its* normalisation - reconstruction on a [0, 255] pixel
    scale with KL summed over latent dimensions. Under the mean-normalised [0, 1]
    losses used here the equivalent weight is about 0.015 (see train_joint.py);
    passing 30 collapses the posterior to the prior and destroys reconstruction.
    """
    recon = nn.functional.mse_loss(x_recon, x, reduction="mean")
    logvar = logvar.clamp(-10.0, 10.0)  # same divergence guard as reparameterize
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl


def kl_schedule(epoch, target, zero_epochs=0, ramp_epochs=0):
    """Effective KL weight for a 1-indexed `epoch` under the journal paper's annealing.

    The paper holds beta at zero for the first 10 epochs, then raises it linearly
    to its target over the next 50, so the encoder learns to reconstruct before
    the prior is imposed; it names a constant high beta from the start as the thing
    that "risks posterior collapse where the encoder ignores inputs".

    ``ramp_epochs=0`` turns annealing off and returns the constant target, which is
    the behaviour the tagged v0.2.0 runs were produced with.
    """
    if epoch < 1:
        raise ValueError(f"epoch is 1-indexed, got {epoch}")
    if zero_epochs < 0 or ramp_epochs < 0:
        raise ValueError(f"annealing lengths must be non-negative, got "
                         f"zero_epochs={zero_epochs}, ramp_epochs={ramp_epochs}")
    if ramp_epochs == 0:
        return target
    if epoch <= zero_epochs:
        return 0.0
    return target * min(1.0, (epoch - zero_epochs) / ramp_epochs)


class PerceptualLoss(nn.Module):
    """Frozen ImageNet VGG19 feature-space distance (the paper's perceptual term).

    Neither paper names which VGG19 layers to compare; this implementation
    distances the final ``.features`` output. VGG19 is RGB-only, so
    single-channel inputs are replicated to three
    channels; both sides are expected in [0, 1] and are ImageNet-normalised
    here. The target features are computed under no_grad because VGG19 is a
    fixed feature extractor - gradients only need to flow through the
    generated side, back into the decoder.
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def _prepare(self, x):
        if self.channels == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.mean) / self.std

    def forward(self, generated, target):
        with torch.no_grad():
            target_features = self.features(self._prepare(target))
        return nn.functional.mse_loss(self.features(self._prepare(generated)), target_features)
