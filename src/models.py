# -*- coding: utf-8 -*-
"""
Model definitions for the hybrid CVAE-CGAN pipeline
(method-level re-implementation of Yang et al., 2025).

Two-stage design:
  Stage 1: train a Conditional VAE so that the decoder learns
           "noise z + class label y -> image of class y".
  Stage 2: reuse the trained CVAE decoder as the generator G of a
           conditional GAN and fine-tune adversarially. The
           discriminator D sees (image, class label) pairs.
The "hybrid" aspect is exactly this weight hand-off: G is NOT
initialized from scratch but from the CVAE decoder.
"""
import torch
import torch.nn as nn


def weights_init(m):
    """DCGAN-style initialization; noticeably improves GAN training stability."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):
    """CVAE encoder: (image, class label) -> (mu, logvar) of the latent distribution."""

    def __init__(self, img_channels, num_classes, latent_dim, base_ch=64, img_size=128):
        super().__init__()
        # Class label is embedded as an extra channel and concatenated with the image.
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        c = base_ch
        # Four stride-2 convolutions: e.g. 128 -> 64 -> 32 -> 16 -> 8
        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_size = img_size // 16
        self.fc_mu = nn.Linear(c * 8 * self.final_size ** 2, latent_dim)
        self.fc_logvar = nn.Linear(c * 8 * self.final_size ** 2, latent_dim)

    def forward(self, x, y):
        y_map = self.label_emb(y).view(x.size(0), 1, x.size(2), x.size(3))
        h = self.net(torch.cat([x, y_map], dim=1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """CVAE decoder; later reused as the CGAN generator G: (z, class label) -> image."""

    def __init__(self, img_channels, num_classes, latent_dim, base_ch=64, img_size=128):
        super().__init__()
        c = base_ch
        self.final_size = img_size // 16
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim * 2, c * 8 * self.final_size ** 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, img_channels, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z, y):
        y_vec = self.label_emb(y)
        h = self.fc(torch.cat([z, y_vec], dim=1))
        h = h.view(z.size(0), -1, self.final_size, self.final_size)
        return self.net(h)


class Discriminator(nn.Module):
    """CGAN discriminator: decides whether an (image, class label) pair is real or generated."""

    def __init__(self, img_channels, num_classes, base_ch=64, img_size=128):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        c = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1), nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_size = img_size // 16
        self.fc = nn.Linear(c * 8 * self.final_size ** 2, 1)

    def forward(self, x, y):
        y_map = self.label_emb(y).view(x.size(0), 1, x.size(2), x.size(3))
        return self.fc(self.net(torch.cat([x, y_map], dim=1)).flatten(1)).view(-1)


def reparameterize(mu, logvar):
    """Reparameterization trick: makes sampling differentiable so the VAE can be trained."""
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def cvae_loss(x, x_recon, mu, logvar, kl_weight=1.0):
    """CVAE loss = pixel reconstruction + KL divergence toward N(0, I)."""
    recon = nn.functional.l1_loss(x_recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl_weight * kl, recon, kl
