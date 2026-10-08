"""Convolutional Variational Autoencoder for Part C.

Compresses a (B, 3, 128, 128) image into a (B, 4, 16, 16) continuous latent.
Follows the architecture in `docs/architecture_vae_ldm.html` exactly:

    Encoder : 3x3 Conv(3->32) -> [stage x3: 2 ResBlocks + stride-2 down] ch 32,64,128
    Mid     : 2 ResBlocks -> 3x3 Conv(128->8)
    Latent  : split(8) -> mu(4) + logvar(4) -> reparameterize -> z (B,4,16,16)
    Decoder : 3x3 Conv(4->128) -> 2 ResBlocks
              -> [stage x3: Upsample2x+Conv + 2 ResBlocks] ch 128,64,32
              -> GN -> SiLU -> 3x3 Conv(32->3) -> Tanh

ResBlocks and the output projection use SiLU and GroupNorm (num_groups=32 unless
channels < 32, in which case we fall back to channels itself).
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(c: int) -> int:
    """Pick num_groups for GroupNorm: 32 if feasible, else fall back."""
    if c % 32 == 0:
        return 32
    if c % 8 == 0:
        return 8
    return c  # LayerNorm-like fallback


class ResBlock(nn.Module):
    """Pre-norm residual block shared by VAE (and the LDM U-Net via a subclass).

    For the VAE this block does NOT ingest a time embedding; the U-Net version
    (in `unet.py`) adds an optional `time_emb` argument.

    Flow : GN -> SiLU -> Conv3x3 -> GN -> SiLU -> Conv3x3 + skip
    The skip is Identity when in==out, else a 1x1 Conv projection.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Strided 3x3 conv that halves spatial resolution."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbour 2x upsample followed by a 3x3 conv (spec: 'Nearest + Conv')."""

    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


@dataclass
class VAEOutput:
    recon: torch.Tensor    # (B, 3, 128, 128), Tanh-bounded to [-1, 1]
    mu: torch.Tensor       # (B, 4, 16, 16)
    logvar: torch.Tensor   # (B, 4, 16, 16)
    z: torch.Tensor        # (B, 4, 16, 16) - sampled latent


class ConvVAE(nn.Module):
    """128x128x3 <-> 16x16x4 convolutional VAE.

    Channel ladder and block counts match the assignment spec:
        encoder : 32 -> 64 -> 128 (3 stride-2 downs)
        decoder : 128 -> 64 -> 32 (3 nearest upsamples)
    """

    LATENT_CH = 4
    LATENT_HW = 16

    def __init__(self, img_size: int = 128, base_ch: int = 32):
        super().__init__()
        assert img_size == 128, "architecture is fixed to 128x128 input"
        enc_channels = [base_ch, base_ch * 2, base_ch * 4]  # 32, 64, 128

        # ---------------- Encoder ----------------
        self.enc_init = nn.Conv2d(3, enc_channels[0], kernel_size=3, padding=1)
        enc_blocks: list[nn.Module] = []
        prev = enc_channels[0]
        for c in enc_channels:
            enc_blocks.append(ResBlock(prev, c))
            enc_blocks.append(ResBlock(c, c))
            enc_blocks.append(Downsample(c))
            prev = c
        self.enc_blocks = nn.ModuleList(enc_blocks)

        # ---------------- Mid (encoder side) ----------------
        self.enc_mid = nn.Sequential(
            ResBlock(prev, prev),
            ResBlock(prev, prev),
        )
        self.to_latent = nn.Conv2d(prev, 2 * self.LATENT_CH, kernel_size=3, padding=1)

        # ---------------- Decoder ----------------
        dec_channels = list(reversed(enc_channels))  # 128, 64, 32
        self.dec_init = nn.Conv2d(self.LATENT_CH, dec_channels[0], kernel_size=3, padding=1)
        self.dec_mid = nn.Sequential(
            ResBlock(dec_channels[0], dec_channels[0]),
            ResBlock(dec_channels[0], dec_channels[0]),
        )
        dec_blocks: list[nn.Module] = []
        prev = dec_channels[0]
        for c in dec_channels:
            dec_blocks.append(Upsample(prev))
            dec_blocks.append(ResBlock(prev, c))
            dec_blocks.append(ResBlock(c, c))
            prev = c
        self.dec_blocks = nn.ModuleList(dec_blocks)

        # ---------------- Output ----------------
        self.out_norm = nn.GroupNorm(_groups(prev), prev)
        self.out_conv = nn.Conv2d(prev, 3, kernel_size=3, padding=1)

    # -- encode / decode helpers -------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_init(x)
        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_mid(h)
        h = self.to_latent(h)                       # (B, 8, 16, 16)
        mu, logvar = h.chunk(2, dim=1)              # (B, 4, 16, 16) each
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_init(z)
        h = self.dec_mid(h)
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return torch.tanh(h)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        # Clamp logvar to a safe range (matches common VAE training practice; avoids
        # NaNs under fp16 autocast while still letting the posterior collapse/expand).
        logvar = logvar.clamp(-30.0, 20.0)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return VAEOutput(recon=recon, mu=mu, logvar=logvar, z=z)


def vae_loss(out: VAEOutput, x: torch.Tensor, kl_weight: float = 1e-6) -> tuple[torch.Tensor, dict]:
    """Eq. 10 in Kingma & Welling (2013), scaled per assignment: MSE + lambda * KL.

    - recon: per-element MSE, averaged across all elements (B*C*H*W).
    - KL   : analytic closed form for isotropic Gaussian posterior vs N(0,I);
             per-element, averaged across all elements.
    """
    recon = F.mse_loss(out.recon, x, reduction="mean")
    kl = -0.5 * (1.0 + out.logvar - out.mu.pow(2) - out.logvar.exp())
    kl = kl.mean()
    total = recon + kl_weight * kl
    return total, {"loss": total.detach(), "recon": recon.detach(), "kl": kl.detach()}
