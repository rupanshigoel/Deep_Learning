"""Conditional U-Net denoiser for Part C (Latent Diffusion Model).

Operates on VAE latents (B, 4, 16, 16). Conditioned on:
    * timestep embedding (sinusoidal -> 2-layer MLP) added inside every ResBlock
    * frozen CLIP text embeddings  (B, seq_len=77, text_dim=512) injected via
      cross-attention in the SpatialTransformer blocks.

Architecture exactly matches `docs/architecture_vae_ldm.html`:

    DownStage 0 : init conv (4->64)  + 2 ResBlock(time)                              + stride-2 down
    DownStage 1 :                      2 ResBlock(time) + SpatialTransformer          + stride-2 down
    DownStage 2 :                      2 ResBlock(time) + SpatialTransformer          + (no down)

    MidStage    : ResBlock(time) + SpatialTransformer + ResBlock(time)

    UpStage 0   : concat skip2 + 2 ResBlock(time) + SpatialTransformer + up 2x + conv
    UpStage 1   : concat skip1 + 2 ResBlock(time) + SpatialTransformer + up 2x + conv
    UpStage 2   : concat skip0 + 2 ResBlock(time)                       + output conv (-> 4)

Channels : 64, 128, 256.   Attention only at 8x8 and 4x4 (spec).
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import _groups, Downsample, Upsample


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------
def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal embedding for integer diffusion timesteps.

    timesteps: (B,) int64 or float
    returns  : (B, dim)
    """
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedMLP(nn.Module):
    """sinusoidal(dim) -> Linear(dim, 4*dim) -> SiLU -> Linear(4*dim, emb_dim)."""

    def __init__(self, sin_dim: int = 256, emb_dim: int = 256):
        super().__init__()
        self.sin_dim = sin_dim
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, 4 * emb_dim),
            nn.SiLU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_timestep_embedding(timesteps, self.sin_dim))


# ---------------------------------------------------------------------------
# ResBlock with time embedding injection
# ---------------------------------------------------------------------------
class TimeResBlock(nn.Module):
    """Same as VAE ResBlock but with a FiLM-style additive time-embedding
    projected to `out_ch` inserted after the first conv.
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Spatial Transformer : LN -> SelfAttn -> + ; LN -> CrossAttn -> + ; LN -> FFN -> +
# ---------------------------------------------------------------------------
class SpatialTransformer(nn.Module):
    """Text-conditioned transformer block over flattened spatial tokens.

    Input feature map (B, C, H, W) is first projected (1x1 conv) into an
    `inner_dim`-dim token sequence (B, H*W, inner_dim), passes through one
    transformer block (self-attn -> cross-attn -> FFN) and is projected back
    to (B, C, H, W) with a residual connection to the input (standard
    Stable Diffusion style).

    We use `inner_dim == C` (no dim change) and `num_heads` = `heads`.
    """

    def __init__(
        self,
        in_ch: int,
        *,
        num_heads: int = 8,
        text_dim: int = 512,
        mlp_mult: int = 4,
    ):
        super().__init__()
        assert in_ch % num_heads == 0, f"channels {in_ch} not divisible by heads {num_heads}"
        self.norm_in = nn.GroupNorm(_groups(in_ch), in_ch)
        self.proj_in = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        # --- Self-attention sub-block ---
        self.ln_self = nn.LayerNorm(in_ch)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=in_ch, num_heads=num_heads, batch_first=True,
        )
        # --- Cross-attention sub-block (K,V from text) ---
        self.ln_cross = nn.LayerNorm(in_ch)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_ch, num_heads=num_heads,
            kdim=text_dim, vdim=text_dim, batch_first=True,
        )
        # --- Feed-forward ---
        self.ln_ff = nn.LayerNorm(in_ch)
        self.ff = nn.Sequential(
            nn.Linear(in_ch, mlp_mult * in_ch),
            nn.GELU(),
            nn.Linear(mlp_mult * in_ch, in_ch),
        )

        self.proj_out = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        # Zero-init the output proj so the transformer starts as an identity --
        # matches the Stable Diffusion convention and stabilises early training.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) ; context: (B, S, text_dim)."""
        B, C, H, W = x.shape
        residual = x
        h = self.proj_in(self.norm_in(x))
        tokens = h.flatten(2).transpose(1, 2)  # (B, HW, C)

        # Self-attention
        q = self.ln_self(tokens)
        a, _ = self.self_attn(q, q, q, need_weights=False)
        tokens = tokens + a

        # Cross-attention (queries from image tokens, keys/values from text)
        q = self.ln_cross(tokens)
        a, _ = self.cross_attn(q, context, context, need_weights=False)
        tokens = tokens + a

        # Feed-forward
        tokens = tokens + self.ff(self.ln_ff(tokens))

        h = tokens.transpose(1, 2).reshape(B, C, H, W)
        return residual + self.proj_out(h)


# ---------------------------------------------------------------------------
# Conditional U-Net
# ---------------------------------------------------------------------------
class ConditionalUNet(nn.Module):
    """Latent-diffusion denoiser. Predicts noise epsilon given (z_t, t, text).

    Input shapes:
        z_t     : (B, 4, 16, 16)
        t       : (B,) long
        context : (B, seq_len, text_dim) = frozen CLIP text embeddings
    Output:
        eps_hat : (B, 4, 16, 16)
    """

    def __init__(
        self,
        latent_ch: int = 4,
        base_ch: int = 64,
        time_emb_dim: int = 256,
        num_heads: int = 8,
        text_dim: int = 512,
    ):
        super().__init__()
        c0, c1, c2 = base_ch, base_ch * 2, base_ch * 4  # 64, 128, 256
        self.time_embed = TimeEmbedMLP(sin_dim=time_emb_dim, emb_dim=time_emb_dim)

        # ---------------- Down 0  (16x16, no attn) ----------------
        self.d0_init = nn.Conv2d(latent_ch, c0, kernel_size=3, padding=1)
        self.d0_res = nn.ModuleList([
            TimeResBlock(c0, c0, time_emb_dim),
            TimeResBlock(c0, c0, time_emb_dim),
        ])
        self.d0_down = Downsample(c0)

        # ---------------- Down 1  (8x8, attn) ----------------
        self.d1_res = nn.ModuleList([
            TimeResBlock(c0, c1, time_emb_dim),
            TimeResBlock(c1, c1, time_emb_dim),
        ])
        self.d1_attn = SpatialTransformer(c1, num_heads=num_heads, text_dim=text_dim)
        self.d1_down = Downsample(c1)

        # ---------------- Down 2  (4x4, attn, no down) ----------------
        self.d2_res = nn.ModuleList([
            TimeResBlock(c1, c2, time_emb_dim),
            TimeResBlock(c2, c2, time_emb_dim),
        ])
        self.d2_attn = SpatialTransformer(c2, num_heads=num_heads, text_dim=text_dim)

        # ---------------- Mid  (4x4, attn) ----------------
        self.mid_res1 = TimeResBlock(c2, c2, time_emb_dim)
        self.mid_attn = SpatialTransformer(c2, num_heads=num_heads, text_dim=text_dim)
        self.mid_res2 = TimeResBlock(c2, c2, time_emb_dim)

        # ---------------- Up 0  (4x4 -> 8x8, attn) ----------------
        # Concat skip2 (c2) -> 2 ResBlock -> SpatialTransformer -> Upsample
        self.u0_res = nn.ModuleList([
            TimeResBlock(c2 + c2, c2, time_emb_dim),
            TimeResBlock(c2, c2, time_emb_dim),
        ])
        self.u0_attn = SpatialTransformer(c2, num_heads=num_heads, text_dim=text_dim)
        self.u0_up = Upsample(c2)

        # ---------------- Up 1  (8x8 -> 16x16, attn) ----------------
        # Input channels = c2 (from u0 upsampled) + c1 (skip1) after concat
        self.u1_res = nn.ModuleList([
            TimeResBlock(c2 + c1, c1, time_emb_dim),
            TimeResBlock(c1, c1, time_emb_dim),
        ])
        self.u1_attn = SpatialTransformer(c1, num_heads=num_heads, text_dim=text_dim)
        self.u1_up = Upsample(c1)

        # ---------------- Up 2  (16x16, no attn, final) ----------------
        # Input channels = c1 (from u1 upsampled) + c0 (skip0) after concat
        self.u2_res = nn.ModuleList([
            TimeResBlock(c1 + c0, c0, time_emb_dim),
            TimeResBlock(c0, c0, time_emb_dim),
        ])
        self.out_norm = nn.GroupNorm(_groups(c0), c0)
        self.out_conv = nn.Conv2d(c0, latent_ch, kernel_size=3, padding=1)
        # Zero-init the final layer so the model starts by predicting zero noise
        # (common DDPM trick for stable early training).
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, zt: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        # Down 0
        h = self.d0_init(zt)
        for blk in self.d0_res:
            h = blk(h, t_emb)
        skip0 = h                                   # (B, 64, 16, 16)
        h = self.d0_down(h)                         # (B, 64, 8, 8)

        # Down 1
        for blk in self.d1_res:
            h = blk(h, t_emb)
        h = self.d1_attn(h, context)
        skip1 = h                                   # (B, 128, 8, 8)
        h = self.d1_down(h)                         # (B, 128, 4, 4)

        # Down 2  (no downsample)
        for blk in self.d2_res:
            h = blk(h, t_emb)
        h = self.d2_attn(h, context)
        skip2 = h                                   # (B, 256, 4, 4)

        # Mid
        h = self.mid_res1(skip2, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, t_emb)

        # Up 0  (concat skip2, then upsample to 8)
        h = torch.cat([h, skip2], dim=1)
        for blk in self.u0_res:
            h = blk(h, t_emb)
        h = self.u0_attn(h, context)
        h = self.u0_up(h)                           # (B, 256, 8, 8)

        # Up 1  (concat skip1, then upsample to 16)
        h = torch.cat([h, skip1], dim=1)
        for blk in self.u1_res:
            h = blk(h, t_emb)
        h = self.u1_attn(h, context)
        h = self.u1_up(h)                           # (B, 128, 16, 16)

        # Up 2  (concat skip0, no attn)
        h = torch.cat([h, skip0], dim=1)
        for blk in self.u2_res:
            h = blk(h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
