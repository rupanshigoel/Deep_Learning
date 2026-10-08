"""Vision Transformer (ViT-S/16) built from scratch.

Follows the standard ViT design (Dosovitskiy et al. 2021). Uses
`nn.MultiheadAttention` for the attention op (allowed by the assignment)
but implements the full transformer block, patch embed, class token,
and positional embedding manually. Fully-abstracted `nn.Transformer` /
`nn.TransformerEncoderLayer` are deliberately NOT used.

Spec (from assignment):
    depth=12, heads=6, dim=384, mlp_dim=1536, patch=16, img=224,
    output projection to `proj_dim` (default 512) on the [CLS] token.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image -> sequence of patch embeddings via a strided conv."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, dim: int = 384):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: MHSA + MLP with residuals."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT backbone. Returns a dict of useful tensors.

    forward(x) -> {
        'cls':          (B, D)      final-LN [CLS] token,
        'patch_tokens': (B, N, D)   final-LN patch tokens (no CLS),
        'tokens':       (B, N+1, D) final-LN all tokens,
        'proj':         (B, proj_dim) projected CLS (for CLIP image embedding).
    }
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        proj_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_dim, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        # Projection head to shared embedding space (used for CLIP image side).
        self.proj = nn.Linear(dim, proj_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _interpolate_pos_embed(self, N_new: int, H_new: int, W_new: int) -> torch.Tensor:
        """Bicubic-interpolate positional embedding to match a non-default
        input resolution. Used for DINO local crops (e.g. 96x96)."""
        if N_new == self.patch_embed.num_patches:
            return self.pos_embed
        cls_pe = self.pos_embed[:, :1]
        patch_pe = self.pos_embed[:, 1:]
        dim = patch_pe.size(-1)
        orig = int(self.patch_embed.num_patches ** 0.5)
        patch_pe = patch_pe.reshape(1, orig, orig, dim).permute(0, 3, 1, 2)
        patch_pe = torch.nn.functional.interpolate(
            patch_pe, size=(H_new, W_new), mode="bicubic", align_corners=False
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, H_new * W_new, dim)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        H_in, W_in = x.shape[-2], x.shape[-1]
        x = self.patch_embed(x)                                   # (B, N, D)
        N = x.size(1)
        H = H_in // self.patch_embed.proj.stride[0]
        W = W_in // self.patch_embed.proj.stride[1]
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, D)
        x = torch.cat([cls, x], dim=1) + self._interpolate_pos_embed(N, H, W)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                                       # (B, N+1, D)

    def forward(self, x: torch.Tensor) -> dict:
        tokens = self.forward_features(x)
        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return {
            "cls": cls,
            "patch_tokens": patch_tokens,
            "tokens": tokens,
            "proj": self.proj(cls),
        }
