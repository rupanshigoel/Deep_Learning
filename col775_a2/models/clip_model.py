"""CLIP wrapper: ViT image encoder + text transformer + learnable temperature."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import VisionTransformer
from .text_transformer import TextTransformer


class CLIPModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 64,
        embed_dim: int = 512,
        # Vision:
        img_size: int = 224,
        patch_size: int = 16,
        v_dim: int = 384,
        v_depth: int = 12,
        v_heads: int = 6,
        v_mlp: int = 1536,
        # Text:
        t_dim: int = 384,
        t_depth: int = 6,
        t_heads: int = 6,
        t_mlp: int = 1536,
    ):
        super().__init__()
        self.visual = VisionTransformer(
            img_size=img_size, patch_size=patch_size, dim=v_dim, depth=v_depth,
            num_heads=v_heads, mlp_dim=v_mlp, proj_dim=embed_dim,
        )
        self.text = TextTransformer(
            vocab_size=vocab_size, context_length=context_length,
            dim=t_dim, depth=t_depth, num_heads=t_heads, mlp_dim=t_mlp, proj_dim=embed_dim,
        )
        # Learnable temperature initialized to log(1/0.07), as per the paper.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)["proj"]

    def encode_text(self, tokens: torch.Tensor, eot_idx: torch.Tensor) -> torch.Tensor:
        return self.text(tokens, eot_idx)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, eot_idx: torch.Tensor):
        img = F.normalize(self.encode_image(images), dim=-1)
        txt = F.normalize(self.encode_text(tokens, eot_idx), dim=-1)
        # Clamp to prevent runaway temperature (as in the official CLIP code).
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logits_per_image = scale * img @ txt.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
