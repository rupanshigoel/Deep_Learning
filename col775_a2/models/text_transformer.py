"""Text transformer for the CLIP text encoder.

Architecture (per assignment):
    6 layers, 6 heads, hidden dim 384, MLP dim 1536, causal mask,
    learned token + positional embeddings. Final hidden state of the
    [EOS] token is projected to the shared 512-d embedding space.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .vit import TransformerBlock


class TextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 64,
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_dim: int = 1536,
        proj_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, mlp_dim, dropout=dropout) for _ in range(depth)]
        )
        self.ln_final = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, proj_dim, bias=False)
        self._init_weights()
        # Causal mask: (L, L), upper triangle = -inf so tokens only attend to past.
        mask = torch.full((context_length, context_length), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _init_weights(self):
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, eot_idx: torch.Tensor) -> torch.Tensor:
        """
        tokens:  (B, L) long — padded with 0s.
        eot_idx: (B,)   long — position of the [EOS] token per sample.
        returns: (B, proj_dim) projected text embedding.
        """
        B, L = tokens.shape
        x = self.token_embed(tokens) + self.pos_embed[:, :L]
        attn_mask = self.causal_mask[:L, :L]
        for blk in self.blocks:
            # Feed per-block causal mask through nn.MultiheadAttention.
            # Replicate the block forward but with attn_mask.
            y = blk.norm1(x)
            a, _ = blk.attn(y, y, y, attn_mask=attn_mask, need_weights=False)
            x = x + a
            x = x + blk.mlp(blk.norm2(x))
        x = self.ln_final(x)
        # Gather the hidden state at the EOS position for each sample.
        eot = x[torch.arange(B, device=x.device), eot_idx]        # (B, D)
        return self.proj(eot)
