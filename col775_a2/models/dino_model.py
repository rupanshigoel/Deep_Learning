"""DINO: student + teacher ViTs with projection heads.

Head design (Caron et al. 2021, section 3 + Appendix C):
    backbone (384) -> 3-layer MLP with GELU -> L2-normalize ->
    weight-normed linear to `out_dim` (4096 per assignment).
    The last linear has no bias. We freeze its weight-norm *magnitude*
    (`weight_g`) to 1 for the whole training, per the paper; that turns
    it into a cosine classifier with learnable direction only.
"""
from __future__ import annotations
import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import VisionTransformer


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 4096,
        n_layers: int = 3,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINOWrapper(nn.Module):
    """Encoder + head. Forward accepts either a single tensor or a list of
    crop-tensors (possibly with different resolutions), batched by resolution
    to avoid many tiny forwards. Returns concatenated logits in input order.
    """

    def __init__(self, backbone: VisionTransformer, head: DINOHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def _encode(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.backbone(imgs)["cls"]

    def forward(self, crops):
        if isinstance(crops, torch.Tensor):
            return self.head(self._encode(crops))

        # Group crops by spatial size to batch equal-shape tensors.
        sizes = [c.shape[-1] for c in crops]
        outs = [None] * len(crops)
        unique_sizes = sorted(set(sizes))
        for s in unique_sizes:
            idxs = [i for i, sz in enumerate(sizes) if sz == s]
            batch = torch.cat([crops[i] for i in idxs], dim=0)
            feats = self._encode(batch)                    # (sum_B, D)
            logits = self.head(feats)                      # (sum_B, K)
            # Split back per original crop tensor.
            splits = torch.split(logits, [crops[i].size(0) for i in idxs], dim=0)
            for i, out in zip(idxs, splits):
                outs[i] = out
        return outs  # list of (B, K) tensors, one per crop


def build_dino(img_size: int = 224, patch_size: int = 16, out_dim: int = 4096) -> nn.ModuleDict:
    """Returns a dict with `student` and `teacher` DINOWrapper modules.

    The teacher is a deep copy of the student with grads disabled; both
    start with identical weights (the paper recommends this).
    """
    def _make():
        bb = VisionTransformer(
            img_size=img_size, patch_size=patch_size,
            dim=384, depth=12, num_heads=6, mlp_dim=1536,
            proj_dim=384,  # unused for DINO; head eats backbone `cls` directly.
        )
        head = DINOHead(in_dim=384, out_dim=out_dim)
        return DINOWrapper(bb, head)

    student = _make()
    teacher = _make()
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    return nn.ModuleDict({"student": student, "teacher": teacher})
