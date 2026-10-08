"""Loss functions."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(torch.nn.Module):
    """DINO cross-entropy loss between centered teacher softmax and student
    log-softmax. Maintains a running `center` updated via EMA after each
    forward call. Sums loss over all valid (teacher_global, student_view)
    pairs where the student view is NOT the same crop as the teacher view.

    Args at construction:
        out_dim     : projection dim (4096)
        n_global    : number of teacher views (2)
        student_temp: tau_s (default 0.1)
        center_momentum: EMA momentum for center (default 0.9)
    `teacher_temp` must be passed per call because it's warmed up.
    """

    def __init__(self, out_dim: int, n_global: int = 2, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_global = n_global
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """teacher_output: (n_global * B, K) — concatenated teacher logits."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

    def forward(self, student_outputs, teacher_outputs, teacher_temp: float):
        """
        student_outputs: list of `n_crops` tensors (B, K), order = [g1, g2, l1, ..., l8].
        teacher_outputs: list of `n_global` tensors (B, K) for the global crops.
        Returns scalar loss. The caller should use the concatenation of
        teacher_outputs (pre-softmax, pre-centering) for `update_center`.
        """
        student_log_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_outputs]
        teacher_probs = [F.softmax((t - self.center) / teacher_temp, dim=-1).detach()
                         for t in teacher_outputs]
        total, n_pairs = 0.0, 0
        for ti, tprob in enumerate(teacher_probs):       # teacher global view index
            for si, slog in enumerate(student_log_probs):  # student view index (all crops)
                if si == ti:  # skip matching same-view pair
                    continue
                total = total + (-(tprob * slog).sum(dim=-1).mean())
                n_pairs += 1
        return total / n_pairs


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE: average of image->text and text->image cross-entropy.

    Assumes the contrastive batch has no duplicated positives (the dataset's
    DedupBatchSampler enforces this), so the target labels are simply
    arange(B).
    """
    B = logits_per_image.size(0)
    target = torch.arange(B, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, target)
    loss_t = F.cross_entropy(logits_per_text, target)
    return 0.5 * (loss_i + loss_t)
