"""Learning-rate and DINO-specific (EMA momentum, teacher temp, weight decay) schedulers."""
from __future__ import annotations
import math
from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule_value(base: float, final: float, step: int, total_steps: int) -> float:
    """Cosine interpolation from `base` (at step 0) to `final` (at `total_steps`)."""
    if total_steps <= 0:
        return final
    progress = min(1.0, max(0.0, step / total_steps))
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


def linear_schedule_value(base: float, final: float, step: int, total_steps: int) -> float:
    """Linear interpolation from base at 0 to final at total_steps (clamped)."""
    if total_steps <= 0:
        return final
    t = min(1.0, max(0.0, step / total_steps))
    return base + t * (final - base)


def cosine_warmup_schedule(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.0):
    """Linear warmup from 0 -> base_lr over `num_warmup_steps`, then cosine
    decay to `min_lr_ratio * base_lr` at `num_training_steps`.
    """
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)
