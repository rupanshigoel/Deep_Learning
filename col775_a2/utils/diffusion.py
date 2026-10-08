"""Gaussian diffusion utilities for the Part C LDM.

Implements:
    * Cosine noise schedule (Nichol & Dhariwal 2021).
    * `q_sample` closed-form forward diffusion step.
    * Ancestral DDPM sampler (Ho et al. 2020, Alg. 2) with Classifier-Free
      Guidance (Ho & Salimans 2022, Eq. 6) at every timestep.

The schedule stores precomputed tensors as registered buffers on a module so
they follow the training device automatically (and get checkpointed if the
user saves the schedule).

Conventions:
    timesteps `t` are integer indices in [0, T-1]. t=0 is the least-noisy step.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal.

    alpha_bar(t/T) = cos((t/T + s)/(1 + s) * pi/2)^2
    beta_t = clip(1 - alpha_bar_t / alpha_bar_{t-1}, 0, 0.999)
    """
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(0.0, 0.999).float()


class GaussianDiffusion(nn.Module):
    """Container for schedule tensors + forward/reverse helpers.

    Buffer shapes:
        betas, alphas, alphas_bar: (T,)
    All derived quantities are recomputed from these three on load.
    """

    def __init__(self, num_timesteps: int = 500, schedule: str = "cosine"):
        super().__init__()
        assert schedule == "cosine", "only cosine schedule supported (spec)."
        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.T = num_timesteps

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())
        self.register_buffer("one_over_sqrt_alphas", (1.0 / alphas.sqrt()))
        # alphas_bar_{t-1} with alphas_bar_{-1} := 1 so variance at t=0 is zero.
        alphas_bar_prev = torch.cat([alphas_bar.new_ones(1), alphas_bar[:-1]])
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)
        # Sampling variance = beta_tilde_t (the "FixedSmall" choice in Ho 2020;
        # equivalent to the true posterior variance under the q(z_{t-1}|z_t,z_0)
        # parameterisation). The other valid choice is beta_t ("FixedLarge").
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))

    # ------------------------------------------------------------------ forward
    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * eps"""
        sqrt_ab = _gather(self.sqrt_alphas_bar, t, z0.dim())
        sqrt_1mab = _gather(self.sqrt_one_minus_alphas_bar, t, z0.dim())
        # Cast noise to z0's dtype so this is safe under autocast (fp16/bf16).
        noise = noise.to(z0.dtype)
        return sqrt_ab * z0 + sqrt_1mab * noise

    # ------------------------------------------------------------------ reverse
    @torch.no_grad()
    def p_sample_loop(
        self,
        model,                                   # ConditionalUNet
        shape: tuple,
        *,
        cond_context: torch.Tensor,              # (B, S, D)
        null_context: torch.Tensor,              # (1, S, D) or (B, S, D)
        guidance_scale: float = 4.0,
        device: torch.device | str = "cuda",
        progress: bool = False,
    ) -> torch.Tensor:
        """Full DDPM ancestral sampling with CFG. Returns (B, C, H, W)."""
        B = shape[0]
        if null_context.shape[0] == 1:
            null_context = null_context.expand(B, -1, -1)
        z = torch.randn(shape, device=device)
        iterator = reversed(range(self.T))
        if progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="ddpm")
        for t_idx in iterator:
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            eps_cond = model(z, t, cond_context)
            eps_uncond = model(z, t, null_context)
            eps_hat = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

            alpha_t = _gather(self.alphas, t, z.dim())
            alpha_bar_t = _gather(self.alphas_bar, t, z.dim())
            beta_t = _gather(self.betas, t, z.dim())

            # mean of p(z_{t-1} | z_t) under the predicted-noise parameterisation.
            mean = (z - beta_t / (1 - alpha_bar_t).sqrt() * eps_hat) / alpha_t.sqrt()

            if t_idx > 0:
                var = _gather(self.posterior_variance, t, z.dim())
                z = mean + var.sqrt() * torch.randn_like(z)
            else:
                z = mean
        return z


def _gather(buf: torch.Tensor, t: torch.Tensor, n_dim: int) -> torch.Tensor:
    """Gather buf[t] and broadcast to match tensor of rank n_dim."""
    val = buf.gather(0, t)
    return val.view(-1, *([1] * (n_dim - 1)))


def sample_timesteps(batch_size: int, T: int, device: torch.device) -> torch.Tensor:
    """Uniform random integer timesteps in [0, T-1]."""
    return torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
