"""One-off: encode all Part_A/train images through the trained VAE and compute
per-channel mean and std of the un-noised latents.

By default uses the posterior-mean mu. Pass `--use-sampled-z` to instead use
the reparameterised z = mu + sigma * eps (closer to a strict reading of "un-noised
VAE latents" per the assignment spec). With a fixed random seed for reproducibility.

Use `--out-name` to write to a different file (e.g. latent_stats_sampled.pt) so
the original stats remain available for backwards compatibility.

Writes `<ckpt-dir>/<out-name>` with fields:
    {"mean": (1, 4, 1, 1), "std": (1, 4, 1, 1), "n_images": int,
     "n_elements_per_channel": int, "source": "mu" | "sampled_z"}

Usage:
    python -m col775_a2.compute_latent_stats
    python -m col775_a2.compute_latent_stats --use-sampled-z --out-name latent_stats_sampled.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from col775_a2.data.ldm_dataset import ClevrImageOnly, IMG_SIZE
from col775_a2.models.vae import ConvVAE


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--captions-json", default=None, help="override path to the train captions JSON")
    ap.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    ap.add_argument("--vae-ckpt", default=None, help="Path to VAE checkpoint (default: <ckpt-dir>/vae_last.pt)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--use-sampled-z", action="store_true",
                    help="compute stats over reparameterised z = mu + sigma*eps "
                         "instead of the posterior mean mu (default: mu)")
    ap.add_argument("--out-name", default="latent_stats.pt",
                    help="filename to write under --ckpt-dir (default: latent_stats.pt)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for reparameterisation noise when --use-sampled-z")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir)
    vae_ckpt = Path(args.vae_ckpt) if args.vae_ckpt else ckpt_dir / "vae_last.pt"
    assert vae_ckpt.is_file(), f"VAE checkpoint missing: {vae_ckpt}"

    # ---- Load VAE ----
    vae = ConvVAE(img_size=args.img_size).to(device)
    state = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(state["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ---- Data ----
    data_root = Path(args.data_root)
    train_json = Path(args.captions_json) if args.captions_json else data_root / "train" / "clevr_train_captions.json"
    dataset = ClevrImageOnly(
        captions_json=str(train_json),
        images_dir=str(data_root / "train" / "images"),
        img_size=args.img_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print(f"[data] {len(dataset)} images across {len(loader)} batches")

    # ---- Online mean/var via Welford-like sufficient statistics ----
    # We accumulate sum and sum_sq over all spatial positions. Latent is (C=4, H=16, W=16);
    # per-channel statistics are computed across B*H*W.
    C = vae.LATENT_CH
    sum_c = torch.zeros(C, dtype=torch.float64, device=device)
    sumsq_c = torch.zeros(C, dtype=torch.float64, device=device)
    count = 0  # total number of (B*H*W) scalar elements per channel

    if args.use_sampled_z:
        # Seed a CUDA generator so the reparameterisation noise is reproducible.
        gen = torch.Generator(device=device).manual_seed(args.seed)
        print(f"[stats] sampling z = mu + sigma*eps (seed={args.seed})")
    else:
        gen = None
        print("[stats] using posterior mean mu")

    with torch.no_grad():
        for i, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            mu, logvar = vae.encode(imgs)          # (B, 4, 16, 16) each
            if args.use_sampled_z:
                logvar = logvar.clamp(-30.0, 20.0)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(std.shape, device=device, generator=gen)
                lat = mu + std * eps
            else:
                lat = mu
            lat64 = lat.to(torch.float64)
            sum_c += lat64.sum(dim=(0, 2, 3))
            sumsq_c += (lat64 ** 2).sum(dim=(0, 2, 3))
            count += lat.shape[0] * lat.shape[2] * lat.shape[3]
            if (i + 1) % 50 == 0:
                print(f"[stats] {i+1}/{len(loader)} batches processed")

    mean = sum_c / count
    var = sumsq_c / count - mean ** 2
    std = var.clamp_min(1e-8).sqrt()
    mean = mean.view(1, C, 1, 1).float().cpu()
    std = std.view(1, C, 1, 1).float().cpu()

    out_path = ckpt_dir / args.out_name
    torch.save({
        "mean": mean,
        "std": std,
        "n_images": len(dataset),
        "n_elements_per_channel": count,
        "source": "sampled_z" if args.use_sampled_z else "mu",
    }, out_path)
    print(f"[done] wrote {out_path}  (source={'sampled_z' if args.use_sampled_z else 'mu'})")
    print(f"  per-channel mean : {mean.flatten().tolist()}")
    print(f"  per-channel std  : {std.flatten().tolist()}")


if __name__ == "__main__":
    main()
