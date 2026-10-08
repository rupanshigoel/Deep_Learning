"""Text-to-image sampling for the Part C LDM. Also computes FID against
the ground-truth Part_A/val images.

Pipeline:
    1. Load trained VAE, trained U-Net, latent stats, frozen CLIP text encoder.
    2. For each (filename, caption) in Part_A/val/clevr_val_captions.json:
         a. Encode caption -> context (B, 77, 512).
         b. DDPM sample z_T -> z_0 with CFG scale=s.
         c. De-normalize z = z * std + mean, decode via VAE, save PNG.
    3. (Optional) Compute clean-fid between (--real-dir) and (--out-dir).

Usage examples:
    python -m col775_a2.sample_ldm --num-samples 16 --out-dir samples/debug
    python -m col775_a2.sample_ldm --full --batch-size 16 --fid
"""
from __future__ import annotations
import argparse
import math
import shutil
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from PIL import Image

from col775_a2.data.ldm_dataset import ClevrCaptionOnly, caption_only_collate, IMG_SIZE
from col775_a2.models.unet import ConditionalUNet
from col775_a2.models.vae import ConvVAE
from col775_a2.models.text_encoder import FrozenCLIPTextEncoder
from col775_a2.utils.diffusion import GaussianDiffusion


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_A"


def denormalize_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 3, H, W) in [-1, 1] (Tanh output) -> uint8 in [0, 255]."""
    x = (x.clamp(-1, 1) + 1.0) * 127.5
    return x.to(torch.uint8)


def save_image_tensor(img: torch.Tensor, path: Path):
    """img: (3, H, W) uint8."""
    arr = img.permute(1, 2, 0).contiguous().cpu().numpy()
    Image.fromarray(arr).save(path)


def _build_real_dir_for_fid(val_imgs_dir: Path, filenames: List[str], dst_dir: Path, img_size: int):
    """Write 128x128 resized copies of the ground-truth val images to dst_dir
    using the same (Resize->CenterCrop) transform the generative model targets.
    This ensures FID compares like-for-like distributions (both 128x128).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
    ])
    for fn in filenames:
        dst = dst_dir / fn
        if dst.is_file():
            continue
        img = Image.open(val_imgs_dir / fn).convert("RGB")
        tfm(img).save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--ldm-ckpt", default=None)
    ap.add_argument("--latent-stats", default=None)
    ap.add_argument("--clip-path", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "samples" / "ldm_val"))
    ap.add_argument("--real-dir", default=str(Path(__file__).resolve().parent / "samples" / "real_val_128"),
                    help="directory where 128x128 ground-truth val images are mirrored (created on demand)")
    ap.add_argument("--num-samples", type=int, default=16,
                    help="sample this many captions from the val set (ignored when --full)")
    ap.add_argument("--full", action="store_true", help="generate the full 10k val set")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=500)
    ap.add_argument("--guidance", type=float, default=4.0, help="CFG scale s")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--text-seq-len", type=int, default=77)
    ap.add_argument("--text-dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fid", action="store_true", help="compute clean-fid after sampling")
    ap.add_argument("--progress", action="store_true", help="tqdm over diffusion steps")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir)
    vae_ckpt = Path(args.vae_ckpt) if args.vae_ckpt else ckpt_dir / "vae_last.pt"
    ldm_ckpt = Path(args.ldm_ckpt) if args.ldm_ckpt else ckpt_dir / "ldm_last.pt"
    stats_path = Path(args.latent_stats) if args.latent_stats else ckpt_dir / "latent_stats.pt"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load VAE ----
    vae = ConvVAE(img_size=args.img_size).to(device).eval()
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=False)["model"])
    for p in vae.parameters():
        p.requires_grad = False

    # ---- Load latent stats ----
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    lat_mean = stats["mean"].to(device)
    lat_std = stats["std"].to(device)

    # ---- Load LDM ----
    ldm_state = torch.load(ldm_ckpt, map_location="cpu", weights_only=False)
    model = ConditionalUNet(
        latent_ch=vae.LATENT_CH, base_ch=64, time_emb_dim=256,
        num_heads=args.num_heads, text_dim=args.text_dim,
    ).to(device).eval()
    model.load_state_dict(ldm_state["model"])
    for p in model.parameters():
        p.requires_grad = False
    null_context = ldm_state["null_context"].to(device)   # (1, 77, 512)
    print(f"[model] LDM loaded from {ldm_ckpt}  epoch={ldm_state.get('epoch')}")

    # ---- Text encoder ----
    text_encoder = FrozenCLIPTextEncoder(model_path=args.clip_path, device=device)

    # ---- Diffusion schedule ----
    diffusion = GaussianDiffusion(num_timesteps=args.timesteps).to(device)

    # ---- Val captions ----
    data_root = Path(args.data_root)
    val_json = data_root / "val" / "clevr_val_captions.json"
    val_imgs = data_root / "val" / "images"
    ds = ClevrCaptionOnly(captions_json=str(val_json), images_dir=str(val_imgs))
    if not args.full:
        # Deterministic subset for debugging.
        ds.samples = ds.samples[: args.num_samples]
    print(f"[data] sampling {len(ds)} captions -> {out_dir}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=caption_only_collate,
    )

    # ---- Sampling loop ----
    latent_shape = (None, vae.LATENT_CH, vae.LATENT_HW, vae.LATENT_HW)
    for bi, (fns, caps) in enumerate(loader):
        B = len(fns)
        with torch.no_grad():
            ctx = text_encoder.encode(caps)   # (B, 77, 512)
        z = diffusion.p_sample_loop(
            model,
            shape=(B, vae.LATENT_CH, vae.LATENT_HW, vae.LATENT_HW),
            cond_context=ctx,
            null_context=null_context,
            guidance_scale=args.guidance,
            device=device,
            progress=args.progress and bi == 0,   # progress bar only for 1st batch
        )
        # De-normalize and decode.
        z = z * lat_std + lat_mean
        with torch.no_grad():
            imgs = vae.decode(z)              # (B, 3, 128, 128) in [-1, 1]
        imgs_u8 = denormalize_to_uint8(imgs)
        for i, fn in enumerate(fns):
            save_image_tensor(imgs_u8[i], out_dir / fn)
        if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
            print(f"[sample] {(bi+1)*args.batch_size}/{len(ds)} done")

    # ---- Optional FID ----
    if args.fid:
        real_dir = Path(args.real_dir)
        print(f"[fid] preparing 128x128 reals -> {real_dir}")
        _build_real_dir_for_fid(val_imgs, [s["filename"] for s in ds.samples], real_dir, args.img_size)
        from col775_a2.utils.fid import compute_fid_between_folders
        score = compute_fid_between_folders(str(real_dir), str(out_dir), device=device)
        print(f"[fid] LDM vs. val  FID = {score:.3f}")
        with open(out_dir / "fid.txt", "w") as f:
            f.write(f"fid={score:.3f}\nreal={real_dir}\nfake={out_dir}\n"
                    f"num_samples={len(ds)}\nguidance={args.guidance}\ntimesteps={args.timesteps}\n")
    print("[done] sampling complete")


if __name__ == "__main__":
    main()
