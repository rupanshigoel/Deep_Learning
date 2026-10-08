"""Generate the qualitative ablation figures for the Part C report.

Two independent panels (one HPC job covers both):

  1. DIVERSITY GRID
       Same set of N captions, M random seeds each.
       Output: samples_ablation/diversity_grid.png      (rows = seeds, cols = captions)

  2. CFG-SCALE ABLATION
       Same set of N captions, sweep guidance scale s in {0, 1, 2, 4, 7.5}.
       Output: samples_ablation/cfg_grid.png            (rows = guidance scale, cols = captions)

Also dumps individual PNGs (per row, per seed/scale) and a captions.txt next to
each grid so the report writer can quote the prompts verbatim.

Reuses the same checkpoints/latent_stats as sample_ldm.py:
    checkpoints/{vae_last.pt, ldm_last.pt, latent_stats.pt}

Usage on HPC:
    python -m col775_a2.sample_ablations \
        --data-root  $PBS_O_WORKDIR/A2_dataset/Part_A \
        --ckpt-dir   $PBS_O_WORKDIR/checkpoints \
        --out-dir    $PBS_O_WORKDIR/samples_ablation
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

from col775_a2.data.ldm_dataset import ClevrCaptionOnly, IMG_SIZE
from col775_a2.models.unet import ConditionalUNet
from col775_a2.models.vae import ConvVAE
from col775_a2.models.text_encoder import FrozenCLIPTextEncoder
from col775_a2.utils.diffusion import GaussianDiffusion


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_A"


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> uint8 [0,255]."""
    return ((x.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """img: (3, H, W) uint8."""
    return Image.fromarray(img.permute(1, 2, 0).contiguous().cpu().numpy())


def make_grid(images: list[list[Image.Image]], row_labels: list[str],
              col_labels: list[str] | None = None,
              cell: int = 128, pad: int = 8, label_w: int = 110, label_h: int = 22) -> Image.Image:
    """images[r][c] -> grid image with row labels (left) and optional col labels (top)."""
    from PIL import ImageDraw, ImageFont
    nr = len(images); nc = len(images[0]) if images else 0
    top_h = label_h if col_labels else 0
    W = label_w + nc * (cell + pad) - pad
    H = top_h + nr * (cell + pad) - pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
    if col_labels:
        for ci, lab in enumerate(col_labels):
            draw.text((label_w + ci * (cell + pad) + 4, 2), lab, fill="black", font=font)
    for ri, row in enumerate(images):
        y = top_h + ri * (cell + pad)
        draw.text((4, y + cell // 2 - 8), row_labels[ri], fill="black", font=font)
        for ci, im in enumerate(row):
            canvas.paste(im.resize((cell, cell)), (label_w + ci * (cell + pad), y))
    return canvas


@torch.no_grad()
def sample_one(model, vae, diffusion, ctx: torch.Tensor, null_ctx: torch.Tensor,
               *, guidance: float, lat_mean, lat_std, device, seed: int,
               img_size: int = 128) -> torch.Tensor:
    """Sample a batch of images. ctx: (B, S, D); null_ctx: (1 or B, S, D)."""
    torch.manual_seed(seed)
    B = ctx.shape[0]
    if null_ctx.shape[0] == 1:
        null_ctx_b = null_ctx.expand(B, -1, -1)
    else:
        null_ctx_b = null_ctx
    z = diffusion.p_sample_loop(
        model,
        shape=(B, 4, img_size // 8, img_size // 8),
        cond_context=ctx,
        null_context=null_ctx_b,
        guidance_scale=guidance,
        device=device,
        progress=False,
    )
    z = z * lat_std + lat_mean
    imgs = vae.decode(z)
    return to_uint8(imgs)   # (B, 3, 128, 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--ldm-ckpt", default=None)
    ap.add_argument("--latent-stats", default=None)
    ap.add_argument("--clip-path", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "samples_ablation"))
    ap.add_argument("--num-captions", type=int, default=4,
                    help="# captions used (= # cols of each grid)")
    ap.add_argument("--num-seeds", type=int, default=4,
                    help="# random seeds for the diversity grid")
    ap.add_argument("--cfg-scales", type=str, default="0,1,2,4,7.5",
                    help="comma-separated guidance scales for the CFG ablation")
    ap.add_argument("--diversity-cfg", type=float, default=4.0,
                    help="guidance used for the diversity grid")
    ap.add_argument("--timesteps", type=int, default=500)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--text-dim", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir)
    vae_ckpt = Path(args.vae_ckpt) if args.vae_ckpt else ckpt_dir / "vae_last.pt"
    ldm_ckpt = Path(args.ldm_ckpt) if args.ldm_ckpt else ckpt_dir / "ldm_last.pt"
    stats_path = Path(args.latent_stats) if args.latent_stats else ckpt_dir / "latent_stats.pt"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Models ----
    vae = ConvVAE(img_size=args.img_size).to(device).eval()
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=False)["model"])
    for p in vae.parameters():
        p.requires_grad = False

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    lat_mean = stats["mean"].to(device)
    lat_std = stats["std"].to(device)

    ldm_state = torch.load(ldm_ckpt, map_location="cpu", weights_only=False)
    model = ConditionalUNet(
        latent_ch=vae.LATENT_CH, base_ch=64, time_emb_dim=256,
        num_heads=args.num_heads, text_dim=args.text_dim,
    ).to(device).eval()
    model.load_state_dict(ldm_state["model"])
    for p in model.parameters():
        p.requires_grad = False
    null_context = ldm_state["null_context"].to(device)
    print(f"[model] LDM loaded epoch={ldm_state.get('epoch')}")

    text_encoder = FrozenCLIPTextEncoder(model_path=args.clip_path, device=device)
    diffusion = GaussianDiffusion(num_timesteps=args.timesteps).to(device)

    # ---- Pick captions (deterministic: first N from val) ----
    data_root = Path(args.data_root)
    ds = ClevrCaptionOnly(
        captions_json=str(data_root / "val" / "clevr_val_captions.json"),
        images_dir=str(data_root / "val" / "images"),
    )
    captions: List[str] = [ds.samples[i]["caption"] for i in range(args.num_captions)]
    fnames: List[str] = [ds.samples[i]["filename"] for i in range(args.num_captions)]
    with open(out_dir / "captions.txt", "w", encoding="utf-8") as f:
        for fn, c in zip(fnames, captions):
            f.write(f"{fn}\t{c}\n")
    print(f"[data] {len(captions)} captions selected")

    ctx = text_encoder.encode(captions)   # (N, 77, 512)

    # ---- (1) Diversity grid: rows = seeds, cols = captions ----
    print(f"[diversity] {args.num_seeds} seeds x {len(captions)} captions @ CFG s={args.diversity_cfg}")
    rows: list[list[Image.Image]] = []
    for s_idx in range(args.num_seeds):
        seed = 1000 + s_idx
        u8 = sample_one(model, vae, diffusion, ctx, null_context,
                        guidance=args.diversity_cfg, lat_mean=lat_mean, lat_std=lat_std,
                        device=device, seed=seed, img_size=args.img_size)
        # Save individual PNGs (handy for debugging / supplementary material)
        for ci in range(u8.shape[0]):
            (out_dir / "diversity").mkdir(parents=True, exist_ok=True)
            tensor_to_pil(u8[ci]).save(out_dir / "diversity" / f"seed{seed}_cap{ci}.png")
        rows.append([tensor_to_pil(u8[ci]) for ci in range(u8.shape[0])])
        print(f"  seed {seed} done")
    col_labels = [f"caption {i}" for i in range(len(captions))]
    row_labels = [f"seed {1000 + i}" for i in range(args.num_seeds)]
    grid = make_grid(rows, row_labels=row_labels, col_labels=col_labels)
    grid.save(out_dir / "diversity_grid.png")
    print(f"[wrote] {out_dir / 'diversity_grid.png'} ({grid.size})")

    # ---- (2) CFG ablation: rows = scale, cols = captions (one fixed seed) ----
    scales = [float(s) for s in args.cfg_scales.split(",") if s.strip()]
    print(f"[cfg] sweeping s in {scales}, fixed seed=2024")
    rows = []
    for s in scales:
        u8 = sample_one(model, vae, diffusion, ctx, null_context,
                        guidance=s, lat_mean=lat_mean, lat_std=lat_std,
                        device=device, seed=2024, img_size=args.img_size)
        for ci in range(u8.shape[0]):
            (out_dir / "cfg").mkdir(parents=True, exist_ok=True)
            tensor_to_pil(u8[ci]).save(out_dir / "cfg" / f"s{s}_cap{ci}.png")
        rows.append([tensor_to_pil(u8[ci]) for ci in range(u8.shape[0])])
        print(f"  s={s} done")
    row_labels = [f"s = {s}" for s in scales]
    grid = make_grid(rows, row_labels=row_labels, col_labels=col_labels)
    grid.save(out_dir / "cfg_grid.png")
    print(f"[wrote] {out_dir / 'cfg_grid.png'} ({grid.size})")

    print("[done] ablation sampling complete")


if __name__ == "__main__":
    main()
