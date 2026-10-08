"""Reconstruct Part_A/val images through the frozen VAE and compute FID.

Writes decoded PNGs (128x128, uint8) to `samples/vae_recon_val/` and
128x128-resized ground-truth PNGs to `samples/real_val_128/`, then (optionally)
runs clean-fid between the two folders.

Usage:
    python -m col775_a2.eval_vae_fid
    python -m col775_a2.eval_vae_fid --no-fid     # just reconstruct
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from col775_a2.data.ldm_dataset import ClevrImageCaption, ClevrCaptionOnly, image_caption_collate, IMG_SIZE
from col775_a2.models.vae import ConvVAE


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_A"


def denormalize_to_uint8(x: torch.Tensor) -> torch.Tensor:
    x = (x.clamp(-1, 1) + 1.0) * 127.5
    return x.to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "samples" / "vae_recon_val"))
    ap.add_argument("--real-dir", default=str(Path(__file__).resolve().parent / "samples" / "real_val_128"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--no-fid", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir)
    vae_ckpt = Path(args.vae_ckpt) if args.vae_ckpt else ckpt_dir / "vae_last.pt"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = Path(args.real_dir); real_dir.mkdir(parents=True, exist_ok=True)

    # ---- VAE ----
    vae = ConvVAE(img_size=args.img_size).to(device).eval()
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=False)["model"])
    for p in vae.parameters():
        p.requires_grad = False

    # ---- Data ----
    data_root = Path(args.data_root)
    val_json = data_root / "val" / "clevr_val_captions.json"
    val_imgs = data_root / "val" / "images"
    ds = ClevrImageCaption(
        captions_json=str(val_json),
        images_dir=str(val_imgs),
        img_size=args.img_size,
    )
    # we only need filenames for output; keep the order from the JSON
    fnames = [s["filename"] for s in ds.samples]
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=image_caption_collate,
    )
    print(f"[data] reconstructing {len(ds)} val images @ {args.img_size}")

    # Pre-compute 128x128 ground-truth PNGs for FID comparison.
    tfm = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
    ])
    for fn in fnames:
        dst = real_dir / fn
        if not dst.is_file():
            img = Image.open(val_imgs / fn).convert("RGB")
            tfm(img).save(dst)

    # ---- Reconstruct ----
    idx = 0
    with torch.no_grad():
        for imgs, _caps in loader:
            imgs = imgs.to(device, non_blocking=True)
            out = vae(imgs)
            u8 = denormalize_to_uint8(out.recon)
            for i in range(u8.shape[0]):
                fn = fnames[idx + i]
                arr = u8[i].permute(1, 2, 0).contiguous().cpu().numpy()
                Image.fromarray(arr).save(out_dir / fn)
            idx += u8.shape[0]
            if idx % (args.batch_size * 20) == 0 or idx == len(ds):
                print(f"[recon] {idx}/{len(ds)}")

    # ---- FID ----
    if not args.no_fid:
        from col775_a2.utils.fid import compute_fid_between_folders
        score = compute_fid_between_folders(str(real_dir), str(out_dir), device=device)
        print(f"[fid] VAE recon vs. val  FID = {score:.3f}")
        with open(out_dir / "fid.txt", "w") as f:
            f.write(f"fid={score:.3f}\nreal={real_dir}\nfake={out_dir}\nnum_samples={len(ds)}\n")
    print("[done] VAE FID evaluation complete")


if __name__ == "__main__":
    main()
