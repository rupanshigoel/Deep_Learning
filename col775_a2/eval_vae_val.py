"""One-shot VAE val pass on the existing `vae_last.pt` checkpoint.

Computes (val_loss, val_recon, val_kl) over the full Part_A/val split and:
  1. Prints the numbers,
  2. Optionally backfills the LAST row of `<log-dir>/vae_train.csv`
     (val_loss / val_recon / val_kl columns) with these numbers.

This avoids re-training the VAE just to populate val columns. The VAE training
loop did not track val per epoch, so we get one final-epoch val tuple — enough
to demonstrate generalisation in the report (no overfitting check works fine
with one number; a curve isn't strictly required).

Usage:
    python -m col775_a2.eval_vae_val
    python -m col775_a2.eval_vae_val --backfill-csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from col775_a2.data.ldm_dataset import ClevrImageOnly, IMG_SIZE
from col775_a2.models.vae import ConvVAE, vae_loss


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_A"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    ap.add_argument("--csv-name", default="vae_train.csv")
    ap.add_argument("--backfill-csv", action="store_true",
                    help="overwrite the val_loss/val_recon/val_kl columns of "
                         "the LAST row of <log-dir>/<csv-name> with these numbers")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--kl-weight", type=float, default=1e-6)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir)
    vae_ckpt = Path(args.vae_ckpt) if args.vae_ckpt else ckpt_dir / "vae_last.pt"
    assert vae_ckpt.is_file(), f"VAE checkpoint missing: {vae_ckpt}"

    # ---- Model ----
    vae = ConvVAE(img_size=args.img_size).to(device).eval()
    state = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(state["model"])
    print(f"[ckpt] loaded {vae_ckpt}  epoch={state.get('epoch')}")

    # ---- Val loader ----
    data_root = Path(args.data_root)
    val_json = data_root / "val" / "clevr_val_captions.json"
    val_imgs = data_root / "val" / "images"
    ds = ClevrImageOnly(captions_json=str(val_json), images_dir=str(val_imgs),
                        img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"[data] val images: {len(ds)}")

    # ---- Run ----
    tot_loss = tot_recon = tot_kl = 0.0
    tot_n = 0
    for bi, imgs in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        out = vae(imgs)
        _, parts = vae_loss(out, imgs, kl_weight=args.kl_weight)
        bsz = imgs.size(0)
        tot_loss += parts["loss"].item() * bsz
        tot_recon += parts["recon"].item() * bsz
        tot_kl += parts["kl"].item() * bsz
        tot_n += bsz
        if (bi + 1) % 25 == 0:
            print(f"  {(bi+1)*args.batch_size}/{len(ds)}")
    n = max(1, tot_n)
    val_loss = tot_loss / n
    val_recon = tot_recon / n
    val_kl = tot_kl / n
    print(f"[result] val_loss = {val_loss:.6f}")
    print(f"[result] val_recon = {val_recon:.6f}")
    print(f"[result] val_kl    = {val_kl:.4f}")

    if args.backfill_csv:
        csv_path = Path(args.log_dir) / args.csv_name
        if not csv_path.is_file():
            print(f"[warn] {csv_path} not found, skipping backfill")
            return
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            fieldnames = list(rows[0].keys())
        # If CSV is the older 6-col schema, migrate it in-place to the 9-col
        # schema (`recon`->`train_recon`, `kl`->`train_kl`, add empty val cols).
        rename_map = {"recon": "train_recon", "kl": "train_kl"}
        if any(k in fieldnames for k in rename_map):
            new_fields = []
            for f in fieldnames:
                new_fields.append(rename_map.get(f, f))
            fieldnames = new_fields
            rows = [{rename_map.get(k, k): v for k, v in r.items()} for r in rows]
        for col in ("val_loss", "val_recon", "val_kl"):
            if col not in fieldnames:
                # insert val cols right after train_kl (or at end if not found)
                idx = fieldnames.index("train_kl") + 1 if "train_kl" in fieldnames else len(fieldnames)
                fieldnames.insert(idx, col)
                for r in rows:
                    r.setdefault(col, "")
        # backfill last row
        rows[-1]["val_loss"] = round(val_loss, 6)
        rows[-1]["val_recon"] = round(val_recon, 6)
        rows[-1]["val_kl"] = round(val_kl, 4)
        tmp = csv_path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
        tmp.replace(csv_path)
        print(f"[csv] backfilled last row of {csv_path}: epoch={rows[-1]['epoch']}")


if __name__ == "__main__":
    main()
