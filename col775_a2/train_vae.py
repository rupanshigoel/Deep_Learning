"""Train the Convolutional VAE on Part_A CLEVR images (128x128).

Resumable checkpoint at `checkpoints/vae_last.pt` - identical plumbing to
`train_clip.py`.

Per epoch writes to `logs/vae_train.csv`:
    epoch, train_loss, train_recon, train_kl, val_loss, val_recon, val_kl,
    lr, epoch_time_min

Every `--sample-every` epochs writes a reconstruction grid to
`samples/vae_recon_ep{:03d}.png` (8 val images, original + reconstruction
stacked vertically) for qualitative inspection.

Usage:
    python -m col775_a2.train_vae
    python -m col775_a2.train_vae --epochs 100 --batch-size 64 --amp
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from col775_a2.data.ldm_dataset import ClevrImageOnly, IMG_SIZE
from col775_a2.models.vae import ConvVAE, vae_loss
from col775_a2.utils.scheduler import cosine_warmup_schedule


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_A"


def init_csv_log(path: Path, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv_row(path: Path, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)


def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, epoch, step, args, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_load(path: Path):
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[resume] failed to load {path}: {e}")
        return None


@torch.no_grad()
def evaluate_vae(model, loader, device, amp: bool, kl_weight: float) -> dict:
    """Run one pass over the val loader; return mean loss/recon/kl (scalars)."""
    model.eval()
    tot_loss = tot_recon = tot_kl = 0.0
    tot_n = 0
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(amp and device == "cuda")):
            out = model(imgs)
            _, parts = vae_loss(out, imgs, kl_weight=kl_weight)
        bsz = imgs.size(0)
        tot_loss += parts["loss"].item() * bsz
        tot_recon += parts["recon"].item() * bsz
        tot_kl += parts["kl"].item() * bsz
        tot_n += bsz
    model.train()
    n = max(1, tot_n)
    return {"loss": tot_loss / n, "recon": tot_recon / n, "kl": tot_kl / n, "n": tot_n}


@torch.no_grad()
def save_recon_grid(model, imgs: torch.Tensor, out_path: Path, device, amp: bool):
    """Save a 2-row PNG grid: top row originals, bottom row reconstructions.

    imgs: (N, 3, H, W) tensor in [-1, 1] (on CPU or device).
    """
    from PIL import Image
    import numpy as np
    model.eval()
    imgs = imgs.to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(amp and device == "cuda")):
        out = model(imgs)
    recon = out.recon.clamp(-1, 1).float()
    model.train()
    # Build grid: (2, N, 3, H, W) -> (3, 2*H, N*W)
    row_orig = imgs.clamp(-1, 1).float().cpu()
    row_recon = recon.cpu()
    N, C, H, W = row_orig.shape
    grid = torch.zeros(C, 2 * H, N * W)
    for i in range(N):
        grid[:, :H, i*W:(i+1)*W] = row_orig[i]
        grid[:, H:, i*W:(i+1)*W] = row_recon[i]
    # [-1, 1] -> uint8
    arr = ((grid + 1.0) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--captions-json", default=None,
                    help="override path to the train captions JSON (default: <data-root>/train/clevr_train_captions.json)")
    ap.add_argument("--val-captions-json", default=None,
                    help="override path to the val captions JSON (default: <data-root>/val/clevr_val_captions.json)")
    ap.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--kl-weight", type=float, default=1e-6)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1,
                    help="run val pass every N epochs (0 disables)")
    ap.add_argument("--sample-every", type=int, default=5,
                    help="save recon grid every N epochs (0 disables)")
    ap.add_argument("--num-val-samples", type=int, default=8,
                    help="how many val images to include in the recon grid")
    ap.add_argument("--sample-dir", default=str(Path(__file__).resolve().parent / "samples" / "vae_train"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "vae_train.csv"
    _CSV_FIELDS = ["epoch", "train_loss", "train_recon", "train_kl",
                   "val_loss", "val_recon", "val_kl",
                   "lr", "epoch_time_min"]
    init_csv_log(csv_path, _CSV_FIELDS)

    data_root = Path(args.data_root)
    train_json = Path(args.captions_json) if args.captions_json else data_root / "train" / "clevr_train_captions.json"
    train_imgs = data_root / "train" / "images"
    val_json = Path(args.val_captions_json) if args.val_captions_json else data_root / "val" / "clevr_val_captions.json"
    val_imgs = data_root / "val" / "images"
    assert train_json.is_file(), f"Missing {train_json}"
    assert train_imgs.is_dir(), f"Missing {train_imgs}"
    assert val_json.is_file(), f"Missing {val_json}"
    assert val_imgs.is_dir(), f"Missing {val_imgs}"

    dataset = ClevrImageOnly(
        captions_json=str(train_json),
        images_dir=str(train_imgs),
        img_size=args.img_size,
    )
    val_dataset = ClevrImageOnly(
        captions_json=str(val_json),
        images_dir=str(val_imgs),
        img_size=args.img_size,
    )
    print(f"[data] train={len(dataset)}  val={len(val_dataset)}  @ {args.img_size}x{args.img_size}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # Deterministic small batch used for the recon-grid visualisation (same indices every epoch).
    _sample_indices = list(range(min(args.num_val_samples, len(val_dataset))))
    sample_batch = torch.stack([val_dataset[i] for i in _sample_indices], dim=0) if _sample_indices else None
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    print(f"[data] steps/epoch={steps_per_epoch}  total_steps={total_steps}")

    model = ConvVAE(img_size=args.img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ConvVAE  params={n_params/1e6:.2f} M")

    # Split into decay / no-decay groups (LayerNorm / bias / GroupNorm scales = no decay).
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            nodecay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    last_ckpt = ckpt_dir / "vae_last.pt"
    start_epoch = 0; global_step = 0
    state = try_load(last_ckpt)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        global_step = state["step"]
        print(f"[resume] resuming from epoch {start_epoch} (step {global_step})")
    else:
        print("[resume] no checkpoint found, training from scratch")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        r_loss = r_recon = r_kl = 0.0
        r_n = 0
        for it, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(args.amp and device == "cuda")):
                out = model(imgs)
                loss, parts = vae_loss(out, imgs, kl_weight=args.kl_weight)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            bsz = imgs.size(0)
            r_loss += parts["loss"].item() * bsz
            r_recon += parts["recon"].item() * bsz
            r_kl += parts["kl"].item() * bsz
            r_n += bsz
            if (it + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"ep {epoch:3d}  it {it+1:5d}/{steps_per_epoch}  "
                      f"loss {r_loss/r_n:.5f}  recon {r_recon/r_n:.5f}  kl {r_kl/r_n:.2f}  "
                      f"lr {lr:.2e}")

        ep_dt_train = time.time() - epoch_t0
        avg = {"loss": r_loss / max(1, r_n), "recon": r_recon / max(1, r_n), "kl": r_kl / max(1, r_n)}
        cur_lr = scheduler.get_last_lr()[0]

        # ---- Val pass (no grad, no aug) ----
        val = {"loss": float("nan"), "recon": float("nan"), "kl": float("nan")}
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            val = evaluate_vae(model, val_loader, device, amp=args.amp, kl_weight=args.kl_weight)

        # ---- Optional reconstruction grid on a fixed val subset ----
        if args.sample_every > 0 and sample_batch is not None and (epoch + 1) % args.sample_every == 0:
            sample_path = Path(args.sample_dir) / f"vae_recon_ep{epoch:03d}.png"
            save_recon_grid(model, sample_batch, sample_path, device, amp=args.amp)
            print(f"[sample] wrote {sample_path}")

        ep_dt = time.time() - epoch_t0
        print(f"[epoch {epoch}] train=(loss={avg['loss']:.5f} recon={avg['recon']:.5f} kl={avg['kl']:.2f})  "
              f"val=(loss={val['loss']:.5f} recon={val['recon']:.5f} kl={val['kl']:.2f})  "
              f"lr={cur_lr:.2e}  time={ep_dt/60:.1f} min")
        append_csv_row(csv_path, {
            "epoch": epoch,
            "train_loss": round(avg["loss"], 6),
            "train_recon": round(avg["recon"], 6),
            "train_kl": round(avg["kl"], 4),
            "val_loss": round(val["loss"], 6) if val["loss"] == val["loss"] else "",
            "val_recon": round(val["recon"], 6) if val["recon"] == val["recon"] else "",
            "val_kl": round(val["kl"], 4) if val["kl"] == val["kl"] else "",
            "lr": round(cur_lr, 8),
            "epoch_time_min": round(ep_dt / 60, 2),
        })
        save_checkpoint(
            last_ckpt,
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch, step=global_step, args=args,
            extra={"avg_loss": avg["loss"], "val": val, "epoch_time_train_min": ep_dt_train / 60},
        )

    save_checkpoint(
        ckpt_dir / "vae_final.pt",
        model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, epoch=args.epochs - 1, step=global_step, args=args,
    )
    print("[done] VAE training complete.")


if __name__ == "__main__":
    main()
