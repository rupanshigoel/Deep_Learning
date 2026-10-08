"""Train the conditional U-Net (latent diffusion model) on Part_A CLEVR.

Requires:
    * a trained VAE checkpoint  at `checkpoints/vae_last.pt`
    * latent statistics        at `checkpoints/latent_stats.pt`
    * a local copy of the CLIP text encoder (or an internet-capable env):
      `openai/clip-vit-base-patch32`. Set `HF_HOME` and `TRANSFORMERS_OFFLINE=1`
      when the HF cache is pre-seeded.

Checkpoint: `checkpoints/ldm_last.pt`, resumable, written after every epoch.
CSV log   : `logs/ldm_train.csv`
    epoch, train_loss, val_loss, lr, epoch_time_min

Every `--sample-every` epochs we also:
    * DDPM-sample a small fixed set of val captions with CFG, decode via VAE,
      write a grid to `samples/ldm_train/ldm_ep{:03d}.png` for qualitative
      monitoring. Uses fewer DDPM steps (`--sample-steps-preview`, default
      100) to keep the cost per milestone bounded.

Usage:
    python -m col775_a2.train_ldm
    python -m col775_a2.train_ldm --epochs 150 --batch-size 64
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from col775_a2.data.ldm_dataset import (
    ClevrImageCaption, image_caption_collate, IMG_SIZE,
)
from col775_a2.models.unet import ConditionalUNet
from col775_a2.models.vae import ConvVAE
from col775_a2.models.text_encoder import FrozenCLIPTextEncoder
from col775_a2.utils.diffusion import GaussianDiffusion, sample_timesteps
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


def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, epoch, step, args,
                    null_context: torch.Tensor, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "args": vars(args),
        "null_context": null_context.detach().cpu(),
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
def evaluate_ldm(model, vae, text_encoder, diffusion, val_loader, lat_mean, lat_std,
                  device, amp: bool, max_batches: int, seed: int = 1234) -> dict:
    """Compute eps-MSE on (a capped subset of) the val loader with a fixed
    noise/timestep draw (via seeded generator) so the number is stable
    across epochs.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    total = 0.0; n = 0
    for bi, (imgs, caps) in enumerate(val_loader):
        if max_batches and bi >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        B = imgs.size(0)
        mu, _ = vae.encode(imgs)
        z0 = (mu - lat_mean) / lat_std
        ctx = text_encoder.encode(caps)
        t = torch.randint(0, diffusion.T, (B,), device=device, dtype=torch.long, generator=gen)
        noise = torch.randn(z0.shape, device=device, generator=gen)
        zt = diffusion.q_sample(z0, t, noise)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(amp and device == "cuda")):
            eps_pred = model(zt, t, ctx)
        total += torch.nn.functional.mse_loss(eps_pred, noise, reduction="mean").item() * B
        n += B
    model.train()
    return {"loss": total / max(1, n), "n": n}


@torch.no_grad()
def save_sample_grid(model, vae, text_encoder, null_context, captions,
                     lat_mean, lat_std, out_path: Path, device, *,
                     guidance: float, steps: int, img_size: int = 128):
    """DDPM-sample `captions` with CFG, decode via VAE, tile into a 1-row PNG.

    Uses a truncated-schedule sampler to keep this cheap for the preview. We
    build a fresh GaussianDiffusion with `T=steps` (still cosine schedule).
    """
    from PIL import Image
    import numpy as np
    from col775_a2.utils.diffusion import GaussianDiffusion
    model.eval()
    B = len(captions)
    small_diff = GaussianDiffusion(num_timesteps=steps).to(device)
    ctx = text_encoder.encode(captions)
    null_exp = null_context.expand(B, -1, -1)
    z = small_diff.p_sample_loop(
        model,
        shape=(B, 4, img_size // 8, img_size // 8),
        cond_context=ctx,
        null_context=null_exp,
        guidance_scale=guidance,
        device=device,
        progress=False,
    )
    z = z * lat_std + lat_mean
    imgs = vae.decode(z).clamp(-1, 1).float().cpu()
    model.train()
    # Assemble into a single row.
    C, H, W = imgs.shape[1], imgs.shape[2], imgs.shape[3]
    grid = torch.zeros(C, H, B * W)
    for i in range(B):
        grid[:, :, i*W:(i+1)*W] = imgs[i]
    arr = ((grid + 1.0) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)
    # Also write the captions alongside for reference.
    with open(out_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        for i, c in enumerate(captions):
            f.write(f"{i}: {c}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--captions-json", default=None,
                    help="override path to the train captions JSON")
    ap.add_argument("--val-captions-json", default=None,
                    help="override path to the val captions JSON")
    ap.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    ap.add_argument("--vae-ckpt", default=None, help="path to trained VAE (default: <ckpt-dir>/vae_last.pt)")
    ap.add_argument("--latent-stats", default=None, help="path to latent_stats.pt (default: <ckpt-dir>/latent_stats.pt)")
    ap.add_argument("--clip-path", default="openai/clip-vit-base-patch32",
                    help="HF id or local path to CLIP text encoder")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=500)
    ap.add_argument("--null-prob", type=float, default=0.1,
                    help="probability of replacing text context with the null context per sample")
    ap.add_argument("--text-seq-len", type=int, default=77)
    ap.add_argument("--text-dim", type=int, default=512)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=1,
                    help="compute val eps-MSE every N epochs (0 disables)")
    ap.add_argument("--val-batches", type=int, default=50,
                    help="cap val to N batches to keep it cheap (0 = full val)")
    ap.add_argument("--sample-every", type=int, default=10,
                    help="DDPM-sample a small val set every N epochs (0 disables)")
    ap.add_argument("--num-sample-captions", type=int, default=8)
    ap.add_argument("--sample-steps-preview", type=int, default=100,
                    help="DDPM steps for the training-time preview sampler (fewer than full --timesteps for speed)")
    ap.add_argument("--sample-guidance", type=float, default=4.0)
    ap.add_argument("--sample-dir", default=str(Path(__file__).resolve().parent / "samples" / "ldm_train"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "ldm_train.csv"
    _CSV_FIELDS = ["epoch", "train_loss", "val_loss", "lr", "epoch_time_min"]
    init_csv_log(csv_path, _CSV_FIELDS)

    # ---- Data ----
    data_root = Path(args.data_root)
    train_json = Path(args.captions_json) if args.captions_json else data_root / "train" / "clevr_train_captions.json"
    train_imgs = data_root / "train" / "images"
    val_json = Path(args.val_captions_json) if args.val_captions_json else data_root / "val" / "clevr_val_captions.json"
    val_imgs = data_root / "val" / "images"
    assert train_json.is_file(), f"Missing {train_json}"
    assert train_imgs.is_dir(), f"Missing {train_imgs}"
    assert val_json.is_file(), f"Missing {val_json}"
    assert val_imgs.is_dir(), f"Missing {val_imgs}"

    dataset = ClevrImageCaption(
        captions_json=str(train_json),
        images_dir=str(train_imgs),
        img_size=args.img_size,
    )
    val_dataset = ClevrImageCaption(
        captions_json=str(val_json),
        images_dir=str(val_imgs),
        img_size=args.img_size,
    )
    print(f"[data] train={len(dataset)}  val={len(val_dataset)} image-caption pairs")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=image_caption_collate,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=image_caption_collate,
        persistent_workers=args.num_workers > 0,
    )
    # Deterministic fixed captions for the preview sampling.
    _si = list(range(min(args.num_sample_captions, len(val_dataset))))
    sample_captions = [val_dataset.samples[i]["caption"] for i in _si]
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    # ---- Frozen VAE + latent stats ----
    vae_ckpt = Path(args.vae_ckpt) if args.vae_ckpt else ckpt_dir / "vae_last.pt"
    stats_path = Path(args.latent_stats) if args.latent_stats else ckpt_dir / "latent_stats.pt"
    assert vae_ckpt.is_file(), f"VAE checkpoint missing: {vae_ckpt}"
    assert stats_path.is_file(), f"latent_stats.pt missing: {stats_path}"

    vae = ConvVAE(img_size=args.img_size).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu", weights_only=False)["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    lat_mean = stats["mean"].to(device)   # (1, 4, 1, 1)
    lat_std = stats["std"].to(device)     # (1, 4, 1, 1)
    print(f"[latent] mean={lat_mean.flatten().tolist()}  std={lat_std.flatten().tolist()}")

    # ---- Frozen CLIP text encoder ----
    text_encoder = FrozenCLIPTextEncoder(model_path=args.clip_path, device=device)
    assert text_encoder.text_dim == args.text_dim, \
        f"CLIP hidden size {text_encoder.text_dim} != --text-dim {args.text_dim}"
    print(f"[text] seq_len={text_encoder.seq_len}  dim={text_encoder.text_dim}")

    # ---- Diffusion schedule ----
    diffusion = GaussianDiffusion(num_timesteps=args.timesteps).to(device)

    # ---- Model ----
    model = ConditionalUNet(
        latent_ch=vae.LATENT_CH,
        base_ch=64, time_emb_dim=256,
        num_heads=args.num_heads, text_dim=args.text_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] U-Net  params={n_params/1e6:.2f} M")

    # ---- Learned null context (B=1, seq_len, text_dim) ----
    null_context = torch.nn.Parameter(
        torch.randn(1, args.text_seq_len, args.text_dim, device=device) * 0.02
    )

    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            nodecay.append(p)
        else:
            decay.append(p)
    # null_context is a separate param; treat as no-decay.
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
            {"params": [null_context], "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    last_ckpt = ckpt_dir / "ldm_last.pt"
    start_epoch = 0; global_step = 0
    state = try_load(last_ckpt)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        if "null_context" in state:
            with torch.no_grad():
                null_context.data.copy_(state["null_context"].to(device))
        start_epoch = state["epoch"] + 1
        global_step = state["step"]
        print(f"[resume] resuming from epoch {start_epoch} (step {global_step})")
    else:
        print("[resume] no checkpoint found, training from scratch")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        r_loss = 0.0; r_n = 0
        for it, (imgs, captions) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            B = imgs.size(0)

            # ---- Encode to normalized latent (no grad through VAE) ----
            with torch.no_grad():
                mu, _ = vae.encode(imgs)
                z0 = (mu - lat_mean) / lat_std

            # ---- Text context + CFG dropout ----
            with torch.no_grad():
                ctx = text_encoder.encode(captions)   # (B, 77, 512)
            # Per-sample null replacement with probability `null_prob`.
            if args.null_prob > 0.0:
                mask = (torch.rand(B, device=device) < args.null_prob).view(B, 1, 1)
                null_exp = null_context.expand(B, -1, -1)
                ctx = torch.where(mask, null_exp, ctx)

            # ---- Sample t, add noise ----
            t = sample_timesteps(B, args.timesteps, device)
            noise = torch.randn_like(z0)
            zt = diffusion.q_sample(z0, t, noise)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(args.amp and device == "cuda")):
                eps_pred = model(zt, t, ctx)
                loss = torch.nn.functional.mse_loss(eps_pred, noise)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + [null_context], args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            r_loss += loss.item() * B
            r_n += B
            if (it + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"ep {epoch:3d}  it {it+1:5d}/{steps_per_epoch}  "
                      f"loss {r_loss/r_n:.5f}  lr {lr:.2e}")

        ep_dt_train = time.time() - epoch_t0
        avg = r_loss / max(1, r_n)
        cur_lr = scheduler.get_last_lr()[0]

        # ---- Val pass (eps-MSE on a capped subset of the val set) ----
        val_loss = float("nan")
        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            val = evaluate_ldm(model, vae, text_encoder, diffusion, val_loader,
                               lat_mean, lat_std, device, amp=args.amp,
                               max_batches=args.val_batches)
            val_loss = val["loss"]

        # ---- Optional sample grid on fixed val captions ----
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0 and sample_captions:
            sample_path = Path(args.sample_dir) / f"ldm_ep{epoch:03d}.png"
            save_sample_grid(
                model, vae, text_encoder, null_context, sample_captions,
                lat_mean, lat_std, sample_path, device,
                guidance=args.sample_guidance, steps=args.sample_steps_preview,
                img_size=args.img_size,
            )
            print(f"[sample] wrote {sample_path} ({args.sample_steps_preview} steps, "
                  f"guidance={args.sample_guidance})")

        ep_dt = time.time() - epoch_t0
        print(f"[epoch {epoch}] train_loss={avg:.5f}  val_loss={val_loss:.5f}  "
              f"lr={cur_lr:.2e}  time={ep_dt/60:.1f} min")
        append_csv_row(csv_path, {
            "epoch": epoch,
            "train_loss": round(avg, 6),
            "val_loss": round(val_loss, 6) if val_loss == val_loss else "",
            "lr": round(cur_lr, 8),
            "epoch_time_min": round(ep_dt / 60, 2),
        })
        save_checkpoint(
            last_ckpt,
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch, step=global_step, args=args,
            null_context=null_context.detach(),
            extra={"avg_loss": avg, "val_loss": val_loss,
                   "epoch_time_train_min": ep_dt_train / 60},
        )

    save_checkpoint(
        ckpt_dir / "ldm_final.pt",
        model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, epoch=args.epochs - 1, step=global_step, args=args,
        null_context=null_context.detach(),
    )
    print("[done] LDM training complete.")


if __name__ == "__main__":
    main()
