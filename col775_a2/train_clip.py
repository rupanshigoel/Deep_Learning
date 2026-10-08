"""Train CLIP on Part_A (CLEVR captions). Resumable.

Key features:
  * Auto-resumes from `checkpoints/clip_last.pt` if present.
  * Writes a checkpoint at the end of every epoch.
  * Appends per-epoch metrics to `logs/clip_train.csv` (epoch, loss, lr,
    logit_scale, epoch_time_min) for report plotting.
  * Ctrl-C exits cleanly; re-running resumes from the last saved epoch.

Usage:
    python -m col775_a2.train_clip
    python -m col775_a2.train_clip --epochs 100 --batch-size 512 --amp
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from col775_a2.data.clip_dataset import (
    ClevrCaptionDataset, DedupBatchSampler, clip_collate,
    default_train_transform,
)
from col775_a2.data.tokenizer import SimpleTokenizer
from col775_a2.models.clip_model import CLIPModel
from col775_a2.utils.losses import clip_contrastive_loss
from col775_a2.utils.scheduler import cosine_warmup_schedule


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_A"


def build_tokenizer(train_json: str, ckpt_dir: Path, context_length: int) -> SimpleTokenizer:
    tok_path = ckpt_dir / "tokenizer.json"
    if tok_path.is_file():
        tok = SimpleTokenizer.load(str(tok_path))
        if tok.context_length == context_length:
            return tok
    import json
    with open(train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions = [r["caption"] for r in data]
    tok = SimpleTokenizer.build(captions, context_length=context_length)
    tok.save(str(tok_path))
    print(f"[tokenizer] built vocab_size={tok.vocab_size} ctx={tok.context_length}")
    return tok


def init_csv_log(path: Path, fieldnames: list) -> None:
    """Create CSV with header if it doesn't exist yet (safe to call every run)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--ckpt-dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.2)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--context-length", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--amp", action="store_true", default=True, help="use fp16 autocast (default on)")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "clip_train.csv"
    _CSV_FIELDS = ["epoch", "train_loss", "lr", "logit_scale", "epoch_time_min"]
    init_csv_log(csv_path, _CSV_FIELDS)

    data_root = Path(args.data_root)
    train_json = data_root / "train" / "clevr_train_captions.json"
    train_imgs = data_root / "train" / "images"
    assert train_json.is_file(), f"Missing {train_json}"
    assert train_imgs.is_dir(), f"Missing {train_imgs}"

    tokenizer = build_tokenizer(str(train_json), ckpt_dir, args.context_length)

    dataset = ClevrCaptionDataset(
        captions_json=str(train_json),
        images_dir=str(train_imgs),
        tokenizer=tokenizer,
        transform=default_train_transform(args.img_size),
    )
    print(f"[data] {len(dataset)} images, {len(dataset.unique_captions)} unique captions")

    sampler = DedupBatchSampler(dataset, batch_size=args.batch_size, drop_last=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=clip_collate(tokenizer),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    print(f"[data] steps/epoch={steps_per_epoch}  total_steps={total_steps}  warmup={warmup_steps}")

    model = CLIPModel(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        img_size=args.img_size,
    ).to(device)

    # Keep logit_scale and LayerNorm/bias out of weight decay.
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "logit_scale" in n:
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
    )
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    last_ckpt = ckpt_dir / "clip_last.pt"
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
        sampler.set_epoch(epoch)
        epoch_t0 = time.time()
        running_loss = 0.0; running_n = 0
        for it, (imgs, tokens, eot) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            eot = eot.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(args.amp and device == "cuda")):
                logits_i, logits_t = model(imgs, tokens, eot)
                loss = clip_contrastive_loss(logits_i, logits_t)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running_loss += loss.item() * imgs.size(0)
            running_n += imgs.size(0)
            if (it + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                scale = model.logit_scale.detach().exp().item()
                print(f"ep {epoch:3d}  it {it+1:5d}/{steps_per_epoch}  "
                      f"loss {running_loss/running_n:.4f}  lr {lr:.2e}  τ^-1 {scale:.2f}")

        ep_dt = time.time() - epoch_t0
        avg_loss = running_loss / max(1, running_n)
        cur_lr = scheduler.get_last_lr()[0]
        cur_scale = model.logit_scale.detach().exp().item()
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f}  lr={cur_lr:.2e}  τ^-1={cur_scale:.2f}  time={ep_dt/60:.1f} min")
        append_csv_row(csv_path, {"epoch": epoch, "train_loss": round(avg_loss, 6),
                                   "lr": round(cur_lr, 8), "logit_scale": round(cur_scale, 4),
                                   "epoch_time_min": round(ep_dt / 60, 2)})
        save_checkpoint(
            last_ckpt,
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, epoch=epoch, step=global_step, args=args,
            extra={"avg_loss": avg_loss},
        )

    # Final tidy checkpoint (identical content — convenience name).
    save_checkpoint(
        ckpt_dir / "clip_final.pt",
        model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, epoch=args.epochs - 1, step=global_step, args=args,
    )
    print("[done] training complete.")


if __name__ == "__main__":
    main()
