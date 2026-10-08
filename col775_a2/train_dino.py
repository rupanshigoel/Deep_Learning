"""Train DINO on Part_A (image-only, multi-crop).

Resumable: every epoch end it writes `checkpoints/dino_last.pt`. Re-running
resumes from the last completed epoch. Ctrl-C kills the process cleanly;
re-running resumes from the last saved epoch.

Per-epoch metrics (loss, lr, teacher_temp, momentum, epoch_time_min) are
appended to `logs/dino_train.csv` for report plotting.

Usage:
    python -m col775_a2.train_dino
    python -m col775_a2.train_dino --epochs 100 --batch-size 128
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from col775_a2.data.dino_dataset import (
    ClevrImageOnlyDataset, DINOMultiCropTransform, dino_collate,
)
from col775_a2.models.dino_model import build_dino
from col775_a2.utils.losses import DINOLoss
from col775_a2.utils.scheduler import (
    cosine_warmup_schedule, cosine_schedule_value, linear_schedule_value,
)


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_A"


def init_csv_log(path: Path, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv_row(path: Path, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)


def save_checkpoint(path: Path, *, models, optimizer, scheduler, scaler, dino_loss, epoch, step, args, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "student": models["student"].state_dict(),
        "teacher": models["teacher"].state_dict(),
        "center":  dino_loss.center.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }
    if extra: payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_load(path: Path):
    if not path.is_file(): return None
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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.04)
    ap.add_argument("--weight-decay-end", type=float, default=0.4)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--min-lr-ratio", type=float, default=1e-6 / 5e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--local-size", type=int, default=96)
    ap.add_argument("--n-local", type=int, default=8)
    ap.add_argument("--out-dim", type=int, default=4096)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--teacher-temp-end", type=float, default=0.07)
    ap.add_argument("--warmup-teacher-temp-epochs", type=int, default=30)
    ap.add_argument("--student-temp", type=float, default=0.1)
    ap.add_argument("--center-momentum", type=float, default=0.9)
    ap.add_argument("--momentum-teacher", type=float, default=0.996)
    ap.add_argument("--momentum-teacher-end", type=float, default=1.0)
    ap.add_argument("--freeze-last-layer-epochs", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=3.0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "dino_train.csv"
    _CSV_FIELDS = ["epoch", "train_loss", "lr", "teacher_temp", "momentum", "epoch_time_min"]
    init_csv_log(csv_path, _CSV_FIELDS)

    data_root = Path(args.data_root)
    train_json = data_root / "train" / "clevr_train_captions.json"  # used only for file list
    train_imgs = data_root / "train" / "images"
    assert train_imgs.is_dir(), f"Missing {train_imgs}"

    transform = DINOMultiCropTransform(
        global_size=args.img_size, local_size=args.local_size, n_local_crops=args.n_local,
    )
    dataset = ClevrImageOnlyDataset(str(train_json), str(train_imgs), transform)
    print(f"[data] {len(dataset)} images")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, collate_fn=dino_collate,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    print(f"[data] steps/epoch={steps_per_epoch}  total_steps={total_steps}")

    models = build_dino(img_size=args.img_size, out_dim=args.out_dim).to(device)
    student = models["student"]; teacher = models["teacher"]
    dino_loss = DINOLoss(
        out_dim=args.out_dim, n_global=2,
        student_temp=args.student_temp, center_momentum=args.center_momentum,
    ).to(device)

    # Param groups: keep biases/LN/logit_scale out of weight decay.
    decay, nodecay = [], []
    for n, p in student.named_parameters():
        if not p.requires_grad: continue
        if p.ndim <= 1 or n.endswith(".bias"):
            nodecay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
    )
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=args.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    last_ckpt = ckpt_dir / "dino_last.pt"
    start_epoch = 0; global_step = 0
    state = try_load(last_ckpt)
    if state is not None:
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])
        dino_loss.center.copy_(state["center"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        global_step = state["step"]
        print(f"[resume] resuming from epoch {start_epoch} (step {global_step})")
    else:
        print("[resume] no checkpoint found, training from scratch")

    def momentum_at(step):
        return cosine_schedule_value(args.momentum_teacher, args.momentum_teacher_end, step, total_steps)

    def teacher_temp_at(epoch):
        return linear_schedule_value(args.teacher_temp, args.teacher_temp_end,
                                     epoch, args.warmup_teacher_temp_epochs)

    def wd_at(step):
        return cosine_schedule_value(args.weight_decay, args.weight_decay_end, step, total_steps)

    student.train(); teacher.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        t_temp = teacher_temp_at(epoch)
        running_loss = 0.0; running_n = 0
        for it, crops in enumerate(loader):
            # crops = [g1, g2, l1..l8], each (B, 3, H, W).
            crops = [c.to(device, non_blocking=True) for c in crops]
            B = crops[0].size(0)

            # Update weight decay on the decayed param group only.
            optimizer.param_groups[0]["weight_decay"] = wd_at(global_step)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(args.amp and device == "cuda")):
                student_outs = student(crops)                      # list len=n_crops
                with torch.no_grad():
                    teacher_outs = teacher(crops[:2])              # teacher sees only globals
                loss = dino_loss(student_outs, teacher_outs, teacher_temp=t_temp)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            # Freeze last layer of head for first few epochs (paper trick).
            if epoch < args.freeze_last_layer_epochs:
                for p in student.head.last_layer.parameters():
                    if p.grad is not None: p.grad.zero_()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update center (using un-centered teacher outputs, pre-temperature).
            with torch.no_grad():
                dino_loss.update_center(torch.cat(teacher_outs, dim=0).float())
                # EMA the teacher from the student.
                m = momentum_at(global_step)
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data.mul_(m).add_(ps.data, alpha=1 - m)

            global_step += 1
            running_loss += loss.item() * B; running_n += B
            if (it + 1) % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"ep {epoch:3d}  it {it+1:5d}/{steps_per_epoch}  "
                      f"loss {running_loss/running_n:.4f}  lr {lr:.2e}  "
                      f"τ_t {t_temp:.3f}  m {momentum_at(global_step):.4f}")

        ep_dt = time.time() - epoch_t0
        avg_loss = running_loss / max(1, running_n)
        cur_lr = scheduler.get_last_lr()[0]
        cur_mom = momentum_at(global_step)
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f}  lr={cur_lr:.2e}  "
              f"τ_t={t_temp:.3f}  m={cur_mom:.4f}  time={ep_dt/60:.1f} min")
        append_csv_row(csv_path, {"epoch": epoch, "train_loss": round(avg_loss, 6),
                                   "lr": round(cur_lr, 8), "teacher_temp": round(t_temp, 4),
                                   "momentum": round(cur_mom, 6), "epoch_time_min": round(ep_dt / 60, 2)})
        save_checkpoint(
            last_ckpt,
            models=models, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            dino_loss=dino_loss, epoch=epoch, step=global_step, args=args,
            extra={"avg_loss": avg_loss},
        )

    save_checkpoint(
        ckpt_dir / "dino_final.pt",
        models=models, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        dino_loss=dino_loss, epoch=args.epochs - 1, step=global_step, args=args,
    )
    print("[done] training complete.")


if __name__ == "__main__":
    main()
