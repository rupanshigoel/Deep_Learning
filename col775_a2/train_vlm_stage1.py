"""Stage-1 VLM trainer — projector-only, image-caption alignment.

Frozen: ViT, Qwen LLM.
Trainable: MLPProjector (≈4.6 M params).

Single-GPU, BF16 autocast, gradient checkpointing on Qwen. Resumable: saves
(and auto-loads on restart) projector + optimizer + scheduler + step counter.

Usage:
    python -m col775_a2.train_vlm_stage1 \\
        --data-root $PBS_O_WORKDIR/A2_dataset/Part_Aa/Clevr_official \\
        --dino-ckpt $PBS_O_WORKDIR/checkpoints/dino_final.pt \\
        --qwen-path Qwen/Qwen3-4B-Instruct-2507 \\
        --ckpt-dir  $PBS_O_WORKDIR/checkpoints \\
        --log-dir   $PBS_O_WORKDIR/logs \\
        --batch-size 2 --grad-accum 16 --lr 1e-3 --epochs 1
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from col775_a2.data.caption_dataset import ClevrCaptionVLMDataset, VLMCollator, vlm_image_transform
from col775_a2.models.vlm import build_vlm
from col775_a2.utils.scheduler import cosine_warmup_schedule


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to Clevr_official")
    ap.add_argument("--dino-ckpt", required=True)
    ap.add_argument("--qwen-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=2, help="micro batch size per step")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all 70K")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--projector-hidden", type=int, default=1536)
    ap.add_argument("--max-steps", type=int, default=0, help="cap training steps (smoke tests); 0 = use full epoch count")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Stage-1 training requires a CUDA GPU"

    # ---------- Model ----------
    print("[s1] Building VLM…", flush=True)
    model, tok = build_vlm(
        dino_ckpt=args.dino_ckpt,
        qwen_name_or_path=args.qwen_path,
        dtype=torch.bfloat16,
        projector_hidden=args.projector_hidden,
        device=device,
    )
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.config.use_cache = False

    # Assert frozen-ness
    bd = model.trainable_param_breakdown()
    print(f"[s1] trainable params: {bd} | total = {sum(bd.values()):,}", flush=True)
    assert bd["projector"] > 0 and bd["lora"] == 0 and bd["llm_other"] == 0 and bd["vit"] == 0, bd

    # ---------- Data ----------
    print("[s1] Loading training captions…", flush=True)
    train_ds = ClevrCaptionVLMDataset(
        captions_json=str(data_root / "captions" / "clevr_train_captions.json"),
        images_dir=str(data_root / "images" / "train"),
        max_samples=(args.max_samples or None),
    )
    print(f"[s1] train size: {len(train_ds)}", flush=True)

    collate = VLMCollator(tok, image_tag=model.image_tag, include_target=True)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate, persistent_workers=args.num_workers > 0, drop_last=True,
    )
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    print(f"[s1] steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  warmup={warmup_steps}", flush=True)

    # ---------- Optimizer / schedule ----------
    opt = torch.optim.AdamW(model.trainable_parameters(),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    sched = cosine_warmup_schedule(opt, warmup_steps, total_steps, min_lr_ratio=0.05)

    # ---------- Resume ----------
    ckpt_last = ckpt_dir / "vlm_stage1_last.pt"
    start_step = 0
    if ckpt_last.is_file():
        state = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        model.projector.load_state_dict(state["projector"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_step = state["step"]
        print(f"[s1] Resumed from step {start_step}", flush=True)

    # ---------- Log CSV ----------
    csv_path = log_dir / "vlm_stage1_train.csv"
    csv_writer = None
    csv_file = open(csv_path, "a", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    if csv_path.stat().st_size == 0:
        csv_writer.writerow(["step", "loss", "lr", "tokens_per_s", "epoch_time_min"])

    # ---------- Training loop ----------
    print("[s1] Training started.", flush=True)
    model.train()
    # Keep ViT/LLM in eval mode (they're frozen); projector in train.
    model.vit.eval()
    # Qwen: grad_ckpt requires model.training=True but all params frozen, so gradients flow only through LoRA-wrapped paths (stage 2) or through input_embeds path back to projector (stage 1).
    model.llm.train()

    step = start_step
    running_loss = 0.0
    running_count = 0
    t0 = time.time()
    tokens_seen = 0
    grad_accum = args.grad_accum

    micro = 0
    opt.zero_grad(set_to_none=True)

    # Enable input-require-grads so gradient checkpointing flows through the embedding
    # into our injected image embeds (projector output).
    if hasattr(model.llm, "enable_input_require_grads"):
        model.llm.enable_input_require_grads()

    for epoch in range(args.epochs):
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pre_ids = batch["pre_ids"]; post_ids = batch["post_ids"]; target_ids = batch["target_ids"]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(pixel_values, pre_ids, post_ids, target_ids)
                loss = out.loss / grad_accum
            loss.backward()
            running_loss += out.loss.detach().float().item()
            running_count += 1
            tokens_seen += sum(t.numel() for t in target_ids)

            micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    tps = tokens_seen / max(1.0, elapsed)
                    avg_loss = running_loss / max(1, running_count)
                    lr_now = opt.param_groups[0]["lr"]
                    csv_writer.writerow([step, f"{avg_loss:.6f}", f"{lr_now:.2e}", f"{tps:.1f}", f"{elapsed/60:.2f}"])
                    print(f"[s1] step {step}/{total_steps}  loss={avg_loss:.4f}  lr={lr_now:.2e}  tok/s={tps:.0f}  t={elapsed/60:.1f}m",
                          flush=True)
                    running_loss = 0.0; running_count = 0

                if step % args.ckpt_every_steps == 0 or step == total_steps:
                    torch.save(
                        {"projector": model.projector.state_dict(),
                         "opt": opt.state_dict(),
                         "sched": sched.state_dict(),
                         "step": step,
                         "args": vars(args)},
                        ckpt_last,
                    )

                if step >= total_steps:
                    break
                if args.max_steps and step >= args.max_steps:
                    break
        if step >= total_steps:
            break
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(
        {"projector": model.projector.state_dict(), "opt": opt.state_dict(),
         "sched": sched.state_dict(), "step": step, "args": vars(args)},
        ckpt_dir / "vlm_stage1_final.pt",
    )
    torch.save({"projector": model.projector.state_dict(), "step": step, "args": vars(args)},
               ckpt_dir / "vlm_stage1_projector.pt")
    print(f"[s1] Training complete at step {step}. Saved vlm_stage1_final.pt + vlm_stage1_projector.pt.", flush=True)
    csv_file.close()


if __name__ == "__main__":
    main()
