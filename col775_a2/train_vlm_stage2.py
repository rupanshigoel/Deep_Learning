"""Stage-2 VLM trainer — projector + LoRA adapters, CoT QA with factual explanations.

Initialises from Stage-1 projector, applies LoRA (r=16, target_modules
q/k/v/o_proj) to the Qwen3 LLM via `peft.get_peft_model`, and trains both
simultaneously. Vision encoder stays frozen.

Usage:
    python -m col775_a2.train_vlm_stage2 \\
        --clevrx-json  $PBS_O_WORKDIR/A2_dataset/Part_B/CLEVR_X/CLEVR_train_explanations_v0.7.10.json \\
        --recut-pkl    $PBS_O_WORKDIR/A2_dataset/Part_B/CLEVR_X/train_images_ids_v0.7.10-recut.pkl \\
        --images-dir   $PBS_O_WORKDIR/A2_dataset/Part_Aa/Clevr_official/images/train \\
        --dino-ckpt    $PBS_O_WORKDIR/checkpoints/dino_final.pt \\
        --stage1-ckpt  $PBS_O_WORKDIR/checkpoints/vlm_stage1_projector.pt \\
        --qwen-path    Qwen/Qwen3-4B-Instruct-2507 \\
        --ckpt-dir     $PBS_O_WORKDIR/checkpoints \\
        --log-dir      $PBS_O_WORKDIR/logs \\
        --max-samples 80000 --batch-size 1 --grad-accum 16 --lr 2e-4 --epochs 1
"""
from __future__ import annotations
import argparse, csv, os, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from col775_a2.data.clevrx_dataset import ClevrXQADataset
from col775_a2.data.caption_dataset import VLMCollator
from col775_a2.models.vlm import build_vlm
from col775_a2.utils.scheduler import cosine_warmup_schedule


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clevrx-json", required=True)
    ap.add_argument("--recut-pkl", default=None, help="If set, restrict to images in this pkl set.")
    ap.add_argument("--images-dir", required=True, help="Directory of CLEVR_train_*.png (Clevr_official/images/train)")
    ap.add_argument("--dino-ckpt", required=True)
    ap.add_argument("--stage1-ckpt", required=True, help="vlm_stage1_projector.pt")
    ap.add_argument("--qwen-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=80000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--projector-hidden", type=int, default=1536)
    ap.add_argument("--max-steps", type=int, default=0)
    # Chained-job full-data training: each job processes a slice of the
    # pre-shuffled (slice_seed) entries. total_steps_override fixes the LR
    # schedule horizon to the whole-epoch step count so cosine decay is
    # consistent across jobs.
    ap.add_argument("--slice-start", type=int, default=-1)
    ap.add_argument("--slice-end",   type=int, default=-1)
    ap.add_argument("--slice-seed",  type=int, default=42)
    ap.add_argument("--total-train-steps", type=int, default=0,
                    help="If >0, use this as the LR schedule total_steps "
                         "(overrides steps_per_epoch * epochs). For chained "
                         "full-data training this is the global epoch step count.")
    return ap.parse_args()


def main():
    from peft import LoraConfig

    args = parse_args()
    torch.manual_seed(args.seed)

    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda"

    # ---------- Model ----------
    print("[s2] Building VLM…", flush=True)
    model, tok = build_vlm(
        dino_ckpt=args.dino_ckpt,
        qwen_name_or_path=args.qwen_path,
        dtype=torch.bfloat16,
        projector_hidden=args.projector_hidden,
        device=device,
    )

    # Warm-start projector from Stage 1
    if args.stage1_ckpt and Path(args.stage1_ckpt).is_file():
        s1 = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
        model.projector.load_state_dict(s1["projector"])
        print(f"[s2] Loaded Stage-1 projector from {args.stage1_ckpt}", flush=True)
    else:
        print(f"[s2] WARNING: no Stage-1 projector at {args.stage1_ckpt}; using random init", flush=True)

    # Apply LoRA — must come after freeze_llm (build_vlm already did that).
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model.apply_lora(lora_cfg)

    # Gradient checkpointing on Qwen
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.llm, "config"):
        model.llm.config.use_cache = False

    # Make sure projector is trainable (it stays on by default)
    for p in model.projector.parameters():
        p.requires_grad = True

    bd = model.trainable_param_breakdown()
    print(f"[s2] trainable params: {bd} | total = {sum(bd.values()):,}", flush=True)
    assert bd["projector"] > 0 and bd["lora"] > 0 and bd["llm_other"] == 0 and bd["vit"] == 0, bd

    # ---------- Data ----------
    print("[s2] Loading CLEVR-X training set…", flush=True)
    use_slice = args.slice_start >= 0 or args.slice_end >= 0
    train_ds = ClevrXQADataset(
        explanations_json=args.clevrx_json,
        images_dir=args.images_dir,
        recut_pkl=args.recut_pkl,
        max_samples=(None if use_slice else args.max_samples),
        stratified=(not use_slice),
        seed=args.seed,
        slice_start=(args.slice_start if args.slice_start >= 0 else None),
        slice_end=(args.slice_end if args.slice_end >= 0 else None),
        slice_seed=args.slice_seed,
    )
    if use_slice:
        print(f"[s2] data slice [{args.slice_start}:{args.slice_end}) (seed={args.slice_seed})", flush=True)
    print(f"[s2] train size: {len(train_ds)}", flush=True)
    try:
        dist = train_ds.answer_distribution()
        print(f"[s2] top-5 answers: {dist.most_common(5)}", flush=True)
    except Exception:
        pass

    collate = VLMCollator(tok, image_tag=model.image_tag, include_target=True,
                          max_target_len=256)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate, persistent_workers=args.num_workers > 0, drop_last=True,
    )
    steps_per_epoch = len(loader) // args.grad_accum
    if args.total_train_steps and args.total_train_steps > 0:
        total_steps = args.total_train_steps
    else:
        total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    print(f"[s2] steps_per_epoch={steps_per_epoch}  total_steps={total_steps}  warmup={warmup_steps}", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    sched = cosine_warmup_schedule(opt, warmup_steps, total_steps, min_lr_ratio=0.05)

    # ---------- Resume ----------
    ckpt_last = ckpt_dir / "vlm_stage2_last.pt"
    lora_last_dir = ckpt_dir / "vlm_stage2_lora_last"
    start_step = 0
    if ckpt_last.is_file():
        state = torch.load(ckpt_last, map_location="cpu", weights_only=False)
        model.projector.load_state_dict(state["projector"])
        if lora_last_dir.is_dir():
            model.llm.load_adapter(str(lora_last_dir), adapter_name="default", is_trainable=True)
            # peft `load_adapter` returns the adapter; we're OK as long as weights were loaded.
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_step = state["step"]
        print(f"[s2] Resumed from step {start_step}", flush=True)

    csv_path = log_dir / "vlm_stage2_train.csv"
    csv_file = open(csv_path, "a", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    if csv_path.stat().st_size == 0:
        csv_writer.writerow(["step", "loss", "lr", "tokens_per_s", "epoch_time_min"])

    print("[s2] Training started.", flush=True)
    model.train(); model.vit.eval()

    if hasattr(model.llm, "enable_input_require_grads"):
        model.llm.enable_input_require_grads()

    step = start_step
    running_loss = 0.0; running_count = 0
    tokens_seen = 0
    t0 = time.time()
    grad_accum = args.grad_accum
    micro = 0
    opt.zero_grad(set_to_none=True)

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
                    print(f"[s2] step {step}/{total_steps}  loss={avg_loss:.4f}  lr={lr_now:.2e}  tok/s={tps:.0f}  t={elapsed/60:.1f}m",
                          flush=True)
                    running_loss = 0.0; running_count = 0

                if step % args.ckpt_every_steps == 0 or step == total_steps:
                    # Save LoRA adapter separately (for submission) + projector + optim state
                    model.llm.save_pretrained(str(lora_last_dir))
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

    final_lora = ckpt_dir / "vlm_stage2_lora_final"
    model.llm.save_pretrained(str(final_lora))
    torch.save(
        {"projector": model.projector.state_dict(), "step": step, "args": vars(args)},
        ckpt_dir / "vlm_stage2_final.pt",
    )
    print(f"[s2] Done. LoRA saved to {final_lora}; projector saved to vlm_stage2_final.pt.", flush=True)
    csv_file.close()


if __name__ == "__main__":
    main()
