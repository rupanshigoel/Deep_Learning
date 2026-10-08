"""Smoke test — runs all the bits that can silently fail at scale.

Checks:
  1. Model builds, Qwen loads from cache, projector is the only trainable in Stage-1 config.
  2. LoRA wrap succeeds and only adds `lora_*` params to trainables.
  3. Forward + backward works with BF16 autocast + grad checkpointing.
  4. `model.generate(inputs_embeds=...)` produces decodable strings.
  5. Stage-1 loss drops over a handful of steps on one batch (overfit smoke).
  6. Stage-2 LoRA receives gradients; base Qwen weights do not.

Exit 0 on success, non-zero on assertion failure.
"""
from __future__ import annotations
import argparse, os, sys, traceback
from pathlib import Path

import torch

from col775_a2.data.caption_dataset import ClevrCaptionVLMDataset, VLMCollator
from col775_a2.data.clevrx_dataset import ClevrXQADataset
from col775_a2.models.vlm import build_vlm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino-ckpt", required=True)
    ap.add_argument("--qwen-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--captions-json", required=True)
    ap.add_argument("--captions-images-dir", required=True)
    ap.add_argument("--clevrx-json", required=True)
    ap.add_argument("--clevrx-images-dir", required=True)
    ap.add_argument("--recut-pkl", default=None)
    return ap.parse_args()


def check(cond: bool, msg: str):
    if not cond:
        print(f"[smoke] FAIL: {msg}", flush=True)
        sys.exit(1)
    print(f"[smoke] OK  : {msg}", flush=True)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check(device == "cuda", "CUDA is available")

    print("\n=== [1] build VLM ===", flush=True)
    model, tok = build_vlm(
        dino_ckpt=args.dino_ckpt,
        qwen_name_or_path=args.qwen_path,
        dtype=torch.bfloat16,
        projector_hidden=1536,
        device=device,
    )
    print(f"[smoke] llm.config.hidden_size = {model.llm.config.hidden_size}", flush=True)
    check(model.llm.config.hidden_size == model.projector.fc2.out_features,
          "Projector out_dim == LLM hidden_size")

    bd = model.trainable_param_breakdown()
    print(f"[smoke] trainable param breakdown (s1): {bd}", flush=True)
    check(bd["projector"] > 0 and bd["lora"] == 0 and bd["llm_other"] == 0 and bd["vit"] == 0,
          "Stage-1: only projector is trainable")

    # Enable grad-ckpt for realistic memory profile
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.config.use_cache = False
    if hasattr(model.llm, "enable_input_require_grads"):
        model.llm.enable_input_require_grads()

    print("\n=== [2] Stage-1: 20 training steps on first 32 captions ===", flush=True)
    ds = ClevrCaptionVLMDataset(args.captions_json, args.captions_images_dir, max_samples=32)
    collate = VLMCollator(tok, image_tag=model.image_tag, include_target=True)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    model.train(); model.vit.eval()
    losses = []
    it = iter(loader)
    for step in range(20):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(pixel_values, batch["pre_ids"], batch["post_ids"], batch["target_ids"])
        out.loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(out.loss.item())
        print(f"[smoke] s1 step {step:2d}  loss={out.loss.item():.4f}", flush=True)

    check(losses[-1] < losses[0] * 0.9,
          f"Stage-1 loss dropped ≥ 10% ({losses[0]:.4f} → {losses[-1]:.4f})")

    print("\n=== [3] generate(inputs_embeds=...) sanity ===", flush=True)
    model.eval()
    if hasattr(model.llm, "config"):
        model.llm.config.use_cache = True
    batch = next(iter(loader))
    pv = batch["pixel_values"][:2].to(device)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        gens = model.generate(pv, batch["pre_ids"][:2], batch["post_ids"][:2],
                              max_new_tokens=64)
    for g in gens:
        print(f"[smoke] gen: {g[:200]!r}", flush=True)
    check(all(isinstance(g, str) and len(g) > 0 for g in gens),
          "generate returned non-empty decoded strings")

    print("\n=== [4] Stage-2: wrap LoRA, 10 train steps ===", flush=True)
    from peft import LoraConfig
    # Re-build to get a fresh LLM unwrapped by LoRA (stage-2 trainer does this via separate build).
    model.llm.config.use_cache = False
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model.apply_lora(lora_cfg)
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.projector.parameters():
        p.requires_grad = True
    # Check trainability
    bd2 = model.trainable_param_breakdown()
    print(f"[smoke] trainable param breakdown (s2): {bd2}", flush=True)
    check(bd2["lora"] > 0 and bd2["llm_other"] == 0 and bd2["vit"] == 0,
          "Stage-2: LoRA + projector trainable, base LLM/ViT frozen")

    ds2 = ClevrXQADataset(
        explanations_json=args.clevrx_json,
        images_dir=args.clevrx_images_dir,
        recut_pkl=args.recut_pkl,
        max_samples=32,
        stratified=False,
        seed=0,
    )
    print(f"[smoke] clevrx smoke entries: {len(ds2)}", flush=True)
    # Print the first entry schema for posterity
    sample = ds2[0]
    print(f"[smoke] stage-2 sample keys: {list(sample.keys())}", flush=True)
    print(f"[smoke] stage-2 sample target: {sample['target'][:200]!r}", flush=True)

    loader2 = DataLoader(ds2, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)

    # Fresh optimizer now that LoRA is attached
    opt2 = torch.optim.AdamW(model.trainable_parameters(), lr=2e-4)
    model.train(); model.vit.eval()

    # Snapshot reference base-Qwen param (must NOT move) + a LoRA param (MUST move)
    ref_base_name = None; ref_base_before = None
    ref_lora_name = None; ref_lora_before = None
    for n, p in model.llm.named_parameters():
        if ref_base_name is None and "lora_" not in n and p.dim() >= 2:
            ref_base_name = n; ref_base_before = p.detach().clone()
        if ref_lora_name is None and "lora_" in n and p.requires_grad:
            ref_lora_name = n; ref_lora_before = p.detach().clone()
        if ref_base_name and ref_lora_name:
            break

    losses2 = []
    it = iter(loader2)
    for step in range(10):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader2); batch = next(it)
        pv = batch["pixel_values"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(pv, batch["pre_ids"], batch["post_ids"], batch["target_ids"])
        out.loss.backward()
        opt2.step(); opt2.zero_grad()
        losses2.append(out.loss.item())
        print(f"[smoke] s2 step {step:2d}  loss={out.loss.item():.4f}", flush=True)

    # Confirm base param did NOT move, LoRA DID move
    for n, p in model.llm.named_parameters():
        if n == ref_base_name:
            delta = (p - ref_base_before.to(p.device)).abs().max().item()
            print(f"[smoke] base param {n!r} max |Δ|={delta:.2e}", flush=True)
            check(delta < 1e-6, f"Base Qwen param {n} unmodified")
        if n == ref_lora_name:
            delta = (p - ref_lora_before.to(p.device)).abs().max().item()
            print(f"[smoke] lora param {n!r} max |Δ|={delta:.2e}", flush=True)
            check(delta > 0, f"LoRA param {n} was updated by optimizer")

    print("\n=== SMOKE PASSED ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
