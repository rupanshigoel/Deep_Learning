"""Unified evaluation for Part B Stages 1 and 2.

Stage 1: image-caption EM + BLEU (sacrebleu) on train-subset + val.
Stage 2: QA exact-match on train-subset + val; answer parsed from generation
         after the "Answer:" marker.

Usage (Stage 1):
    python -m col775_a2.eval_vlm \\
        --stage 1 \\
        --projector-ckpt $PBS_O_WORKDIR/checkpoints/vlm_stage1_projector.pt \\
        --data-root      $PBS_O_WORKDIR/A2_dataset/Part_Aa/Clevr_official \\
        --dino-ckpt      $PBS_O_WORKDIR/checkpoints/dino_final.pt \\
        --out-dir        $PBS_O_WORKDIR/logs/vlm_eval_stage1

Usage (Stage 2):
    python -m col775_a2.eval_vlm \\
        --stage 2 \\
        --projector-ckpt $PBS_O_WORKDIR/checkpoints/vlm_stage2_final.pt \\
        --lora-dir       $PBS_O_WORKDIR/checkpoints/vlm_stage2_lora_final \\
        --clevrx-train-json $PBS_O_WORKDIR/A2_dataset/Part_B/CLEVR_X/CLEVR_train_explanations_v0.7.10.json \\
        --recut-pkl         $PBS_O_WORKDIR/A2_dataset/Part_B/CLEVR_X/train_images_ids_v0.7.10-recut.pkl \\
        --clevrx-val-json   $PBS_O_WORKDIR/A2_dataset/Part_B/CLEVR_X/CLEVR_val_explanations_v0.7.10.json \\
        --train-images-dir  $PBS_O_WORKDIR/A2_dataset/Part_Aa/Clevr_official/images/train \\
        --val-images-dir    $PBS_O_WORKDIR/A2_dataset/Part_Aa/Clevr_official/images/val \\
        --dino-ckpt         $PBS_O_WORKDIR/checkpoints/dino_final.pt \\
        --out-dir           $PBS_O_WORKDIR/logs/vlm_eval_stage2
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from col775_a2.data.caption_dataset import ClevrCaptionVLMDataset, VLMCollator
from col775_a2.data.clevrx_dataset import ClevrXQADataset
from col775_a2.models.vlm import build_vlm


ANSWER_RE = re.compile(r"[Aa]nswer\s*:\s*(.*?)(?:\n|$)", re.DOTALL)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--dino-ckpt", required=True)
    ap.add_argument("--projector-ckpt", required=True)
    ap.add_argument("--lora-dir", default=None)
    ap.add_argument("--qwen-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--projector-hidden", type=int, default=1536)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--splits", default="train,val", help="comma-sep: train,val")
    ap.add_argument("--train-subsample", type=int, default=2000,
                    help="Train split is big; cap for eval.")
    ap.add_argument("--val-subsample", type=int, default=0, help="0 = use all")
    # Stage 1 args
    ap.add_argument("--data-root", help="Clevr_official root (Stage 1)")
    # Stage 2 args
    ap.add_argument("--clevrx-train-json")
    ap.add_argument("--recut-pkl")
    ap.add_argument("--train-images-dir")
    ap.add_argument("--clevrx-val-json")
    ap.add_argument("--val-images-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-qualitative", type=int, default=5)
    return ap.parse_args()


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    # strip terminal punctuation
    s = re.sub(r"[.!?,;:]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_answer_from_generation(gen: str) -> str:
    """Pull the answer token out of 'Reasoning: ...\\nAnswer: X' style output."""
    m = ANSWER_RE.search(gen)
    if m:
        return normalize_answer(m.group(1).split("\n")[0])
    # Fallback: first non-empty line
    for line in gen.strip().splitlines():
        if line.strip():
            return normalize_answer(line)
    return ""


def load_model(args, device: str):
    print("[eval] Building VLM…", flush=True)
    model, tok = build_vlm(
        dino_ckpt=args.dino_ckpt,
        qwen_name_or_path=args.qwen_path,
        dtype=torch.bfloat16,
        projector_hidden=args.projector_hidden,
        device=device,
    )
    # Load projector
    state = torch.load(args.projector_ckpt, map_location="cpu", weights_only=False)
    proj_sd = state["projector"] if "projector" in state else state
    model.projector.load_state_dict(proj_sd)
    print(f"[eval] Loaded projector from {args.projector_ckpt}", flush=True)

    if args.lora_dir:
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, args.lora_dir)
        model.llm = model.llm.to(device)
        print(f"[eval] Loaded LoRA from {args.lora_dir}", flush=True)

    # Inference mode
    model.eval()
    if hasattr(model.llm, "config"):
        model.llm.config.use_cache = True
    # Force left padding for generation.
    tok.padding_side = "left"
    return model, tok


def _collate_generation(batch, collator):
    """Run collator with include_target=False (inference) plus keep refs."""
    collator.include_target = False
    out = collator(batch)
    # Still stash refs + targets for scoring
    out["target_str"] = [b["target"] for b in batch]
    out["ref"] = [b.get("ref_caption") or b.get("target") for b in batch]
    out["image_filename"] = [b.get("image_filename", "") for b in batch]
    if "question" in batch[0]:
        out["question"] = [b["question"] for b in batch]
        out["answer"] = [b["answer"] for b in batch]
        out["factual_explanation"] = [b["factual_explanation"] for b in batch]
    return out


@torch.no_grad()
def run_inference(model, loader, device, max_new_tokens: int, split_name: str, out_file):
    preds = []
    for i, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pre_ids = batch["pre_ids"]; post_ids = batch["post_ids"]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outs = model.generate(pixel_values, pre_ids, post_ids, max_new_tokens=max_new_tokens)
        B = len(outs)
        for j in range(B):
            rec = {
                "split": split_name,
                "image_filename": batch["image_filename"][j],
                "pred_raw": outs[j].strip(),
                "target_str": batch["target_str"][j],
            }
            if "question" in batch:
                rec.update({
                    "question": batch["question"][j],
                    "answer_gt": batch["answer"][j],
                    "factual_explanation_gt": batch["factual_explanation"][j],
                })
            preds.append(rec)
            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (i + 1) % 20 == 0:
            print(f"[eval:{split_name}] {(i+1)*B} samples generated", flush=True)
    return preds


def eval_stage1(args, model, tok, device, out_dir: Path):
    data_root = Path(args.data_root)
    splits = args.splits.split(",")
    collator = VLMCollator(tok, image_tag=model.image_tag, include_target=False, max_target_len=256)

    def make_loader(split_dir: str, json_name: str, cap: int):
        ds = ClevrCaptionVLMDataset(
            captions_json=str(data_root / "captions" / json_name),
            images_dir=str(data_root / "images" / split_dir),
        )
        if cap and cap < len(ds):
            idxs = np.random.default_rng(args.seed).choice(len(ds), cap, replace=False).tolist()
            ds = Subset(ds, idxs)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=lambda b: _collate_generation(b, collator))

    results: Dict = {}
    # Line-buffered (buffering=1) so predictions persist even if PBS kills on walltime.
    preds_file = open(out_dir / "predictions.jsonl", "w", encoding="utf-8", buffering=1)
    for split in splits:
        if split == "train":
            loader = make_loader("train", "clevr_train_captions.json", args.train_subsample)
        else:
            loader = make_loader("val", "clevr_val_captions.json", args.val_subsample)
        preds = run_inference(model, loader, device, args.max_new_tokens, split, preds_file)
        # Metrics
        refs = [p["target_str"] for p in preds]
        hypos = [p["pred_raw"] for p in preds]
        em = sum(int(normalize_answer(h) == normalize_answer(r)) for h, r in zip(hypos, refs)) / max(1, len(preds))
        try:
            import sacrebleu
            bleu = sacrebleu.corpus_bleu(hypos, [refs]).score
        except Exception as e:
            print("[eval] sacrebleu failed:", e, flush=True)
            bleu = None
        results[split] = {"em": em, "bleu": bleu, "n": len(preds)}
        print(f"[eval] Stage-1 {split}: EM={em*100:.2f}%  BLEU={bleu:.2f}  (n={len(preds)})", flush=True)
    preds_file.close()
    return results


def eval_stage2(args, model, tok, device, out_dir: Path):
    splits = args.splits.split(",")
    collator = VLMCollator(tok, image_tag=model.image_tag, include_target=False, max_target_len=512)

    def make_loader(json_path, images_dir, recut_pkl, cap):
        ds = ClevrXQADataset(
            explanations_json=json_path,
            images_dir=images_dir,
            recut_pkl=recut_pkl,
            stratified=True,
            max_samples=cap if cap else None,
            seed=args.seed,
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=lambda b: _collate_generation(b, collator))

    results: Dict = {}
    preds_file = open(out_dir / "predictions.jsonl", "w", encoding="utf-8", buffering=1)
    for split in splits:
        if split == "train":
            loader = make_loader(args.clevrx_train_json, args.train_images_dir,
                                 args.recut_pkl, args.train_subsample)
        else:
            loader = make_loader(args.clevrx_val_json, args.val_images_dir,
                                 None, args.val_subsample)
        preds = run_inference(model, loader, device, args.max_new_tokens, split, preds_file)
        # EM on extracted answers
        em_hits = 0; total = 0
        for p in preds:
            pred_ans = extract_answer_from_generation(p["pred_raw"])
            gt_ans = normalize_answer(p.get("answer_gt", ""))
            if pred_ans == gt_ans:
                em_hits += 1
            total += 1
            p["pred_answer"] = pred_ans
        em = em_hits / max(1, total)
        results[split] = {"em": em, "n": total}
        print(f"[eval] Stage-2 {split}: EM={em*100:.2f}%  (n={total})", flush=True)
    preds_file.close()
    return results


def dump_qualitative(predictions_jsonl: Path, out_dir: Path, stage: int, n: int = 5):
    """Pick n correct + n incorrect samples per split; write image + text side-by-side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    records: Dict[str, Dict[str, List[Dict]]] = {}   # {split: {"correct": [...], "incorrect": [...]}}
    with open(predictions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            split = r["split"]
            if stage == 1:
                correct = normalize_answer(r["pred_raw"]) == normalize_answer(r["target_str"])
            else:
                # `pred_answer` isn't in the jsonl (it's added post-write in memory);
                # re-extract from pred_raw so the correct/incorrect buckets fill properly.
                pred_ans = extract_answer_from_generation(r.get("pred_raw", ""))
                r["pred_answer"] = pred_ans
                correct = pred_ans == normalize_answer(r.get("answer_gt", ""))
            records.setdefault(split, {"correct": [], "incorrect": []})
            bucket = "correct" if correct else "incorrect"
            if len(records[split][bucket]) < n:
                records[split][bucket].append(r)

    # Find images
    project_root = out_dir.parent.parent
    for split, buckets in records.items():
        for bucket_name, items in buckets.items():
            if not items:
                continue
            fig, axes = plt.subplots(len(items), 2, figsize=(13, 3 * len(items)),
                                     gridspec_kw={"width_ratios": [1, 2.5]})
            if len(items) == 1:
                axes = [axes]
            for row, rec in enumerate(items):
                img_fn = rec["image_filename"]
                # Find the actual image: look under images/train/ or images/val/
                img = None
                for d in ["train", "val"]:
                    candidate = project_root / "A2_dataset" / "Part_Aa" / "Clevr_official" / "images" / d / img_fn
                    if candidate.is_file():
                        img = Image.open(candidate).convert("RGB")
                        break
                if img is not None:
                    axes[row][0].imshow(img)
                axes[row][0].set_axis_off()
                if stage == 1:
                    text = f"GT: {rec['target_str'][:300]}\n\nPRED: {rec['pred_raw'][:300]}"
                else:
                    text = (f"Q: {rec.get('question','')[:200]}\n"
                            f"GT answer: {rec.get('answer_gt','')}\n"
                            f"GT expl.: {rec.get('factual_explanation_gt','')[:200]}\n\n"
                            f"PRED:\n{rec['pred_raw'][:500]}")
                axes[row][1].axis("off")
                axes[row][1].text(0.0, 0.5, text, va="center", fontsize=7, family="monospace")
            fig.suptitle(f"Stage-{stage} {split} / {bucket_name} samples")
            fig.tight_layout()
            out_path = out_dir / f"qualitative_{split}_{bucket_name}.png"
            fig.savefig(out_path, dpi=140); plt.close(fig)
            print(f"[eval] wrote {out_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "Eval needs a GPU for generation at this scale"

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args, device)

    if args.stage == 1:
        assert args.data_root, "--data-root required for Stage 1 eval"
        results = eval_stage1(args, model, tok, device, out_dir)
    else:
        assert args.clevrx_val_json and args.val_images_dir, "Stage 2 needs CLEVR-X val json + images dir"
        results = eval_stage2(args, model, tok, device, out_dir)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Wrote {out_dir / 'metrics.json'}: {results}", flush=True)

    try:
        dump_qualitative(out_dir / "predictions.jsonl", out_dir, args.stage, n=args.n_qualitative)
    except Exception as e:
        print(f"[eval] qualitative dump failed: {e}", flush=True)


if __name__ == "__main__":
    main()
