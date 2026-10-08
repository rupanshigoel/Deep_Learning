"""Stage-2 dataset — CLEVR-X QA with factual explanations.

Inputs (per entry from `CLEVR_<split>_explanations_v0.7.10.json`):
    image_index, image_filename, question_index, question, answer,
    factual_explanation: List[str]  (1..10 variants per QA pair)

The CLEVR-X "recut" splits live in two pickle files of **image filenames**:
    train_images_ids_v0.7.10-recut.pkl  (set of ~56 000 CLEVR_train_*.png)
    dev_images_ids_v0.7.10-recut.pkl    (set of ~14 000 CLEVR_train_*.png)

For Part B Stage 2 training we:
  * Take CLEVR_train_explanations entries whose `image_filename` is in the
    recut-train set.
  * Stratified-subsample down to a budget (e.g. 80 K) by capping per-answer.
  * Each epoch, each QA pair uses a freshly-sampled factual_explanation from
    its list (diversity across epochs, one per step).

For validation we typically use CLEVR_val_explanations_v0.7.10.json directly
(never seen during train; 15 K images × 10 Q ≈ 150 K QAs), but the dataset
class is agnostic — pass the JSON you want.
"""
from __future__ import annotations
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .caption_dataset import vlm_image_transform


STAGE2_SYSTEM = (
    "You are a visual reasoning assistant. Look at the image, reason step-by-step "
    "about the question, and then give a concise final answer."
)
STAGE2_USER_TEMPLATE = "<<IMAGE>>\nQuestion: {question}"


def format_stage2_target(factual_explanation: str, answer: str) -> str:
    """Canonical CoT target: reasoning + answer marker."""
    return f"Reasoning: {factual_explanation.strip()}\nAnswer: {answer.strip()}"


def stratified_subsample(entries: List[Dict], total: int, seed: int = 0) -> List[Dict]:
    """Cap samples per answer class so counts saturate the budget evenly.

    Rationale: 'yes'/'no' dominate the CLEVR-X answer distribution. Without
    stratification, an 80K subsample would be >40% yes/no and under-represent
    counts/colors/shapes — hurting Stage 2 generalization.
    """
    if total >= len(entries):
        return list(entries)
    rng = random.Random(seed)
    by_answer: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        by_answer[e["answer"]].append(i)
    for k in by_answer:
        rng.shuffle(by_answer[k])
    # Proportional allocation with a per-class cap: never let any class exceed
    # 1.5× the uniform share.
    K = len(by_answer)
    uniform = total // K
    cap = int(uniform * 1.5)
    picks: List[int] = []
    for k, idxs in by_answer.items():
        picks.extend(idxs[: min(cap, len(idxs))])
    # If we've under-filled (small classes), top up from the majority classes.
    if len(picks) < total:
        remaining = []
        for k, idxs in by_answer.items():
            remaining.extend(idxs[min(cap, len(idxs)):])
        rng.shuffle(remaining)
        need = total - len(picks)
        picks.extend(remaining[:need])
    rng.shuffle(picks)
    picks = picks[:total]
    return [entries[i] for i in picks]


class ClevrXQADataset(Dataset):
    def __init__(
        self,
        explanations_json: str,
        images_dir: str,
        recut_pkl: Optional[str] = None,      # restrict to image filenames in this set; None = use all
        transform=None,
        max_samples: Optional[int] = None,     # final cap (after recut filter + stratified subsample)
        stratified: bool = True,
        seed: int = 0,
        slice_start: Optional[int] = None,     # data-slice start index (after deterministic shuffle)
        slice_end: Optional[int] = None,       # data-slice end index (exclusive)
        slice_seed: int = 42,                  # master shuffle seed; same across chained jobs
    ):
        with open(explanations_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        questions = raw["questions"] if isinstance(raw, dict) else raw

        img_set: Optional[Set[str]] = None
        if recut_pkl:
            with open(recut_pkl, "rb") as f:
                img_set = set(pickle.load(f))

        images_dir = Path(images_dir)
        entries: List[Dict] = []
        # Don't stat files over Lustre — filter by img_set only. Bad rows raise cleanly
        # at __getitem__.
        for q in questions:
            fn = q.get("image_filename")
            if not fn:
                continue
            if img_set is not None and fn not in img_set:
                continue
            fe = q.get("factual_explanation") or []
            if not fe:
                continue
            entries.append({
                "image_filename": fn,
                "question": q["question"],
                "answer": str(q["answer"]),
                "factual_explanations": list(fe),
            })

        # Slice mode for chained-job full-data training: shuffle once with master
        # seed, then take [slice_start, slice_end). Disjoint slices across jobs
        # cover the full dataset exactly once (one true epoch).
        if slice_start is not None or slice_end is not None:
            rng = random.Random(slice_seed)
            rng.shuffle(entries)
            s = slice_start or 0
            e = slice_end if slice_end is not None else len(entries)
            entries = entries[s:e]
        elif max_samples and max_samples < len(entries):
            if stratified:
                entries = stratified_subsample(entries, max_samples, seed=seed)
            else:
                rng = random.Random(seed)
                rng.shuffle(entries)
                entries = entries[:max_samples]

        self.entries = entries
        self.images_dir = images_dir
        self.transform = transform or vlm_image_transform()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.entries[idx]
        img = Image.open(self.images_dir / rec["image_filename"]).convert("RGB")
        img = self.transform(img)
        # Sample one factual_explanation. Use Python's default `random` so each
        # DataLoader worker (fork) gets fresh randomness from its own PID seed.
        fes = rec["factual_explanations"]
        fe = fes[random.randrange(len(fes))] if len(fes) > 1 else fes[0]
        target = format_stage2_target(fe, rec["answer"])
        return {
            "pixel_values": img,
            "system": STAGE2_SYSTEM,
            "user": STAGE2_USER_TEMPLATE.format(question=rec["question"]),
            "target": target,
            "image_filename": rec["image_filename"],
            "question": rec["question"],
            "answer": rec["answer"],
            "factual_explanation": fe,
            "ref_caption": rec["answer"],  # EM compares against raw answer, extracted from generation
        }

    def answer_distribution(self) -> Counter:
        return Counter(e["answer"] for e in self.entries)
