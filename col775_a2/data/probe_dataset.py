"""Part_Aa probe datasets.

The supplied JSONs contain an absolute HPC path in `image_path`. Per the
pinned FAQ, we strip that and resolve against `Part_Aa/Clevr_official/images/<split>/`.

Two tasks:
  * count  (single-label 0..max) — clevr_count_{train,val}.json
  * colors (multi-label 8-d)     — clevr_colors_{train,val}.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image

from .clip_dataset import default_eval_transform


class ProbeDataset(Dataset):
    """Generic loader for probe JSONs. Returns (image_tensor, target)."""

    def __init__(
        self,
        probe_json: str,
        images_root: str,
        task: str,                 # "count" | "colors"
        split: str,                # "train" | "val" (matches image subdir)
        transform=None,
        max_count: int | None = None,  # for 'count' to size the label vocab
    ):
        with open(probe_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.meta = data["metadata"]
        self.examples = data["examples"]
        self.images_dir = Path(images_root) / split
        self.transform = transform or default_eval_transform(224)
        self.task = task
        if task == "count":
            labels = [e["label"] for e in self.examples]
            self.max_count = max_count if max_count is not None else max(labels)
            self.num_classes = self.max_count + 1
        elif task == "colors":
            self.label_vocab: List[str] = self.meta["label_vocab"]
            self.num_classes = len(self.label_vocab)
        else:
            raise ValueError(f"unknown task {task}")

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        img = Image.open(self.images_dir / ex["image_filename"]).convert("RGB")
        img = self.transform(img)
        if self.task == "count":
            target = torch.tensor(int(ex["label"]), dtype=torch.long)
        else:
            target = torch.tensor(ex["multi_hot"], dtype=torch.float32)
        return img, target, ex["image_filename"]


def probe_collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    names = [b[2] for b in batch]
    return imgs, targets, names
