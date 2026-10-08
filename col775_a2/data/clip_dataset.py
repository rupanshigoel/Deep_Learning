"""CLIP training dataset and dedup-aware batch sampler for CLEVR captions.

Key concern (from the assignment):
    "Since no spatial information is used in the captions, many different
    images can share the same caption. If several identical captions appear
    in one contrastive batch, the learning signal becomes less clean. You
    must take this into account while constructing your batches."

We therefore group samples by caption string. Each batch draws at most one
sample per unique caption, guaranteeing the contrastive labels (arange(B))
are truly one-hot in-batch.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
from torchvision import transforms

from .tokenizer import SimpleTokenizer


# Standard ImageNet statistics (CLIP uses its own; these are a reasonable proxy
# and work well for CLEVR which has natural-looking RGB renders).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_train_transform(img_size: int = 224):
    # RandomResizedCrop at scale 0.8-1.0 gives mild random cropping without
    # losing objects at image edges (CLEVR renders mostly occupy the full frame).
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ClevrCaptionDataset(Dataset):
    """Loads (image, caption) pairs from Part_A."""

    def __init__(
        self,
        captions_json: str,
        images_dir: str,
        tokenizer: SimpleTokenizer,
        transform=None,
    ):
        with open(captions_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.transform = transform or default_train_transform()

        # Filter to samples whose image file actually exists (caption JSON may
        # contain slightly more entries than images on disk).
        kept = []
        caption_to_indices: dict[str, List[int]] = {}
        for rec in raw:
            p = self.images_dir / rec["image_filename"]
            if not p.is_file():
                continue
            kept.append({"filename": rec["image_filename"], "caption": rec["caption"]})
            caption_to_indices.setdefault(rec["caption"], []).append(len(kept) - 1)
        self.samples = kept
        self.caption_to_indices = caption_to_indices
        self.unique_captions = list(caption_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img = Image.open(self.images_dir / rec["filename"]).convert("RGB")
        img = self.transform(img)
        return img, rec["caption"]


class DedupBatchSampler(Sampler[List[int]]):
    """Yields batches of dataset indices in which every caption is unique.

    Strategy:
      * Each epoch shuffle the list of unique captions.
      * For each caption, pick ONE index at random from its duplicate group.
      * Then chunk the resulting (shuffled) sequence into batches of `batch_size`.
    This gives an epoch length of `num_unique_captions // batch_size` batches.
    """

    def __init__(
        self,
        dataset: ClevrCaptionDataset,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed
        self._len = len(dataset.unique_captions) // batch_size if drop_last \
            else (len(dataset.unique_captions) + batch_size - 1) // batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        caps = list(self.dataset.unique_captions)
        rng.shuffle(caps)
        picks: List[int] = []
        for cap in caps:
            group = self.dataset.caption_to_indices[cap]
            picks.append(group[rng.randrange(len(group))])
        for i in range(0, len(picks), self.batch_size):
            batch = picks[i : i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch


class clip_collate:
    """Picklable collate callable (Windows spawn-safe).

    Stacks images and tokenizes captions using the provided tokenizer.
    """
    def __init__(self, tokenizer: SimpleTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        caps = [b[1] for b in batch]
        tokens, eot = self.tokenizer.encode_batch(caps)
        return imgs, tokens, eot
