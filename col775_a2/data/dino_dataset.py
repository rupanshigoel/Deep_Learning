"""DINO multi-crop dataset + augmentation pipeline (Caron et al. 2021).

Per image we yield a list of 10 tensors:
    * 2 global crops at 224x224 (scale 0.4-1.0 of original area)
    * 8 local crops at 96x96   (scale 0.05-0.4)

Each crop gets independent random augmentations: color jitter,
random grayscale, Gaussian blur; solarization is only applied to
global crop #2; horizontal flip on all.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from .clip_dataset import IMAGENET_MEAN, IMAGENET_STD


class GaussianBlur:
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        self.p = p; self.radius_min = radius_min; self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            r = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        return img


class Solarize:
    def __init__(self, p: float = 0.2): self.p = p

    def __call__(self, img):
        return ImageOps.solarize(img) if random.random() < self.p else img


def _color_jitter():
    return transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8,
    )


def _normalize():
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


class DINOMultiCropTransform:
    def __init__(
        self,
        global_size: int = 224,
        local_size: int = 96,
        n_local_crops: int = 8,
        global_scale=(0.4, 1.0),
        local_scale=(0.05, 0.4),
    ):
        self.n_local_crops = n_local_crops

        # PIL-side: flip + custom GaussianBlur/Solarize (operate on PIL.Image).
        # Tensor-side: color jitter + grayscale + normalize. ColorJitter is run
        # after ToTensor so it takes torchvision's tensor code path, avoiding
        # `np.uint8(hue_factor * 255)` which overflows on numpy >= 2.0 when
        # hue_factor is negative.
        flip = transforms.RandomHorizontalFlip(p=0.5)
        tensor_color = transforms.Compose([
            transforms.ToTensor(),
            _color_jitter(),
            transforms.RandomGrayscale(p=0.2),
            _normalize(),
        ])

        self.global1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            flip, GaussianBlur(p=1.0), tensor_color,
        ])
        self.global2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=global_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            flip, GaussianBlur(p=0.1), Solarize(p=0.2), tensor_color,
        ])
        self.local = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_scale,
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            flip, GaussianBlur(p=0.5), tensor_color,
        ])

    def __call__(self, img) -> List[torch.Tensor]:
        crops = [self.global1(img), self.global2(img)]
        for _ in range(self.n_local_crops):
            crops.append(self.local(img))
        return crops


class ClevrImageOnlyDataset(Dataset):
    """Image-only dataset for DINO. Uses captions JSON only for the file list
    (image_filename entries). Falls back to listing the image dir if no JSON."""

    def __init__(self, captions_json: str | None, images_dir: str, transform):
        self.images_dir = Path(images_dir)
        self.transform = transform
        if captions_json and Path(captions_json).is_file():
            with open(captions_json, "r", encoding="utf-8") as f:
                raw = json.load(f)
            files = [r["image_filename"] for r in raw]
            files = [f for f in files if (self.images_dir / f).is_file()]
        else:
            files = sorted(p.name for p in self.images_dir.iterdir()
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        # Deduplicate while preserving order.
        seen = set(); uniq = []
        for f in files:
            if f in seen: continue
            seen.add(f); uniq.append(f)
        self.files = uniq

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.images_dir / self.files[idx]).convert("RGB")
        return self.transform(img)


def dino_collate(batch):
    """`batch` is a list (len B) of lists of crop tensors (len n_crops).
    Return a list (len n_crops) of stacked tensors, each (B, 3, H, W).
    """
    n_crops = len(batch[0])
    return [torch.stack([sample[c] for sample in batch], dim=0) for c in range(n_crops)]
