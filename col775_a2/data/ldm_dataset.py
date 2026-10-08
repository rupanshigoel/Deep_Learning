"""Datasets for Part C (VAE + LDM).

Resizes CLEVR images on the fly to 128x128 and normalises them to [-1, 1]
(Tanh-aligned for the VAE decoder output). Captions come from the same
`clevr_{train,val}_captions.json` files as Part A.

Three dataset flavours:
    * `ClevrImageOnly`         -> image tensor only (VAE training).
    * `ClevrImageCaption`      -> (image, caption_str) pair (LDM training).
    * `ClevrCaptionOnly`       -> caption strings + filenames (LDM sampling).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


IMG_SIZE = 128


def default_image_transform(img_size: int = IMG_SIZE):
    """Resize->CenterCrop->ToTensor->Normalize to [-1, 1] (Tanh range)."""
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def _load_caption_records(captions_json: str, images_dir: Path) -> list[dict]:
    """Load caption JSON and filter to records whose image is on disk."""
    with open(captions_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    kept: list[dict] = []
    for rec in raw:
        fn = rec["image_filename"]
        if (images_dir / fn).is_file():
            kept.append({"filename": fn, "caption": rec["caption"]})
    return kept


class ClevrImageOnly(Dataset):
    """Images at 128x128 (for VAE training)."""

    def __init__(self, captions_json: str, images_dir: str, img_size: int = IMG_SIZE):
        self.images_dir = Path(images_dir)
        self.samples = _load_caption_records(captions_json, self.images_dir)
        self.transform = default_image_transform(img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec = self.samples[idx]
        img = Image.open(self.images_dir / rec["filename"]).convert("RGB")
        return self.transform(img)


class ClevrImageCaption(Dataset):
    """(image at 128x128, raw caption str) pairs (for LDM training).

    Text tokenisation is done inside the frozen CLIP text encoder; we simply
    pass the string through.
    """

    def __init__(self, captions_json: str, images_dir: str, img_size: int = IMG_SIZE):
        self.images_dir = Path(images_dir)
        self.samples = _load_caption_records(captions_json, self.images_dir)
        self.transform = default_image_transform(img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        rec = self.samples[idx]
        img = Image.open(self.images_dir / rec["filename"]).convert("RGB")
        return self.transform(img), rec["caption"]


class ClevrCaptionOnly(Dataset):
    """(filename, caption) pairs without reading images (for LDM sampling)."""

    def __init__(self, captions_json: str, images_dir: str):
        self.images_dir = Path(images_dir)
        self.samples = _load_caption_records(captions_json, self.images_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        rec = self.samples[idx]
        return rec["filename"], rec["caption"]


def image_caption_collate(batch):
    """Stack images and keep captions as a list of strings."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    caps = [b[1] for b in batch]
    return imgs, caps


def caption_only_collate(batch):
    fns = [b[0] for b in batch]
    caps = [b[1] for b in batch]
    return fns, caps
