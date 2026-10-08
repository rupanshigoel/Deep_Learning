"""Stage-1 dataset & collate: (image, caption) pairs from Clevr_official.

Captions schema (from `Part_Aa/Clevr_official/captions/clevr_{train,val}_captions.json`):
    [{image_index, image_filename, split, object_count, caption}, ...]

Images live at `Part_Aa/Clevr_official/images/{train,val}/CLEVR_<split>_NNNNNN.png`.

The dataset returns raw Python types; tokenization + image-tag splicing is done in
`VLMCollator` so the tokenizer dependency stays in one place.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .clip_dataset import IMAGENET_MEAN, IMAGENET_STD


def vlm_image_transform(img_size: int = 224):
    """No augmentation — CLEVR captions are content-sensitive."""
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


STAGE1_SYSTEM = "You are a helpful vision-language assistant. Describe the given image factually and concisely."
STAGE1_USER_TEMPLATE = "<<IMAGE>>\nDescribe this image."


class ClevrCaptionVLMDataset(Dataset):
    """Stage-1 (image, caption) dataset. Identical source files to Part A
    retrieval: `clevr_{train,val}_captions.json` under Clevr_official.
    """

    def __init__(
        self,
        captions_json: str,
        images_dir: str,
        transform=None,
        max_samples: Optional[int] = None,
    ):
        with open(captions_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.images_dir = Path(images_dir)
        self.transform = transform or vlm_image_transform()

        # Don't stat 70K files over Lustre — trust the JSON. Bad rows surface at
        # __getitem__ with a clean IOError rather than blocking init for minutes.
        if max_samples:
            raw = raw[:max_samples]
        self.entries = [{"filename": rec["image_filename"], "caption": rec["caption"]} for rec in raw]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.entries[idx]
        img = Image.open(self.images_dir / rec["filename"]).convert("RGB")
        img = self.transform(img)
        return {
            "pixel_values": img,
            "system": STAGE1_SYSTEM,
            "user": STAGE1_USER_TEMPLATE,
            "target": rec["caption"],
            "image_filename": rec["filename"],
            "ref_caption": rec["caption"],
        }


class VLMCollator:
    """Tokenizes prompt halves + target at collate time.

    Splits the rendered chat-template prompt at the `IMAGE_TAG` sentinel so the
    trainer can splice in projected image embeddings at that offset. Returns
    lists of 1D LongTensors (variable-length per sample) plus the stacked
    pixel-values; batching/padding happens inside the VLM's `build_batch`.
    """

    def __init__(self, tokenizer, image_tag: str = "<<IMAGE>>",
                 max_prompt_len: int = 256, max_target_len: int = 512,
                 include_target: bool = True):
        self.tok = tokenizer
        self.image_tag = image_tag
        self.max_prompt_len = max_prompt_len
        self.max_target_len = max_target_len
        self.include_target = include_target

    def _render_prefix(self, system: str, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    def _split_prefix(self, prefix: str) -> Tuple[str, str]:
        assert self.image_tag in prefix, f"IMAGE_TAG {self.image_tag!r} not found in rendered prompt"
        a, b = prefix.split(self.image_tag, 1)
        return a, b

    def __call__(self, batch: List[Dict]):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        pre_ids, post_ids, target_ids = [], [], []
        refs, fnames = [], []

        for b in batch:
            prefix = self._render_prefix(b["system"], b["user"])
            pre_str, post_str = self._split_prefix(prefix)
            pre = self.tok(pre_str, add_special_tokens=False, return_tensors="pt").input_ids[0]
            post = self.tok(post_str, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if pre.numel() + post.numel() > self.max_prompt_len:
                # Shouldn't happen for CLEVR — but defend.
                keep = self.max_prompt_len - post.numel()
                pre = pre[-max(keep, 1):]
            pre_ids.append(pre)
            post_ids.append(post)

            if self.include_target:
                eos = self.tok.eos_token or ""
                tgt_str = f"{b['target']}{eos}"
                tgt = self.tok(tgt_str, add_special_tokens=False, return_tensors="pt").input_ids[0]
                if tgt.numel() > self.max_target_len:
                    tgt = tgt[:self.max_target_len]
                target_ids.append(tgt)

            refs.append(b.get("ref_caption") or b.get("target"))
            fnames.append(b.get("image_filename", ""))

        out = {
            "pixel_values": pixel_values,
            "pre_ids": pre_ids,
            "post_ids": post_ids,
            "ref": refs,
            "image_filename": fnames,
        }
        if self.include_target:
            out["target_ids"] = target_ids
        return out
