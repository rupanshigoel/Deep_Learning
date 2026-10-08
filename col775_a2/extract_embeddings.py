"""Extract frozen features from CLIP / DINO-student / DINO-teacher backbones
for Part_Aa probe datasets (count + colors, both splits).

For each (encoder, split) we cache ONE shared image-order feature file
containing both CLS (384-d) and GAP-over-patches (384-d) representations,
plus aligned labels for both tasks. Later probing/t-SNE/retrieval just
load these `.npy`/`.npz` files.

Output layout:
    embeddings/<encoder>_<split>.npz
        cls        (N, 384)
        gap        (N, 384)
        count      (N,)          int, object-count label
        colors     (N, 8)        float32 multi-hot
        filenames  (N,)          list[str]

encoder ∈ {clip, dino_student, dino_teacher}
split   ∈ {train, val}

Usage:
    python -m col775_a2.extract_embeddings
        --clip-ckpt col775_a2/checkpoints/clip_last.pt
        --dino-ckpt col775_a2/checkpoints/dino_last.pt
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from col775_a2.data.clip_dataset import default_eval_transform
from col775_a2.models.vit import VisionTransformer
from col775_a2.models.clip_model import CLIPModel
from col775_a2.models.dino_model import build_dino


ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_Aa"
PROBE_DIR = ROOT / "Probe-Datasets"
IMG_ROOT = ROOT / "Clevr_official" / "images"


class _JointProbeDataset(Dataset):
    """Reads count+colors JSONs for a split, merges them by image_filename so
    we can iterate each image exactly once and store both labels aligned.
    Missing labels (if files diverge) are filled with -1 / zeros."""

    def __init__(self, split: str, transform):
        count_js = PROBE_DIR / f"clevr_count_{split}.json"
        colors_js = PROBE_DIR / f"clevr_colors_{split}.json"
        with open(count_js, "r", encoding="utf-8") as f: cd = json.load(f)
        with open(colors_js, "r", encoding="utf-8") as f: kd = json.load(f)
        count_map = {e["image_filename"]: int(e["label"]) for e in cd["examples"]}
        color_map = {e["image_filename"]: e["multi_hot"] for e in kd["examples"]}
        self.color_vocab = kd["metadata"]["label_vocab"]
        files = [e["image_filename"] for e in cd["examples"]]  # canonical order
        for f in files:
            if f not in color_map: color_map[f] = [0] * len(self.color_vocab)
        self.files = files
        self.counts = np.array([count_map[f] for f in files], dtype=np.int64)
        self.colors = np.array([color_map[f] for f in files], dtype=np.float32)
        self.img_dir = IMG_ROOT / split
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / self.files[idx]).convert("RGB")
        return self.transform(img), idx


@torch.no_grad()
def _extract(backbone: VisionTransformer, split: str, device: str, batch_size: int, workers: int):
    backbone.eval()
    ds = _JointProbeDataset(split, default_eval_transform(224))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    cls_all = np.zeros((len(ds), backbone.dim), dtype=np.float32)
    gap_all = np.zeros((len(ds), backbone.dim), dtype=np.float32)
    seen = 0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = backbone(imgs)
        cls = out["cls"].float().cpu().numpy()
        gap = out["patch_tokens"].float().mean(dim=1).cpu().numpy()
        B = cls.shape[0]
        cls_all[seen:seen+B] = cls
        gap_all[seen:seen+B] = gap
        seen += B
        if seen % (batch_size * 20) == 0 or seen == len(ds):
            print(f"    [{split}] {seen}/{len(ds)}")
    return cls_all, gap_all, ds


def _save_split(out_dir: Path, tag: str, split: str, cls_arr, gap_arr, ds: _JointProbeDataset):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_{split}.npz"
    np.savez_compressed(
        out_path,
        cls=cls_arr, gap=gap_arr,
        count=ds.counts, colors=ds.colors,
        filenames=np.array(ds.files),
        color_vocab=np.array(ds.color_vocab),
    )
    print(f"  wrote {out_path}  cls={cls_arr.shape}  gap={gap_arr.shape}")


def _load_clip_backbone(ckpt_path: Path, device: str) -> VisionTransformer:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = state.get("args", {})
    vocab_size = args.get("vocab_size", None)
    if vocab_size is None:
        # fall back to tokenizer.json alongside the ckpt
        tok_json = ckpt_path.parent / "tokenizer.json"
        with open(tok_json, "r", encoding="utf-8") as f: vocab_size = len(json.load(f)["vocab"])
    ctx = args.get("context_length", 64); img = args.get("img_size", 224)
    model = CLIPModel(vocab_size=vocab_size, context_length=ctx, img_size=img)
    model.load_state_dict(state["model"])
    model = model.to(device)
    return model.visual


def _load_dino_backbones(ckpt_path: Path, device: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = state.get("args", {})
    models = build_dino(img_size=args.get("img_size", 224), out_dim=args.get("out_dim", 4096))
    models["student"].load_state_dict(state["student"])
    models["teacher"].load_state_dict(state["teacher"])
    models = models.to(device)
    return models["student"].backbone, models["teacher"].backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-ckpt", default=str(Path(__file__).resolve().parent / "checkpoints" / "clip_last.pt"))
    ap.add_argument("--dino-ckpt", default=str(Path(__file__).resolve().parent / "checkpoints" / "dino_last.pt"))
    ap.add_argument("--out-dir",   default=str(Path(__file__).resolve().parent / "embeddings"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--skip", nargs="*", default=[], help="Encoders to skip: clip dino_student dino_teacher")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)

    encoders: Dict[str, VisionTransformer] = {}
    clip_ckpt = Path(args.clip_ckpt)
    dino_ckpt = Path(args.dino_ckpt)

    if "clip" not in args.skip and clip_ckpt.is_file():
        print(f"[load] CLIP backbone from {clip_ckpt}")
        encoders["clip"] = _load_clip_backbone(clip_ckpt, device)
    elif "clip" not in args.skip:
        print(f"[warn] CLIP ckpt not found at {clip_ckpt}, skipping")

    if dino_ckpt.is_file():
        s, t = _load_dino_backbones(dino_ckpt, device)
        if "dino_student" not in args.skip:
            print(f"[load] DINO student from {dino_ckpt}"); encoders["dino_student"] = s
        if "dino_teacher" not in args.skip:
            print(f"[load] DINO teacher from {dino_ckpt}"); encoders["dino_teacher"] = t
    else:
        print(f"[warn] DINO ckpt not found at {dino_ckpt}, skipping DINO")

    for tag, bb in encoders.items():
        for split in args.splits:
            out_path = out_dir / f"{tag}_{split}.npz"
            if out_path.is_file():
                print(f"[skip] {out_path} exists")
                continue
            print(f"[extract] encoder={tag}  split={split}")
            cls_a, gap_a, ds = _extract(bb, split, device, args.batch_size, args.num_workers)
            _save_split(out_dir, tag, split, cls_a, gap_a, ds)


if __name__ == "__main__":
    main()
