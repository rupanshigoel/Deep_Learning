"""Smoke test for friend's HPC setup. Run after `cd` into friend's anav_help dir."""
import torch
import col775_a2.train_vlm_stage2  # import-only; no training
from col775_a2.data.clevrx_dataset import ClevrXQADataset
from col775_a2.models.vlm import build_vlm  # noqa: F401

print("torch", torch.__version__)
state = torch.load("checkpoints/full/vlm_stage2_last.pt",
                   map_location="cpu", weights_only=False)
print("resume step =", state["step"])
print("keys =", list(state.keys()))

ds = ClevrXQADataset(
    "A2_dataset/Part_B/CLEVR_X/CLEVR_train_explanations_v0.7.10.json",
    "A2_dataset/Part_Aa/Clevr_official/images/train",
    recut_pkl="A2_dataset/Part_B/CLEVR_X/train_images_ids_v0.7.10-recut.pkl",
    slice_start=445452, slice_end=509088, slice_seed=42,
)
print("slice 8 size =", len(ds))
print("first sample fn =", ds.entries[0]["image_filename"])
print("ALL IMPORTS + CHECKPOINT + DATASET SLICE OK")
