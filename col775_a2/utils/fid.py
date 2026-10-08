"""Thin wrapper around `clean-fid` for Part C evaluation.

Usage:
    from col775_a2.utils.fid import compute_fid_between_folders
    score = compute_fid_between_folders("real_dir", "fake_dir")

Both directories should contain PNG/JPG files at 128x128 (our working
resolution). clean-fid handles its own resize/interpolation internally.
"""
from __future__ import annotations
import os
from pathlib import Path


def compute_fid_between_folders(real_dir: str, fake_dir: str, mode: str = "clean", device: str = "cuda") -> float:
    """Compute FID between two folders of images. Returns a Python float.

    `mode="clean"` uses the standardised bicubic resize + PIL pipeline from
    the clean-fid paper; the default across modern papers.
    """
    from cleanfid import fid as cleanfid  # local import -> no import cost if unused

    assert Path(real_dir).is_dir(), f"no such dir: {real_dir}"
    assert Path(fake_dir).is_dir(), f"no such dir: {fake_dir}"
    score = cleanfid.compute_fid(
        fdir1=real_dir,
        fdir2=fake_dir,
        mode=mode,
        device=device,
        verbose=True,
    )
    return float(score)
