"""Pre-download the clean-fid Inception-V3 torchscript weights into
`upload_bundle/cleanfid_cache/`.

On Linux HPC compute nodes, clean-fid's InceptionV3W expects the file at
`/tmp/inception-2015-12-05.pt`. Since compute nodes can't reach the nvlabs
CDN through the proxy, we:

    1. Trigger the download here on Windows (it writes to repo CWD).
    2. Move the file into `upload_bundle/cleanfid_cache/`.
    3. On HPC, `submit_*_fid.pbs` copies this file into `/tmp` before
       running the FID job.

Run from the repo root on Windows:
    python scripts/prewarm_cleanfid.py
"""
from __future__ import annotations
import shutil
from pathlib import Path

import torch


INCEPTION_FILENAME = "inception-2015-12-05.pt"


def main():
    repo = Path(__file__).resolve().parent.parent
    dst = repo / "upload_bundle" / "cleanfid_cache"
    dst.mkdir(parents=True, exist_ok=True)
    dst_file = dst / INCEPTION_FILENAME
    if dst_file.is_file() and dst_file.stat().st_size > 10_000_000:
        print(f"[skip] already cached at {dst_file}  ({dst_file.stat().st_size/1e6:.1f} MB)")
        return

    print("[warmup] loading clean-fid Inception feature extractor (triggers download to CWD) ...")
    from cleanfid.features import build_feature_extractor
    feat = build_feature_extractor(mode="clean", device="cpu")
    # Actually load the JIT module (the returned object is a closure/DataParallel).
    feat(torch.zeros(1, 3, 299, 299))

    # Windows clean-fid writes to CWD.
    src = Path.cwd() / INCEPTION_FILENAME
    if not src.is_file():
        src = repo / INCEPTION_FILENAME     # repo root if CWD != repo
    if not src.is_file():
        raise RuntimeError(f"couldn't find {INCEPTION_FILENAME} after download")

    shutil.move(str(src), str(dst_file))
    print(f"[ok] moved {src.name} -> {dst_file}  ({dst_file.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
