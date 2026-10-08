"""Pre-download `openai/clip-vit-base-patch32` into upload_bundle/hf_cache/.

The HPC compute nodes can't reach HuggingFace through the IITD proxy, so we
materialize the model cache locally on Windows and SFTP it up as part of
`upload_bundle/`.

The resulting directory layout is the standard HF hub cache:
    upload_bundle/hf_cache/hub/models--openai--clip-vit-base-patch32/...

The PBS scripts set HF_HOME=$PBS_O_WORKDIR/hf_cache and
TRANSFORMERS_OFFLINE=1 before running training.

Run from the repo root on Windows:
    python scripts/download_clip_text_encoder.py
"""
from __future__ import annotations
import os
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parent.parent
    cache = repo / "upload_bundle" / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)

    from transformers import CLIPTextModel, CLIPTokenizer
    model_id = "openai/clip-vit-base-patch32"
    print(f"[cache] HF_HOME = {cache}")
    print(f"[download] tokenizer ...")
    CLIPTokenizer.from_pretrained(model_id)
    print(f"[download] text model ...")
    CLIPTextModel.from_pretrained(model_id)
    print(f"[ok] cache ready at {cache}")
    print("Contents:")
    for p in sorted(cache.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(cache)}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
