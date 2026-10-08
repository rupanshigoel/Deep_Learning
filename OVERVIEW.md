# COL775 Assignment 2 — Code Walkthrough

This zip contains the full source for my COL775 (IIT Delhi, Spring 2026) Assignment 2
solution — CLIP + DINO representation learning, a LLaVA-style VLM, and a text-conditional
Latent Diffusion Model on CLEVR.

> **Reading order:** start with `col775_a2.pdf` (assignment brief) → this file →
> `FOLDER.md` (repo map) → `report/REPORT.md` (master write-up) → per-part READMEs.

## What's in the zip

| Path | What it is |
|---|---|
| `col775_a2.pdf` | Original assignment brief from the course staff |
| `OVERVIEW.md` | This file |
| `README.md` | Top-level project status (final numbers, how to run) |
| `FOLDER.md` | Source-of-truth map of every file in the repo |
| `col775_a2/` | **The Python package — all training/eval/data code** |
| `hpc/` | IITD HPC env setup + PBS submission scripts |
| `report/` | Final write-up (Markdown + plots + CSV training logs) |
| `piazza_faqs.txt`, `piazza_faqs_b_c.txt` | Course Q&A excerpts that disambiguated the spec |

## What's intentionally NOT in the zip

- **Trained checkpoints** (`*.pt`, LoRA `.safetensors`) — total ~290 MB. The code is
  self-contained; rerun training to reproduce, or ping me for the weights separately.
- **`A2_dataset/`** — CLEVR images/captions provided by course staff. ~10 GB.
- **`torch25_wheels/`** — 4 GB of pip wheels for offline HPC env build.
- **`archive/`, `__pycache__/`, `*_full/` log duplicates** — not needed.

## Assignment in one paragraph

CLEVR is a synthetic dataset of rendered scenes with 3–10 objects (8 colors, 2 sizes,
2 materials, 3 shapes). The assignment has four parts:

- **Part A — Representation learning.** Train a CLIP-style image–text contrastive model
  *and* a DINO self-distillation model (both with a shared ViT backbone) on CLEVR from
  scratch, 100 epochs each.
- **Part Aa — Probing the encoders.** Linear probe for object count + color set, t-SNE,
  and CLIP image↔text retrieval on the held-out val split.
- **Part B — Vision-language model.** LLaVA-style: frozen DINO image tower → 2-layer MLP
  projector → frozen Qwen3-4B-Instruct decoder. Stage 1 trains only the projector on
  image captioning; Stage 2 adds a LoRA adapter (r=16) and fine-tunes on CLEVR-X CoT QA.
- **Part C — Generative.** A small convolutional VAE (3.25 M params) that compresses
  128×128 → 16×16×4 latents, then a 16 M-param U-Net latent diffusion model conditioned
  on frozen CLIP text embeddings, trained with classifier-free guidance.

## Headline results (everything done; see `report/REPORT.md` for full tables)

| Part | Metric | Value |
|---|---|---|
| A — CLIP | final contrastive loss | 0.0009 |
| A — DINO | final distillation loss | 3.142 |
| Aa | linear probe count acc / color F1 | (full table in `report/part_a/README.md`) |
| Aa | retrieval R@1 / R@3 (i2t) | 96.03 / 99.25 |
| Aa | retrieval R@1 / R@3 (t2i) | 96.50 / 99.53 |
| B — Stage 1 | EM / BLEU on full 15 K val | 50.37 % / 93.10 |
| B — Stage 2 | EM on 2.5 K stratified val | 79.40 % |
| C — VAE | reconstruction FID (10 K val) | 4.808 |
| C — LDM | generation FID (500-step DDPM, CFG=4) | 18.437 |

## Codebase tour — the 30-second version

The whole training stack lives in `col775_a2/` and is a single Python package. PBS
scripts on HPC invoke modules by fully-qualified name (`python -m col775_a2.train_clip`),
so the layout is fixed.

```
col775_a2/
├── data/         dataset + transform classes (one per part)
├── models/       network definitions (ViT, CLIP, DINO, VAE, U-Net, VLM, ...)
├── utils/        losses, LR schedulers, FID, diffusion math
├── train_*.py    one entry-point per training run
├── eval_*.py     evaluation entry-points
└── *_plot.py / extract_*.py    Part Aa probing scripts
```

`FOLDER.md` has the per-file purpose for every script. `report/REPORT.md` documents
*why* each design choice was made (architecture, optimizer settings, augmentation
recipe, etc.) — that's the doc to read for "implementation details".

## How to run (high level)

Everything was trained on a single A100 (40 GB) on the IIT Delhi HPC under a PBS
queue. The conda env is reproducible from `hpc/setup/build_torch25_env.sh`
(torch 2.5.0+cu118, the version mandated by the assignment). PBS scripts in
`hpc/pbs/` show the exact command + resource request for each training run.

If you just want to read the code: every file under `col775_a2/` is plain PyTorch,
no framework magic — a `python -m col775_a2.train_xxx --help` would show all CLI
flags.

## A few non-obvious design decisions worth flagging

These are written up in detail in `report/REPORT.md` and `report/part_b/design_choices.md`
but the headlines:

1. **DINO multi-crop has 10 views/image** (2 global + 8 local) instead of the canonical
   2+6 — gave slightly better CLEVR linear-probe transfer.
2. **CLIP uses a dedup batch sampler** so two captions of the same image never collide
   in a single contrastive batch (CLEVR has many caption variants per image).
3. **VLM Stage 1 freezes the LLM and image tower entirely** — only the 2-layer MLP
   projector is trained (~13 M params). This was deliberately cheaper than full
   fine-tuning and the EM/BLEU numbers show it generalises fine.
4. **VLM Stage 2 LoRA r=16, α=32** on Qwen3-4B's `q_proj`/`k_proj`/`v_proj`/`o_proj`.
5. **LDM uses cosine noise schedule + 500-step DDPM at sampling** with classifier-free
   guidance scale 4. The latents are normalised by per-channel mean/std computed once
   on the train set (`compute_latent_stats.py`).

## Contact

If you find a bug or want any of the trained checkpoints, ping me on Piazza /
email — happy to send things over.
