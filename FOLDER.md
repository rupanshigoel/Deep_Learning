# Folder Structure

This is the organization doc for the COL775 Assignment 2 project. Read this first if you're picking up work on this repo.

The project covers all four parts of the assignment (`col775_a2.pdf`):
- **Part A** — CLIP + DINO representation learning on CLEVR (`A2_dataset/Part_A/`).
- **Part Aa** — Linear probing, t-SNE, cross-modal retrieval on the frozen encoders (`A2_dataset/Part_Aa/`).
- **Part B** — LLaVA-style VLM: DINO Teacher + 2-layer MLP projector + Qwen3-4B-Instruct-2507 + LoRA (`A2_dataset/Part_B/CLEVR_X/`).
- **Part C** — VAE + text-conditional Latent Diffusion on CLEVR, FID evaluation.

## Top-level layout

```
COL Assignment 2/
├── A2_dataset/                  CLEVR data (gitignored; provided by course staff)
├── CLAUDE.md                    local behavioral rules for Claude; not submission
├── FOLDER.md                    this file
├── README.md                    project overview + current status
├── col775_a2.pdf                assignment brief
├── col775_a2/                   THE Python package (single source of code)
├── hpc/                         IITD HPC setup scripts + PBS submission files
├── report/                      everything that goes into the final report PDF
├── archive/                     gitignored; non-submission artifacts kept for reference
├── torch25_wheels/              gitignored; pip wheels used to build the HPC env
└── piazza_faqs.txt              course Q&A excerpts (reference)
```

## col775_a2/  —  Python package

**Do not restructure this directory.** The PBS scripts on HPC invoke modules by fully-qualified name (`python -m col775_a2.train_clip`). Moving files would break every queued/running job.

```
col775_a2/
├── __init__.py
├── data/
│   ├── clip_dataset.py          Part A   ClevrCaptionDataset + DedupBatchSampler
│   ├── dino_dataset.py          Part A   DINOMultiCropTransform (10 views/image)
│   ├── ldm_dataset.py           Part C   128×128 Clevr + CLIP-tokenized captions
│   ├── probe_dataset.py         Part Aa  count + color-set JSON loader
│   └── tokenizer.py             Part A   SimpleTokenizer (word-level, from train captions)
├── models/
│   ├── clip_model.py            Part A
│   ├── dino_model.py            Part A
│   ├── vit.py                   Part A/Aa  shared ViT backbone
│   ├── text_encoder.py          Part A/C   frozen CLIP text encoder wrapper
│   ├── vae.py                   Part C
│   └── unet.py                  Part C     conditional U-Net with spatial transformer
├── utils/
│   ├── diffusion.py             Part C     cosine noise schedule + DDPM sampler
│   ├── fid.py                   Part C     clean-fid wrappers
│   ├── losses.py                Part A     contrastive + DINO distillation losses
│   └── scheduler.py             Part A     cosine-warmup LR schedule (also reused for C)
├── train_clip.py                Part A     100-epoch CLIP run, resumable
├── train_dino.py                Part A     100-epoch DINO run, resumable
├── extract_embeddings.py        Part Aa    write CLS+GAP + count/color labels to .npz
├── linear_probe.py              Part Aa    Table 1 (count accuracy + color F1)
├── tsne_plot.py                 Part Aa    70K t-SNE, colored by object count
├── retrieval.py                 Part Aa    CLIP R@1/R@3 both directions + qualitative grid
├── compute_latent_stats.py      Part C     mean/std of un-noised VAE latents over train set
├── train_vae.py                 Part C
├── train_ldm.py                 Part C
├── sample_ldm.py                Part C     DDPM + classifier-free guidance sampler
├── eval_vae_fid.py              Part C     FID for VAE recon + LDM gen
├── data/caption_dataset.py      Part B     Stage-1 (image, caption) dataset + VLMCollator
├── data/clevrx_dataset.py       Part B     Stage-2 CLEVR-X QA with factual_explanation
├── models/vlm.py                Part B     LlavaLikeVLM + MLPProjector + build_vlm
├── train_vlm_stage1.py          Part B     projector-only trainer (captioning)
├── train_vlm_stage2.py          Part B     projector + LoRA trainer (CoT QA)
├── eval_vlm.py                  Part B     unified EM + BLEU + qualitative grids
└── smoke_vlm.py                 Part B     15-min end-to-end HPC sanity (assertion-based)
```

## hpc/  —  HPC setup & submission

```
hpc/
├── hpc_upload.py                local SFTP helper (gitignored; has HPC password fallback)
├── setup/
│   ├── setup_hpc.sh             one-shot HPC environment bootstrap (conda env, dataset symlinks)
│   ├── build_torch25_env.sh     rebuild col775_t25 env (torch 2.5.0+cu118) — matches assignment spec
│   ├── install_partc_deps.sh    Part C extras (clean-fid, transformers, diffusers-adjacent)
│   ├── probe_cuda11.sh          diagnose CUDA 11.x availability on a GPU node
│   ├── environment.yml          conda env spec
│   ├── requirements.txt         pip pins
│   ├── fetch_captions.sh        download programmatic captions into dataset tree
│   ├── download_clip_text_encoder.py   pre-warm HuggingFace cache with openai/clip-vit-base-patch32
│   └── prewarm_cleanfid.py              pre-cache Inception-V3 weights for clean-fid offline runs
└── pbs/
    ├── condaBaseSetup.sh        sourced inside every PBS job to init conda
    ├── proxy.sh                 IITD CNTLM proxy config (Kerberos auth)
    ├── submit_interactive.sh    start an interactive A100 session for debugging
    ├── submit_test.pbs          Part A   CLIP 2-epoch smoke test
    ├── submit_test_dino.pbs     Part A   DINO 2-epoch smoke test
    ├── submit_test_vae.pbs      Part C   VAE smoke test
    ├── submit_clip.pbs          Part A   CLIP full 100-epoch training
    ├── submit_dino.pbs          Part A   DINO full 100-epoch training (resumable)
    ├── submit_probe.pbs         Part Aa  extract → linear_probe → tsne → retrieval
    ├── submit_vae.pbs           Part C   VAE 100-epoch training
    ├── submit_latent_stats.pbs  Part C   compute mean/std of train VAE latents
    ├── submit_ldm.pbs           Part C   LDM 150-epoch training
    ├── submit_vae_fid.pbs       Part C   VAE reconstruction FID
    ├── submit_ldm_fid.pbs       Part C   LDM sampling + FID
    ├── submit_vlm_smoke.pbs     Part B   15-min build + 20 S1 steps + 10 S2 steps + generate
    ├── submit_vlm_stage1.pbs    Part B   MLP projector training on 70K captions
    ├── submit_vlm_stage2.pbs    Part B   Projector + LoRA on CLEVR-X QA
    ├── submit_vlm_eval_s1.pbs   Part B   Stage-1 EM + BLEU (full 15K val)
    └── submit_vlm_eval_s2.pbs   Part B   Stage-2 EM (2.5K val)
```

All PBS scripts activate the `col775_t25` conda env on HPC. See `report/REPORT.md` for how jobs chain.

Sync local PBS edits to HPC with:
```
python hpc/hpc_upload.py hpc/pbs /home/maths/btech/mt1230785/scratch/col775_a2_hpc/hpc
python hpc/hpc_upload.py col775_a2 /home/maths/btech/mt1230785/scratch/col775_a2_hpc/col775_a2
```

## report/

```
report/
├── REPORT.md                    main write-up (will become the submission PDF)
├── part_a/                      Part A + Aa artifacts (populated after submit_probe.pbs completes)
├── part_b/                      Part B VLM artifacts (everything-needed-for-report, self-contained)
│   ├── README.md                numbers, per-category breakdown, analysis, qualitative examples
│   ├── make_plots.py            regenerate loss curves from CSVs (no GPU)
│   ├── logs/                    training CSVs, eval metrics.json, predictions.jsonl, stdout/stderr
│   ├── plots/                   loss_curves.png, stage{1,2}_loss.png, qualitative_*.png (8 grids)
│   └── checkpoints/
│       ├── vlm_stage1_projector.pt          17 MB  projector-only state
│       └── vlm_stage2_lora_final/           submittable LoRA adapter
│           ├── adapter_config.json
│           └── adapter_model.safetensors    46 MB
└── part_c/
    ├── README.md
    ├── PART_C.md                per-section notes
    ├── architecture_vae_ldm.html  design diagram for review
    ├── logs/                    training CSVs + stdout captures
    │   ├── vae_train.csv        vae_fid.out  vae_recon_fid.txt
    │   ├── ldm_train.csv        ldm_fid.out  ldm_gen_fid.txt
    │   └── vae.out ldm.out
    ├── plots/                   loss curves, LR schedules, epoch times, sample grids
    ├── samples/                 gitignored; real / vae_recon / ldm_gen PNGs used to build grids
    ├── checkpoints/             vae_last.pt, ldm_last.pt, latent_stats.pt (submission copies)
    ├── code/                    frozen snapshot of col775_a2 at Part-C submission time
    ├── make_plots.py            regenerate plots/*.png from logs/*.csv
    └── make_sample_grids.py     regenerate *_grid.png from samples/
```

## archive/  —  gitignored, kept for reference

Not needed for submission / report / debugging. Kept to avoid losing context that might help future sessions.

```
archive/
├── HPC_Website/                 saved .htm pages of HPC docs (hardware, PBS, policies, charges)
├── cleanfid_cache/              Inception-V3 FID cache (92 MB, already synced to HPC)
└── hf_cache/                    HuggingFace hub cache for CLIP text encoder (581 MB, already synced to HPC)
```

If you need any of these on HPC again, re-upload with `hpc_upload.py`. Don't re-download on HPC — the IITD proxy is captive-portal-blocked for PyPI/HF/Drive (see memory `col775_t25_env.md`).

## torch25_wheels/  —  gitignored, 4 GB

All the pip wheels needed to rebuild `col775_t25` env (torch 2.5.0+cu118 + torchvision 0.20.0+cu118 + 11 `nvidia-*-cu11` runtime wheels). Kept because IITD proxy blocks PyPI — any env rebuild must install offline from these wheels.

## Where to find common things

| I want to... | Path |
|---|---|
| ...read the assignment spec | `col775_a2.pdf` |
| ...see current state / what's done | `README.md` |
| ...understand the codebase layout | this file |
| ...run a PBS job | `hpc/pbs/*.pbs` |
| ...rebuild the HPC conda env | `hpc/setup/build_torch25_env.sh` |
| ...see training loss curves | `report/part_*/plots/*_loss.png` |
| ...see FID numbers | `report/part_c/logs/*_fid.txt` |
| ...write the report | `report/REPORT.md` |

## Running state (reference time 2026-04-24)

- Parts A, Aa, B, C **all complete**. No jobs running.
- Part B final numbers (see [`report/part_b/README.md`](report/part_b/README.md)):
  - Stage 1 val (full 15K): EM 50.37 %, BLEU 93.10
  - Stage 2 val (2.5K stratified): EM 79.40 %, with color/material/size/shape/yes-no ≥ 89 % and count 52 %
- Only remaining work: render `report/REPORT.md` to PDF for submission.

See `README.md` for the latest status — it's updated as jobs complete.
