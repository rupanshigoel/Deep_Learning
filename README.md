# COL775 A2

CLEVR representation learning + vision-language modeling + text-conditional latent diffusion.
Three parts of the assignment (`col775_a2.pdf`): **A** (CLIP + DINO), **Aa** (analysis of A), **B** (VLM), **C** (VAE + LDM).

**Read [`FOLDER.md`](FOLDER.md) first** — it is the source of truth for where each file lives and what it's for.

**HPC account:** `mt1230785` at IITD HPC · Project `col775.mt1230785.course` · Budget ≈ ₹1000 · Work dir `$HOME/scratch/col775_a2_hpc/`.

## Status (last updated 2026-04-24 20:00 IST)

| Part | Status | Artifacts |
|---|---|---|
| A — CLIP | ✅ Done, 100 epochs (loss 0.0009) | `checkpoints/clip_final.pt` on HPC · `report/part_a/` |
| A — DINO | ✅ Done, 100 epochs (loss 3.142) | `checkpoints/dino_final.pt` on HPC · `report/part_a/` |
| Aa — linear probe (Table 1) | ✅ Done | `report/part_a/README.md` |
| Aa — t-SNE (3 encoders) | ✅ Done | `report/part_a/plots/tsne_*_cls.png` |
| Aa — retrieval R@1/R@3 + qualitative | ✅ Done (i2t 96.03 / 99.25, t2i 96.50 / 99.53) | `report/part_a/plots/retrieval_*.png` |
| B — VLM Stage 1 (DINO-Teacher + MLP projector + frozen Qwen3-4B) | ✅ Done, loss 0.042 (EM 50.37 / BLEU 93.10 on full 15K val) | `report/part_b/checkpoints/vlm_stage1_projector.pt` (17MB) + HPC `vlm_stage1_final.pt` |
| B — VLM Stage 2 (Stage 1 + LoRA r=16 on CoT QA CLEVR-X) | ✅ Done, loss 0.096 (EM 79.4 % on 2.5K val) | `report/part_b/checkpoints/vlm_stage2_lora_final/` (46MB LoRA adapter) |
| C — VAE | ✅ Done, 100 epochs, FID 4.808 | `report/part_c/` + HPC weights |
| C — LDM | ✅ Done, 150 epochs, FID 18.437 | `report/part_c/` + HPC weights |

## Compliance

- **Env spec** (from assignment §Notes): Python 3.10, PyTorch 2.5.0+cu11.8. 
- **HPC env used**: `col775_t25` (conda) with `torch==2.5.0+cu118`, `torchvision==0.20.0+cu118`, matching cu11 nvidia runtime wheels.
- CLIP was originally trained on the old `col775_a2` env (torch 2.3.1+cu121). The checkpoint is API-compatible — verified by loading `clip_final.pt` in torch 2.5 and running a forward pass. Same applies to DINO weights at epoch 77.
- All new jobs (including the current DINO resume) use `col775_t25`.

## How to run

All pipelines run on HPC via PBS. Local-side steps are just file edits + SFTP sync.

**Set up HPC (one-time):**
```
python hpc/hpc_upload.py col775_a2          /home/maths/btech/mt1230785/scratch/col775_a2_hpc/col775_a2
python hpc/hpc_upload.py hpc/pbs            /home/maths/btech/mt1230785/scratch/col775_a2_hpc/hpc
python hpc/hpc_upload.py hpc/setup          /home/maths/btech/mt1230785/scratch/col775_a2_hpc/setup   # first-time only
ssh hpc.iitd.ac.in 'bash $HOME/scratch/col775_a2_hpc/setup/setup_hpc.sh'
ssh hpc.iitd.ac.in 'bash $HOME/scratch/col775_a2_hpc/setup/build_torch25_env.sh'
```

**Submit a job:** edit the PBS file in `hpc/pbs/`, upload, `qsub`.
```
python hpc/hpc_upload.py hpc/pbs /home/maths/btech/mt1230785/scratch/col775_a2_hpc/hpc
ssh hpc.iitd.ac.in 'cd $HOME/scratch/col775_a2_hpc && qsub hpc/submit_dino.pbs'
```

**Chain jobs:** pass `-W depend=afterok:<prev_jobid>.pbshpc` to `qsub`. The probe job does this against DINO.

**Check queue:** `qstat -u mt1230785` — shows live PBS state.

**Cost breakdown:** `high` queue costs 5× `standard`. Use `standard` for anything long (training) and `high` for time-sensitive small jobs (smoke tests, resumes close to deadline). Budget so far: ~₹150 of ₹1000.

## Pending work

- **Report PDF**: rendering the master `report/REPORT.md` to PDF at submission. All Part A/Aa/B/C numbers and plots are frozen.

## Common gotchas (see memory files for details)

- **IITD proxy blocks PyPI and HuggingFace** — all new packages must be downloaded locally and SFTP'd into `torch25_wheels/` or `archive/hf_cache/`.
- **`pip install --no-deps` of torch 2.5 wheel breaks the env** because it leaves CUDA-12 nvidia runtime wheels in place; `hpc/setup/build_torch25_env.sh` does the cu11 swap correctly. One past pip run also *truncated `python3.10` to 0 bytes*; restore by `cp` from a working env.
- **`hpc_upload.py` uses size-based skip.** If two local copies of the same file exist with different contents but the same size, it silently desyncs HPC. Always keep a single source of truth — never duplicate code under `upload_bundle/` or similar. (Historical: `upload_bundle/col775_a2/data/dino_dataset.py` existed with the buggy version and overwrote the working HPC file between DINO jobs 884013 and 884014, triggering the numpy 2.0 ColorJitter crash.)
- **PBS pins the script at `qsub` time.** Editing a submitted script's file has no effect until you `qdel` + resubmit.
