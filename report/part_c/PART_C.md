# Part C — VAE + Latent Diffusion Model (CLEVR text→image)

Written for future Claude sessions. Part A (CLIP + DINO) is in `col775_a2/train_clip.py` and `col775_a2/train_dino.py`; Part C implements the VAE + LDM pipeline on the same Part_A dataset at 128×128 resolution.

## Module layout

```
col775_a2/
├── models/
│   ├── vae.py            # ConvVAE: (3,128,128) <-> (4,16,16) + ResBlock, Upsample, Downsample shared with U-Net
│   ├── unet.py           # ConditionalUNet + SpatialTransformer(self,cross,FFN) + TimeResBlock
│   └── text_encoder.py   # FrozenCLIPTextEncoder (HF transformers, frozen)
├── utils/
│   ├── diffusion.py      # GaussianDiffusion (cosine), q_sample, DDPM+CFG sampler
│   └── fid.py            # clean-fid wrapper
├── data/
│   └── ldm_dataset.py    # ClevrImageOnly / ClevrImageCaption / ClevrCaptionOnly at 128x128, [-1,1]
├── train_vae.py          # >=100 epochs, resumable, CSV logs -> logs/vae_train.csv
├── compute_latent_stats.py    # writes checkpoints/latent_stats.pt
├── train_ldm.py          # 150 epochs, resumable, CSV logs -> logs/ldm_train.csv
├── sample_ldm.py         # DDPM+CFG text->image; also computes FID
└── eval_vae_fid.py       # VAE reconstruction FID

scripts/                  # Run locally on Windows (pre-populate caches for HPC)
├── download_clip_text_encoder.py   # -> upload_bundle/hf_cache/
└── prewarm_cleanfid.py             # -> upload_bundle/cleanfid_cache/

upload_bundle/
├── col775_a2/            # mirror of code (kept in sync with root col775_a2/)
├── hpc/
│   ├── submit_test_vae.pbs     # 1h smoke
│   ├── submit_vae.pbs          # 24h, 100 epochs
│   ├── submit_latent_stats.pbs # 1h
│   ├── submit_ldm.pbs          # 24h, 150 epochs (resubmit 2-3x)
│   ├── submit_vae_fid.pbs      # 2h
│   └── submit_ldm_fid.pbs      # 6h
├── install_partc_deps.sh       # run once on HPC login node to add transformers+cleanfid
├── hf_cache/                    # (gitignored) pre-downloaded CLIP weights
└── cleanfid_cache/              # (gitignored) pre-downloaded Inception weights
```

## Architectures (full spec is in `docs/architecture_vae_ldm.html`)

**VAE**: 3→32→64→128 encoder (3 stride-2 downs), 2-ResBlock mid, split to μ/logvar (4 ch each), reparameterize → z(4,16,16). Decoder 4→128→64→32 (3 nearest 2× upsamples), 2 ResBlocks per stage, final GN-SiLU-Conv(3)-Tanh. SiLU + GroupNorm throughout. Loss: MSE + 1e-6·KL.

**U-Net (LDM)**: channels 64→128→256, attention only at 8×8 and 4×4.
- Down0: init conv(4→64) + 2 TimeResBlock + down. Skip0 @ (64,16,16).
- Down1: 2 TimeResBlock + SpatialTransformer + down. Skip1 @ (128,8,8).
- Down2: 2 TimeResBlock + SpatialTransformer (no down). Skip2 @ (256,4,4).
- Mid: TimeResBlock → SpatialTransformer → TimeResBlock.
- Up0: concat Skip2 → 2 TimeResBlock → SpatialTransformer → up.
- Up1: concat Skip1 → 2 TimeResBlock → SpatialTransformer → up.
- Up2: concat Skip0 → 2 TimeResBlock → GN/SiLU/Conv(→4). No attention.
- Time: sinusoidal(256) → MLP(→256), FiLM-style add after first Conv in each ResBlock.
- Text: frozen CLIP `openai/clip-vit-base-patch32` → (B, 77, 512). K,V for cross-attn.
- Null context: `nn.Parameter(1, 77, 512)`, swapped in per-sample with p=0.1 during training.

**Diffusion**: cosine schedule (Nichol & Dhariwal), T=500. Predict ε. DDPM sampling with CFG scale 4.0 at inference (run U-Net twice per step).

## End-to-end workflow

### One-time local prep (Windows)

```powershell
pip install transformers cleanfid
python scripts/download_clip_text_encoder.py   # populates upload_bundle/hf_cache/
python scripts/prewarm_cleanfid.py             # populates upload_bundle/cleanfid_cache/
```

### Upload to HPC

```powershell
python hpc_upload.py upload_bundle /home/maths/btech/mt1230785/scratch/col775_a2_hpc
```

### One-time HPC prep (login node)

```bash
cd ~/scratch/col775_a2_hpc
bash install_partc_deps.sh        # adds transformers + cleanfid to col775_t25
```

### Train + eval (sequential; each step depends on the previous)

```bash
qsub hpc/submit_test_vae.pbs      # 1h smoke (optional)
qsub hpc/submit_vae.pbs           # 24h, 100 epochs VAE
qsub hpc/submit_latent_stats.pbs  # 1h, needs vae_last.pt
qsub hpc/submit_ldm.pbs           # 24h; re-submit 2-3x until 150 epochs done
qsub hpc/submit_vae_fid.pbs       # 2h VAE recon FID on 10k val
qsub hpc/submit_ldm_fid.pbs       # 6h LDM sample + FID on 10k val captions
```

Monitor with `qstat -u mt1230785`; logs stream to `logs/*.out`.

### Pull results back (Windows)

```bash
scp mt1230785@hpc.iitd.ac.in:~/scratch/col775_a2_hpc/logs/{vae,ldm,vae_fid,ldm_fid}.out .
scp mt1230785@hpc.iitd.ac.in:~/scratch/col775_a2_hpc/logs/{vae,ldm}_train.csv .
scp mt1230785@hpc.iitd.ac.in:~/scratch/col775_a2_hpc/samples/ldm_val/fid.txt .
scp mt1230785@hpc.iitd.ac.in:~/scratch/col775_a2_hpc/samples/vae_recon_val/fid.txt .
scp mt1230785@hpc.iitd.ac.in:~/scratch/col775_a2_hpc/checkpoints/{vae_last,ldm_last,latent_stats}.pt .
# sample grids (tar first for convenience)
ssh mt1230785@hpc.iitd.ac.in "cd ~/scratch/col775_a2_hpc/samples && tar czf samples.tgz ldm_val vae_recon_val"
scp mt1230785@hpc.iitd.ac.in:~/scratch/col775_a2_hpc/samples/samples.tgz .
```

## Key design notes

- **λ=1e-6**: per assignment spec. With MSE on Tanh-bounded output and near-standard Gaussian posteriors, KL ≈ 10²–10³ and MSE ≈ 10⁻³; λ ~1e-6 keeps reconstruction dominant, which is exactly the intent.
- **Latent normalization**: `z = (μ_vae - mean) / std` before LDM training (per-channel, computed over full train set in `compute_latent_stats.py`). De-normalized at inference before decoding.
- **Zero-init** on the `proj_out` of SpatialTransformer and on the final `out_conv` of the U-Net — starts the network as near-identity noise-predictor, stabilises early training.
- **Per-sample CFG dropout** (not per-batch): `mask = torch.rand(B) < 0.1` then `torch.where(mask, null, ctx)`. Matches Ho & Salimans.
- **Eval resolution**: FID between `val/images` (resized to 128) and generated/reconstructed 128×128 PNGs using `clean-fid` in `clean` mode.

## Known gotchas

- **transformers + torch 2.5**: some transformers releases pin torch<2.5; we cap `<4.50` which covers 4.47.1 (works) — tested locally.
- **HPC proxy blocks HF/Drive**: that's why `hf_cache` is bundled and `TRANSFORMERS_OFFLINE=1` is set in the PBS job.
- **clean-fid first-run download**: blocked on compute nodes; bundle is symlinked into `~/.cache/clean-fid` at the start of each FID job.
- **`nn.MultiheadAttention` + fp16 autocast**: works on A100 but the kdim/vdim variant used in cross-attn pre-allocates separate weight matrices — memory is fine at these resolutions.
- **DDPM sampling is slow** at 500 steps × 2 (CFG) × 10k images: ~5h budget on A100 with bs=16. If FID ends up needing iteration, `sample_ldm.py --num-samples N` lets you iterate on small sets.

## Verification steps

Before submitting big jobs, run the smoke tests from the repo root:

```bash
python -m col775_a2.train_vae --epochs 1 --batch-size 4 --num-workers 0
python -m col775_a2.compute_latent_stats --batch-size 8 --num-workers 0
python -m col775_a2.train_ldm --epochs 1 --batch-size 2 --num-workers 0
python -m col775_a2.sample_ldm --num-samples 2 --batch-size 2 --progress
```

(Each needs the previous step's checkpoint.) The `progress` flag in the LDM sampler shows a tqdm over diffusion timesteps — useful for catching immediate errors.

## Cost estimate (from project budget, ₹1 per GPU-hour)

| Step | Hours | Cost |
|------|-------|------|
| VAE (100 epochs) | ~24 | ₹24 |
| Latent stats | ~0.3 | <₹1 |
| LDM (150 epochs, 3 submissions) | ~66 | ₹66 |
| VAE FID | ~0.5 | <₹1 |
| LDM FID (10k × 500 × CFG) | ~5 | ₹5 |
| **Total** | **~96** | **~₹97** |

Within the ₹1000 budget with plenty of headroom for re-runs.
