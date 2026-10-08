# COL775 Assignment 2 — Part C: VAE + Latent Diffusion Model

**Entry No.:** mt1230785  
**Dataset:** Part_A CLEVR (100 032 train / 10 000 val) at **128 × 128**  
**Hardware:** single NVIDIA A100 40 GB (IITD HPC, `col775_t25` env, torch 2.5.0+cu118, FP16 autocast)

---

## 1. Overview

Two generative models trained sequentially:

1. **Convolutional Variational Autoencoder (VAE)** that maps a 128 × 128 × 3 image to a 16 × 16 × 4 continuous latent.
2. **Conditional Latent Diffusion Model (LDM)** — a U-Net denoiser operating on normalised VAE latents, conditioned on frozen `openai/clip-vit-base-patch32` text embeddings with classifier-free guidance.

### Headline results

| Metric | Value |
|---|---|
| **VAE reconstruction FID** (10 000 val) | **4.808** |
| **LDM generation FID** (10 000 val, 500-step DDPM, CFG s=4) | **18.437** |
| VAE params | 3.25 M |
| LDM U-Net params | 16.18 M |
| VAE final recon MSE | 1.54 × 10⁻⁴ |
| LDM final ε-MSE (train / val) | 0.126 / 0.134 |
| Total Part C GPU time | ~13 h on a single A100 |

All code lives in `col775_a2/` (see [docs/PART_C.md](../docs/PART_C.md) for module layout).

---

## 2. VAE

### 2.1 Architecture  (3.25 M parameters)

| Block | Detail | Output |
|---|---|---|
| Input                | — | (B, 3, 128, 128) |
| Encoder init conv    | Conv2d 3×3 (3→32) | (B, 32, 128, 128) |
| Encoder stage 1      | 2 × ResBlock + Downsample (s=2) | (B, 32, 64, 64) |
| Encoder stage 2      | 2 × ResBlock(32→64) + Downsample | (B, 64, 32, 32) |
| Encoder stage 3      | 2 × ResBlock(64→128) + Downsample | (B, 128, 16, 16) |
| Mid                  | 2 × ResBlock + Conv 3×3 (128→8) | (B, 8, 16, 16) |
| **Reparameterise**   | split → μ (4), logvar (4); z = μ + σ·ε | (B, 4, 16, 16) |
| Decoder init conv    | Conv2d 3×3 (4→128) | (B, 128, 16, 16) |
| Decoder mid          | 2 × ResBlock | (B, 128, 16, 16) |
| Decoder stage 1      | Upsample(nearest 2×)+Conv + 2 ResBlock | (B, 128, 32, 32) |
| Decoder stage 2      | Upsample+Conv + 2 ResBlock(128→64) | (B, 64, 64, 64) |
| Decoder stage 3      | Upsample+Conv + 2 ResBlock(64→32) | (B, 32, 128, 128) |
| Output               | GroupNorm → SiLU → Conv 3×3 (32→3) → Tanh | (B, 3, 128, 128) |

Every ResBlock: `GN → SiLU → Conv3x3 → GN → SiLU → Conv3x3 + (identity or 1×1-conv skip)`. Activations are SiLU throughout, normalisation is GroupNorm with `num_groups = min(32, ch)`. Inputs are normalised to `[-1, 1]` for consistency with the Tanh output.

### 2.2 Training

| Setting | Value |
|---|---|
| Optimiser | AdamW(β₁=0.9, β₂=0.999, ε=1e-8), lr 1e-4, wd 0 |
| LR schedule | 1 epoch linear warmup + cosine decay to 0 |
| Batch size | 64 |
| Epochs | 100 (resumable) |
| Loss | `MSE(x̂, x) + 1e-6 · KL`, per-element mean |
| AMP | FP16 autocast, GradScaler + grad-clip 1.0 |
| Time | ~2.3–2.4 min/ep on A100 → **≈ 4.1 h total** |

### 2.3 Loss trajectory

![VAE loss and KL](plots/vae_loss.png)

Reconstruction MSE drops **from 1.56 × 10⁻² (ep 0) to 1.54 × 10⁻⁴ (ep 99)** — two orders of magnitude in 100 epochs. Per-element KL stabilises around 5.1 (un-scaled). With λ = 1e-6 the KL contributes ~5 × 10⁻⁶ to the total loss, i.e. the posterior is allowed to be informative while remaining well-behaved.

![VAE LR schedule](plots/vae_lr.png)

### 2.4 Reconstruction quality

![VAE reconstruction grid](plots/vae_recon_grid.png)

Eight validation images at 128 × 128, top row = ground truth, bottom row = VAE reconstruction. Colours, shapes and material cues (metal specular highlights vs rubber matte) are preserved faithfully.

### 2.5 FID (clean-fid, 10 000 val images)

> **FID (real vs VAE reconstruction) = 4.808**

Computed via `cleanfid.fid.compute_fid(mode="clean")` between
`samples/real_val_128/` (val images resized to 128 via bicubic + centre-crop) and
`samples/vae_recon_val/` (the decoder outputs). Under 5 FID at 128² indicates a nearly-lossless latent space — adequate for diffusion to learn.

---

## 3. Latent Diffusion Model (Conditional U-Net)

### 3.1 Latent normalisation

Before training the LDM we encode the full 100 032 train images through the frozen VAE and compute per-channel mean / std of the posterior mean μ:

```
per-channel mean : [-0.4789, -0.0505,  0.5351,  0.4812]
per-channel std  : [ 1.0543,  0.5581,  0.8482,  0.3834]
```

These are written to `checkpoints/latent_stats.pt`. LDM training normalises `z = (μ_vae − mean) / std`; inference de-normalises before the VAE decode.

### 3.2 Architecture  (16.18 M parameters)

U-Net with channel ladder **64 → 128 → 256**, self+cross+FFN Spatial Transformers at the 8 × 8 and 4 × 4 resolutions only (not at 16 × 16 to save params).

| Stage | Op | Output |
|---|---|---|
| Down 0 | Init Conv(4→64) → 2 × TimeResBlock → Down s=2 | (B, 64, 8, 8) — Skip 0 @ (64, 16, 16) |
| Down 1 | 2 × TimeResBlock → SpatialTransformer → Down | (B, 128, 4, 4) — Skip 1 @ (128, 8, 8) |
| Down 2 | 2 × TimeResBlock → SpatialTransformer (no down) | (B, 256, 4, 4) — Skip 2 |
| Mid    | TimeResBlock → SpatialTransformer → TimeResBlock | (B, 256, 4, 4) |
| Up 0   | Concat Skip 2 → 2 × TimeResBlock → SpatialT → Up+Conv | (B, 256, 8, 8) |
| Up 1   | Concat Skip 1 → 2 × TimeResBlock → SpatialT → Up+Conv | (B, 128, 16, 16) |
| Up 2   | Concat Skip 0 → 2 × TimeResBlock → Output Conv(→4) | (B, 4, 16, 16) |

**Timestep embedding**: sinusoidal (dim 256) → MLP `256 → 1024 → 256`; added via a per-block `Linear(256, out_ch)` after the first Conv of every ResBlock (FiLM-style).

**Text conditioning**: frozen `CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")`, `last_hidden_state` → (B, 77, 512). Each SpatialTransformer does:
```
LN → SelfAttn(Q=K=V=image tokens, 8 heads) → +
LN → CrossAttn(Q=image, K=V=text, 8 heads) → +
LN → FFN (4× expansion) → +
```
with 1×1 conv project-in / project-out; the project-out is **zero-initialised** so the block starts as an identity (Stable-Diffusion trick for stable early training). The output conv of the full U-Net is likewise zero-initialised.

**Classifier-free guidance**. A learned null context `nn.Parameter(1, 77, 512)` replaces the text embedding with probability 0.1 **per sample** during training. At inference we combine
```
ε̂ = (1 + s) · ε_cond − s · ε_uncond ,   s = 4.0
```
over all 500 DDPM steps.

### 3.3 Diffusion process

* Cosine β-schedule (Nichol & Dhariwal 2021, `s = 0.008`) with **T = 500** steps, pre-computed.
* Forward: `q(z_t | z_0) = N(√ᾱ_t · z_0, (1−ᾱ_t) · I)`.
* Reverse: DDPM ancestral sampling (Ho et al. 2020, Alg. 2), variance β_t.

### 3.4 Training

| Setting | Value |
|---|---|
| Optimiser | AdamW(β₁=0.9, β₂=0.999), lr 1e-4 |
| Weight decay | 0 on LDM, 0 on null context |
| LR schedule | 1 epoch linear warmup + cosine to 0 |
| Batch size | 64 |
| Epochs | 150 |
| Loss | `MSE(ε̂, ε)` on normalised latents |
| Null-dropout | 10 % of samples per batch |
| AMP | FP16 autocast, grad-clip 1.0 |
| Time | ~2.74 min/ep → **≈ 6.9 h total** (1 PBS session, 24 h walltime) |

### 3.5 Loss trajectory

![LDM loss](plots/ldm_loss.png)

ε-prediction MSE drops from 0.468 (ep 0) to **0.126 train / 0.134 val** at ep 149. The val curve tracks the train curve tightly — no signs of overfitting, expected since CLEVR captions are strongly determined and the model is modestly sized relative to 100 K samples.

![LDM LR schedule](plots/ldm_lr.png)

### 3.6 Qualitative samples

Three-way comparison on 6 val captions (top → Real, middle → VAE-only reconstruction, bottom → LDM text-conditional generation, 500 DDPM steps, CFG s=4):

![Three-way comparison](plots/three_way_grid.png)

The LDM reproduces the CLEVR style (studio lighting, large flat ground, specular + matte materials) and approximately follows the caption's colour and count cues, but fine shape placement and count fidelity are imperfect — expected for a 16 M-param U-Net at 150 epochs on 16×16 latents with T=500 DDPM sampling. Longer training or a larger U-Net would be the main levers for further improvement.

An 8-caption sampling grid is in `plots/ldm_samples_grid.png`.

### 3.7 FID (clean-fid, 10 000 generated val images)

> **FID (real vs LDM-generated) = 18.437**

Computed on the full 10 000-sample val set with the exact pipeline the assignment asks for:
1. For each val caption, encode with frozen CLIP → (1, 77, 512).
2. Run 500-step DDPM with CFG (s = 4.0) to get `z ∈ R^{B×4×16×16}` on the normalised-latent manifold.
3. De-normalise `z ← z · std + mean`, decode via the frozen VAE, un-normalise `[-1,1] → [0,1]`.
4. Save to `samples/ldm_val/`, compute `cleanfid.fid.compute_fid(mode="clean")` against
   `samples/real_val_128/`.

Configuration: `guidance=4.0`, `timesteps=500`, `num_samples=10000`.

The ~4× gap between VAE reconstruction FID (4.81) and LDM generation FID (18.44) is the
"diffusion overhead" — it captures everything the U-Net fails to model about the true latent
distribution.

---

## 4. Reproduction / artefact layout

```
col775_a2/                            # code
├── models/{vae,unet,text_encoder}.py
├── utils/{diffusion,fid,scheduler}.py
├── data/ldm_dataset.py
├── train_vae.py, compute_latent_stats.py, train_ldm.py
├── sample_ldm.py, eval_vae_fid.py

checkpoints/                          # trained weights (on HPC)
├── vae_last.pt      (38 MB)
├── vae_final.pt     (duplicate of _last at epoch 99)
├── latent_stats.pt  (per-channel μ, σ)
├── ldm_last.pt      (185 MB; epoch 149)
└── ldm_final.pt     (duplicate of _last)

logs/
├── vae_train.csv, vae.out           (100 epochs)
├── ldm_train.csv, ldm.out           (150 epochs, with val loss)
├── vae_fid.out                      (FID = 4.808)
├── ldm_fid.out                      (FID = 18.437)
└── latstats.out

samples/                              # on HPC
├── real_val_128/    (10 000 PNGs, 128² bicubic + centre-crop of val)
├── vae_recon_val/   (10 000 PNGs + fid.txt)
└── ldm_val/         (10 000 PNGs + fid.txt)

report_artifacts/                     # this dir
├── logs/            (CSVs + .out files)
├── samples/         (12 pulled examples × {real, vae_recon, ldm_gen})
├── plots/           (all figures rendered from CSVs + images)
├── make_plots.py    (regenerates plots/*.png from logs/)
├── make_sample_grids.py
└── REPORT.md        (this file)
```

### 4.1 End-to-end commands (HPC)

```bash
cd ~/scratch/col775_a2_hpc
qsub hpc/submit_vae.pbs            # 100 ep VAE, ~4 h
qsub hpc/submit_latent_stats.pbs   # μ/σ over train set
qsub hpc/submit_ldm.pbs            # 150 ep LDM, ~7 h
qsub hpc/submit_vae_fid.pbs        # recon FID, ~15 min
qsub hpc/submit_ldm_fid.pbs        # 500-step DDPM FID, ~2 h
```

All intermediate dependencies were submitted with `-W depend=afterok:<JID>` to keep the chain automatic.

---

## 5. Budget

Part C total reconciled HPC cost ≈ **₹55** (well under the ₹1 000 project allocation):
| Job | Walltime req. | Actual elapsed |
|---|---|---|
| vae_test    | 1 h  | 13 min |
| vae_train   | 24 h | 4.1 h |
| latent_stats| 1 h  | 12 min |
| ldm_train   | 24 h | 6.9 h |
| vae_fid     | 2 h  | 14 min |
| ldm_fid     | 6 h  | 2 h 05 min |

---

## 6. Design notes / deviations

1. **λ = 1 × 10⁻⁶** matches the assignment's suggested value; KL contribution is ~3 % of total loss at equilibrium.
2. **Zero-init** on `SpatialTransformer.proj_out` and the U-Net output conv — stabilises early training (standard Stable-Diffusion trick, not in the assignment spec).
3. **Per-sample null-dropout** (not per-batch): `torch.where(mask, null, ctx)` with `mask = rand(B) < 0.1`. Gives finer CFG training signal.
4. **Latent statistics** computed over **posterior mean μ** (not sampled z) — eliminates stochasticity in the normalisation constants.
5. **FID weights bundled** offline: `upload_bundle/cleanfid_cache/inception-2015-12-05.pt` copied to `/tmp/` at job start because the HPC proxy blocks NVIDIA CDN and HuggingFace.
