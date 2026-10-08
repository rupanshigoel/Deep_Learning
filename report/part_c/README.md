# Part C — report artifacts

Self-contained directory with everything needed to write / hand in the report.

```
REPORT.md                  ← main deliverable, written in GitHub-flavoured Markdown
make_plots.py              ← regenerates plots/*loss.png, *_lr.png, *_epoch_time.png from logs/*.csv
make_sample_grids.py       ← regenerates plots/vae_recon_grid, ldm_samples_grid, three_way_grid
                             (requires samples/{real,vae_recon,ldm_gen}/*.png)

logs/
├── vae_train.csv          100 rows: epoch, train_loss, recon, kl, lr, epoch_time_min
├── ldm_train.csv          150 rows: epoch, train_loss, val_loss, lr, epoch_time_min
├── vae.out                stdout from vae_train job (per-iter printout)
├── ldm.out                stdout from ldm_train job
├── vae_fid.out            "[fid] VAE recon vs. val  FID = 4.808"
├── ldm_fid.out            "[fid] LDM vs. val  FID = 18.437"
├── vae_recon_fid.txt      machine-readable summary of VAE recon FID (4.808)
└── ldm_gen_fid.txt        machine-readable summary of LDM gen FID (18.437)

plots/
├── vae_loss.png            train+val recon MSE (log-y) — KL split into its own plot
├── vae_kl.png              per-element KL (linear-y) — separate from recon for legibility
├── vae_lr.png              cosine-warmup LR schedule
├── vae_epoch_time.png      A100 wallclock per epoch (y clipped to drop ep-0 spike)
├── ldm_loss.png            train + val eps-MSE, log scale
├── ldm_loss_linear.png     same, linear y zoomed to [0.10, 0.20] band
├── ldm_lr.png              LDM LR schedule
├── ldm_epoch_time.png      LDM wallclock per epoch (y clipped)
├── vae_recon_grid.png      8 val images: real (top) vs VAE recon (bottom), 4-col x 2-panel
├── ldm_samples_grid.png    8 val captions: real (top) vs LDM CFG-sample (bottom), 4-col x 2-panel
├── three_way_grid_p1.png   3 examples (page 1): real | VAE | LDM with full captions
├── three_way_grid_p2.png   3 examples (page 2)
├── three_way_grid.png      concatenation of the two pages (one tall image)
├── diversity_grid.png      4 captions × 4 random seeds @ CFG s=4 (stochasticity demo)
├── cfg_grid.png            4 captions × 5 CFG scales (s ∈ {0,1,2,4,7.5}, fixed seed)
└── ablation_captions.txt   the 4 prompt strings used for the ablation grids

samples/
├── real/            12 × 128×128 real val PNGs
├── vae_recon/       12 × 128×128 VAE reconstructions (same filenames)
└── ldm_gen/         12 × 128×128 LDM text-conditional generations

checkpoints/
├── vae_last.pt          (38 MB) full ConvVAE state_dict + optimizer
├── latent_stats.pt      (1.5 KB) per-channel μ, σ over train set
└── ldm_last.pt          (185 MB) ConditionalUNet state_dict + null context
                         + optimizer + scheduler

code/
└── (full Part C python package copied from ../col775_a2/)
    ├── models/ data/ utils/
    ├── train_vae.py compute_latent_stats.py train_ldm.py
    └── sample_ldm.py eval_vae_fid.py
```

## Key numbers

| Metric | Value |
|---|---|
| VAE params | 3.25 M |
| VAE train MSE (ep 99) | 0.000154 |
| VAE training time (A100) | ~3.9 h (100 epochs × 2.34 min) |
| **VAE reconstruction FID** | **4.808** |
| LDM params | 16.18 M |
| LDM train MSE (ep 149) | 0.1263 |
| LDM val MSE (ep 149) | 0.1339 |
| LDM training time (A100) | ~6.9 h (150 epochs × 2.74 min) |
| LDM sampling + FID time | ~2.05 h (10 000 × 500-step DDPM + CFG) |
| **LDM generation FID** | **18.437** |

## FID numbers (assignment-required)

The spec only asks for one FID value for VAE reconstruction and one for the LDM
text-conditional generation, both computed with `cleanfid.fid.compute_fid(mode="clean")`
on the full 10 000 val captions / images at 128 × 128 (bicubic resize +
centre crop on the real-image side).

| System | FID (10k val) | Notes |
|---|---|---|
| **VAE reconstruction** | **4.808** | Upper-bound on what the latent diffusion can reach on top of this VAE — anything > 4.81 is "diffusion overhead". |
| **LDM, full pipeline** | **18.437** | 500-step DDPM, CFG s = 4.0, normalised latents from `latent_stats.pt`. |

A CFG-scale FID sweep (s = 0/1/2/7.5) was considered as a strengthening ablation
but is **not asked by the spec**, so we did not run it.

## Latent-stats μ-vs-z sanity check (job 888436, completed)

The spec literally says "un-noised VAE latents" when defining the LDM
normalisation statistics. We computed stats two ways:

| Source | mean (per channel) | std (per channel) |
|---|---|---|
| posterior mean μ (used for the trained LDM, `latent_stats.pt`) | [-0.4789, -0.0505, 0.5351, 0.4812] | [1.0543, 0.5581, 0.8482, 0.3834] |
| reparameterised z = μ + σ·ε (`latent_stats_sampled.pt`) | [-0.4789, -0.0505, 0.5351, 0.4813] | [1.0543, 0.5583, 0.8482, 0.3835] |

The two sets match to 4 decimal places — with λ = 1e-6 the KL pull on σ is
negligibly small, so the posterior is effectively deterministic per image and
the sample averaging across 100 K train images converges to the same statistics
either way. The trained LDM did **not** need to be re-trained.

## VAE generalisation check (job 888521, completed)

The original VAE training only persisted train losses per epoch. To verify
no overfitting, we ran a single val pass on `vae_last.pt` over the full 10K
val set after-the-fact (no retrain — the existing weights are kept; this
keeps the LDM's latent statistics valid):

| | Train (ep 99) | Val (ep 99) | Gap |
|---|---|---|---|
| recon MSE | 1.54e-4 | **1.69e-4** | +9.7 % |
| KL (per-elem) | 5.092 | **5.093** | +0.02 % |
| total loss | 1.59e-4 | 1.74e-4 | +9.4 % |

A ~10 % train→val gap on reconstruction MSE with effectively zero KL gap
indicates the encoder has not collapsed and the VAE has not overfit. The val
point is rendered as a labelled star at epoch 99 in `plots/vae_loss.png`
and `plots/vae_kl.png`.

**Why no per-epoch val curve?** We didn't track val during training and we
only persisted `vae_last.pt` (no per-epoch checkpoints). Retraining to
recover the curve would (a) cost ~4 h and (b) cascade to retraining the LDM
(different VAE → different latent distribution → existing LDM can't decode
correctly), costing another ~7 h plus a fresh FID run. A single end-of-training
val tuple is sufficient evidence of generalisation.

## Qualitative ablations (high queue, all complete)

| Job ID | PBS script | Produces | Status |
|---|---|---|---|
| 888436 | submit_latent_stats_sampled.pbs | `checkpoints/latent_stats_sampled.pt` | ✅ done |
| 888446 | submit_ablations.pbs | `plots/diversity_grid.png` + `plots/cfg_grid.png` | ✅ done |
| 888521 | submit_vae_val_eval.pbs | one-shot `(val_loss, val_recon, val_kl)` for `vae_last.pt` | ✅ done |

### Diversity grid (`plots/diversity_grid.png`)
Same 4 val captions, 4 different random seeds, all at CFG s=4. Each row uses a
distinct seed (1000-1003), so visible per-row variation = the model is genuinely
sampling, not memorising. See `plots/ablation_captions.txt` for the prompt
strings.

### CFG-scale ablation (`plots/cfg_grid.png`)
4 val captions × 5 guidance scales (s ∈ {0, 1, 2, 4, 7.5}), single fixed seed
(2024). s=0 is unconditional sampling; s=7.5 is over-guidance with visible
saturation/blur artefacts. The default s=4.0 used for the headline FID number
sits at the visual sweet spot — colors and counts honour the prompt without
collapsing diversity.

## Regenerating figures

```bash
python make_plots.py          # rewrites plots/{vae,ldm}_loss.png etc.
python make_sample_grids.py   # rewrites plots/three_way_grid.png etc.
```
