# report/

All submission artifacts that will go into the final PDF + code package.

## Structure

- [`REPORT.md`](REPORT.md) — the master write-up. Gets converted to PDF at submission.
- [`part_a/`](part_a/) — Part A (CLIP + DINO) + Part Aa (linear probing, t-SNE, retrieval) results. Currently empty — populated by pulling artifacts from HPC after `submit_probe.pbs` runs.
- [`part_c/`](part_c/) — Part C (VAE + LDM) results. Complete.

## Part A / Aa — what goes here after probe job finishes

From HPC `$HOME/scratch/col775_a2_hpc/` copy into `report/part_a/`:

| HPC path | Local path | Purpose |
|---|---|---|
| `logs/clip_train.csv` | `part_a/logs/clip_train.csv` | CLIP loss curve |
| `logs/dino_train.csv` | `part_a/logs/dino_train.csv` | DINO loss curve |
| `logs/clip.out` `logs/dino.out` `logs/probe.out` | `part_a/logs/` | Full stdout captures |
| `plots/tsne_*.png` | `part_a/plots/` | t-SNE scatterplots (CLIP, DINO student, DINO teacher) |
| `plots/retrieval_*.png` | `part_a/plots/` | Qualitative success/failure grids for image→text, text→image |
| Table 1 numbers (printed to `probe.out`) | Paste into `REPORT.md` §Part Aa | — |

Submission weights (`clip_final.pt`, `dino_final.pt`) are large; they go in `part_a/checkpoints/` and are referenced from the README. Same submission copy as HPC — pull once at the end.

## Part C — already complete

See [`part_c/README.md`](part_c/README.md) for the inventory. Final numbers:
- VAE reconstruction FID: **4.808**
- LDM generation FID (10K val captions, CFG s=4.0): **18.437**

## Regenerating plots

The scripts under `part_c/` (`make_plots.py`, `make_sample_grids.py`) are idempotent. Run them if the underlying CSVs or samples change.
