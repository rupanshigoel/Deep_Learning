# Part B — Next Steps

Generated 2026-04-25, after submitting Stage-1 full eval + Stage-2 full-data 5-chunk training.

## Currently submitted (qstat 2026-04-25)

| Job ID | Name | State | Walltime | Notes |
|---|---|---|---:|---|
| 888449 | vlm_eval_s1_full       | R    | 9 h | Stage-1 EM/BLEU on full 70 K train + 15 K val (greedy, batch 8). |
| 888450 | vlm_s2_full_c1         | Q    | 6 h | Stage-2 chunk 1/5: data slice [0, 140 000) of 700 K (master shuffle seed 42). |
| 888451 | vlm_s2_full_c2         | H    | 6 h | afterok:888450 — slice [140 000, 280 000). |
| 888452 | vlm_s2_full_c3         | H    | 6 h | afterok:888451 — slice [280 000, 420 000). |
| 888453 | vlm_s2_full_c4         | H    | 6 h | afterok:888452 — slice [420 000, 560 000). |
| 888454 | vlm_s2_full_c5         | H    | 6 h | afterok:888453 — slice [560 000, 700 000); promotes `_last` → `_full_final` after job. |

All jobs run on `standard` queue (cost factor 1× per `qstat -f`); high queue is 5× more.

## Estimated GPU time / cost (standard queue, ₹3.57/GPU-h after 5× discount vs high)

| Workload | Samples | Throughput | Wall-clock | Estimated cost |
|---|---:|---|---:|---:|
| Stage-1 full eval (running) | 85 000 | ~3.2 samp/s | ~7.4 h | ₹26 |
| Stage-2 full train (1 epoch, queued) | 700 000 | ~7.6 samp/s | ~25.6 h split into 5 jobs | ₹91 |
| **Subtotal in flight** |  |  | **~33 h** | **~₹117** |

## Bug fixes shipped in this round

| Bug | File | Fix |
|---|---|---|
| Trainer over-runs `total_steps` after a walltime resume; data overlap at the end | `col775_a2/train_vlm_stage{1,2}.py` | Added `if step >= total_steps: break` in the inner+outer loops. |
| `make_plots.py` Stage-1 title swapped optim-step / micro-step labels | `report/part_b/make_plots.py` | Use the last logged step from the CSV; report effective batch in title. |
| No way to chain jobs over disjoint data slices | `col775_a2/data/clevrx_dataset.py` + `train_vlm_stage2.py` | Added `--slice-start/--slice-end/--slice-seed` (deterministic shuffle then slice) and `--total-train-steps` to keep the cosine LR schedule consistent across chained jobs. |

After chunks 1–5 complete, the global step counter will be exactly **43 750 = 700 000 / 16** (one true epoch over the recut-train CLEVR-X split with effective batch 16). LR follows a single global cosine schedule across all 5 jobs.

## Plots already added (no GPU needed)

- [`plots/stage2_category_em.png`](plots/stage2_category_em.png) — bar chart, color/material/size/shape/yes-no/count, with overall line.
- [`plots/stage2_count_confmat.png`](plots/stage2_count_confmat.png) — predicted vs true count, 0–10, with diagonal + off-by-one band shown in title.

Numbers from the existing 2 500-sample val: diagonal 52.4 %, off-by-one band 35.9 %, zero unparsed. **88.3 % of count predictions are within ±1 of ground-truth** — confirms the "model can almost count" failure mode you call out in the README, and is by far the strongest single data point for the Stage-2 analysis section of the report.

---

# After the chained training finishes

## 1. Confirm chunk 5 promoted the checkpoints

```bash
ssh hpc 'ls -la $HOME/scratch/col775_a2_hpc/checkpoints/full/'
# Expect:
#   vlm_stage2_full_final.pt              (projector + step + args)
#   vlm_stage2_full_lora_final/           (adapter_config.json + adapter_model.safetensors)
#   vlm_stage2_last.pt                    (last-step incremental, same step counter)
#   vlm_stage2_lora_last/                 (last-step LoRA snapshot)
```

If the post-loop `cp` failed (e.g. the job was walltime-killed in the middle of step 5/5), promote manually:
```bash
cd $HOME/scratch/col775_a2_hpc/checkpoints/full
cp -r vlm_stage2_lora_last vlm_stage2_full_lora_final
cp    vlm_stage2_last.pt    vlm_stage2_full_final.pt
```

## 2. Run Stage-2 evaluation on the full-data checkpoint

PBS script already written at [`hpc/pbs/submit_vlm_eval_s2_full.pbs`](../../hpc/pbs/submit_vlm_eval_s2_full.pbs).

It evaluates on **10 K stratified train + 10 K stratified val** (each at cap = 1.5× uniform per-class). 10 K is a deliberate compromise:

| Eval choice | n | Wall | Cost (normal) | Recommendation |
|---|---:|---:|---:|---|
| Submitted PBS as-is | 10 K + 10 K | ~1.05 h | ₹4 | **Recommended.** Per-category CIs ~2.5 %; matches train/val on the same sampler so train EM > val EM cleanly. |
| Full val | 150 K | ~8 h | ₹29 | Maximum-rigour headline. Replace `--val-subsample 10000` with `0`. |
| Full train EM | 700 K | ~37 h | ₹130 | **Don't.** Training-set EM is diagnostic — go for 10 K-stratified or skip entirely. |

Submit:
```bash
python hpc/hpc_upload.py hpc/pbs /home/maths/btech/mt1230785/scratch/col775_a2_hpc/hpc
python hpc/hpc_submit_chain.py hpc/pbs/submit_vlm_eval_s2_full.pbs
```

After it finishes:
- Pull `logs/vlm_eval_stage2_full/{predictions.jsonl,metrics.json}` to `report/part_b/logs_full/`
- Update `report/part_b/make_category_plots.py` to point at the new predictions.jsonl
- Re-run `python make_category_plots.py` → fresh bar chart + confusion matrix at the higher n.

## 3. Update the report

Files to update in [`report/REPORT.md`](../REPORT.md) and `report/part_b/README.md`:

1. **Headline numbers**: replace 80 K-stratified Stage-2 train numbers with 700 K full-train numbers; replace 2 500-stratified val with 10 K-stratified val.
2. **Loss curves**: re-run `python report/part_b/make_plots.py` — the CSV at `logs/full/vlm_stage2_train.csv` will now have ~43 750 rows.
3. **Per-category EM bar chart**: re-run `make_category_plots.py` against the new predictions.
4. **Count confusion matrix**: same.
5. **Data-scale paragraph**: change "80 K stratified subsample (≈11 % of 700 K)" → "**full 700 K, 1 epoch, single global cosine LR schedule**".
6. **Plot of LR schedule** (new, easy win): can be added by extending `make_plots.py` to also plot the `lr` column from `vlm_stage2_train.csv`.
7. **Re-render qualitative grids**: rebuild with **one example per PNG** (image-on-top + wrapped text below) for LaTeX readability — see Top-5 #3 in the audit. Consider a new `make_qualitative_v2.py` next round.

## 4. Things deferred for now (would-be improvements)

- **Stage-1 full train EM/BLEU**: now in flight (job 888449); just wait for it.
- **Stage-2 full val (150 K)**: skip unless you want to spend ~₹29 / ~8 h. The 10 K-stratified number is statistically tight enough.
- **Re-rendering qualitative plots**: not on critical path, but high marks-per-hour.

---

## Estimates if you want to push further

| Action | GPU-h (standard) | Cost |
|---|---:|---:|
| Stage-2 full val 150 K instead of 10 K stratified | 8 h | ₹29 |
| Stage-2 full train EM 700 K | 37 h | ₹130 (skip) |
| Re-train Stage-2 for **2 epochs** on full data | 50 h | ₹180 (very tight, only worth it if count EM hasn't budged) |
| Stage-2 full val 150 K + full train 700 K | 45 h | ₹160 (don't — better spend on the next bullet) |
| Stage-2 with `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` (LoRA on MLP too) | re-train 25 h | ₹91 — could improve count EM. Only do if the headline comes back disappointing. |

## Sanity check while jobs run

Periodically:
```bash
ssh hpc 'qstat -u mt1230785 | tail -10'
ssh hpc 'tail -5 $HOME/scratch/col775_a2_hpc/logs/full/vlm_stage2_train.csv 2>/dev/null'
ssh hpc 'tail -20 $HOME/scratch/col775_a2_hpc/logs/vlm_eval_s1_full.out'
```

You should see the chunk-1 CSV start filling around the 5–15 minute mark after the job goes from Q→R.
