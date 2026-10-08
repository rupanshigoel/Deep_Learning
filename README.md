# COL775 A2

CLEVR representation learning + vision-language modeling + text-conditional latent diffusion.
Three parts of the assignment (`col775_a2.pdf`): **A** (CLIP + DINO), **Aa** (analysis of A), **B** (VLM), **C** (VAE + LDM).

## Status 

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

