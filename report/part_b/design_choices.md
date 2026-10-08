# Part B — Design Choices, Assumptions & Simplifications

Everything that the assignment PDF (§3) didn't fix in stone. Each entry: the choice, the **rationale**, and whether it's a **budget-driven simplification** or a free design decision.

---

## 1. Data scale & subsampling

### 1.1 Stage 1 training — **full 70 K captions** (no subset)
- Used every caption in `clevr_train_captions.json` for one epoch.
- **Not a simplification.** Single-epoch training over the full 70 K is how Stage 1 ran (2 187 optim steps at micro-batch 2 × grad_accum 16).

### 1.2 Stage 2 training — **80 K stratified subsample** of ≈ 700 K CLEVR-X train QAs (≈ 11.4 % of full)
- CLEVR-X training set has 699 964 questions across ~56 000 images (recut-train split); each question has up to 10 `factual_explanation` strings.
- We subsampled to **80 000 QA rows**, stratified per-answer with a per-class cap at 1.5 × uniform share, to prevent the natural yes/no dominance (≈ 41 % of full) from crowding out colour/shape/count classes.
- **This is a budget-driven simplification.** The assignment doesn't prescribe dataset size; we had ~₹300 remaining at the start of Part B. Full 700 K at our measured ~0.36 s/QA would have been ~70 GPU-hours (~₹1250 on the high queue, 5× cost factor). The 80 K subsample was chosen because (a) it still covers every answer class with decent sample size (50+ per class after stratification), and (b) one epoch at 80 K = 5 000 optim steps fits in a single 6 h PBS slot.
- **One factual_explanation is sampled uniformly per `__getitem__` call** from the list of up to 10 per QA. Assumption: treating all k explanations as separate training examples would inflate the dataset k-fold with high redundancy; single-sample-per-step gives variety across epochs without the blow-up. (We train one epoch anyway.)

### 1.3 Stage 1 evaluation — train **2 000** random-sample / val **15 000** (full)
- Train subset: 2 000 captions drawn uniformly at random from 70 K, with fixed `--seed 0`. **This is a sub-sample for reporting convenience** — the assignment says "on train and validation sets" without prescribing size, and train EM/BLEU are diagnostic (overfitting check); val is the headline metric.
- Val: full 15 000 captions. **No simplification** for the main reported number.

### 1.4 Stage 2 evaluation — train **1 000** stratified / val **2 500** stratified
- Both splits use stratified subsampling by answer class (same cap rule as 1.2).
- **Train 1 K is a sub-sample for speed** (full 80 K eval would be ~8 h per category). 1 K is small but reports a representative per-class mean; the val number is where the headline EM comes from.
- **Val 2 500 is a budget-driven simplification.** Full CLEVR-X val = 149 984 questions → ~15 GPU-hours per eval pass, which would consume the bulk of the remaining budget on a single metric. 2 500 stratified covers 28 answer classes at ~89 samples/class — tight enough for the per-category table in the main README (category CIs are well below 5 % at that sample size).
- **Stratified vs uniform val sampling — deliberate.** Uniform sampling would be dominated by yes/no (~41 % baseline), making the headline EM misleadingly easy. Stratified forces the model to be evaluated on rare classes (counts, shapes, colours) proportionally. This is **stricter than uniform sampling** — a uniform-sampled val would give a higher EM number.

### 1.5 CLEVR-X "recut" — used for Stage 2 train data; val comes from **original CLEVR val**
- CLEVR-X ships `train_images_ids_v0.7.10-recut.pkl` (56 K images) and `dev_images_ids_v0.7.10-recut.pkl` (14 K images) — a re-split of the 70 K CLEVR train images.
- We used the recut-train image set to filter `CLEVR_train_explanations_v0.7.10.json` for training.
- For validation we used `CLEVR_val_explanations_v0.7.10.json` directly (images from the original CLEVR val split, which were never in any training partition).
- **Assumption**: this gives a cleaner train/val boundary than CLEVR-X's own recut-dev (which is still drawn from CLEVR train images).

---

## 2. Architecture

### 2.1 Projector hidden dim = **1536**
- Assignment prescribes "2-layer MLP with reverse-bottleneck"; does not prescribe hidden dim. "Reverse bottleneck" means hidden > min(in, out).
- We picked **4 × in_dim** = 1536, matching the ViT's own internal MLP ratio (ViT MLP is 384 → 1536 → 384).
- Alternative considered: 2 × llm_hidden = 5120, which is ~3× more parameters (~15 M vs 4.5 M) and more Stage 2 activation memory. Rejected — no evidence the wider projector helps at CLEVR scale, and Stage 2 memory budget (A100-40GB) was tight.

### 2.2 Projector activation = **GELU**
- Assignment doesn't specify. GELU matches Qwen's internal MLP choice and LLaVA-1.5's projector. No ablation run.

### 2.3 Projector dtype = **fp32 params, bf16 output**
- Standard LLaVA pattern: the projector has ~5 M trainable params — small enough to afford fp32 for numerical stability, but the output is cast to bf16 before concatenation with the bf16 LLM embeddings. Assignment says "mixed precision" generically.

### 2.4 Image-embed splicing uses **no new tokenizer ID**
- Rather than add `<|image|>` to the tokenizer, resize the embedding table, and insert image embeds at that token's position, we splice at a position-based offset. Concretely: we tokenize `system + user-text-before-image` and `user-text-after-image + assistant-header` as two halves, then build `inputs_embeds = [pre | image_embeds | post | target]`.
- **Assumption**: this is functionally equivalent to using a sentinel token and avoids any tokenizer mutation. Works with unmodified Qwen3 weights, simpler to reload adapters. Downside: can't generate with raw `input_ids` — `generate(inputs_embeds=...)` is mandatory.

### 2.5 LoRA hyperparameters — α = 32, dropout = 0.05, target modules
- Assignment fixes **r = 16** and "attention blocks"; nothing else.
- We picked α = 32 (standard 2r), dropout = 0.05 (LLaVA-1.5 default), `target_modules=["q_proj","k_proj","v_proj","o_proj"]` — all four attention projections in every Qwen3 block.
- `bias="none"` — no LoRA on biases; default.
- **Assumption**: "attention blocks" means all four Q/K/V/O linear projections. Alternatives would be just Q+V (LoRA paper) or Q+V+MLP. We picked all-four as a reasonable middle ground given the CoT task complexity.

### 2.6 Qwen3 loaded with **trust_remote_code=True** and default attention backend (SDPA)
- Required for Qwen3 loading (custom modeling code). `attn_implementation` left at the default (SDPA), with **cuDNN SDPA disabled globally** via `torch.backends.cuda.enable_cudnn_sdp(False)` — cuDNN 9 has a known bug on the causal-mask pattern Qwen3 uses (same bug bit us in Part A retrieval). Disabling cuDNN SDPA leaves flash / efficient / math backends available.

---

## 3. Training hyperparameters

### 3.1 Optimizer & schedule
Assignment specifies **AdamW** for CLIP (§1.1); we re-used for Part B. Assignment's Part B doesn't prescribe optimizer or schedule.

- **Optimizer**: AdamW, betas = (0.9, 0.95), weight_decay = 0.
- **LR Stage 1**: **1e-3** (matches LLaVA-1.5 Stage 1 projector pre-training).
- **LR Stage 2**: **2e-4** (lower because we're co-training projector with LoRA on a pre-aligned model; LLaVA-1.5 Stage 2 uses 2e-5 for full-model finetuning but we use 2e-4 given LoRA is "sparser" and benefits from a higher LR).
- **Schedule**: cosine decay with 3 % linear warmup, min-LR-ratio 0.05 (so min LR = 5 % of base).
- **Grad clipping**: max-norm 1.0.

### 3.2 Batch sizing
- **Stage 1**: micro-batch 2, grad_accum 16 → effective batch 32. Matches LLaVA Stage 1 effective batch.
- **Stage 2**: micro-batch 1, grad_accum 16 → effective batch 16. Lower because Stage 2 sequences are longer (~500 tokens with the explanation in target) and LoRA + grad-ckpt activations are tighter in memory.
- **Assumption**: smaller Stage 2 effective batch (16 vs 32) is fine because LoRA has fewer trainable params and benefits less from batch-level noise reduction.

### 3.3 Epochs = **1** for both stages
- Assignment doesn't prescribe epoch count.
- Stage 1: 1 epoch on 70 K = 2 187 optimizer steps; loss went 3.7 → 0.042 (convergence).
- Stage 2: 1 epoch on 80 K = 5 000 optimizer steps; loss stabilised at ~0.10 by step 4 500.
- **Simplification**: we could have trained longer (e.g. 3 epochs on Stage 2). Rejected because (a) loss curve was flat at the end, and (b) budget. The actual step count went to 5 360 due to a minor loop bug (loop bounded by data-exhaustion, not `total_steps`; doesn't materially matter — LR was at floor).

### 3.4 Gradient-checkpointing, accumulation, mixed precision — all applied
- `gradient_checkpointing_enable(use_reentrant=False)` on Qwen3.
- `enable_input_require_grads()` after LoRA wrap so grad flows through grad-ckpt into LoRA weights.
- BF16 autocast across the whole forward on A100 (native BF16 tensor cores).

---

## 4. Prompts

Not specified by the assignment. We chose simple LLaVA-style prompts:

### 4.1 Stage 1
```
system:     You are a helpful vision-language assistant. Describe the given image factually and concisely.
user:       <<IMAGE>>
            Describe this image.
assistant:  {caption}
```

### 4.2 Stage 2
```
system:     You are a visual reasoning assistant. Look at the image, reason step-by-step about the
            question, and then give a concise final answer.
user:       <<IMAGE>>
            Question: {question}
assistant:  Reasoning: {sampled_factual_explanation}
            Answer: {answer}
```

`<<IMAGE>>` is a literal placeholder string we split on before tokenising — it's never actually tokenised (see §2.4).

**Assumptions baked into the prompt:**
- Target starts with "Reasoning:" then "Answer:" on a new line, so we can regex-extract the answer at eval time.
- No few-shot examples in the prompt. Assignment's CoT description says "train to predict reasoning and then answer"; we interpret this as zero-shot formatting trained from scratch.

---

## 5. Image preprocessing

### 5.1 No augmentation at VLM training time
- Stage 1 and Stage 2 both use the same deterministic transform as Part A retrieval: resize 224 → center-crop 224 → ImageNet mean/std normalize.
- **Deliberate choice**: caption / CoT targets are tightly content-dependent. RandomResizedCrop or ColorJitter can change which objects are visible, desynchronising the target.

### 5.2 Resolution = 224 × 224, patch 16 → 14 × 14 = 196 patch tokens
- Inherited from the Part A ViT. Assignment doesn't allow changing this.

---

## 6. Evaluation methodology

### 6.1 Greedy decoding
- `do_sample=False` for both stages. Makes EM / BLEU reproducible and is the conservative choice (no sampling-noise boost). Assignment doesn't specify.

### 6.2 `max_new_tokens` — 96 for Stage 1, 192 for Stage 2
- Tuned to the target length distribution. CLEVR Stage 1 captions are ≤ ~80 tokens. Stage 2 "Reasoning: …\nAnswer: X" averages ~130 tokens. Padding above is safety margin.

### 6.3 Exact-match normalization
- Lower-case, strip trailing punctuation, collapse whitespace, then string-compare.
- Stage 2: we additionally regex-extract the substring after `/[Aa]nswer\s*:/` before the normalization, so the reasoning prefix doesn't penalise a correct answer.

### 6.4 BLEU via sacrebleu corpus_bleu
- `sacrebleu.corpus_bleu(hypos, [refs]).score`. Default tokeniser (13a). Matches the assignment's "via sacrebleu package" directive.

### 6.5 Qualitative = 5 correct + 5 incorrect per split × 2 splits = 4 PNG grids per stage
- Assignment says "visualize correct and incorrect samples from validation set". We include train-split grids too for comparison, but the val grids are the submission-relevant ones.

---

## 7. Resumability & infrastructure

### 7.1 Both trainers auto-resume from `_last` checkpoints
- Preserves optimizer, scheduler, and LoRA adapter state across walltime boundaries.
- Saved every 200 steps (a compromise between I/O and loss-on-kill).

### 7.2 `vlm_stage2_final.pt` is a **promoted `_last` at step 5 200**
- Stage 2 walltime-killed before the natural loop end, so the post-loop save code never ran. We `cp`'d `vlm_stage2_last.pt` → `vlm_stage2_final.pt` and `vlm_stage2_lora_last/` → `vlm_stage2_lora_final/`.
- **Justification**: loss had been stable at ~0.09 since step 4 900 (LR at its floor 1 e-5), so the last per-200-step save is effectively the final. Not a correctness concern.

### 7.3 SFTP quirks
- `paramiko` silently dropped some large uploads initially (parent dir not created, `mkdir_p` fell through). We switched to explicit `sftp.put(..., confirm=True)` + post-put `stat` verification for big files (`hpc_upload.py` was patched mid-run). Not a simplification — an implementation detail of our upload workflow.

### 7.4 Eval `predictions.jsonl` is **line-buffered** (buffering=1)
- First eval job walltime-killed lost 5 760 generated samples because the default buffer hadn't flushed. Fix: `open(..., "w", buffering=1)`.

---

## 8. What we *did not* do (explicit negative choices)

- **No flash-attention 2** — left default SDPA; performance sufficient.
- **No bitsandbytes / 4-bit loading** — Qwen3-4B in BF16 fits on A100-40GB with grad-ckpt + LoRA.
- **No QLoRA** — LoRA over full BF16 Qwen weights.
- **No RLHF / DPO** — only SFT-style causal LM loss.
- **No tokenizer mutation** (see §2.4).
- **No validation during training** — we skip a val pass between epochs because we only train 1 epoch.
- **No early stopping** — single epoch anyway.
- **No learning-rate ablation** — single configuration per stage.
- **No projector-dim ablation** — single configuration (1536).
- **No comparison vs CLIP as the frozen encoder** — Piazza clarified "best encoder" = avg probe accuracy, which is DINO Teacher; running a CLIP version as a baseline would double Stage 1 + Stage 2 cost.

---

## 9. Summary: where we deviated from strictest interpretation of the guidelines

| Deviation | Why |
|---|---|
| Stage 2 train = 80 K subsample of 700 K | Budget (₹300 remaining on high queue) |
| Stage 2 val = 2 500 subsample of 150 K | Budget; full eval would consume ~₹400 |
| Stage 1 train-EM computed on 2 K of 70 K | Diagnostic metric only; val is headline |
| Stage 2 train-EM computed on 1 K of 80 K | Same |
| 1 epoch per stage (no explicit epoch count in PDF) | Loss converged; budget |
| Stratified (not uniform) val subsampling | Stricter, not looser — exposes per-class failure modes |
| Projector hidden = 1536 (4 × in) | Free choice; smallest reasonable "reverse-bottleneck" |
| α = 32, dropout = 0.05 for LoRA | Free choice; LLaVA-1.5 defaults |
| LR = 1e-3 / 2e-4 for Stages 1 / 2 | Free choice; LLaVA-1.5 inspired |
| Image-embed splicing without special token | Free choice; functionally equivalent |
| `max_new_tokens` = 96 / 192 | Free choice; covers the observed length distribution |
