"""LlavaLikeVLM — vision-language model built from frozen DINO ViT + MLP projector + Qwen3.

Design (LLaVA recipe, adapted to CLEVR + Qwen3-4B-Instruct-2507):

    image  ─► DINO teacher ViT backbone (frozen) ─► patch tokens (B, 196, 384)
                                                     │
                                                     ▼  MLPProjector (trainable in s1/s2)
                                             (B, 196, llm_hidden)
                                                     │
         system+user text ─► Qwen tokenizer+embed ─► text embeds          ─► concat
         assistant target ─► Qwen tokenizer+embed ─► target embeds ───────────────▲
                                                                                   │
                                                                               Qwen3 LLM
                                                                                   │
                                                                               logits + loss
Stage-1 trains only `MLPProjector`. Stage-2 additionally wraps the LLM in PEFT LoRA
(r=16) and trains LoRA + projector together. The ViT stays frozen in both stages.

Image tokens are *spliced* into the text-embedding stream at a fixed offset — we
do not introduce any new tokenizer ID. This side-steps `resize_token_embeddings`
and works cleanly with Qwen's chat template (Jinja produces the system/user/
assistant markers, and we splice image embeds into the resulting embedding
sequence).
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# cuDNN 9.x's scaled_dot_product_attention graph-rewriter fails on Qwen3's
# causal-mask attention pattern ("No execution plans support the graph").
# Force HF SDPA to use flash / efficient / math backends instead. We bit
# this same bug in Part A retrieval — see col775_a2/retrieval.py.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)


IMAGE_TAG = "<<IMAGE>>"  # placeholder in the raw prompt; never tokenized — we splice at this offset.


class MLPProjector(nn.Module):
    """2-layer reverse-bottleneck MLP: ViT dim → hidden → LLM hidden.

    Params kept in fp32 (standard LLaVA pattern — a few million trainable
    parameters benefit from fp32 stability, cast output to bf16 at use site).
    """
    def __init__(self, in_dim: int = 384, hidden_dim: int = 1536, out_dim: int = 2560):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02); nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02); nn.init.zeros_(self.fc2.bias)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # (B, N, in_dim) -> (B, N, out_dim)
        return self.fc2(self.act(self.fc1(patch_tokens)))


class LlavaLikeVLM(nn.Module):
    """Frozen ViT + trainable MLP + (optionally LoRA-adapted) Qwen3 LLM."""

    def __init__(self, vit: nn.Module, projector: MLPProjector, llm: nn.Module, tokenizer):
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer
        # Cache the LLM input-embedding lookup for fast token → embed conversion.
        self._embed_fn = llm.get_input_embeddings()
        self.num_image_tokens = 196  # 14 × 14 patches at 224/16
        self.llm_hidden = self.llm.config.hidden_size
        self.image_tag = IMAGE_TAG

    # --------------------------------------------------------------------- #
    # Encoding
    # --------------------------------------------------------------------- #
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 196, llm_hidden), cast to LLM dtype."""
        with torch.no_grad():
            feats = self.vit(pixel_values)["patch_tokens"]  # (B, 196, 384), fp32 or bf16 per ViT dtype
        feats = feats.to(self.projector.fc1.weight.dtype)
        proj = self.projector(feats)                        # (B, 196, llm_hidden), fp32
        return proj.to(self._embed_fn.weight.dtype)         # cast to match LLM embeds (bf16)

    # --------------------------------------------------------------------- #
    # Prompt helpers (chat template)
    # --------------------------------------------------------------------- #
    def build_training_strings(self, system: str, user: str, target: str) -> Tuple[str, str]:
        """Returns (prompt_before_target, target_with_eos).

        `user` may contain `IMAGE_TAG`; we split the chat-template-rendered prompt
        at that tag to find the splice offset later.
        """
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        # The rendered `prefix` ends with `<|im_start|>assistant\n` — ready for the answer.
        # We append target + eos and let `labels` supervise only the target tokens.
        eos = self.tokenizer.eos_token or "<|endoftext|>"
        target_str = f"{target}{eos}"
        return prefix, target_str

    def build_inference_string(self, system: str, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    # --------------------------------------------------------------------- #
    # Input construction (image-embed splicing)
    # --------------------------------------------------------------------- #
    def _splice_embeds(
        self,
        prefix_ids: torch.Tensor,        # (B, Lp)
        prefix_attn: torch.Tensor,       # (B, Lp)
        image_embeds: torch.Tensor,      # (B, Ni, H)
        target_ids: Optional[torch.Tensor] = None,      # (B, Lt) or None (inference)
        target_attn: Optional[torch.Tensor] = None,
    ):
        """Build `inputs_embeds`, `attention_mask`, `labels` by concatenating:
            [ left-pad | prefix (sys+user w/ image-tag replaced by embeds) | target ]

        The prefix contains the IMAGE_TAG string, which we tokenized as a sentinel
        sequence. The dataset guarantees IMAGE_TAG tokenizes to a **known fixed**
        single-token sequence (we add it to the tokenizer as a special token if
        needed). For simplicity we instead tokenize the prefix as two halves
        ('pre' and 'post' around IMAGE_TAG) at dataset time, and the prefix_ids
        passed here are ALREADY split — see `build_batch` for the canonical path.
        """
        raise NotImplementedError("Use build_batch; legacy stub retained for reference.")

    def build_batch(
        self,
        pixel_values: torch.Tensor,         # (B, 3, H, W)
        pre_ids: List[torch.Tensor],        # list of (Lpre_i,)  tokens before IMAGE_TAG
        post_ids: List[torch.Tensor],       # list of (Lpost_i,) tokens after IMAGE_TAG (inclusive of assistant header)
        target_ids: Optional[List[torch.Tensor]] = None,   # list of (Lt_i,) target-with-eos (training only)
        padding_side: str = "right",
    ):
        """Build padded `(inputs_embeds, attention_mask, labels)` from per-sample
        token segments. When `target_ids` is None we're in inference mode:
        labels is None and padding is 'left' so generation continues at the right.
        """
        device = pixel_values.device
        image_embeds = self.encode_image(pixel_values)            # (B, Ni, H)
        Ni = image_embeds.size(1)
        H = image_embeds.size(2)
        embed_dtype = image_embeds.dtype

        # Per-sample total length
        lens = []
        B = pixel_values.size(0)
        for i in range(B):
            Lpre = pre_ids[i].numel()
            Lpost = post_ids[i].numel()
            Ltgt = target_ids[i].numel() if target_ids is not None else 0
            lens.append(Lpre + Ni + Lpost + Ltgt)
        Lmax = max(lens)

        inputs_embeds = torch.zeros(B, Lmax, H, dtype=embed_dtype, device=device)
        attn_mask = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        labels = torch.full((B, Lmax), -100, dtype=torch.long, device=device) if target_ids is not None else None

        for i in range(B):
            pre = pre_ids[i].to(device)
            post = post_ids[i].to(device)
            pre_emb = self._embed_fn(pre).to(embed_dtype)          # (Lpre, H)
            post_emb = self._embed_fn(post).to(embed_dtype)        # (Lpost, H)
            parts = [pre_emb, image_embeds[i], post_emb]
            real_len = lens[i]

            if target_ids is not None:
                tgt = target_ids[i].to(device)
                tgt_emb = self._embed_fn(tgt).to(embed_dtype)
                parts.append(tgt_emb)

            real = torch.cat(parts, dim=0)                          # (real_len, H)
            if padding_side == "left":
                pad = Lmax - real_len
                inputs_embeds[i, pad:] = real
                attn_mask[i, pad:] = 1
                if labels is not None:
                    Lpre, Lpost, Ltgt = pre.numel(), post.numel(), tgt.numel()
                    tgt_start = Lmax - Ltgt
                    labels[i, tgt_start:] = tgt
            else:
                inputs_embeds[i, :real_len] = real
                attn_mask[i, :real_len] = 1
                if labels is not None:
                    Lpre, Lpost, Ltgt = pre.numel(), post.numel(), tgt.numel()
                    tgt_start = Lpre + Ni + Lpost
                    labels[i, tgt_start:tgt_start + Ltgt] = tgt

        return inputs_embeds, attn_mask, labels

    # --------------------------------------------------------------------- #
    # Forward (training)
    # --------------------------------------------------------------------- #
    def forward(
        self,
        pixel_values: torch.Tensor,
        pre_ids: List[torch.Tensor],
        post_ids: List[torch.Tensor],
        target_ids: List[torch.Tensor],
    ):
        inputs_embeds, attention_mask, labels = self.build_batch(
            pixel_values, pre_ids, post_ids, target_ids, padding_side="right"
        )
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

    # --------------------------------------------------------------------- #
    # Generation (inference)
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        pre_ids: List[torch.Tensor],
        post_ids: List[torch.Tensor],
        max_new_tokens: int = 256,
        do_sample: bool = False,
        **generate_kwargs,
    ) -> List[str]:
        inputs_embeds, attention_mask, _ = self.build_batch(
            pixel_values, pre_ids, post_ids, target_ids=None, padding_side="left"
        )
        eos_id = self.tokenizer.eos_token_id
        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or eos_id,
            eos_token_id=eos_id,
            use_cache=True,
            **generate_kwargs,
        )
        # `out` has shape (B, max_new_tokens) — generate() with inputs_embeds
        # returns *only* the newly generated tokens, not the prompt.
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    # --------------------------------------------------------------------- #
    # Freeze / unfreeze helpers
    # --------------------------------------------------------------------- #
    def freeze_vit(self):
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad = False

    def apply_lora(self, lora_config):
        """Wrap the LLM in a PEFT LoRA adapter. Stage-2 only.

        Must be called AFTER `freeze_llm()` so only LoRA weights become trainable.
        `self._embed_fn` already points at the (shared) base embedding layer and
        continues to work — LoRA does not alter the embedding weights.
        """
        from peft import get_peft_model
        self.llm = get_peft_model(self.llm, lora_config)
        # Needed for gradient checkpointing + LoRA to flow grads into adapters.
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def trainable_param_breakdown(self) -> dict:
        """Return a dict of {group: total_params} for observability."""
        groups = {"projector": 0, "lora": 0, "vit": 0, "llm_other": 0}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("projector."):
                groups["projector"] += p.numel()
            elif "lora_" in n:
                groups["lora"] += p.numel()
            elif n.startswith("vit."):
                groups["vit"] += p.numel()
            else:
                groups["llm_other"] += p.numel()
        return groups


# -------------------------------------------------------------------------- #
# Builders
# -------------------------------------------------------------------------- #
def load_dino_teacher_backbone(ckpt_path: str, device: str = "cpu") -> nn.Module:
    """Load only the ViT backbone of the DINO *teacher* from a final checkpoint."""
    from .dino_model import build_dino
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoint layout (written by train_dino.save_checkpoint): {"student": ..., "teacher": ...}
    models = build_dino(img_size=224, patch_size=16, out_dim=4096)
    models["teacher"].load_state_dict(state["teacher"])
    backbone = models["teacher"].backbone
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone.to(device)


def build_vlm(
    dino_ckpt: str,
    qwen_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507",
    dtype: torch.dtype = torch.bfloat16,
    projector_hidden: int = 1536,
    device: str = "cuda",
    attn_implementation: Optional[str] = None,
) -> Tuple[LlavaLikeVLM, object]:
    """Build a fresh VLM. Returns (model, tokenizer).

    The projector is newly-initialized; load a Stage-1 state_dict externally
    via `model.projector.load_state_dict(...)` for Stage 2.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vit = load_dino_teacher_backbone(dino_ckpt, device="cpu")

    tok = AutoTokenizer.from_pretrained(qwen_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    llm_kwargs = dict(dtype=dtype, trust_remote_code=True)
    if attn_implementation:
        llm_kwargs["attn_implementation"] = attn_implementation
    llm = AutoModelForCausalLM.from_pretrained(qwen_name_or_path, **llm_kwargs)
    llm_hidden = llm.config.hidden_size

    projector = MLPProjector(in_dim=384, hidden_dim=projector_hidden, out_dim=llm_hidden)

    model = LlavaLikeVLM(vit=vit, projector=projector, llm=llm, tokenizer=tok)
    model.freeze_vit()
    model.freeze_llm()  # projector is the only trainable in Stage 1
    # Move to device; keep projector in fp32.
    model.vit.to(device)                        # already frozen, cast doesn't matter
    model.llm.to(device)
    model.projector.to(device, dtype=torch.float32)
    return model, tok
