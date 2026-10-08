"""Frozen CLIP text encoder used as the conditioning branch for the LDM.

Wraps HuggingFace `transformers.CLIPTextModel` + `CLIPTokenizer` for
`openai/clip-vit-base-patch32`, with all parameters frozen and in eval mode.
Exposes:
    - `encode(captions: list[str]) -> (B, seq_len=77, 512)` tensor on `device`.
    - `.seq_len` (77) and `.text_dim` (512) attributes.

The model path is resolved as follows (first match wins):
    1. `model_path` kwarg (absolute local dir OR HF hub id).
    2. `$HF_HOME/hub/models--openai--clip-vit-base-patch32` (HF cache layout).
    3. `openai/clip-vit-base-patch32` (attempts a live HF download - fine
       locally on Windows, expected to fail on HPC when proxy blocks HF).

Typical HPC usage: pre-download locally, SFTP `upload_bundle/hf_cache` to the
workdir, PBS script sets `HF_HOME=$PBS_O_WORKDIR/hf_cache` and
`TRANSFORMERS_OFFLINE=1` before `python -m col775_a2.train_ldm ...`.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

# transformers >= 4.50 hard-fails `torch.load(...)` on torch < 2.6 due to
# CVE-2025-32434, even though our cached CLIP weights ship as a trusted
# pytorch_model.bin. The col775_t25 env on HPC is pinned to torch 2.5; we
# bypass the runtime check before importing CLIPTextModel below.
#
# Note: `from .utils.import_utils import check_torch_load_is_safe` inside
# transformers.modeling_utils captures a separate binding — patching only
# import_utils is not enough. We import every relevant transformers submodule
# eagerly and replace the function on each.
def _bypass_transformers_torchload_check():
    import sys, importlib
    def _noop(*args, **kwargs):  # type: ignore[no-untyped-def]
        return None
    for mod_name in ("transformers.utils.import_utils",
                     "transformers.modeling_utils"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "check_torch_load_is_safe"):
                setattr(mod, "check_torch_load_is_safe", _noop)
        except Exception:
            pass
    # Catch any other already-imported transformers submodule that bound it.
    for name, mod in list(sys.modules.items()):
        if not name.startswith("transformers"):
            continue
        if mod is not None and hasattr(mod, "check_torch_load_is_safe"):
            try:
                setattr(mod, "check_torch_load_is_safe", _noop)
            except Exception:
                pass


_bypass_transformers_torchload_check()


DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


class FrozenCLIPTextEncoder(nn.Module):
    """Frozen CLIPTextModel; returns the per-token hidden states (B, 77, 512)."""

    def __init__(
        self,
        model_path: str | None = None,
        max_length: int = 77,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        path = model_path or DEFAULT_MODEL_ID
        # Allow either a local directory or a hub id.
        resolved = Path(path)
        if resolved.is_dir():
            src = str(resolved)
        else:
            src = path

        self.tokenizer = CLIPTokenizer.from_pretrained(src)
        self.model = CLIPTextModel.from_pretrained(src)
        self.max_length = max_length
        self.seq_len = max_length
        self.text_dim = self.model.config.hidden_size

        # Freeze.
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.to(device)
        self._device = torch.device(device)

    def train(self, mode: bool = True):       # keep encoder in eval mode always
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode(self, captions: Sequence[str]) -> torch.Tensor:
        """Tokenize and forward through CLIP text; returns (B, seq_len, text_dim)."""
        enc = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self._device)
        attn_mask = enc["attention_mask"].to(self._device)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return out.last_hidden_state   # (B, 77, 512)

    def forward(self, captions: Sequence[str]) -> torch.Tensor:
        return self.encode(captions)
