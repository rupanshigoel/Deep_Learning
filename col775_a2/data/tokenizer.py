"""Word-level tokenizer built from CLEVR-style captions.

Captions look like:
    "An image with 3 objects: 1 small red metal cube, 2 large blue rubber spheres"

Since the caption vocabulary is tiny and fixed (numbers 1-10, sizes, colors,
materials, shapes, plus a few stop-words), a simple whitespace / punctuation
tokenizer suffices. Special tokens:
    [PAD] = 0   — used for padding (also causal-mask agnostic)
    [SOS] = 1   — start-of-text
    [EOS] = 2   — end-of-text (the token whose hidden state is projected)
    [UNK] = 3   — fallback for unseen words (should not occur if vocab is
                  built from full training captions)
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Iterable, List


_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")


def basic_tokenize(text: str) -> List[str]:
    """Split on words / numbers, drop punctuation, lowercase."""
    return _TOKEN_RE.findall(text.lower())


PAD = "[PAD]"
SOS = "[SOS]"
EOS = "[EOS]"
UNK = "[UNK]"
SPECIAL = [PAD, SOS, EOS, UNK]


class SimpleTokenizer:
    def __init__(self, vocab: List[str], context_length: int = 64):
        self.context_length = context_length
        self.itos = list(vocab)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.sos_id = self.stoi[SOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, captions: Iterable[str], context_length: int = 64) -> "SimpleTokenizer":
        vocab: List[str] = list(SPECIAL)
        seen = set(vocab)
        for cap in captions:
            for w in basic_tokenize(cap):
                if w not in seen:
                    seen.add(w)
                    vocab.append(w)
        return cls(vocab, context_length=context_length)

    def encode(self, text: str) -> List[int]:
        """Return [SOS] + tokens(truncated) + [EOS]; caller pads."""
        ids = [self.sos_id]
        for w in basic_tokenize(text):
            ids.append(self.stoi.get(w, self.unk_id))
            if len(ids) >= self.context_length - 1:
                break
        ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: List[str]):
        """Return (tokens LongTensor [B,L], eot_idx LongTensor [B])."""
        import torch
        L = self.context_length
        B = len(texts)
        tokens = torch.full((B, L), self.pad_id, dtype=torch.long)
        eot = torch.zeros(B, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = self.encode(t)
            tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            eot[i] = len(ids) - 1  # position of [EOS]
        return tokens, eot

    # -- persistence --
    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.itos, "context_length": self.context_length}, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], context_length=d["context_length"])
