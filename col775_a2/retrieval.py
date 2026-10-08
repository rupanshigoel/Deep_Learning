"""Cross-modal retrieval with the trained CLIP on Part_Aa/val.

For each val image, retrieve captions; for each caption, retrieve images.
Report R@1 and R@3 for both directions. Dump qualitative examples
(successes + failures) as PER-ROW subfigures so the report can use them
with `\\begin{subfigure}` in LaTeX. Also print a quantitative failure-mode
breakdown (how often does the top-1 wrong caption share count / color-set
/ size-set / material-set with GT?).

Uses captions from `Part_Aa/Clevr_official/captions/clevr_val_captions.json`.

Usage:
    python -m col775_a2.retrieval
"""
from __future__ import annotations
import argparse, json, re, textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# cuDNN 9.1's scaled_dot_product_attention graph-rewriter fails on the text
# encoder's causal-mask attention pattern. Disable the cuDNN SDPA path.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from col775_a2.data.clip_dataset import default_eval_transform
from col775_a2.data.tokenizer import SimpleTokenizer
from col775_a2.models.clip_model import CLIPModel


ROOT = Path(__file__).resolve().parent.parent / "A2_dataset" / "Part_Aa"
IMG_DIR = ROOT / "Clevr_official" / "images" / "val"
CAP_PATH = ROOT / "Clevr_official" / "captions" / "clevr_val_captions.json"


class _ValImageDataset(Dataset):
    def __init__(self, entries, img_dir, transform):
        self.entries = entries; self.img_dir = img_dir; self.transform = transform
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        rec = self.entries[idx]
        img = Image.open(self.img_dir / rec["image_filename"]).convert("RGB")
        return self.transform(img), idx


def _load_clip(ckpt_path: Path, device: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = state.get("args", {})
    tok_json = ckpt_path.parent / "tokenizer.json"
    tok = SimpleTokenizer.load(str(tok_json))
    model = CLIPModel(vocab_size=tok.vocab_size, context_length=tok.context_length,
                      img_size=args.get("img_size", 224))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), tok


@torch.no_grad()
def _encode_images(model, entries, img_dir, device, batch_size, workers):
    ds = _ValImageDataset(entries, img_dir, default_eval_transform(224))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=True,
                        shuffle=False, persistent_workers=workers > 0)
    feats = np.zeros((len(ds), 512), dtype=np.float32); seen = 0
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            v = model.encode_image(imgs)
        v = F.normalize(v.float(), dim=-1).cpu().numpy()
        feats[seen:seen+v.shape[0]] = v; seen += v.shape[0]
    return feats


@torch.no_grad()
def _encode_texts(model, tok, captions, device, batch_size):
    feats = np.zeros((len(captions), 512), dtype=np.float32)
    for i in range(0, len(captions), batch_size):
        chunk = captions[i:i+batch_size]
        tokens, eot = tok.encode_batch(chunk)
        tokens = tokens.to(device); eot = eot.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            t = model.encode_text(tokens, eot)
        t = F.normalize(t.float(), dim=-1).cpu().numpy()
        feats[i:i+len(chunk)] = t
    return feats


def _recall(sim: np.ndarray, gt: np.ndarray, ks=(1, 3)):
    order = np.argsort(-sim, axis=1)
    results = {}
    for k in ks:
        hit = np.any(order[:, :k] == gt[:, None], axis=1)
        results[k] = hit.mean()
    return results


# ---- caption parsing for failure-mode analysis ----
_COLORS = {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}
_SIZES = {"small", "large"}
_MATERIALS = {"metal", "rubber"}
_SHAPES = {"cube", "cubes", "sphere", "spheres", "cylinder", "cylinders"}
_COUNT_RE = re.compile(r"with\s+(\d+)\s+objects?", re.I)


def _parse_caption(c: str):
    """Return (count:int, color-multiset, size-multiset, material-multiset, shape-multiset).
    A multiset is a sorted tuple so two parses with the same composition compare equal.
    """
    m = _COUNT_RE.search(c)
    count = int(m.group(1)) if m else -1
    toks = re.findall(r"[a-z]+", c.lower())
    colors = tuple(sorted(t for t in toks if t in _COLORS))
    sizes = tuple(sorted(t for t in toks if t in _SIZES))
    mats = tuple(sorted(t for t in toks if t in _MATERIALS))
    shapes = tuple(sorted(t for t in toks if t in _SHAPES))
    return count, colors, sizes, mats, shapes


def _failure_stats(captions, order, direction: str) -> dict:
    gt = np.arange(len(captions))
    top1 = order[:, 0]
    fails = np.where(top1 != gt)[0]
    n = len(fails)
    out = {"direction": direction, "n_failures": int(n), "n_total": int(len(captions))}
    if n == 0:
        print(f"  [{direction}] no failures")
        return out
    same_c = same_col = same_sz = same_mat = same_sh = 0
    for q in fails:
        gc, gcol, gsz, gmat, gsh = _parse_caption(captions[q])
        pc, pcol, psz, pmat, psh = _parse_caption(captions[top1[q]])
        same_c += int(gc == pc)
        same_col += int(gcol == pcol)
        same_sz += int(gsz == psz)
        same_mat += int(gmat == pmat)
        same_sh += int(gsh == psh)
    out.update({
        "same_count_pct": round(100 * same_c / n, 2),
        "same_color_set_pct": round(100 * same_col / n, 2),
        "same_size_set_pct": round(100 * same_sz / n, 2),
        "same_material_set_pct": round(100 * same_mat / n, 2),
        "same_shape_set_pct": round(100 * same_sh / n, 2),
    })
    print(f"  [{direction}] {n} failures / {len(captions)} queries")
    print(f"     same count        : {out['same_count_pct']:5.1f}%")
    print(f"     same color-set    : {out['same_color_set_pct']:5.1f}%")
    print(f"     same size-set     : {out['same_size_set_pct']:5.1f}%")
    print(f"     same material-set : {out['same_material_set_pct']:5.1f}%")
    print(f"     same shape-set    : {out['same_shape_set_pct']:5.1f}%")
    return out


# ---- per-row qualitative subfigures ----
def _plot_i2t_row(q: int, order_i2t, entries, captions, k: int, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5),
                             gridspec_kw={"width_ratios": [1, 2]})
    img = Image.open(IMG_DIR / entries[q]["image_filename"]).convert("RGB")
    axes[0].imshow(img); axes[0].set_axis_off()
    gt_wrap = textwrap.fill(captions[q], width=40)
    axes[0].set_title(f"query\nGT: {gt_wrap}", fontsize=8)

    lines = []
    for r, t in enumerate(order_i2t[q, :k]):
        marker = "✓" if t == q else "✗"
        wrapped = textwrap.fill(str(captions[t]), width=58,
                                subsequent_indent="       ")
        lines.append(f"{r+1}. [{marker}] {wrapped}")
    axes[1].axis("off")
    axes[1].text(0.0, 0.5, "\n\n".join(lines), va="center",
                 fontsize=8, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_t2i_row(q: int, order_t2i, entries, captions, k: int, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Fix the overflow: query column is 2× as wide as each image column.
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * k + 4, 3.2),
                             gridspec_kw={"width_ratios": [2] + [1] * k})
    query_wrap = textwrap.fill(str(captions[q]), width=28)
    axes[0].axis("off")
    axes[0].text(0.02, 0.5, f"query:\n{query_wrap}", va="center",
                 fontsize=8.5, family="monospace")
    for ci, t in enumerate(order_t2i[q, :k]):
        img = Image.open(IMG_DIR / entries[t]["image_filename"]).convert("RGB")
        axes[ci + 1].imshow(img); axes[ci + 1].set_axis_off()
        marker = "✓" if t == q else "✗"
        axes[ci + 1].set_title(f"{marker} rank {ci + 1}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_qualitative(entries, captions, sim_i2t, sim_t2i, out_dir: Path,
                      k: int = 3, n_show: int = 5, seed: int = 0):
    """Emits ONE png per example, organized as:
        out_dir/retrieval_rows/retrieval_i2t_{success,failure}_{0..n-1}.png
        out_dir/retrieval_rows/retrieval_t2i_{success,failure}_{0..n-1}.png
    """
    rows_dir = out_dir / "retrieval_rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    N = len(entries)
    order_i2t = np.argsort(-sim_i2t, axis=1)
    order_t2i = np.argsort(-sim_t2i, axis=1)
    gt = np.arange(N)

    def _pick(mask, n):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        return rng.choice(idx, size=min(n, len(idx)), replace=False)

    # image -> text
    hits = order_i2t[:, 0] == gt
    for i, q in enumerate(_pick(hits, n_show)):
        _plot_i2t_row(int(q), order_i2t, entries, captions, k,
                      rows_dir / f"retrieval_i2t_success_{i}.png")
    for i, q in enumerate(_pick(~hits, n_show)):
        _plot_i2t_row(int(q), order_i2t, entries, captions, k,
                      rows_dir / f"retrieval_i2t_failure_{i}.png")

    # text -> image
    hits = order_t2i[:, 0] == gt
    for i, q in enumerate(_pick(hits, n_show)):
        _plot_t2i_row(int(q), order_t2i, entries, captions, k,
                      rows_dir / f"retrieval_t2i_success_{i}.png")
    for i, q in enumerate(_pick(~hits, n_show)):
        _plot_t2i_row(int(q), order_t2i, entries, captions, k,
                      rows_dir / f"retrieval_t2i_failure_{i}.png")
    print(f"  wrote per-row subfigures to {rows_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-ckpt", default=str(Path(__file__).resolve().parent / "checkpoints" / "clip_last.pt"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "plots"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = use all val")
    ap.add_argument("--qualitative", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with open(CAP_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    entries = [e for e in entries if (IMG_DIR / e["image_filename"]).is_file()]
    if args.max_samples and args.max_samples < len(entries):
        entries = entries[: args.max_samples]
    captions = [e["caption"] for e in entries]
    print(f"[retrieval] {len(entries)} val (image, caption) pairs")

    model, tok = _load_clip(Path(args.clip_ckpt), device)
    print(f"[retrieval] CLIP loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    img_feats = _encode_images(model, entries, IMG_DIR, device, args.batch_size, args.num_workers)
    txt_feats = _encode_texts(model, tok, captions, device, args.batch_size)
    print(f"[retrieval] img_feats {img_feats.shape}  txt_feats {txt_feats.shape}")

    sim_i2t = img_feats @ txt_feats.T
    sim_t2i = txt_feats @ img_feats.T
    gt = np.arange(len(entries))

    r_i2t = _recall(sim_i2t, gt, ks=(1, 3))
    r_t2i = _recall(sim_t2i, gt, ks=(1, 3))
    print("\n============ Recall@K ============")
    print(f"  Image -> Text   R@1={r_i2t[1]*100:.2f}%   R@3={r_i2t[3]*100:.2f}%")
    print(f"  Text  -> Image  R@1={r_t2i[1]*100:.2f}%   R@3={r_t2i[3]*100:.2f}%")

    # Caption-equivalence relaxed R@1 (same-caption images count as a hit).
    cap_to_gt: dict = {}
    for i, c in enumerate(captions): cap_to_gt.setdefault(c, []).append(i)
    equiv = np.zeros((len(captions), len(captions)), dtype=bool)
    for group in cap_to_gt.values():
        for i in group: equiv[i, group] = True
    top1_i2t = np.argmax(sim_i2t, axis=1)
    top1_t2i = np.argmax(sim_t2i, axis=1)
    eq_i2t = equiv[np.arange(len(captions)), top1_i2t].mean()
    eq_t2i = equiv[np.arange(len(captions)), top1_t2i].mean()
    print(f"  (caption-equivalence) i2t R@1={eq_i2t*100:.2f}%   t2i R@1={eq_t2i*100:.2f}%")

    # Failure-mode breakdown: for each failure, how often does the top-1 wrong
    # caption share the GT's count / color-set / size-set / material-set / shape-set?
    order_i2t = np.argsort(-sim_i2t, axis=1)
    order_t2i = np.argsort(-sim_t2i, axis=1)
    print("\n============ Failure-mode breakdown ============")
    stats = {
        "i2t": _failure_stats(captions, order_i2t, "i2t"),
        "t2i": _failure_stats(captions, order_t2i, "t2i"),
        "recall": {
            "i2t_R@1": round(float(r_i2t[1]) * 100, 2),
            "i2t_R@3": round(float(r_i2t[3]) * 100, 2),
            "t2i_R@1": round(float(r_t2i[1]) * 100, 2),
            "t2i_R@3": round(float(r_t2i[3]) * 100, 2),
            "i2t_R@1_cap_equiv": round(float(eq_i2t) * 100, 2),
            "t2i_R@1_cap_equiv": round(float(eq_t2i) * 100, 2),
        },
    }
    stats_path = out_dir / "retrieval_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {stats_path}")

    _save_qualitative(entries, captions, sim_i2t, sim_t2i, out_dir,
                      k=3, n_show=args.qualitative)


if __name__ == "__main__":
    main()
