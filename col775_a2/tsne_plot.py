"""t-SNE visualization of training embeddings, color-coded by object count.

Generates one scatter per (encoder, pool). Defaults to CLS pooling since
the assignment table is titled "CLS Embedding", but both are available.

Needs `scikit-learn` and `matplotlib`. Use `openTSNE` via `--backend opentsne`
if sklearn's TSNE is too slow on 70k points (sklearn runs O(N^2) or Barnes-Hut
with perplexity; 70k in ~20 min is typical).

Usage:
    python -m col775_a2.tsne_plot --pool cls
    python -m col775_a2.tsne_plot --subsample 10000
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def _tsne_sklearn(x: np.ndarray, perplexity: float, n_iter: int, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    # For newer sklearn the arg is `max_iter`; fall back gracefully.
    kw = dict(n_components=2, perplexity=perplexity, init="pca",
              learning_rate="auto", random_state=seed, verbose=1)
    try:
        return TSNE(max_iter=n_iter, **kw).fit_transform(x)
    except TypeError:
        return TSNE(n_iter=n_iter, **kw).fit_transform(x)


def _plot_one(xy: np.ndarray, counts: np.ndarray, title: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=counts, cmap="viridis", s=3, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="object count")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=str(Path(__file__).resolve().parent / "embeddings"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "plots"))
    ap.add_argument("--encoders", nargs="+",
                    default=["clip", "dino_student", "dino_teacher"])
    ap.add_argument("--pool", choices=["cls", "gap"], default="cls")
    ap.add_argument("--split", default="train")
    ap.add_argument("--subsample", type=int, default=0,
                    help="0 = use all; else randomly pick this many points")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for enc in args.encoders:
        p = emb_dir / f"{enc}_{args.split}.npz"
        if not p.is_file():
            print(f"[skip] missing {p}"); continue
        d = np.load(p, allow_pickle=True)
        x = d[args.pool].astype(np.float32); counts = d["count"].astype(np.int64)
        if args.subsample and args.subsample < x.shape[0]:
            idx = rng.choice(x.shape[0], size=args.subsample, replace=False)
            x = x[idx]; counts = counts[idx]
        print(f"[tsne] {enc}: {x.shape}")
        xy = _tsne_sklearn(x, args.perplexity, args.n_iter, args.seed)
        np.savez_compressed(out_dir / f"tsne_{enc}_{args.pool}.npz", xy=xy, count=counts)
        _plot_one(xy, counts, f"{enc}  |  pool={args.pool}", out_dir / f"tsne_{enc}_{args.pool}.png")


if __name__ == "__main__":
    main()
