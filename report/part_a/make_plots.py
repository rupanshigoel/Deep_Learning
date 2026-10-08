"""Regenerate every Part_A / Part_Aa plot from the CSVs / NPZs in this folder.

Pure matplotlib; no GPU needed. Run locally:

    python report/part_a/make_plots.py

Inputs (already on disk):
    report/part_a/logs/clip_train.csv
    report/part_a/logs/dino_train.csv
    report/part_a/plots/tsne_{clip,dino_student,dino_teacher}_cls.npz

Outputs (written to report/part_a/plots/):
    clip_loss.png              train-loss curve (log-y)
    clip_lr_tau.png            LR + learnable τ⁻¹ trajectory
    dino_loss.png              train-loss curve (linear)
    dino_schedules.png         teacher τ warmup + EMA momentum
    epoch_time.png             per-epoch wall clock, both models
    tsne_{enc}_cls.png         re-rendered with discrete colorbar
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)


def _load_csv(path: Path) -> Dict[str, List[float]]:
    rows: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k, v in r.items():
                rows.setdefault(k, []).append(float(v))
    return rows


def _plot_clip_loss(clip: Dict[str, List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.semilogy(clip["epoch"], clip["train_loss"], color="#1f77b4", lw=1.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss (log scale)")
    ax.set_title("CLIP — symmetric InfoNCE training loss (100 epochs)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_xlim(0, max(clip["epoch"]))
    fig.tight_layout()
    out = PLOTS / "clip_loss.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  wrote {out.name}")


def _plot_clip_lr_tau(clip: Dict[str, List[float]]) -> None:
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax1.plot(clip["epoch"], clip["lr"], color="#1f77b4", lw=1.8, label="learning rate")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("LR", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, ls=":", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(clip["epoch"], clip["logit_scale"], color="#d62728", lw=1.8,
             label="learnable τ⁻¹")
    ax2.set_ylabel("τ⁻¹  (= exp(logit_scale))", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.axhline(100.0, color="#d62728", ls="--", lw=1.0, alpha=0.4)
    ax2.text(max(clip["epoch"]) * 0.02, 101.0, "clamp @ 100",
             color="#d62728", fontsize=8, alpha=0.7)

    ax1.set_title("CLIP — LR (cosine + warmup) and learnable temperature")
    fig.tight_layout()
    out = PLOTS / "clip_lr_tau.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  wrote {out.name}")


def _plot_dino_loss(dino: Dict[str, List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(dino["epoch"], dino["train_loss"], color="#2ca02c", lw=1.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss (multi-crop CE)")
    ax.set_title("DINO — teacher↔student distillation loss (100 epochs)")
    ax.axhline(np.log(4096), color="gray", ls="--", lw=1.0, alpha=0.5)
    ax.text(max(dino["epoch"]) * 0.45, np.log(4096) - 0.30,
            r"log(4096) $\approx 8.32$  (uniform-center ceiling)",
            color="gray", fontsize=8, ha="center")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlim(0, max(dino["epoch"]))
    ax.set_ylim(min(dino["train_loss"]) - 0.2, np.log(4096) + 0.3)
    fig.tight_layout()
    out = PLOTS / "dino_loss.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  wrote {out.name}")


def _plot_dino_schedules(dino: Dict[str, List[float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))

    ax = axes[0]
    ax.plot(dino["epoch"], dino["teacher_temp"], color="#ff7f0e", lw=1.8)
    ax.set_xlabel("epoch"); ax.set_ylabel(r"teacher $\tau_t$")
    ax.set_title("Teacher temperature warmup")
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1]
    ax.plot(dino["epoch"], dino["momentum"], color="#9467bd", lw=1.8)
    ax.set_xlabel("epoch"); ax.set_ylabel(r"teacher EMA momentum $m$")
    ax.set_title("Teacher EMA momentum schedule")
    ax.set_ylim(0.995, 1.0005)
    ax.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    out = PLOTS / "dino_schedules.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  wrote {out.name}")


def _plot_epoch_times(clip, dino):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(clip["epoch"], clip["epoch_time_min"], color="#1f77b4",
            lw=1.6, label=f"CLIP ({sum(clip['epoch_time_min']):.0f} min total)")
    ax.plot(dino["epoch"], dino["epoch_time_min"], color="#2ca02c",
            lw=1.6, label=f"DINO ({sum(dino['epoch_time_min']):.0f} min total)")
    ax.set_xlabel("epoch"); ax.set_ylabel("epoch wall-clock (min)")
    ax.set_title("Per-epoch training time — A100-40GB")
    ax.grid(True, ls=":", alpha=0.5); ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = PLOTS / "epoch_time.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- t-SNE re-rendering with a DISCRETE colormap ----------
def _plot_tsne(enc: str) -> None:
    npz = PLOTS / f"tsne_{enc}_cls.npz"
    if not npz.is_file():
        print(f"  [skip] {npz.name} not found"); return
    d = np.load(npz, allow_pickle=True)
    xy = d["xy"]
    counts = d["count"].astype(int)
    lo, hi = int(counts.min()), int(counts.max())     # CLEVR: 3..10
    n_cls = hi - lo + 1

    # Build a discrete colormap with exactly n_cls shades from viridis.
    base = plt.get_cmap("viridis", n_cls)
    cmap = ListedColormap([base(i) for i in range(n_cls)])
    boundaries = np.arange(lo - 0.5, hi + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, ncolors=n_cls)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=counts, cmap=cmap, norm=norm,
                    s=3, alpha=0.75, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, ticks=np.arange(lo, hi + 1), fraction=0.046, pad=0.04)
    cb.set_label("object count", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    pretty = {"clip": "CLIP", "dino_student": "DINO Student", "dino_teacher": "DINO Teacher"}
    ax.set_title(f"{pretty.get(enc, enc)}  —  t-SNE on [CLS] (70K train images)",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out = PLOTS / f"tsne_{enc}_cls.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    clip_csv = LOGS / "clip_train.csv"
    dino_csv = LOGS / "dino_train.csv"
    if clip_csv.is_file():
        clip = _load_csv(clip_csv)
        _plot_clip_loss(clip)
        _plot_clip_lr_tau(clip)
    if dino_csv.is_file():
        dino = _load_csv(dino_csv)
        _plot_dino_loss(dino)
        _plot_dino_schedules(dino)
    if clip_csv.is_file() and dino_csv.is_file():
        _plot_epoch_times(clip, dino)

    for enc in ("clip", "dino_student", "dino_teacher"):
        _plot_tsne(enc)

    _redraw_confusions()

    print(f"\n  all plots written to {PLOTS}/")


def _redraw_confusions() -> None:
    """Re-render every count_confusion_*.csv as a PNG with correct axis labels.

    The probe writes raw 11x11 matrices where row/col index == count value.
    Empty leading/trailing classes (counts 0..2) are cropped, axes are labelled
    with actual count values (3..10 for CLEVR). Lets us fix mislabelled PNGs
    without re-running the HPC probe.
    """
    cm_dir = PLOTS / "probe_analysis"
    if not cm_dir.is_dir():
        print("  [skip] no probe_analysis/ — run linear_probe on HPC first"); return
    pretty = {"clip": "CLIP", "dino_student": "DINO Student", "dino_teacher": "DINO Teacher"}
    for csv_path in sorted(cm_dir.glob("count_confusion_*.csv")):
        cm = np.loadtxt(csv_path, delimiter=",", dtype=np.int64)
        used = np.flatnonzero(cm.sum(axis=0) + cm.sum(axis=1))
        if len(used) == 0:
            continue
        lo, hi = int(used.min()), int(used.max())
        cm = cm[lo:hi + 1, lo:hi + 1]
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sum

        # Decode "count_confusion_<enc>_<pool>.csv" filename.
        stem = csv_path.stem.replace("count_confusion_", "")
        if stem.endswith("_cls"): enc, pool = stem[:-4], "CLS"
        elif stem.endswith("_gap"): enc, pool = stem[:-4], "GAP"
        else: enc, pool = stem, ""

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] == 0:
                    continue
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=8, color="black" if cm_norm[i, j] < 0.6 else "white")
        labels = [str(i + lo) for i in range(cm.shape[0])]
        ax.set_xticks(range(cm.shape[0])); ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("predicted count"); ax.set_ylabel("ground-truth count")
        ax.set_title(f"{pretty.get(enc, enc)}  ({pool})  —  val count confusion",
                     fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
        fig.tight_layout()
        out_path = cm_dir / f"count_confusion_{enc}_{pool.lower()}.png"
        fig.savefig(out_path, dpi=150); plt.close(fig)
        print(f"  wrote probe_analysis/{out_path.name}")


if __name__ == "__main__":
    main()
