"""Build Stage 1 + Stage 2 training curves from CSV logs.

Stage-2 reads the **full-data** training CSV (`logs_full/stage2_full_train.csv`)
which spans the entire 17 480-optim-step run (chunks 1-7 on user account, then
remainder + top-up on friend account, all appended). LR schedule overlay added
per audit recommendation.

No GPU needed.
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(p):
    rows = list(csv.DictReader(open(p)))
    return [(int(r["step"]), float(r["loss"]), float(r["lr"]),
             float(r["tokens_per_s"]), float(r["epoch_time_min"])) for r in rows]


def _plot_loss(ax, data, label, color):
    steps = [r[0] for r in data]
    losses = [r[1] for r in data]
    ax.plot(steps, losses, color=color, lw=1.0, label=label)


def _plot_lr(ax_right, data, color="0.5"):
    steps = [r[0] for r in data]
    lrs = [r[2] for r in data]
    ax_right.plot(steps, lrs, color=color, lw=0.9, ls="--", label="LR")
    ax_right.set_ylabel("learning rate", color=color)
    ax_right.tick_params(axis="y", labelcolor=color)
    ax_right.set_yscale("log")


def main():
    here = Path(__file__).parent
    s1 = _read_csv(here / "logs/stage1_train.csv")
    s2_old = _read_csv(here / "logs/stage2_train.csv")  # legacy 80K-stratified run, kept for reference
    s2_full = _read_csv(here / "logs_full/stage2_full_train.csv")  # NEW full-data 17 480-step run

    # ---------- side-by-side Stage-1 + Stage-2-full with LR overlay ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))

    s1_steps = s1[-1][0] if s1 else 0
    s2_steps = s2_full[-1][0] if s2_full else 0

    _plot_loss(ax1, s1, "Stage-1 (projector only)", color="#1f77b4")
    ax1.set_xlabel("optim step"); ax1.set_ylabel("train loss")
    ax1.set_title(f"Stage 1 — CLEVR captioning ({s1_steps} optim-steps, eff. batch 32)", fontsize=10)
    ax1.grid(alpha=0.3); ax1.legend(loc="upper right")
    ax1r = ax1.twinx(); _plot_lr(ax1r, s1, color="#888888")
    ax1r.legend(loc="center right", fontsize=8)

    _plot_loss(ax2, s2_full, "Stage-2 (projector + LoRA r=16)", color="#d62728")
    ax2.set_xlabel("optim step"); ax2.set_ylabel("train loss")
    ax2.set_title(f"Stage 2 — CLEVR-X CoT QA full-data ({s2_steps} optim-steps, eff. batch 32)", fontsize=10)
    ax2.grid(alpha=0.3); ax2.legend(loc="upper right")
    ax2r = ax2.twinx(); _plot_lr(ax2r, s2_full, color="#888888")
    ax2r.legend(loc="center right", fontsize=8)

    fig.tight_layout()
    fig.savefig(here / "plots_full/loss_curves_full.png", dpi=150); plt.close(fig)
    print("wrote plots_full/loss_curves_full.png")

    # ---------- Stage 1 alone ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_loss(ax, s1, "Stage-1 train loss", color="#1f77b4")
    ax.set_xlabel("optim step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    axr = ax.twinx(); _plot_lr(axr, s1, color="#888888"); axr.legend(loc="center right", fontsize=8)
    ax.set_title("Stage 1 — CLEVR captioning")
    fig.tight_layout(); fig.savefig(here / "plots_full/stage1_loss.png", dpi=150); plt.close(fig)
    print("wrote plots_full/stage1_loss.png")

    # ---------- Stage 2 full alone ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_loss(ax, s2_full, "Stage-2 train loss (full data)", color="#d62728")
    ax.set_xlabel("optim step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    axr = ax.twinx(); _plot_lr(axr, s2_full, color="#888888"); axr.legend(loc="center right", fontsize=8)
    ax.set_title("Stage 2 — CLEVR-X CoT QA with LoRA (1 epoch over filtered 559 969 entries)")
    fig.tight_layout(); fig.savefig(here / "plots_full/stage2_loss_full.png", dpi=150); plt.close(fig)
    print("wrote plots_full/stage2_loss_full.png")

    # ---------- Old vs new Stage-2 comparison (optional) ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_loss(ax, s2_old, "Stage-2 v1 (80K stratified)", color="#aa7733")
    _plot_loss(ax, s2_full, "Stage-2 v2 (full data, 1 epoch)", color="#d62728")
    ax.set_xlabel("optim step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax.set_title("Stage 2 training: old 80K vs new full-data run")
    fig.tight_layout(); fig.savefig(here / "plots_full/stage2_loss_compare.png", dpi=150); plt.close(fig)
    print("wrote plots_full/stage2_loss_compare.png")

    print("done.")


if __name__ == "__main__":
    main()
