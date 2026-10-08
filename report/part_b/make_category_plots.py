"""Offline plots from stage2_eval_predictions.jsonl: per-category EM bar chart
+ count confusion matrix (predicted vs true count, 0..10). No GPU needed.

Run from `report/part_b/`:
    python make_category_plots.py
Writes:
    plots/stage2_category_em.png
    plots/stage2_count_confmat.png
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ANSWER_RE = re.compile(r"[Aa]nswer\s*:\s*(.*?)(?:\n|$)", re.DOTALL)
COLORS    = {"red", "blue", "green", "yellow", "purple", "cyan", "brown", "gray"}
SHAPES    = {"sphere", "cube", "cylinder"}
MATERIALS = {"rubber", "metal"}
SIZES     = {"small", "large"}
YESNO     = {"yes", "no"}


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[.!?,;:]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_answer(gen: str) -> str:
    m = ANSWER_RE.search(gen)
    if m:
        return normalize(m.group(1).split("\n")[0])
    for line in gen.strip().splitlines():
        if line.strip():
            return normalize(line)
    return ""


def categorize(ans: str) -> str:
    ans = ans.strip().lower()
    if ans in YESNO:     return "yes/no"
    if ans in COLORS:    return "color"
    if ans in SHAPES:    return "shape"
    if ans in MATERIALS: return "material"
    if ans in SIZES:     return "size"
    if ans.isdigit() and 0 <= int(ans) <= 10: return "count"
    return "other"


def load_predictions(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["pred_answer"] = extract_answer(r.get("pred_raw", ""))
            r["gt_answer"]   = normalize(r.get("answer_gt", ""))
            r["category"]    = categorize(r["gt_answer"])
            rows.append(r)
    return rows


def plot_category_em(rows, out_path: Path, split: str = "val"):
    rows = [r for r in rows if r["split"] == split]
    cats = ["color", "material", "size", "shape", "yes/no", "count"]
    em, n = {}, {}
    for c in cats:
        sub = [r for r in rows if r["category"] == c]
        em[c] = (sum(int(r["pred_answer"] == r["gt_answer"]) for r in sub) / len(sub)) if sub else 0.0
        n[c]  = len(sub)

    overall = sum(int(r["pred_answer"] == r["gt_answer"]) for r in rows) / max(1, len(rows))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(cats))
    vals = [em[c]*100 for c in cats]
    bars = ax.bar(xs, vals, color=["#4c72b0","#55a868","#c44e52","#8172b2","#ccb974","#dd8452"], edgecolor="black", lw=0.5)
    ax.axhline(overall*100, ls="--", color="black", lw=1.0, alpha=0.7, label=f"overall = {overall*100:.1f}%")
    ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylabel("Exact-match accuracy (%)", fontsize=11)
    ax.set_title(f"Stage-2 per-category EM ({split} set, n={len(rows)})", fontsize=12)
    ax.set_ylim(0, 105)
    for b, v, c in zip(bars, vals, cats):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%\nn={n[c]}",
                ha="center", va="bottom", fontsize=9)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"wrote {out_path}  (overall {overall*100:.2f}%, per-cat: {em})")


def plot_count_confusion(rows, out_path: Path, split: str = "val"):
    rows = [r for r in rows if r["split"] == split and r["category"] == "count"]
    K = 11  # 0..10
    cm = np.zeros((K, K), dtype=int)
    bad = 0
    for r in rows:
        gt = int(r["gt_answer"])
        try:
            pred = int(r["pred_answer"])
            if not (0 <= pred <= 10):
                bad += 1; continue
        except Exception:
            bad += 1; continue
        cm[gt, pred] += 1

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    diag = np.trace(cm) / max(1, cm.sum())
    off_by_one = sum(cm[i, j] for i in range(K) for j in range(K) if abs(i-j) == 1) / max(1, cm.sum())

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(range(K)); ax.set_yticklabels(range(K))
    ax.set_xlabel("predicted count", fontsize=11)
    ax.set_ylabel("ground-truth count", fontsize=11)
    ax.set_title(f"Stage-2 count confusion ({split}, n={len(rows)})\n"
                 f"diag = {diag*100:.1f}%, off-by-one band = {off_by_one*100:.1f}%, "
                 f"unparsed = {bad}", fontsize=11)
    for i in range(K):
        for j in range(K):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.45 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="row-normalized rate", fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"wrote {out_path}  diag={diag*100:.2f}%  off-by-1={off_by_one*100:.2f}%  unparsed={bad}")


def main():
    here = Path(__file__).parent
    pred_path = here / "logs_full/stage2_full_predictions.jsonl"
    rows = load_predictions(pred_path)
    print(f"loaded {len(rows)} rows from {pred_path}")
    plots_dir = here / "plots_full"
    plots_dir.mkdir(exist_ok=True)
    for split in ("val", "train"):
        plot_category_em(rows, plots_dir / f"stage2_full_category_em_{split}.png", split=split)
        plot_count_confusion(rows, plots_dir / f"stage2_full_count_confmat_{split}.png", split=split)


if __name__ == "__main__":
    main()
