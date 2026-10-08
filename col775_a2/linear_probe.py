"""Train linear probes on frozen embeddings and print Table 1.

Two tasks:
  * Counting    — CE loss, report top-1 accuracy.
  * Color Pred. — BCE-with-logits, report macro-F1 @ threshold 0.5.

Two embedding strategies: CLS and GAP.
Three encoders:            CLIP, DINO Student, DINO Teacher.

Loads precomputed `.npz` from `embeddings/<encoder>_<split>.npz` written by
`extract_embeddings.py`. No model / image is touched here.

Additional artifacts written into `<out-dir>/probe_analysis/`:
  * `count_confusion_<encoder>_<pool>.csv`  — confusion matrix (rows=GT, cols=pred).
  * `count_confusion_<encoder>_<pool>.png`  — rendered heatmap.
  * `color_f1_<encoder>_<pool>.csv`         — per-colour F1 (train + val).
  * `probe_summary.json`                    — all numbers in one structured file.

Usage:
    python -m col775_a2.linear_probe
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load(emb_dir: Path, tag: str, split: str):
    p = emb_dir / f"{tag}_{split}.npz"
    d = np.load(p, allow_pickle=True)
    return d


def _train_count(x_tr, y_tr, x_va, y_va, *, epochs, batch_size, lr, device):
    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    clf = nn.Linear(x_tr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).long().to(device)
    x_va_t = torch.from_numpy(x_va).to(device)
    y_va_t = torch.from_numpy(y_va).long().to(device)
    N = x_tr_t.size(0)
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(clf(x_tr_t[idx]), y_tr_t[idx])
            loss.backward(); opt.step()
    with torch.no_grad():
        tr_logits = clf(x_tr_t); va_logits = clf(x_va_t)
        tr_acc = (tr_logits.argmax(1) == y_tr_t).float().mean().item()
        va_acc = (va_logits.argmax(1) == y_va_t).float().mean().item()
    return tr_acc, va_acc, tr_logits.detach(), va_logits.detach(), y_tr_t, y_va_t, num_classes


def _per_label_f1(logits: torch.Tensor, y: torch.Tensor, thresh: float = 0.5) -> np.ndarray:
    pred = (logits.sigmoid() >= thresh).float()
    tp = (pred * y).sum(dim=0)
    fp = (pred * (1 - y)).sum(dim=0)
    fn = ((1 - pred) * y).sum(dim=0)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return f1.detach().cpu().numpy()


def _train_colors(x_tr, y_tr, x_va, y_va, *, epochs, batch_size, lr, device):
    C = y_tr.shape[1]
    clf = nn.Linear(x_tr.shape[1], C).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).float().to(device)
    x_va_t = torch.from_numpy(x_va).to(device)
    y_va_t = torch.from_numpy(y_va).float().to(device)
    N = x_tr_t.size(0)
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(clf(x_tr_t[idx]), y_tr_t[idx])
            loss.backward(); opt.step()
    with torch.no_grad():
        tr_logits = clf(x_tr_t); va_logits = clf(x_va_t)
        tr_f1_per = _per_label_f1(tr_logits, y_tr_t)
        va_f1_per = _per_label_f1(va_logits, y_va_t)
    return float(tr_f1_per.mean()), float(va_f1_per.mean()), tr_f1_per, va_f1_per


def _confusion_matrix(pred: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred, y):
        cm[t, p] += 1
    return cm


def _save_confusion_png(cm: np.ndarray, title: str, out_path: Path):
    """Heatmap of a count-confusion matrix.

    The probe's class index is the raw count value (CLEVR counts: 3..10), so
    rows/cols below the dataset's min count are always zero. We crop to the
    smallest sub-matrix that contains every non-zero row and column and label
    the axes with the actual count values.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = np.flatnonzero(cm.sum(axis=0) + cm.sum(axis=1))
    if len(used) == 0:
        return
    lo, hi = int(used.min()), int(used.max())
    cm = cm[lo:hi + 1, lo:hi + 1]

    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sum
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
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_per_color_csv(color_vocab, tr_f1, va_f1, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["color", "train_f1", "val_f1"])
        for c, t, v in zip(color_vocab, tr_f1, va_f1):
            w.writerow([c, f"{t:.4f}", f"{v:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=str(Path(__file__).resolve().parent / "embeddings"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "plots"))
    ap.add_argument("--encoders", nargs="+",
                    default=["clip", "dino_student", "dino_teacher"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_dir = Path(args.emb_dir)
    analysis_dir = Path(args.out_dir) / "probe_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows: Dict[str, Dict[str, tuple]] = {}
    summary: Dict[str, dict] = {}

    for enc in args.encoders:
        tr = _load(emb_dir, enc, "train"); va = _load(emb_dir, enc, "val")
        color_vocab = [str(c) for c in tr["color_vocab"].tolist()]
        rows[enc] = {}; summary[enc] = {}
        for pool in ("cls", "gap"):
            x_tr = tr[pool].astype(np.float32); x_va = va[pool].astype(np.float32)
            cnt_tr_l = tr["count"].astype(np.int64); cnt_va_l = va["count"].astype(np.int64)
            col_tr_l = tr["colors"].astype(np.float32); col_va_l = va["colors"].astype(np.float32)
            print(f"\n[probe] {enc}  pool={pool}  X={x_tr.shape}  count_classes={cnt_tr_l.max()+1}")
            cnt_tr, cnt_va, tr_cnt_logits, va_cnt_logits, y_tr_t, y_va_t, num_cnt_cls = \
                _train_count(x_tr, cnt_tr_l, x_va, cnt_va_l,
                             epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.lr, device=device)
            col_tr, col_va, tr_f1_per, va_f1_per = _train_colors(
                x_tr, col_tr_l, x_va, col_va_l,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=device)
            rows[enc][pool] = (cnt_tr, cnt_va, col_tr, col_va)

            # --- confusion matrix (val) ---
            va_pred = va_cnt_logits.argmax(1).cpu().numpy()
            cm = _confusion_matrix(va_pred, cnt_va_l, num_cnt_cls)
            cm_csv = analysis_dir / f"count_confusion_{enc}_{pool}.csv"
            np.savetxt(cm_csv, cm, fmt="%d", delimiter=",")
            cm_png = analysis_dir / f"count_confusion_{enc}_{pool}.png"
            _save_confusion_png(cm,
                                f"{enc}  ({pool})  — val count confusion", cm_png)
            print(f"  wrote {cm_csv.name}  +  {cm_png.name}")

            # --- per-colour F1 ---
            f1_csv = analysis_dir / f"color_f1_{enc}_{pool}.csv"
            _save_per_color_csv(color_vocab, tr_f1_per, va_f1_per, f1_csv)
            print(f"  per-colour F1 (val)  " +
                  "  ".join(f"{c}={v:.3f}" for c, v in zip(color_vocab, va_f1_per)))

            summary[enc][pool] = {
                "count_acc_train": round(cnt_tr, 4),
                "count_acc_val":   round(cnt_va, 4),
                "color_macro_f1_train": round(col_tr, 4),
                "color_macro_f1_val":   round(col_va, 4),
                "color_per_label_train": {c: round(float(v), 4) for c, v in zip(color_vocab, tr_f1_per)},
                "color_per_label_val":   {c: round(float(v), 4) for c, v in zip(color_vocab, va_f1_per)},
                "count_confusion_val": cm.tolist(),
                "count_classes": num_cnt_cls,
            }

    # ---- Table 1 ----
    print("\n======================= Table 1 =======================")
    header = (
        f"{'Model':<14}| CLS-count Tr/Va    | CLS-color Tr/Va    |"
        f" GAP-count Tr/Va    | GAP-color Tr/Va"
    )
    print(header); print("-" * len(header))
    name_map = {"clip": "CLIP", "dino_student": "DINO Student", "dino_teacher": "DINO Teacher"}
    for enc in args.encoders:
        c_cnt_tr, c_cnt_va, c_col_tr, c_col_va = rows[enc]["cls"]
        g_cnt_tr, g_cnt_va, g_col_tr, g_col_va = rows[enc]["gap"]
        print(f"{name_map.get(enc, enc):<14}|"
              f" {c_cnt_tr:.3f} / {c_cnt_va:.3f}   |"
              f" {c_col_tr:.3f} / {c_col_va:.3f}   |"
              f" {g_cnt_tr:.3f} / {g_cnt_va:.3f}   |"
              f" {g_col_tr:.3f} / {g_col_va:.3f}")

    summary_path = analysis_dir / "probe_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote {summary_path}")


if __name__ == "__main__":
    main()
