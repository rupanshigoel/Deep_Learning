"""Render Part C report figures from logged CSVs.

Outputs:
    plots/vae_loss.png        train+val recon-MSE on log-y (separate subplot)
    plots/vae_kl.png          per-element KL (un-scaled)
    plots/vae_lr.png          lr schedule
    plots/vae_epoch_time.png  per-epoch wallclock (clipped y-axis)
    plots/ldm_loss.png        train_loss + val_loss (log-y)
    plots/ldm_loss_linear.png same but linear-y, zoomed
    plots/ldm_lr.png          lr schedule
    plots/ldm_epoch_time.png  per-epoch wallclock (clipped y-axis)

CSV schemas:
    vae_train.csv: epoch,train_loss,train_recon,train_kl,val_loss,val_recon,val_kl,lr,epoch_time_min
    ldm_train.csv: epoch,train_loss,val_loss,lr,epoch_time_min
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
PLOTS = HERE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
DPI = 200


def read_csv(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            try:
                cols[k].append(float(v))
            except (TypeError, ValueError):
                cols[k].append(None)
    return cols


def _filter_pairs(ep, vals):
    """Drop None val entries (val_* columns are sparse for some runs)."""
    return [(e, v) for e, v in zip(ep, vals) if v is not None]


def _plot_val(ax, val_pairs, *, color, line_label, marker_label_fmt):
    """Plot val series as a line if multi-point, else as a labelled marker.

    The VAE training run only persisted a single final-epoch val number
    (the curve was added to the trainer post-hoc); a 1-point series doesn't
    show up on a log-y line plot, so we render it as a star marker with the
    numeric value in the legend instead.
    """
    if len(val_pairs) >= 2:
        ve, vv = zip(*val_pairs)
        ax.plot(ve, vv, label=line_label, color=color)
    elif len(val_pairs) == 1:
        ve, vv = val_pairs[0]
        ax.plot([ve], [vv], marker="*", markersize=14, color=color, linestyle="",
                label=marker_label_fmt.format(epoch=int(ve), val=vv))


def plot_vae(cols: dict):
    ep = cols["epoch"]

    # ---- (1) Reconstruction MSE on log-y, train + val ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, cols["train_recon"], label="train recon (MSE)", color="tab:blue")
    _plot_val(
        ax, _filter_pairs(ep, cols["val_recon"]),
        color="tab:orange",
        line_label="val recon (MSE)",
        marker_label_fmt="val recon @ ep {epoch} = {val:.6f}",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE")
    ax.set_yscale("log")
    ax.set_title("VAE reconstruction loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "vae_loss.png"
    plt.savefig(out, dpi=DPI); plt.close()
    print(f"[wrote] {out}")

    # ---- (2) KL term on its own (linear y) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, cols["train_kl"], label="train KL (per-element)", color="tab:red")
    _plot_val(
        ax, _filter_pairs(ep, cols["val_kl"]),
        color="tab:purple",
        line_label="val KL",
        marker_label_fmt="val KL @ ep {epoch} = {val:.4f}",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL (un-scaled, per-element mean)")
    ax.set_title("VAE KL divergence (lambda = 1e-6 in total loss)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "vae_kl.png"
    plt.savefig(out, dpi=DPI); plt.close()
    print(f"[wrote] {out}")

    # ---- (3) LR schedule ----
    plt.figure(figsize=(7, 3))
    plt.plot(ep, cols["lr"])
    plt.xlabel("epoch"); plt.ylabel("learning rate")
    plt.title("VAE learning-rate schedule (1 ep warmup + cosine decay)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "vae_lr.png"
    plt.savefig(out, dpi=DPI); plt.close()
    print(f"[wrote] {out}")

    # ---- (4) Epoch time (clip y to drop the cold-start spike at ep 0) ----
    plt.figure(figsize=(7, 3))
    plt.plot(ep, cols["epoch_time_min"])
    plt.xlabel("epoch"); plt.ylabel("wallclock (min)")
    plt.title("VAE epoch time (A100). Epoch 0 includes data-loader warmup.")
    plt.ylim(2.0, 3.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "vae_epoch_time.png"
    plt.savefig(out, dpi=DPI); plt.close()
    print(f"[wrote] {out}")


def plot_ldm(cols: dict):
    ep = cols["epoch"]

    # ---- (1) Loss on log-y ----
    plt.figure(figsize=(7, 4))
    plt.plot(ep, cols["train_loss"], label="train MSE (eps vs eps_hat)", color="tab:blue")
    val_pairs = _filter_pairs(ep, cols["val_loss"])
    if val_pairs:
        ve, vv = zip(*val_pairs)
        plt.plot(ve, vv, label="val MSE", color="tab:orange")
    plt.xlabel("epoch")
    plt.ylabel("eps-prediction MSE (log)")
    plt.yscale("log")
    plt.title("LDM training loss (cosine T=500, CFG-dropout 10%)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "ldm_loss.png"
    plt.savefig(out, dpi=DPI); plt.close()
    print(f"[wrote] {out}")

    # ---- (1b) Linear-y companion zoomed to the convergence band ----
    plt.figure(figsize=(7, 4))
    plt.plot(ep, cols["train_loss"], label="train MSE", color="tab:blue")
    if val_pairs:
        ve, vv = zip(*val_pairs)
        plt.plot(ve, vv, label="val MSE", color="tab:orange")
    plt.xlabel("epoch")
    plt.ylabel("eps-prediction MSE")
    plt.title("LDM loss (linear scale, zoomed)")
    plt.ylim(0.10, 0.20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "ldm_loss_linear.png"
    plt.savefig(out, dpi=DPI); plt.close()
    print(f"[wrote] {out}")

    # ---- (2) LR schedule ----
    plt.figure(figsize=(7, 3))
    plt.plot(ep, cols["lr"])
    plt.xlabel("epoch"); plt.ylabel("learning rate")
    plt.title("LDM learning-rate schedule (1 ep warmup + cosine decay)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "ldm_lr.png"
    plt.savefig(out, dpi=DPI); plt.close()
    print(f"[wrote] {out}")

    # ---- (3) Epoch time (clip y) ----
    plt.figure(figsize=(7, 3))
    plt.plot(ep, cols["epoch_time_min"])
    plt.xlabel("epoch"); plt.ylabel("wallclock (min)")
    plt.title("LDM epoch time (A100). Epoch 0 includes warmup.")
    plt.ylim(2.5, 3.6)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "ldm_epoch_time.png"
    plt.savefig(out, dpi=DPI); plt.close()
    print(f"[wrote] {out}")


def main():
    plot_vae(read_csv(LOGS / "vae_train.csv"))
    plot_ldm(read_csv(LOGS / "ldm_train.csv"))


if __name__ == "__main__":
    main()
