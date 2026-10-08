"""Re-render Stage-1 qualitative samples as one PNG per example, LaTeX-ready.

Layout: image on the LEFT, monospace GT/PRED text on the RIGHT. Everything is
sized in inches so nothing gets cut by `bbox_inches='tight'` or by long captions.
The figure height grows with the text-line count.

Reads `logs_full/stage1_predictions.jsonl`, picks N correct + N incorrect from val,
writes:
    plots_full/stage1_qual_v2_val_correct_NN.png
    plots_full/stage1_qual_v2_val_incorrect_NN.png
"""
from __future__ import annotations
import json, re, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ---- Layout knobs (inches) — sized for LaTeX \includegraphics[width=\linewidth] in a 6.5" column ----
IMG_W_IN     = 2.6          # image axis width
IMG_H_IN     = 1.75         # image axis height (CLEVR aspect ratio ~3:2)
TEXT_W_IN    = 3.6          # text column width
TITLE_H_IN   = 0.30         # space for the colored title bar
PAD_TOP      = 0.12
PAD_BOT      = 0.18
PAD_LR       = 0.20
HSPACE       = 0.22         # gap between image and text column
TEXT_LH_IN   = 0.16         # height per wrapped text line at fontsize 9
WRAP_WIDTH   = 42           # chars per line at fontsize 9 in 3.6" column
FONTSIZE     = 9
TITLE_FS     = 10


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[.!?,;:]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_predictions(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def find_image(project_root: Path, fn: str) -> Path | None:
    for split in ("train", "val"):
        p = project_root / "A2_dataset" / "Part_Aa" / "Clevr_official" / "images" / split / fn
        if p.is_file():
            return p
    return None


def _wrap_block(label: str, body: str) -> str:
    wrapped = textwrap.wrap(body, width=WRAP_WIDTH) or [""]
    indent = " " * (len(label) + 2)
    head = f"{label}: " + wrapped[0]
    rest = [indent + w for w in wrapped[1:]]
    return "\n".join([head] + rest)


def _format_text(rec: dict, stage: int) -> str:
    if stage == 1:
        gt   = rec.get("target_str", "").strip()
        pred = rec.get("pred_raw",   "").strip()
        return _wrap_block("GT  ", gt) + "\n\n" + _wrap_block("PRED", pred)
    # Stage 2 — Q/A schema
    q     = rec.get("question", "").strip()
    gt_a  = rec.get("answer_gt", "").strip()
    gt_fe = rec.get("factual_explanation_gt", "").strip()
    pred  = rec.get("pred_raw", "").strip()
    parts = [
        _wrap_block("Q   ", q),
        _wrap_block("GT-A", gt_a),
        _wrap_block("GT-E", gt_fe),
        "",
        _wrap_block("PRED", pred),
    ]
    return "\n".join(parts)


def render_one(rec, out_path: Path, project_root: Path, correct: bool, stage: int = 1):
    img_path = find_image(project_root, rec["image_filename"])
    text = _format_text(rec, stage)
    n_lines = text.count("\n") + 1

    # Compute figure height: max(image area, text area) + title + paddings.
    body_h_in = max(IMG_H_IN, TEXT_LH_IN * n_lines + 0.3)
    fig_h     = TITLE_H_IN + body_h_in + PAD_TOP + PAD_BOT
    fig_w     = PAD_LR + IMG_W_IN + HSPACE + TEXT_W_IN + PAD_LR

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # Title bar
    title_color = "#1a7f37" if correct else "#cf222e"
    title_text  = ("CORRECT" if correct else "INCORRECT") + f"  ·  {rec['image_filename']}"
    fig.text(PAD_LR / fig_w, 1 - PAD_TOP / fig_h, title_text,
             color=title_color, fontsize=TITLE_FS, fontweight="bold", va="top", ha="left")

    # Image axis (left) — pinned to top-left of body region
    body_top = 1 - (PAD_TOP + TITLE_H_IN) / fig_h
    body_bot = PAD_BOT / fig_h
    img_left   = PAD_LR / fig_w
    img_right  = (PAD_LR + IMG_W_IN) / fig_w
    img_top    = body_top
    img_bot    = body_top - IMG_H_IN / fig_h
    ax_img = fig.add_axes([img_left, img_bot, img_right - img_left, img_top - img_bot])
    ax_img.set_axis_off()
    if img_path:
        ax_img.imshow(Image.open(img_path).convert("RGB"))

    # Text axis (right)
    txt_left  = (PAD_LR + IMG_W_IN + HSPACE) / fig_w
    txt_right = 1 - PAD_LR / fig_w
    ax_txt = fig.add_axes([txt_left, body_bot, txt_right - txt_left, body_top - body_bot])
    ax_txt.set_axis_off()
    ax_txt.text(0.0, 1.0, text, va="top", ha="left",
                fontsize=FONTSIZE, family="monospace")

    fig.savefig(out_path, dpi=150)  # NO bbox_inches='tight' — keeps fixed layout
    plt.close(fig)


ANSWER_RE = re.compile(r"[Aa]nswer\s*:\s*(.*?)(?:\n|$)", re.DOTALL)


def _extract_answer(gen: str) -> str:
    m = ANSWER_RE.search(gen)
    if m: return normalize(m.group(1).split("\n")[0])
    for line in gen.strip().splitlines():
        if line.strip(): return normalize(line)
    return ""


def pick_examples(rows, n: int, stage: int):
    correct, incorrect = [], []
    for r in rows:
        if r["split"] != "val": continue
        if stage == 1:
            ok = normalize(r["pred_raw"]) == normalize(r["target_str"])
        else:
            ok = _extract_answer(r["pred_raw"]) == normalize(r.get("answer_gt", ""))
        bucket = correct if ok else incorrect
        if len(bucket) < n:
            bucket.append((ok, r))
        if len(correct) >= n and len(incorrect) >= n:
            break
    return correct + incorrect


def render_set(stage: int, pred_filename: str, prefix: str):
    here = Path(__file__).parent
    project_root = here.parent.parent
    pred_path = here / "logs_full" / pred_filename
    out_dir = here / "plots_full"
    out_dir.mkdir(exist_ok=True)
    examples = pick_examples(load_predictions(pred_path), n=4, stage=stage)
    counts = {"correct": 0, "incorrect": 0}
    for ok, rec in examples:
        bucket = "correct" if ok else "incorrect"
        counts[bucket] += 1
        idx = counts[bucket]
        out = out_dir / f"{prefix}_v2_val_{bucket}_{idx:02d}.png"
        render_one(rec, out, project_root, correct=ok, stage=stage)
        w, h = Image.open(out).size
        print(f"wrote {out.name}  px={w}x{h}")


def main():
    render_set(stage=1, pred_filename="stage1_predictions.jsonl",       prefix="stage1_qual")
    render_set(stage=2, pred_filename="stage2_full_predictions.jsonl",  prefix="stage2_full_qual")


if __name__ == "__main__":
    main()
