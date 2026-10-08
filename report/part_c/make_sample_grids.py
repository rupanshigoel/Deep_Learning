r"""Build side-by-side comparison grids for the Part C report.

Outputs:
    plots/vae_recon_grid.png        Real (top) vs VAE recon (bottom), 4 cols x 2 examples
                                    -> 2 stacked panels of 4 cols each, total 8 examples
    plots/ldm_samples_grid.png      Real (top) vs LDM gen (bottom), same layout
    plots/three_way_grid_p{1,2}.png Real | VAE recon | LDM gen, 3 examples per page
                                    with full captions wrapped underneath (no clipping)

Designed for a LaTeX-rendered PDF: keep each panel <= 4 cols at 128px each so the
figure fits inside the standard \linewidth without losing detail.
"""
from __future__ import annotations
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
REAL = HERE / "samples" / "real"
VAE = HERE / "samples" / "vae_recon"
LDM = HERE / "samples" / "ldm_gen"
PLOTS = HERE / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

CELL = 128            # image cell width/height in px
PAD = 10              # spacing between cells
COLS_PER_ROW = 4      # pdf-friendly count per row
LABEL_W = 120         # left label column
FONT_SIZE = 14
CAPTION_FONT_SIZE = 12


def _font(size: int = FONT_SIZE):
    """Try to load a real TTF; fall back to PIL default."""
    for cand in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_captions() -> dict:
    cap_json = HERE.parent.parent / "A2_dataset" / "Part_A" / "val" / "clevr_val_captions.json"
    if not cap_json.is_file():
        # Fall back to an empty dict — three_way_grid will render without captions
        # if the dataset isn't present locally.
        return {}
    with open(cap_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {r["image_filename"]: r["caption"] for r in raw}


def _wrap(text: str, max_chars: int) -> list[str]:
    """Greedy word wrap (no ellipsis — return all lines)."""
    words = text.split()
    lines, line = [], ""
    for w in words:
        if not line:
            line = w
        elif len(line) + 1 + len(w) <= max_chars:
            line = line + " " + w
        else:
            lines.append(line); line = w
    if line:
        lines.append(line)
    return lines


def _two_row_panel(top: list[Image.Image], bot: list[Image.Image],
                   labels: tuple[str, str]) -> Image.Image:
    """Render `len(top) == len(bot) == COLS_PER_ROW` images as a 2-row panel."""
    n = len(top)
    canvas_w = LABEL_W + n * (CELL + PAD) - PAD
    canvas_h = 2 * (CELL + PAD) - PAD
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for ri, (row_imgs, lab) in enumerate([(top, labels[0]), (bot, labels[1])]):
        y = ri * (CELL + PAD)
        draw.text((6, y + CELL // 2 - 8), lab, fill="black", font=font)
        for ci, img in enumerate(row_imgs):
            canvas.paste(img.resize((CELL, CELL)), (LABEL_W + ci * (CELL + PAD), y))
    return canvas


def _two_panel_stacked(top_all: list[Image.Image], bot_all: list[Image.Image],
                       labels: tuple[str, str]) -> Image.Image:
    """Stack multiple 2-row panels vertically (handles >COLS_PER_ROW examples)."""
    panels = []
    for i in range(0, len(top_all), COLS_PER_ROW):
        t = top_all[i:i + COLS_PER_ROW]
        b = bot_all[i:i + COLS_PER_ROW]
        # Pad shorter rows with blank cells (white) so width is consistent.
        while len(t) < COLS_PER_ROW:
            t.append(Image.new("RGB", (CELL, CELL), "white"))
            b.append(Image.new("RGB", (CELL, CELL), "white"))
        panels.append(_two_row_panel(t, b, labels))
    if len(panels) == 1:
        return panels[0]
    inter_panel_pad = 24
    W = panels[0].width
    H = sum(p.height for p in panels) + inter_panel_pad * (len(panels) - 1)
    out = Image.new("RGB", (W, H), "white")
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height + inter_panel_pad
    return out


def three_way_with_captions(reals: list[Image.Image], vaes: list[Image.Image],
                             ldms: list[Image.Image], caps: list[str],
                             cols_per_page: int = 3) -> list[Image.Image]:
    """Render a 3-row (Real / VAE / LDM) grid with full captions underneath.

    Returns one Image per page (caps split across pages of `cols_per_page` cols).
    """
    pages = []
    font = _font()
    cap_font = _font(CAPTION_FONT_SIZE)
    line_h = CAPTION_FONT_SIZE + 4

    # Determine caption box height = tallest wrap across captions on this page.
    # CELL=128 + PAD=10 ~ 138px column; at 12 px font, ~20 chars per line stays in column.
    max_chars = 20
    for start in range(0, len(reals), cols_per_page):
        page_caps = caps[start:start + cols_per_page]
        wrapped = [_wrap(c, max_chars) for c in page_caps]
        max_lines = max((len(w) for w in wrapped), default=0)
        cap_h = max_lines * line_h + 10

        n = len(page_caps)
        col_w = CELL + PAD
        page_w = LABEL_W + n * col_w - PAD
        page_h = 3 * (CELL + PAD) - PAD + cap_h
        canvas = Image.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(canvas)

        labels = ["Real", "VAE recon", "LDM gen"]
        for ri, lab in enumerate(labels):
            y = ri * (CELL + PAD)
            draw.text((6, y + CELL // 2 - 8), lab, fill="black", font=font)

        for ci in range(n):
            x = LABEL_W + ci * col_w
            for ri, im in enumerate([reals[start + ci], vaes[start + ci], ldms[start + ci]]):
                canvas.paste(im.resize((CELL, CELL)), (x, ri * (CELL + PAD)))
            cap_y0 = 3 * (CELL + PAD)
            for li, ln in enumerate(wrapped[ci]):
                draw.text((x, cap_y0 + li * line_h), ln, fill="black", font=cap_font)
        pages.append(canvas)
    return pages


def main():
    captions = load_captions()
    files = sorted(p.name for p in REAL.glob("*.png"))
    n_show = min(8, len(files))
    fn_list = files[:n_show]

    reals = [Image.open(REAL / fn).convert("RGB") for fn in fn_list]
    vaes = [Image.open(VAE / fn).convert("RGB") for fn in fn_list]
    ldms = [Image.open(LDM / fn).convert("RGB") for fn in fn_list]

    # 8 examples => 2 stacked panels of 4 cols (PDF-friendly)
    out = _two_panel_stacked(reals, vaes, ("Real", "VAE recon"))
    out.save(PLOTS / "vae_recon_grid.png")
    print(f"[wrote] {PLOTS / 'vae_recon_grid.png'} ({out.size})")

    out = _two_panel_stacked(reals, ldms, ("Real", "LDM gen"))
    out.save(PLOTS / "ldm_samples_grid.png")
    print(f"[wrote] {PLOTS / 'ldm_samples_grid.png'} ({out.size})")

    # Three-way: 6 examples = 2 pages of 3 cols each, full caption wrapping (no clipping)
    n_three = min(6, n_show)
    caps = [captions.get(fn_list[i], "") for i in range(n_three)]
    pages = three_way_with_captions(reals[:n_three], vaes[:n_three], ldms[:n_three], caps)
    if len(pages) == 1:
        pages[0].save(PLOTS / "three_way_grid.png")
        print(f"[wrote] {PLOTS / 'three_way_grid.png'} ({pages[0].size})")
    else:
        for i, page in enumerate(pages, 1):
            out_path = PLOTS / f"three_way_grid_p{i}.png"
            page.save(out_path)
            print(f"[wrote] {out_path} ({page.size})")
        # also write a combined image (vertical stack) for convenience
        W = max(p.width for p in pages)
        pad = 30
        H = sum(p.height for p in pages) + pad * (len(pages) - 1)
        combo = Image.new("RGB", (W, H), "white")
        y = 0
        for p in pages:
            combo.paste(p, (0, y))
            y += p.height + pad
        combo.save(PLOTS / "three_way_grid.png")
        print(f"[wrote] {PLOTS / 'three_way_grid.png'} (combined, {combo.size})")


if __name__ == "__main__":
    main()
