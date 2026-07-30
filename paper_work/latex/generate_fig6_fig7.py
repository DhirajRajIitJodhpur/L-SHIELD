"""
generate_fig6_fig7.py
=====================
Generates Figure 6 (system architecture block diagram) and Figure 7
(qualitative sample grid) for the L-SHIELD paper.

Run from anywhere:
    python3 paper_work/latex/generate_fig6_fig7.py
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

HERE        = pathlib.Path(__file__).resolve().parent
OUTDIR      = HERE / "figures"
ROOT        = HERE.parent.parent          # ISES_NIT_GOA/
IMGDIR_COCO = ROOT / "results" / "coco_50" / "images"
COCO_MASTER = pd.read_csv(ROOT / "results" / "coco_50" / "metrics" / "master_metrics.csv")

OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Figure 6 — System Architecture Pipeline  (polished v4, reference-matched)
# ============================================================================
def fig6_system_architecture():
    """
    Polished L-SHIELD pipeline diagram — v5 (double-column / figure*).

    Redesigned for full IEEEtran textwidth (figure*):
      - Canvas enlarged to 16.0 x 6.5 in; all block widths and font sizes
        scaled up ~33% so text renders at comfortable 8-10pt at full width.
      - All seven pipeline blocks widened and given more inter-block spacing.
      - Gaussian bell, formula box, calibration box, legend all rescaled.
      - Every text element registered; after fig.canvas.draw() all pairwise
        bounding boxes are checked — assertion fires on any overlap.
    """
    import matplotlib.patches as mpatches  # local

    # ---- Canvas (enlarged for figure* full-textwidth rendering) ----
    XMAX = 16.5
    YMAX = 6.5
    fig, ax = plt.subplots(figsize=(16.0, 6.5))
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    BG = "#edf2f7"
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ---- Color palette (consistent with paper's other figures) ----
    C_INPUT   = "#b03a2e"   # deep crimson  — adversarial input
    C_EDGE_IN = "#7b241c"
    C_VAE     = "#1e8449"   # forest green  — VAE enc / dec
    C_EDGE_V  = "#0b5345"
    C_SHIELD  = "#ca6f1e"   # burnt orange  — L-SHIELD
    C_EDGE_S  = "#784212"
    C_UNET    = "#1a5276"   # navy blue     — denoising U-Net
    C_EDGE_U  = "#0d2b45"
    C_OUTPUT  = "#6c3483"   # rich purple   — clean output
    C_EDGE_O  = "#3d1a5a"
    C_PREP    = "#117a65"   # deep teal     — preprocessing
    C_EDGE_PR = "#0a5249"
    C_CAL_BG  = "#fef9f0"
    C_CAL_ED  = "#935116"   # dark amber    — calibration dashed box
    C_PIPE    = "#17202a"   # near-black    — main pipeline arrows
    C_ATK     = "#b03a2e"   # adversarial threat (same as input)

    # ---- Block geometry (scaled ~33% wider than v4 for double-column) ----
    YM  = 3.70   # pipeline centerline y
    H_N = 1.35   # normal block height
    H_T = 1.70   # tall block height (Preprocess, L-SHIELD)
    GAP = 0.065  # clearance between block edge and arrow tip

    # (xc, w, h, fc, ec)
    BLK_INPUT  = (1.40,  1.77, H_N, C_INPUT,  C_EDGE_IN)
    BLK_PREP   = (3.60,  1.89, H_T, C_PREP,   C_EDGE_PR)
    BLK_VAEE   = (5.85,  1.59, H_N, C_VAE,    C_EDGE_V)
    BLK_SHIELD = (8.08,  2.02, H_T, C_SHIELD, C_EDGE_S)
    BLK_UNET   = (10.30, 1.59, H_N, C_UNET,   C_EDGE_U)
    BLK_VAED   = (12.40, 1.59, H_N, C_VAE,    C_EDGE_V)
    BLK_OUTPUT = (14.60, 1.77, H_N, C_OUTPUT, C_EDGE_O)
    all_blocks = [BLK_INPUT, BLK_PREP, BLK_VAEE, BLK_SHIELD,
                  BLK_UNET, BLK_VAED, BLK_OUTPUT]

    text_registry = []   # every Text object → checked for overlap after draw

    # ---- Helpers ----
    def shadow_rect(xc, yc, w, h, fc, ec, lw=2.4, ls="solid",
                    radius="round,pad=0.11"):
        """Drop-shadow + filled rounded rectangle centred at (xc, yc)."""
        ax.add_patch(FancyBboxPatch(
            (xc - w/2 + 0.10, yc - h/2 - 0.11), w, h,
            boxstyle=radius, facecolor="#7f8c8d", edgecolor="none",
            alpha=0.35, zorder=1))
        ax.add_patch(FancyBboxPatch(
            (xc - w/2, yc - h/2), w, h,
            boxstyle=radius, facecolor=fc, edgecolor=ec,
            linewidth=lw, linestyle=ls, zorder=2))

    def pipeline_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C_PIPE,
                                   lw=3.2, mutation_scale=22,
                                   connectionstyle="arc3,rad=0.0"), zorder=5)

    def cal_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C_CAL_ED,
                                   lw=2.0, mutation_scale=16,
                                   linestyle=(0, (5, 3)),
                                   connectionstyle="arc3,rad=0.0"), zorder=5)

    def atk_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C_ATK,
                                   lw=2.6, mutation_scale=20,
                                   linestyle="solid"), zorder=6)

    def reg_text(*args, **kwargs):
        """ax.text() wrapper that registers the Text for overlap checking."""
        t = ax.text(*args, **kwargs)
        text_registry.append(t)
        return t

    # ==========================================================
    # 1. Block backgrounds (shadow + fill for all blocks)
    # ==========================================================
    for (xc, w, h, fc, ec) in all_blocks:
        shadow_rect(xc, YM, w, h, fc, ec, lw=2.4)

    # ==========================================================
    # 2. Input / Output — embed real COCO-50 image thumbnails
    # ==========================================================
    IMG_INSET = 0.08

    def embed_image(img_path, xc, w, h):
        x1 = xc - w/2 + IMG_INSET
        x2 = xc + w/2 - IMG_INSET
        y1 = YM  - h/2 + IMG_INSET
        y2 = YM  + h/2 - IMG_INSET
        try:
            pil = Image.open(img_path).convert("RGB")
            ax.imshow(np.array(pil), extent=[x1, x2, y1, y2],
                      aspect="auto", zorder=3, interpolation="lanczos")
        except Exception:
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                facecolor="white", alpha=0.25, zorder=3))

    embed_image(IMGDIR_COCO / "img0_pgd_attacked.png",
                BLK_INPUT[0],  BLK_INPUT[1],  BLK_INPUT[2])
    embed_image(IMGDIR_COCO / "img0_clean_output.png",
                BLK_OUTPUT[0], BLK_OUTPUT[1], BLK_OUTPUT[2])

    # ==========================================================
    # 3. Standard text blocks: VAE Encoder, U-Net, VAE Decoder
    # ==========================================================
    def std_text(xc, line1, line2, math_lbl):
        reg_text(xc, YM + 0.26, line1, ha="center", va="center",
                 fontsize=12.0, fontweight="bold", color="white",
                 zorder=3, fontfamily="DejaVu Sans")
        reg_text(xc, YM + 0.02, line2, ha="center", va="center",
                 fontsize=11.5, fontweight="bold", color="white",
                 zorder=3, fontfamily="DejaVu Sans")
        reg_text(xc, YM - 0.32, math_lbl, ha="center", va="center",
                 fontsize=14.0, color="white", zorder=3)

    std_text(BLK_VAEE[0], "VAE",       "Encoder", r"$\mathcal{E}$")
    std_text(BLK_UNET[0], "Denoising", "U-Net",   r"$\boldsymbol{\theta}$")
    std_text(BLK_VAED[0], "VAE",       "Decoder", r"$\mathcal{D}$")

    # ==========================================================
    # 4. Preprocessing block — bulleted two-line operation list
    # ==========================================================
    xc_pr = BLK_PREP[0]
    reg_text(xc_pr, YM + 0.59, "Preprocessing",
             ha="center", va="center", fontsize=12.0, fontweight="bold",
             color="white", zorder=3, fontfamily="DejaVu Sans")
    reg_text(xc_pr, YM + 0.24,
             "\u2022  5-bit Quantization",
             ha="center", va="center", fontsize=9.0, color="white",
             zorder=3, fontfamily="DejaVu Sans")
    reg_text(xc_pr, YM - 0.10,
             "\u2022  Gaussian Blur (\u03c3\u202f=\u202f1.0)",
             ha="center", va="center", fontsize=9.0, color="white",
             zorder=3, fontfamily="DejaVu Sans")
    reg_text(xc_pr, YM - 0.50,
             r"$\tilde{\mathbf{x}}_e$",
             ha="center", va="center", fontsize=14.0, color="white", zorder=3)

    # ==========================================================
    # 5. L-SHIELD block — header + mini Gaussian bell + formula box
    # ==========================================================
    xc_sh = BLK_SHIELD[0]   # 8.08

    reg_text(xc_sh, YM + 0.75, "L-SHIELD",
             ha="center", va="center", fontsize=12.0, fontweight="bold",
             color="white", zorder=3, fontfamily="DejaVu Sans")
    reg_text(xc_sh, YM + 0.46, "Stat.\u202fClip (Ours)",
             ha="center", va="center", fontsize=10.0, fontweight="bold",
             color="white", zorder=3, fontfamily="DejaVu Sans")

    # Mini Gaussian bell curve (wider, taller to fill larger block)
    BELL_SIG  = 0.28
    BELL_BASE = YM + 0.12
    BELL_AMP  = 0.26
    bx = np.linspace(xc_sh - 0.80, xc_sh + 0.80, 200)
    bx_n = (bx - xc_sh) / BELL_SIG
    by = BELL_BASE + BELL_AMP * np.exp(-0.5 * bx_n**2)

    ax.plot(bx, by, color="white", lw=2.0, alpha=0.90, zorder=4)
    clip_mask = np.abs(bx_n) <= 1.5
    ax.fill_between(bx, BELL_BASE, by,
                    where=clip_mask, color="white", alpha=0.20, zorder=4)

    yv_boundary = BELL_BASE + BELL_AMP * float(np.exp(-0.5 * 1.5**2))
    for sign in [-1, 1]:
        xv = xc_sh + sign * BELL_SIG * 1.5
        ax.plot([xv, xv], [BELL_BASE, yv_boundary],
                color="white", lw=1.2, linestyle="--", alpha=0.80, zorder=4)

    # Formula box (lower portion of L-SHIELD block)
    fx1, fx2 = xc_sh - 0.90, xc_sh + 0.90
    fy1, fy2 = YM - 0.80,    YM - 0.16
    ax.add_patch(FancyBboxPatch(
        (fx1, fy1), fx2 - fx1, fy2 - fy1,
        boxstyle="round,pad=0.05",
        facecolor="#fef5e7", edgecolor="#784212",
        linewidth=1.8, zorder=4))
    reg_text(xc_sh, (fy1 + fy2) / 2,
             r"$\hat{\mathbf{z}}_e = \mathrm{clip}"
             r"\!\left(\mathbf{z}_e,\;\boldsymbol{\mu}"
             r" \pm \lambda\boldsymbol{\sigma}\right)$",
             ha="center", va="center", fontsize=11.0,
             color="#4a235a", zorder=5, fontfamily="DejaVu Sans")

    # ==========================================================
    # 6. Labels ABOVE Input / Output blocks
    # ==========================================================
    blk_top = YM + H_N / 2    # = 4.375
    LBL_Y1  = blk_top + 0.72  # "Adversarial Input" / "Recovered Output"
    LBL_Y2  = blk_top + 0.38  # math label

    reg_text(BLK_INPUT[0], LBL_Y1, "Adversarial Input",
             ha="center", va="center", fontsize=12.0, fontweight="bold",
             color=C_INPUT, zorder=6, fontfamily="DejaVu Sans")
    reg_text(BLK_INPUT[0], LBL_Y2,
             r"$(\mathbf{x}+\boldsymbol{\delta})$",
             ha="center", va="center", fontsize=12.0, color=C_INPUT, zorder=6)
    atk_arrow(BLK_INPUT[0], LBL_Y2 - 0.15,
              BLK_INPUT[0], blk_top + GAP)

    reg_text(BLK_OUTPUT[0], LBL_Y1, "Recovered Output",
             ha="center", va="center", fontsize=12.0, fontweight="bold",
             color=C_OUTPUT, zorder=6, fontfamily="DejaVu Sans")
    reg_text(BLK_OUTPUT[0], LBL_Y2,
             r"$(\hat{\mathbf{x}})$",
             ha="center", va="center", fontsize=12.0, color=C_OUTPUT, zorder=6)

    # Math caption below Input block
    blk_bot = YM - H_N / 2    # = 3.025
    reg_text(BLK_INPUT[0], blk_bot - 0.30,
             r"$\mathbf{x}_e = \mathbf{x} + \boldsymbol{\delta}$",
             ha="center", va="center", fontsize=10.0,
             color="#5d6d7e", style="italic", zorder=5)

    # ==========================================================
    # 7. Pipeline arrows + inter-block tensor labels
    # ==========================================================
    inter_labels = [
        r"$\mathbf{x}_e$",           # Input      → Preprocess
        r"$\tilde{\mathbf{x}}_e$",   # Preprocess → VAE Enc
        r"$\mathbf{z}_e$",           # VAE Enc    → L-SHIELD
        r"$\hat{\mathbf{z}}_e$",     # L-SHIELD   → U-Net
        r"$\hat{\mathbf{z}}_0$",     # U-Net      → VAE Dec
        r"$\hat{\mathbf{x}}$",       # VAE Dec    → Output
    ]
    for i in range(len(all_blocks) - 1):
        xc1, w1 = all_blocks[i][0],     all_blocks[i][1]
        xc2, w2 = all_blocks[i + 1][0], all_blocks[i + 1][1]
        xs = xc1 + w1 / 2 + GAP
        xe = xc2 - w2 / 2 - GAP
        pipeline_arrow(xs, YM, xe, YM)
        reg_text((xs + xe) / 2, YM + 0.52, inter_labels[i],
                 ha="center", fontsize=11.0, color="#1a252f",
                 style="italic", fontweight="bold", zorder=6)

    # ==========================================================
    # 8. Calibration stats box (below L-SHIELD, 3-line text)
    # ==========================================================
    cal_xc = BLK_SHIELD[0]
    cal_yc = 1.65
    cal_w, cal_h = 3.30, 1.05

    ax.add_patch(FancyBboxPatch(
        (cal_xc - cal_w/2 + 0.10, cal_yc - cal_h/2 - 0.11),
        cal_w, cal_h, boxstyle="round,pad=0.09",
        facecolor="#8a9ba8", edgecolor="none", alpha=0.32, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (cal_xc - cal_w/2, cal_yc - cal_h/2), cal_w, cal_h,
        boxstyle="round,pad=0.09", facecolor=C_CAL_BG,
        edgecolor=C_CAL_ED, linewidth=2.0, linestyle=(0, (5, 3)), zorder=2))

    reg_text(cal_xc, cal_yc + 0.32,
             r"Calibration Stats  $(\boldsymbol{\mu},\boldsymbol{\sigma})$",
             ha="center", va="center", fontsize=11.0, color="#6e2c00",
             fontweight="bold", zorder=3, fontfamily="DejaVu Sans")
    reg_text(cal_xc, cal_yc + 0.02,
             "per-position  [offline, pre-computed]",
             ha="center", va="center", fontsize=9.5, color="#935116",
             style="italic", zorder=3)
    reg_text(cal_xc, cal_yc - 0.30,
             "from clean latent representations",
             ha="center", va="center", fontsize=9.5, color="#935116",
             style="italic", zorder=3)

    shield_bot = YM - BLK_SHIELD[2] / 2   # = 2.85
    cal_arrow(cal_xc, cal_yc + cal_h/2 + GAP,
              cal_xc, shield_bot - GAP)

    # ==========================================================
    # 9. Title bar (commented out to avoid duplicate header in IEEE paper)
    # ==========================================================
    # ax.add_patch(FancyBboxPatch(
    #     (0.0, 5.98), XMAX, 0.48,
    #     boxstyle="round,pad=0.05",
    #     facecolor="#17202a", edgecolor="none", zorder=6))
    # reg_text(XMAX / 2, 6.22,
    #          "L-SHIELD Defense Pipeline \u2014 "
    #          "Latent Space Statistical Clipping",
    #          ha="center", va="center", fontsize=13.5, fontweight="bold",
    #          color="white", zorder=7, fontfamily="DejaVu Sans")

    # ==========================================================
    # 10. Legend — bordered box, bounds verified before drawing
    # ==========================================================
    legend_items = [
        (C_INPUT,  "Adversarial Input"),
        (C_PREP,   "Preprocess"),
        (C_VAE,    "VAE Enc / Dec"),
        (C_SHIELD, "L-SHIELD (Ours)"),
        (C_UNET,   "Denoising U-Net"),
        (C_OUTPUT, "Clean Output"),
    ]

    LEG_Y1     = 0.12
    LEG_Y2     = 0.84
    LEG_PAD_X  = 0.30
    SWATCH_W   = 0.38
    SWATCH_H   = 0.34
    SWATCH_GAP = 0.16
    ITEM_SEP   = 2.52
    lx_start   = 0.55

    N = len(legend_items)
    CHAR_W_EST = 0.115
    max_label_w = max(len(lbl) for _, lbl in legend_items) * CHAR_W_EST

    last_swatch_x   = lx_start + ITEM_SEP * (N - 1)
    last_item_right = last_swatch_x + SWATCH_W + SWATCH_GAP + max_label_w
    leg_box_left    = lx_start - LEG_PAD_X
    leg_box_right   = last_item_right + LEG_PAD_X

    assert leg_box_left  >= 0.02, \
        f"Legend left ({leg_box_left:.3f}) clips canvas left"
    assert leg_box_right <= XMAX - 0.05, \
        (f"Legend right ({leg_box_right:.3f}) clips canvas right "
         f"(XMAX={XMAX}). Reduce ITEM_SEP.")
    assert LEG_Y1 >= 0.02
    assert LEG_Y2 <= YMAX - 0.02

    ax.add_patch(FancyBboxPatch(
        (leg_box_left, LEG_Y1),
        leg_box_right - leg_box_left, LEG_Y2 - LEG_Y1,
        boxstyle="round,pad=0.06",
        facecolor="#f4f6f7", edgecolor="#5d6d7e",
        linewidth=1.8, linestyle="solid", zorder=5))

    ly_c = (LEG_Y1 + LEG_Y2) / 2
    lx   = lx_start
    for fc, lbl in legend_items:
        ax.add_patch(FancyBboxPatch(
            (lx, ly_c - SWATCH_H / 2), SWATCH_W, SWATCH_H,
            boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor="#17202a", linewidth=1.4, zorder=6))
        reg_text(lx + SWATCH_W + SWATCH_GAP, ly_c, lbl,
                 ha="left", va="center", fontsize=9.5, color="#17202a",
                 fontweight="bold", zorder=7, fontfamily="DejaVu Sans")
        lx += ITEM_SEP

    # ==========================================================
    # 11. Programmatic bounding-box overlap verification
    #     Renders every Text to window coords and checks all pairs.
    # ==========================================================
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer) for t in text_registry]
    n_overlaps = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if bboxes[i].overlaps(bboxes[j]):
                n_overlaps += 1
                print(f"    OVERLAP detected: text[{i}] vs text[{j}]")
    print(f"  [overlap-check] {len(bboxes)} text elements — "
          f"{n_overlaps} overlapping pair(s)")
    assert n_overlaps == 0, \
        f"fig6: {n_overlaps} text overlap(s) — adjust element positions!"

    # ==========================================================
    # 12. Save
    # ==========================================================
    fig.savefig(OUTDIR / "fig6_system_architecture.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("  OK  fig6_system_architecture.pdf")
    print(f"  [verify] Canvas: xlim=[0, {XMAX}]  ylim=[0, {YMAX}]")
    print(f"  [verify] Legend box x=[{leg_box_left:.3f}, {leg_box_right:.3f}] "
          f"(right margin={XMAX - leg_box_right:.3f})")
    print(f"  [verify] last_item_right={last_item_right:.3f} "
          f"<= XMAX-0.05={XMAX-0.05:.2f}: "
          f"{'PASS' if last_item_right <= XMAX - 0.05 else 'FAIL'}")
    print(f"  [verify] Shield-bot={YM - BLK_SHIELD[2]/2:.3f}  "
          f"cal-top={cal_yc + cal_h/2:.3f}  "
          f"gap={(YM - BLK_SHIELD[2]/2) - (cal_yc + cal_h/2):.3f}")
    print(f"  [verify] cal-bot={cal_yc - cal_h/2:.3f}  "
          f"legend-top={LEG_Y2:.3f}  "
          f"gap={cal_yc - cal_h/2 - LEG_Y2:.3f}")


# ============================================================================
# Figure 7 — Qualitative Sample Grid (4 attacks x 4 columns)
# ============================================================================
def fig7_qualitative_samples():
    # Average SSIM values read live from master_metrics.csv — never hardcoded
    def _ssim(scenario):
        return float(COCO_MASTER[COCO_MASTER.Scenario == scenario]["SSIM"].iloc[0])
    ssim_data = {
        "FGSM":       {"Clean": 1.000, "Attacked": _ssim("FGSM Attack"),
                       "DiffPure": _ssim("FGSM + DiffPure Defense"),
                       "L-SHIELD": _ssim("FGSM + LSSP Defense")},
        "PGD":        {"Clean": 1.000, "Attacked": _ssim("PGD Attack"),
                       "DiffPure": _ssim("PGD + DiffPure Defense"),
                       "L-SHIELD": _ssim("PGD + LSSP Defense")},
        "PhotoGuard": {"Clean": 1.000, "Attacked": _ssim("PhotoGuard Attack"),
                       "DiffPure": _ssim("PhotoGuard + DiffPure Defense"),
                       "L-SHIELD": _ssim("PhotoGuard + LSSP Defense")},
        "C&W":        {"Clean": 1.000, "Attacked": _ssim("C&W Attack"),
                       "DiffPure": _ssim("C&W + DiffPure Defense"),
                       "L-SHIELD": _ssim("C&W + LSSP Defense")},
    }

    img_map = {
        "FGSM": {
            "Clean":    IMGDIR_COCO / "img0_clean_output.png",
            "Attacked": IMGDIR_COCO / "img0_fgsm_attacked.png",
            "DiffPure": IMGDIR_COCO / "img0_fgsm_diffpure_defended.png",
            "L-SHIELD": IMGDIR_COCO / "img0_fgsm_lssp_defended.png",
        },
        "PGD": {
            "Clean":    IMGDIR_COCO / "img0_clean_output.png",
            "Attacked": IMGDIR_COCO / "img0_pgd_attacked.png",
            "DiffPure": IMGDIR_COCO / "img0_pgd_diffpure_defended.png",
            "L-SHIELD": IMGDIR_COCO / "img0_pgd_lssp_defended.png",
        },
        "PhotoGuard": {
            "Clean":    IMGDIR_COCO / "img0_clean_output.png",
            "Attacked": IMGDIR_COCO / "img0_pg_attacked.png",
            "DiffPure": IMGDIR_COCO / "img0_pg_diffpure_defended.png",
            "L-SHIELD": IMGDIR_COCO / "img0_pg_lssp_defended.png",
        },
        "C&W": {
            "Clean":    IMGDIR_COCO / "img0_clean_output.png",
            "Attacked": IMGDIR_COCO / "img0_cw_attacked.png",
            "DiffPure": IMGDIR_COCO / "img0_cw_diffpure_defended.png",
            "L-SHIELD": IMGDIR_COCO / "img0_cw_lssp_defended.png",
        },
    }

    attacks = ["FGSM", "PGD", "PhotoGuard", "C&W"]
    columns = ["Clean", "Attacked", "DiffPure", "L-SHIELD"]
    col_titles = [
        "Clean Output\n(Reference)",
        "Attacked Output\n(No Defense)",
        "DiffPure\nDefended",
        "L-SHIELD Defended\n(Ours)"
    ]
    # Orange highlight for L-SHIELD column
    col_edge_colors = ["#2c3e50", "#c0392b", "#2980b9", "#e67e22"]
    col_lw          = [0.8,        0.8,        0.8,       2.0]

    fig, axes = plt.subplots(
        len(attacks), len(columns),
        figsize=(8.5, 4.8),
        gridspec_kw={"hspace": 0.06, "wspace": 0.03}
    )
    fig.subplots_adjust(top=0.91, bottom=0.06, left=0.07, right=0.99)

    for row_i, atk in enumerate(attacks):
        for col_j, col in enumerate(columns):
            ax = axes[row_i, col_j]

            # Load and display image
            img_path = img_map[atk][col]
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                ax.imshow(img, aspect="auto")
            except Exception:
                ax.set_facecolor("#dddddd")
                ax.text(0.5, 0.5, "Missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="red")

            ax.set_xticks([])
            ax.set_yticks([])

            # Colour-coded spine for column identity
            for spine in ax.spines.values():
                spine.set_edgecolor(col_edge_colors[col_j])
                spine.set_linewidth(col_lw[col_j])

            # SSIM annotation
            ssim_val = ssim_data[atk][col]
            if col == "Clean":
                label = "SSIM: 1.000 (ref)"
                clr = "#1d8348"
            else:
                label = f"SSIM: {ssim_val:.3f}"
                clr = ("#1d8348" if ssim_val >= 0.58 else
                       "#e67e22" if ssim_val >= 0.47 else "#c0392b")
            ax.set_xlabel(label, fontsize=8.5, fontweight="bold", color=clr, labelpad=3)

            # Column title (top row only)
            if row_i == 0:
                ax.set_title(col_titles[col_j], fontsize=9.5,
                             fontweight="bold", color=col_edge_colors[col_j],
                             pad=6, multialignment="center")

            # Attack row label (leftmost column only)
            if col_j == 0:
                ax.set_ylabel(atk, fontsize=10.5, fontweight="bold",
                              rotation=90, labelpad=6)

    # fig.suptitle(
    #     "Qualitative Comparison: Clean / Attacked / DiffPure / L-SHIELD"
    #     " (COCO-50, example image)",
    #     fontsize=10, fontweight="bold")

    fig.savefig(OUTDIR / "fig7_qualitative_samples.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  OK  fig7_qualitative_samples.pdf")


# ============================================================================
if __name__ == "__main__":
    print(f"Output directory: {OUTDIR}\n")
    # fig6_system_architecture()  # PRESERVE ORIGINAL FIG 6
    fig7_qualitative_samples()
    print(f"\nFigure 7 saved to {OUTDIR}")
