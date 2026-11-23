"""Draw figures/summary.png: G, R and D_F for the 9 valid adjacent-tier pairs of the all-pairs run.

All numbers are constants copied from files in this repository; the source of
each one is given in the comment next to it. Run from any directory:
    python figures/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt

BLUE = "#1f4e79"
GREY = "#9aa5b1"
ORANGE = "#d98c1f"
INK = "#222222"
MUTED = "#5f6b76"
AXIS = "#444444"
GRID = "#e6e9ec"
FS_TITLE, FS_SUB, FS_TEXT = 13, 10, 8  # the only three font sizes used

# (pair label, G, R, D_F) in pp, screening slot idx0
CELLS = [
    ("AIME  o3-mini  low→medium", 62.96, 24.07, 25.56),    # results/all_pairs/tables/pairs_all.csv
    ("AIME  o4-mini  low→medium", 63.89, 25.00, 21.11),    # results/all_pairs/tables/pairs_all.csv
    ("AIME  o4-mini  medium→high", 58.33, 25.00, 11.11),   # results/all_pairs/tables/pairs_all.csv
    ("HMMT  o3-mini  low→medium", 34.78, 13.04, 20.00),    # results/all_pairs/tables/pairs_all.csv
    ("HMMT  o3-mini  medium→high", 27.27, 0.00, 16.67),    # results/all_pairs/tables/pairs_all.csv
    ("HMMT  o4-mini  low→medium", 44.44, 26.67, 17.78),    # results/all_pairs/tables/pairs_all.csv
    ("HMMT  o4-mini  medium→high", 57.14, 23.81, 16.67),   # results/all_pairs/tables/pairs_all.csv
    ("BRUMO  o4-mini  low→medium", 56.67, 20.00, 16.67),   # results/all_pairs/tables/pairs_all.csv
    ("BRUMO  o4-mini  medium→high", 41.67, 50.00, 1.11),   # results/all_pairs/tables/pairs_all.csv
]

MACRO_G = 50.70    # results/all_pairs/tables/aggregates.csv
MACRO_R = 25.14    # results/all_pairs/tables/aggregates.csv
MACRO_DF = 15.83   # results/all_pairs/tables/aggregates.csv

SERIES = [  # (index in tuple, colour, direct label used on the macro row)
    (1, BLUE, "G, apparent gain on screened problems"),
    (2, ORANGE, "R, lower tier's own recovery"),
    (3, GREY, "D_F, gap on all problems"),
]


def setup():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": FS_TEXT,
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS,
        "xtick.direction": "out",
        "xtick.major.size": 3,
        "xtick.major.width": 0.6,
        "xtick.color": AXIS,
        "xtick.labelsize": FS_TEXT,
        "ytick.labelsize": FS_TEXT,
    })


def main():
    setup()
    cells = sorted(CELLS, key=lambda c: c[1], reverse=True)
    rows = [("Macro average (3 competitions)", MACRO_G, MACRO_R, MACRO_DF)] + cells

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.6)

    bar_h = 0.25
    ys = [-(i + (0.55 if i > 0 else 0)) for i in range(len(rows))]
    left, right = 0.255, 0.745
    fig.subplots_adjust(left=left, right=right, top=0.80, bottom=0.17)
    ax.set_ylim(ys[-1] - 0.5, ys[0] + 0.5)
    ax.set_xlim(0, 70)
    ax.set_xticks(range(0, 71, 20))
    ax.set_xlabel("Percentage points", color=INK)

    # value columns to the right of the plot: x in axes units, y in data units
    tr = ax.get_yaxis_transform()
    col_right = [(fx - left) / (right - left) for fx in (0.845, 0.910, 0.975)]
    head_off = [w / (right - left) for w in (0.017, 0.017, 0.036)]  # square sits left of each header letter
    for i, row in enumerate(rows):
        weight = "bold" if i == 0 else "normal"
        for j, (k, colour, _) in enumerate(SERIES):
            ax.barh(ys[i] + (1 - j) * bar_h, row[k], height=bar_h * 0.85, color=colour, linewidth=0)
            ax.text(col_right[j], ys[i], f"{row[k]:+.2f}", transform=tr, ha="right", va="center",
                    color=INK, fontweight=weight, clip_on=False)
    for j, (k, colour, _) in enumerate(SERIES):
        y_head = ys[0] + 0.85
        ax.text(col_right[j] - head_off[j], y_head, "\u25A0", transform=tr, ha="right", va="center",
                color=colour, clip_on=False)
        ax.text(col_right[j], y_head, ["G", "R", "D_F"][j], transform=tr, ha="right", va="center",
                color=INK, fontweight="bold", clip_on=False)

    ax.axhline((ys[0] + ys[1]) / 2 - 0.02, color="#cfd5da", linewidth=0.6)
    ax.set_yticks(ys)
    labels = ax.set_yticklabels([r[0] for r in rows])
    labels[0].set_fontweight("bold")

    key = "   ".join(f"{s}: {n.split(', ')[1]}" for s, (_, _, n) in zip(["G", "R", "D_F"], SERIES))
    fig.text(0.015, 0.035, key, ha="left", va="center", color=MUTED)
    fig.text(0.015, 0.955, "Screened problems overstate the tier gap: +50.70 vs +15.83 pp",
             fontsize=FS_TITLE, fontweight="bold", ha="left", va="center", color=INK)
    fig.text(0.015, 0.902, "9 valid adjacent-tier pairs, MathArena AIME/HMMT/BRUMO 2025, slot idx0, "
             "sorted by G",
             fontsize=FS_SUB, ha="left", va="center", color=MUTED)

    out = Path(__file__).resolve().parent / "summary.png"
    fig.savefig(out, dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
