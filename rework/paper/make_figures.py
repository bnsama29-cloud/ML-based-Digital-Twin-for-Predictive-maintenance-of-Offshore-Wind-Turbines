"""Paper figures that are drawn from the result tables (not from the case-study runs).

    python paper/make_figures.py --tables <merged paper_tables> --out paper/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def pipeline(out):
    fig, ax = plt.subplots(figsize=(7.1, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    def box(x, y, w, h, text, fill="#eef4fc", edge=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fill, ec=edge, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.2, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=MUTED))

    main = [("SCADA\n10-min rows", 0), ("Preprocess,\nidle filter", 14.2),
            ("Detector\nAE, CAE, LSTM,\nPCA, IF, 3σ", 28.4),
            ("Condition\nnormalisation\n(on $\\mathcal{C}_a$)", 42.6),
            ("Causal\n6 h median", 56.8), ("Threshold\n(on $\\mathcal{C}_b$),\n12 h counter", 71),
            ("Alarm", 86.5)]
    for i, (t, x) in enumerate(main):
        box(x, 17, 12 if i < 6 else 11, 10, t, *(("#fdeee7", ORANGE) if i == 6 else ()))
        if i:
            arrow(main[i - 1][1] + 12.4, 22, x - 0.4, 22)
    side = [("Plant-area\nattribution", 20), ("Health index,\ntime to threshold", 44),
            ("Work-order\ngate", 66), ("Unity\nplayback", 84)]
    for t, x in side:
        box(x - 7, 1.5, 14.5, 8.5, t, fill="#f4f4f2", edge=MUTED)
    arrow(34.4, 16.6, 27, 10.5)
    arrow(62.8, 16.6, 51, 10.5)
    arrow(77, 16.6, 73, 10.5)
    arrow(92, 16.6, 91, 10.5)
    fig.savefig(os.path.join(out, "pipeline.png"), dpi=300, bbox_inches="tight")


def care_bars(tables, out):
    care = pd.read_csv(os.path.join(tables, "care.csv"))
    care = care[(care.farm == "all") & (care.variant == "cn_smooth")].set_index("method")
    lofo = pd.read_csv(os.path.join(tables, "lofo.csv"))
    lofo = lofo[lofo.variant == "cn_smooth"].set_index("method")
    order = [("ae", "AE"), ("cae", "CAE"), ("lstm", "LSTM-AE"), ("pca", "PCA"),
             ("iforest", "IF")]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    w = 0.38
    for i, (m, label) in enumerate(order):
        f, t = care.loc[m], lofo.loc[m]
        ax.bar(i - w / 2 - 0.01, f["care"], w, color=BLUE, yerr=f.get("care_sd", 0) or 0,
               error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK), label="fixed (q=0.99, k=72)" if i == 0 else None)
        ax.bar(i + w / 2 + 0.01, t["care_lofo"], w, color=ORANGE, hatch="////", edgecolor="white",
               lw=0, yerr=t["care_lofo_sd"] or 0, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK),
               label="tuned leave-one-farm-out" if i == 0 else None)
    for y, txt in ((0.5, "random 0.50"), (0.66, "tuned AE of [12]: 0.66")):
        ax.axhline(y, color=MUTED, lw=0.7, ls="--")
        ax.text(len(order) - 0.5, y, " " + txt.replace(": ", "\n"), ha="left", va="center",
                fontsize=6, color=MUTED, clip_on=False)
    ax.set_xticks(range(len(order)), [l for _, l in order])
    ax.set_xlim(-0.55, len(order) - 0.5)
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("CARE score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e4e0", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.13))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_care.png"), dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pipeline(a.out)
    care_bars(a.tables, a.out)


if __name__ == "__main__":
    main()
