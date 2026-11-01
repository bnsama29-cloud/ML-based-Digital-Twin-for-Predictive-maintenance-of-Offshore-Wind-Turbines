"""Summary figures drawn from the result tables (the case-study figures come from
analyze_case.py, the framework diagram is figures/framework.png).

    python pipeline/make_figures.py --tables results/tables --out figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


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
    for y, txt in ((0.5, "random 0.50"), (0.66, "tuned AE of CARE paper: 0.66")):
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
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    care_bars(a.tables, a.out)
    robustness(a.tables, a.out)



def robustness(tables, out):
    """False alarms (of 50 normal periods) and faults detected (of 45) under corruption."""
    r = pd.read_csv(os.path.join(tables, "robustness.csv"))
    conds = [("none", "clean"), ("noise0.1", "σ=0.1"), ("noise0.3", "σ=0.3"),
             ("noise0.5", "σ=0.5"), ("missing0.05", "5% miss."), ("missing0.2", "20% miss.")]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    for ax, (col, title) in zip(axes, (("false_alarms", "Normal periods with alarm (of 50)"),
                                      ("detected", "Faults detected (of 45)"))):
        for j, (key, name, color, hatch) in enumerate((("ae_s0|cn_smooth", "AE", BLUE, None),
                                                       ("pca|cn_smooth", "PCA", ORANGE, "////"))):
            vals = [r[(r.key == key) & (r.corruption == c)][col].iloc[0] for c, _ in conds]
            xs = [i + (j - 0.5) * 0.4 for i in range(len(conds))]
            ax.bar(xs, vals, 0.38, color=color, hatch=hatch, edgecolor="white", lw=0, label=name)
        ax.set_title(title, fontsize=6.5, color=INK)
        ax.set_xticks(range(len(conds)), [n for _, n in conds], rotation=45, ha="right", fontsize=5.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e4e4e0", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "fig_robust.png"), dpi=300)



if __name__ == "__main__":
    main()
