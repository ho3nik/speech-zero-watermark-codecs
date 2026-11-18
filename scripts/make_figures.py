"""Figures for the paper, built from results/*.csv.

fig1_bitrate.png   mean BER against EnCodec bitrate for the four families and SEL-codec, MP3 32 kbps as reference
fig2_verify.png    genuine BER after EnCodec 1.5 kbps against external impostor BER, four families

Usage: python scripts/make_figures.py --results results --out figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fixed categorical order (validated palette, light surface); schemes keep their slot everywhere
PALETTE = {"DWT-DCT": "#2a78d6", "Philips": "#eb6834", "MFCC": "#1baf7a", "WavLM-LSH": "#9b59b6", "SEL-codec": "#555555"}
FAMILIES = ["DWT-DCT", "Philips", "MFCC", "WavLM-LSH"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#222", "xtick.color": "#444", "ytick.color": "#444"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(R / "table_ber.csv")

    # ---- Fig 1 bitrate curve
    enc = {"encodec_1.5k": 1.5, "encodec_3k": 3, "encodec_6k": 6, "encodec_12k": 12, "encodec_24k": 24}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for s in FAMILIES + ["SEL-codec"]:
        d = tab[(tab.scheme == s) & tab.attack.isin(enc)].copy()
        d["kbps"] = d.attack.map(enc); d = d.sort_values("kbps")
        ax.errorbar(d.kbps, d.ber_mean, yerr=[d.ber_mean - d.ber_ci_lo, d.ber_ci_hi - d.ber_mean], color=PALETTE[s],
                    marker="o", ms=4, lw=1.8, capsize=2, label=s, ls="--" if s == "SEL-codec" else "-")
        m = tab[(tab.scheme == s) & (tab.attack == "mp3_32k")].ber_mean.iloc[0]
        ax.plot([0.9 * 1.06 ** (FAMILIES + ["SEL-codec"]).index(s) * 0.86], [m], marker="s", ms=5, color=PALETTE[s], ls="none")
    ax.set_xscale("log"); ax.set_xticks([0.9, 1.5, 3, 6, 12, 24]); ax.set_xticklabels(["MP3\n32", "1.5", "3", "6", "12", "24"])
    ax.minorticks_off()
    ax.set_xlim(0.7, 40); ax.set_ylim(0, 0.5)
    ax.set_xlabel("EnCodec bitrate (kbps)"); ax.set_ylabel("mean BER")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "fig1_bitrate.png", dpi=300); fig.savefig(out / "fig1_bitrate.pdf"); fig.savefig(out / "Fig1.eps"); fig.savefig(out / "Fig1.tif", dpi=600); plt.close(fig)

    # ---- Fig 2 genuine vs external impostor BER after EnCodec 1.5 kbps
    pc = pd.read_csv(R / "per_clip.csv")
    hist = pd.read_csv(R / "impostor_ext_hist.csv").set_index("scheme")
    edges = np.linspace(0, 0.6, 61); centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5), sharey=True, gridspec_kw={"wspace": 0.15})
    for ax, s in zip(axes, FAMILIES):
        g = pc[(pc.scheme == s) & (pc.attack == "encodec_1.5k")].ber
        cnt, _ = np.histogram(g, bins=edges)
        ax.bar(centers, cnt / cnt.sum(), width=0.01, color=PALETTE[s], label="genuine")
        h = hist.loc[s, [f"b{k:02d}" for k in range(60)]].values.astype(float)
        ax.plot(centers, h / h.sum(), color="#444", lw=1.2, label="impostor")
        ax.axvline(hist.loc[s, "tau_1e-4"], color="#222", lw=1, ls=":")
        ax.set_title(s, fontsize=9); ax.set_xlabel("BER"); ax.set_xlim(0, 0.6); ax.set_xticks([0, 0.25, 0.5])
    axes[0].set_ylabel("fraction of pairs"); axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.88, wspace=0.15); fig.savefig(out / "fig2_verify.png", dpi=300); fig.savefig(out / "fig2_verify.pdf"); fig.savefig(out / "Fig2.eps"); fig.savefig(out / "Fig2.tif", dpi=600); plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main()
