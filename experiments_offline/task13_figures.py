"""Task 13 - regenerate manuscript figures with the reviewer's requested changes.

  * F_sen renamed to F_ill (target illumination) on every axis/legend.
  * Larger fonts and markers.
  * Convergence plot (Fig 2) gets mean +/- 95% CI uncertainty bands.
  * The HV reference (nadir) point is stated and identical for every curve.
  * The CDF caption states the number of realisations and the online budget.
  * The online curve is labelled an "online per-realisation reference", never a
    formal upper bound.

Reads results/experiment.pkl (+ experiments_offline/out/task1_convergence.npz for the
Fig-2 bands) and writes to figures_paper/.
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = "figures_paper"
PKL = "results/experiment.pkl"
CONV = "experiments_offline/out/task1_convergence.npz"

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 16,
    "legend.fontsize": 12.5, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 2.2, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
})
LABELS = {"adaptive": "RAMP (proposed)", "no_transfer": "No-transfer",
          "mo_de": "MO-DE", "pooled": "Pooled single-task", "mopso": "MOPSO"}
COLORS = {"adaptive": "#0072B2", "no_transfer": "#009E73", "mo_de": "#D55E00",
          "pooled": "#CC79A7", "mopso": "#E69F00"}
F_COM = r"$F_{\mathrm{com}}$  (worst-user rate, bps/Hz)"
F_ILL = r"$F_{\mathrm{ill}}$  (worst-target illumination, dB)"
HV_REF = (0.0, -30.0)          # cfg.hv_ref_com, cfg.hv_ref_sen_db (consistent everywhere)


def fig1(data):
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for m in ("pooled", "mopso", "mo_de", "no_transfer", "adaptive"):
        f = data.unified_fronts.get(m)
        if f is None or f.shape[0] == 0:
            continue
        f = f[np.argsort(f[:, 0])]
        is_r = m == "adaptive"
        ax.plot(f[:, 0], f[:, 1], "-o", ms=7 if is_r else 5,
                lw=3.0 if is_r else 1.8, color=COLORS[m], label=LABELS[m],
                alpha=1.0 if is_r else 0.7, zorder=6 if is_r else 3)
    usp, rnd = data.reference["uspa"], data.reference["random"]
    if usp.shape[0]:
        ax.scatter(usp[:, 0], usp[:, 1], marker="*", s=340, color="k",
                   zorder=8, label="USPA (fixed)", edgecolor="w", linewidth=0.6)
    if rnd.shape[0]:
        ax.scatter(rnd[:, 0], rnd[:, 1], marker="x", s=55, color="gray",
                   alpha=0.6, zorder=2, label="Random layouts")
    ax.set_xlabel(F_COM)
    ax.set_ylabel(F_ILL)
    ax.legend(loc="lower left", framealpha=0.92)
    ax.set_title("Unified robust Pareto front (min over regimes)", fontsize=14)
    p = os.path.join(OUTDIR, "fig1_unified_front.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig2(data):
    conv = np.load(CONV) if os.path.exists(CONV) else None
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    G = np.asarray(data.hv_gen["adaptive"]).shape[1]
    x = np.arange(1, G + 1)
    for m in ("adaptive", "no_transfer", "mo_de", "pooled", "mopso"):
        arr = np.asarray(data.hv_gen[m])
        mean = arr.mean(axis=0)
        is_r = m == "adaptive"
        ax.plot(x, mean, color=COLORS[m], lw=3.0 if is_r else 1.8,
                label=LABELS[m], zorder=6 if is_r else 3, alpha=1.0 if is_r else 0.85)
        if conv is not None and f"{m}__ci_lo" in conv:
            ax.fill_between(x, conv[f"{m}__ci_lo"], conv[f"{m}__ci_hi"],
                            color=COLORS[m], alpha=0.18, zorder=2, linewidth=0)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Robust hypervolume  (ref. $[0,\\,-30\\,\\mathrm{dB}]$)")
    ax.legend(loc="lower right", framealpha=0.92, ncol=1)
    ax.set_title("Cumulative-best robust HV (mean $\\pm$ 95% CI, 25 seeds)", fontsize=14)
    p = os.path.join(OUTDIR, "fig2_hv_over_gen.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def _cdf(vals):
    v = np.sort(np.asarray(vals))
    return v, np.arange(1, v.size + 1) / v.size


def fig3(data):
    f3 = data.fig3
    off, on = np.asarray(f3["offline_hv"]), np.asarray(f3["online_hv"])
    n_off, n_on = off.size, on.size
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    x, y = _cdf(off)
    ax.plot(x, y, color=COLORS["adaptive"], lw=3.0,
            label=f"Offline unified front ({n_off} realisations)")
    for key, lbl, col in (("uspa_hv", "USPA (fixed)", "k"),
                          ("random_hv", "Random layout", "gray")):
        if key in f3 and np.asarray(f3[key]).size:
            xc, yc = _cdf(np.asarray(f3[key]))
            ax.plot(xc, yc, "--", color=col, lw=1.8, alpha=0.8, label=lbl)
    if n_on:
        med = float(np.median(on))
        lo, hi = np.percentile(on, [10, 90])
        ax.axvspan(lo, hi, color=COLORS["mo_de"], alpha=0.13, zorder=1)
        ax.axvline(med, color=COLORS["mo_de"], lw=2.4, ls=":",
                   label=f"Online per-realisation reference\n(median, {n_on} realisations, G={max(20, 300 // 4)})")
    ax.set_xlabel("Robust hypervolume per realisation")
    ax.set_ylabel("Empirical CDF")
    ax.legend(loc="upper left", framealpha=0.92)
    ax.set_title("Robustness of the one-time offline design", fontsize=14)
    p = os.path.join(OUTDIR, "fig3_hv_cdf.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    paths = [fig1(data), fig2(data), fig3(data)]
    print("Regenerated (Task 13):")
    for p in paths:
        print("  ", p)
    print("\nReviewer changes applied: F_sen->F_ill; larger fonts/markers; Fig-2 "
          "mean+/-95%CI bands; HV ref [0,-30 dB] stated & consistent; CDF caption "
          "states #realisations + online budget; online curve = 'reference' (not upper bound).")


if __name__ == "__main__":
    main()
