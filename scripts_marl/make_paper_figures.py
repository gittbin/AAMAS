"""Generate the paper figures (PDF) from the compiled results JSON.

Greyscale-safe + colour-blind-friendly (distinct line styles + markers).
Run: python scripts_marl/make_paper_figures.py --outdir <AAMAS folder>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight",
})
_BLUE, _ORANGE = "#0072B2", "#D55E00"   # colour-blind safe pair


def fig_learning(J, out):
    lc = J["T5_learning_curve"]
    steps = [d["step"] / 1000 for d in lc]
    mean = [d["mean"] for d in lc]
    std = [d["std"] for d in lc]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    lo = [m - s for m, s in zip(mean, std)]
    hi = [m + s for m, s in zip(mean, std)]
    ax.fill_between(steps, lo, hi, color=_BLUE, alpha=0.2)
    ax.plot(steps, mean, color=_BLUE, marker="o", ms=3, lw=1.5, label="RMAPPO (5 seeds)")
    ax.set_xlabel("Environment steps ($\\times 10^3$)")
    ax.set_ylabel("Eval. episode return")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out / "fig_learning.pdf")
    plt.close(fig)


def fig_robust(J, out):
    T3 = J["T3_tau"]
    taus = sorted(float(t) for t in T3)
    lm = [T3[f"{t}" if f"{t}" in T3 else str(t)]["learned_min"] for t in taus]
    nm = [T3[str(t)]["noop_min"] for t in taus]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(taus, lm, color=_BLUE, marker="o", ms=4, lw=1.6, ls="-", label="Learned (RMAPPO)")
    ax.plot(taus, nm, color=_ORANGE, marker="s", ms=4, lw=1.6, ls="--", label="No-op (static USPA)")
    ax.set_xlabel("CSI error level $\\tau$")
    ax.set_ylabel("Min-user rate (bps/Hz)")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / "fig_robust.pdf")
    plt.close(fig)


def fig_delay(E1, out):
    ds = sorted(int(d) for d in E1)
    lm = [E1[str(d)]["learned_min"] for d in ds]
    nm = [E1[str(d)]["noop_min"] for d in ds]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(ds, lm, color=_BLUE, marker="o", ms=4, lw=1.6, ls="-", label="Learned (RMAPPO)")
    ax.plot(ds, nm, color=_ORANGE, marker="s", ms=4, lw=1.6, ls="--", label="No-op (static USPA)")
    ax.set_xlabel("Actor CSI delay $\\delta$ (steps)")
    ax.set_ylabel("Min-user rate (bps/Hz)")
    ax.set_xticks(ds)
    ax.legend(loc="center right", fontsize=8)
    fig.savefig(out / "fig_delay.pdf")
    plt.close(fig)


def fig_mainbar(J, out):
    labels = ["Random", "Heuristic", "Equal-pow.", "No-op", "Learned"]
    rets = [69.45, 97.14, 111.16, 112.07, 112.96]
    colors = ["#999999", "#999999", "#999999", "#999999", _BLUE]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    y = range(len(labels))
    ax.barh(list(y), rets, color=colors, edgecolor="black", lw=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Eval.\\ episode return")
    ax.set_xlim(60, 118)
    for i, v in enumerate(rets):
        ax.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=7)
    fig.savefig(out / "fig_mainbar.pdf")
    plt.close(fig)


def fig_tradeoff(out):
    pts = {"Learned": (7.651, -0.00), "No-op": (7.457, 0.70), "Equal-pow.": (7.459, 0.70),
           "Heuristic": (6.640, -1.33), "Random": (5.357, -2.61)}
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for name, (x, y) in pts.items():
        c = _BLUE if name == "Learned" else "#888888"
        m = "*" if name == "Learned" else "o"
        ax.scatter([x], [y], s=70 if name == "Learned" else 40, color=c, marker=m,
                   edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Min-user rate (bps/Hz)")
    ax.set_ylabel("Sensing illum.\\ (dB)")
    fig.savefig(out / "fig_tradeoff.pdf")
    plt.close(fig)


def fig_regime(ext, out):
    E2 = ext["E2_regime_sig"]
    regs = ["LoS", "Rayleigh", "Rician"]
    learned = [E2[r]["learned_ret"] for r in regs]
    noop = [E2[r]["noop_ret"] for r in regs]
    import numpy as _np
    x = _np.arange(len(regs))
    w = 0.36
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.bar(x - w / 2, learned, w, color=_BLUE, edgecolor="black", lw=0.4, label="Learned")
    ax.bar(x + w / 2, noop, w, color="#999999", edgecolor="black", lw=0.4, label="No-op")
    ax.set_xticks(list(x))
    ax.set_xticklabels(regs)
    ax.set_ylabel("Eval.\\ episode return")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / "fig_regime.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(_REPO_ROOT / "outputs_marl" / "paper" / "paper_results.json"))
    ap.add_argument("--ext", default=str(_REPO_ROOT / "outputs_marl" / "paper" / "ext_results.json"))
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "AAMAS-2026-Formatting-Instructions-CCBY"))
    args = ap.parse_args()
    J = json.loads(pathlib.Path(args.results).read_text(encoding="utf-8"))
    J["T3_tau"] = {str(k): v for k, v in J["T3_tau"].items()}
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig_learning(J, out)
    fig_robust(J, out)
    fig_mainbar(J, out)
    ext = json.loads(pathlib.Path(args.ext).read_text(encoding="utf-8"))
    fig_delay({str(k): v for k, v in ext["E1_delay"].items()}, out)
    fig_regime(ext, out)
    fig_tradeoff(out)
    print("wrote figs (learning/robust/mainbar/delay/regime/tradeoff) ->", out)


if __name__ == "__main__":
    main()
