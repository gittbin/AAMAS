"""Task 13 (part 2) - sensitivity/scalability + RMP-sweep figures.

Reads experiments_offline/out/sensitivity.json and task4_rmp_sweep.json and writes:
  figures_paper/fig_sens_B.png   HV & IGD vs MC batch B (Task 6)
  figures_paper/fig_scal_L.png   HV & runtime/mem vs RIS L (Task 7)
  figures_paper/fig_scal_M.png   HV & runtime/mem vs MA M (Task 8)
  figures_paper/fig4_rmp_sweep.png  robust HV vs fixed RMP + adaptive line (Task 4)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "experiments_offline/out"
FIG = "figures_paper"
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "legend.fontsize": 11.5,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "lines.linewidth": 2.2,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300, "savefig.bbox": "tight"})
C = {"hv": "#0072B2", "igd": "#D55E00", "t": "#009E73", "mem": "#CC79A7"}


def _sens_fig(res, knob, xlabel, fname, title):
    vals = res["values"]
    rows = res["rows"]
    x = [str(v) for v in vals]
    hv = [rows[str(v)]["hv_mean"] if str(v) in rows else rows[v]["hv_mean"] for v in vals]
    hvsd = [_get(rows, v, "hv_std") for v in vals]
    igd = [_get(rows, v, "igd_mean") for v in vals]
    t = [_get(rows, v, "runtime_s_mean") for v in vals]
    mem = [_get(rows, v, "peak_mem_MB_mean") for v in vals]

    fig, ax1 = plt.subplots(figsize=(6.0, 4.4))
    ax1.errorbar(x, hv, yerr=hvsd, marker="o", ms=8, color=C["hv"], capsize=4, label="Robust HV")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Robust HV", color=C["hv"])
    ax1.tick_params(axis="y", labelcolor=C["hv"])
    ax2 = ax1.twinx()
    ax2.plot(x, igd, marker="s", ms=7, color=C["igd"], ls="--", label="IGD")
    ax2.set_ylabel("IGD (normalised)", color=C["igd"])
    ax2.tick_params(axis="y", labelcolor=C["igd"])
    ax2.grid(False)
    ax1.set_title(title, fontsize=13)
    p = os.path.join(FIG, fname)
    fig.savefig(p)
    plt.close(fig)
    return p


def _get(rows, v, field):
    key = str(v) if str(v) in rows else v
    return rows[key][field]


def sensitivity_figs():
    path = os.path.join(OUT, "sensitivity.json")
    if not os.path.exists(path):
        return []
    data = json.load(open(path))["results"]
    made = []
    specs = {"B": ("MC batch size B", "fig_sens_B.png", "Sensitivity to MC batch (Task 6)"),
             "L": ("RIS elements L", "fig_scal_L.png", "Scalability to RIS size (Task 7)"),
             "M": ("Movable antennas M", "fig_scal_M.png", "Scalability to MA count (Task 8)")}
    for knob, (xl, fn, title) in specs.items():
        if knob in data:
            made.append(_sens_fig(data[knob], knob, xl, fn, title))
    return made


def rmp_sweep_fig():
    path = os.path.join(OUT, "task4_rmp_sweep.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    sweep = d["sweep"]
    xs = sorted(float(k) for k in sweep)
    hv = [sweep[str(v) if str(v) in sweep else f"{v}"]["hv_mean"] for v in xs]
    # keys may be '0.0','0.1',...; normalise lookup
    hv, hvsd = [], []
    for v in xs:
        row = sweep.get(str(v)) or sweep.get(f"{v:g}") or sweep.get(repr(v))
        hv.append(row["hv_mean"]); hvsd.append(row["hv_std"])
    ad = d["adaptive"]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.errorbar([f"{v:g}" for v in xs], hv, yerr=hvsd, marker="o", ms=8, color=C["hv"],
                capsize=4, label="fixed RMP")
    ax.axhline(ad["hv_mean"], color=C["igd"], ls="--", lw=2.4,
               label=f"adaptive (r_G→{ad['final_rG_mean']:.2f})")
    ax.fill_between(range(len(xs)), ad["hv_mean"] - ad["hv_std"], ad["hv_mean"] + ad["hv_std"],
                    color=C["igd"], alpha=0.12)
    ax.set_xlabel("Fixed random-mating probability (RMP)")
    ax.set_ylabel("Final robust HV")
    ax.set_title("Fixed-RMP sweep; adaptive auto-tunes (Task 4)", fontsize=13)
    ax.legend(loc="best")
    p = os.path.join(FIG, "fig4_rmp_sweep.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    os.makedirs(FIG, exist_ok=True)
    made = sensitivity_figs()
    r = rmp_sweep_fig()
    if r:
        made.append(r)
    print("Sensitivity/RMP figures:")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
