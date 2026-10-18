"""Task 10 - standardised offline vs online per-realisation reference.

Explicit protocol:
  N_ONLINE = number of stochastic realisations
  G_ONLINE = generations for each online per-realisation optimisation

For a trained RAMP offline unified front it reports:
  * median offline per-realisation HV (front deployed as-is), with 10-90% interval;
  * median online per-realisation HV (re-optimised for each realisation), 10-90%;
  * online optimisation time per realisation;
  * online SELECTION time (pick the best offline-front operating point per realisation);
  * HV retained = 100 * median(offline)/median(online).

The online curve is an *online per-realisation optimisation reference*, NOT a formal
upper bound.

Usage: python resubmission/task10_online.py --device cuda --n-online 30 --seed 0
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from momgrap import baselines as B
from momgrap.config import default_config
from momgrap.logging_utils import setup_logging
from momgrap.metrics import hv_reference, hypervolume_2d, non_dominated
from momgrap.objectives import draw_generation_environments, robust_evaluate

OUT = "resubmission/out"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-online", type=int, default=30)
    p.add_argument("--n-offline", type=int, default=200)
    p.add_argument("--g-online", type=int, default=None, help="default max(20, G//4)")
    args = p.parse_args(argv)
    log = setup_logging(log_file=os.path.join(OUT, "task10_online.log"))

    cfg = dataclasses.replace(default_config(), device_str=args.device, log_every=0)
    seed = args.seed
    s_eval = cfg.mc_samples * 2
    G_ONLINE = args.g_online if args.g_online is not None else max(20, cfg.max_gen // 4)
    N_ONLINE = args.n_online
    ref = hv_reference(cfg).to(cfg.device)
    log.info(f"Task10: device={cfg.device} seed={seed} N_ONLINE={N_ONLINE} "
             f"G_ONLINE={G_ONLINE} N_offline={args.n_offline}")

    # ---- train the offline RAMP unified front ----
    t0 = time.perf_counter()
    res = B.run_adaptive(cfg, seed, tag="[task10 offline train]")
    egen = torch.Generator(device=cfg.device); egen.manual_seed(seed + 314159)
    front_pts, front_genos = res.archive.unified_robust_front(egen, s_eval)
    log.info(f"offline front trained in {time.perf_counter()-t0:.1f}s |front|={front_genos.shape[0]}")

    # ---- offline HV over realisations (front deployed as-is) ----
    offline_hv = B.offline_hv_over_realisations(front_genos, cfg, seed, args.n_offline).cpu().numpy()

    # ---- online per-realisation: optimise for each realisation + selection timing ----
    online_hv = np.empty(N_ONLINE)
    opt_times = np.empty(N_ONLINE)
    sel_times = np.empty(N_ONLINE)
    sel_hv = np.empty(N_ONLINE)
    is_cuda = cfg.device.type == "cuda"
    for r in range(N_ONLINE):
        g = torch.Generator(device=cfg.device); g.manual_seed(seed + 90000 + r)
        envs = draw_generation_environments(cfg, g, 1)
        # online optimisation for THIS realisation
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ores = B.single_pop_nsga(lambda gg: robust_evaluate(gg, envs, cfg), cfg, g, G_ONLINE,
                                 seed + r, hv_envs=envs, log_progress=False)
        if is_cuda:
            torch.cuda.synchronize()
        opt_times[r] = time.perf_counter() - t1
        online_hv[r] = ores.hv_history[-1]
        # SELECTION from the precomputed offline front (no optimisation)
        if is_cuda:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        pts = robust_evaluate(front_genos, envs, cfg).double()
        hv_sel = hypervolume_2d(non_dominated(pts), ref)
        if is_cuda:
            torch.cuda.synchronize()
        sel_times[r] = time.perf_counter() - t2
        sel_hv[r] = hv_sel
        log.info(f"  realisation {r+1}/{N_ONLINE}: online_HV={online_hv[r]:.1f} "
                 f"opt={opt_times[r]:.2f}s  sel_HV={hv_sel:.1f} sel={sel_times[r]*1000:.1f}ms")

    med_off, med_on = float(np.median(offline_hv)), float(np.median(online_hv))
    hv_ratio = 100.0 * med_off / med_on if med_on > 0 else float("nan")
    speed = float(opt_times.mean() / sel_times.mean()) if sel_times.mean() > 0 else float("nan")

    ph = {
        "N_ONLINE": str(N_ONLINE),
        "G_ONLINE": str(G_ONLINE),
        "HV_RATIO": f"{hv_ratio:.1f}%",
    }
    result = {
        "seed": seed, "N_ONLINE": N_ONLINE, "G_ONLINE": G_ONLINE, "N_offline": args.n_offline,
        "device": str(cfg.device),
        "offline": {"median": med_off,
                    "p10_p90": [float(np.percentile(offline_hv, 10)), float(np.percentile(offline_hv, 90))]},
        "online": {"median": med_on,
                   "p10_p90": [float(np.percentile(online_hv, 10)), float(np.percentile(online_hv, 90))]},
        "online_opt_time_s": {"mean": float(opt_times.mean()), "std": float(opt_times.std(ddof=1))},
        "online_selection_time_ms": {"mean": float(sel_times.mean() * 1000),
                                     "std": float(sel_times.std(ddof=1) * 1000)},
        "opt_vs_select_speedup": speed,
        "hv_retained_pct": hv_ratio,
        "placeholders_task10": ph,
        "note": "online = online per-realisation optimisation reference (NOT a formal upper bound).",
    }
    with open(os.path.join(OUT, "task10_online.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = [
        f"TASK 10 - offline vs online (seed {seed}, device {cfg.device})", "=" * 66,
        f"N_ONLINE = {N_ONLINE}   G_ONLINE = {G_ONLINE}   N_offline = {args.n_offline}",
        f"offline per-realisation HV: median {med_off:.1f}  10-90% "
        f"[{np.percentile(offline_hv,10):.1f}, {np.percentile(offline_hv,90):.1f}]",
        f"online  per-realisation HV: median {med_on:.1f}  10-90% "
        f"[{np.percentile(online_hv,10):.1f}, {np.percentile(online_hv,90):.1f}]",
        f"online optimisation time / realisation: {opt_times.mean():.2f} +/- {opt_times.std(ddof=1):.2f} s",
        f"online selection time / realisation:    {sel_times.mean()*1000:.1f} +/- {sel_times.std(ddof=1)*1000:.1f} ms",
        f"  -> selecting from the offline front is ~{speed:.0f}x faster than re-optimising",
        f"HV retained = 100 * median(offline)/median(online) = {hv_ratio:.1f}%",
        "\nFILLED PLACEHOLDERS (Task 10):",
    ]
    for k, v in ph.items():
        lines.append(f"{k} = {v}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "task10_online.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
