"""Task 1b + Task 2 driver — re-run every method with instrumentation.

For each (method, seed) it records the per-seed unified robust front (for IGD),
timing (search / archive re-eval / total), peak memory, fitness-eval count and
throughput, plus the RMP/survival trajectory for the adaptive method.

Results are checkpointed to experiments_offline/out/rerun_fronts.pkl after every run, so
the job is resumable (re-invoking skips finished (method, seed) pairs).

Usage:
    python experiments_offline/rerun_fronts.py --seeds 15 [--device cuda|cpu|auto]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from momgrap.config import default_config
from momgrap.logging_utils import setup_logging
from experiments_offline.harness import run_instrumented

OUT = "experiments_offline/out"
CKPT = os.path.join(OUT, "rerun_fronts.pkl")
METHODS = ["adaptive", "fixed_rmp", "no_transfer", "pooled", "mo_de", "mopso"]


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            return pickle.load(f)
    return {"meta": {}, "results": {}}


def save_ckpt(state):
    os.makedirs(OUT, exist_ok=True)
    tmp = CKPT + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, CKPT)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--methods", nargs="*", default=METHODS)
    args = p.parse_args(argv)

    log = setup_logging(log_file=os.path.join(OUT, "rerun_fronts.log"))
    cfg = dataclasses.replace(default_config(), device_str=args.device, log_every=100)
    seeds = list(range(args.seeds))
    log.info(f"rerun_fronts: device={cfg.device} methods={args.methods} seeds={seeds} "
             f"(N={cfg.pop_size} G={cfg.max_gen} S={cfg.mc_samples})")

    state = load_ckpt()
    state["meta"] = {"cfg_repr": repr(cfg), "seeds": seeds, "methods": args.methods,
                     "device": str(cfg.device)}
    done = set(state["results"].keys())

    total = len(args.methods) * len(seeds)
    t0 = time.perf_counter()
    n_done = 0
    for sd in seeds:
        for m in args.methods:
            key = f"{m}|{sd}"
            if key in done:
                n_done += 1
                continue
            r = run_instrumented(m, cfg, sd)
            state["results"][key] = r
            save_ckpt(state)
            n_done += 1
            elapsed = time.perf_counter() - t0
            rate = n_done / max(1e-9, elapsed)
            eta = (total - n_done) / max(1e-9, rate)
            log.info(f"[{n_done}/{total}] {m} s{sd}: HV_front~{r['final_hv_curve']:.1f} "
                     f"|front|={r['n_front']} t_search={r['t_search']:.1f}s "
                     f"t_arch={r['t_archive']:.1f}s mem={r['peak_mem_bytes']/1e6:.0f}MB "
                     f"({r['mem_kind']}) evals/s={r['evals_per_sec']:.0f} ETA {eta/60:.1f}min")
    log.info(f"rerun_fronts complete: {total} runs in {(time.perf_counter()-t0)/60:.1f} min "
             f"-> {CKPT}")


if __name__ == "__main__":
    main()
