"""Shared instrumentation harness for the offline experiment runs.

Runs any method (RAMP/adaptive, fixed_rmp, no_transfer, pooled, mo_de, mopso) with:
  * timing split into evolutionary-search time and final archive cross-regime
    re-evaluation time (building the unified robust front);
  * peak memory (CUDA max_memory_allocated on GPU, else peak process RSS on CPU);
  * an analytic fitness-evaluation count + evaluations/second;
  * the per-seed unified robust front (points + genotypes) for IGD;
  * RMP / survival trajectories and transfer counts for the adaptive method.

Also provides IGD helpers (common reference front over the non-dominated union).
Device-agnostic: uses cfg.device (CUDA if the installed torch has it, else CPU).
"""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from momgrap import baselines as B
from momgrap.metrics import hv_reference, hypervolume_2d, non_dominated
from momgrap.nsga import fast_nondominated_sort

# ------------------------------------------------------------------ method map
MFEA_METHODS = {"adaptive", "fixed_rmp", "no_transfer"}
SINGLEPOP_METHODS = {"pooled", "mo_de", "mopso"}


def method_runner(name):
    return {
        "adaptive": B.run_adaptive,
        "fixed_rmp": B.run_fixed_rmp,
        "no_transfer": B.run_no_transfer,
        "pooled": B.run_pooled_single_task,
        "mo_de": B.run_mo_de,
        "mopso": B.run_mopso,
    }[name]


def analytic_fitness_evals(name, cfg):
    """One fitness eval = one genotype scored to [F_com, F_ill] on a generation's MC batch.

    MFEA + pooled/mo_de score pop+offspring (2N) each generation; mopso scores the
    swarm (N) each generation plus the initial swarm.
    """
    N, G = cfg.pop_size, cfg.max_gen
    if name == "mopso":
        return N * (G + 1)
    return 2 * N * G


class _CPUMemSampler:
    """Background sampler of peak process RSS (bytes) for CPU runs."""

    def __init__(self, dt=0.02):
        import psutil
        self._p = psutil.Process()
        self._dt = dt
        self._peak = self._p.memory_info().rss
        self._stop = threading.Event()
        self._t = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, self._p.memory_info().rss)
            except Exception:
                pass
            self._stop.wait(self._dt)

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)

    @property
    def peak_bytes(self):
        return int(self._peak)


def _eval_generator(cfg, seed):
    g = torch.Generator(device=cfg.device)
    g.manual_seed(int(seed) + 314159)
    return g


def run_instrumented(name, cfg, seed, s_eval=None, runner=None):
    """Run one method+seed with full instrumentation. Returns a plain dict.

    ``runner`` overrides the method dispatch (e.g. a fixed-RMP value sweep that
    calls run_mfea on a pre-modified cfg); it must accept ``(cfg, seed, tag)`` and
    return an object exposing ``.archive`` and ``.hv_history``.
    """
    s_eval = (cfg.mc_samples * 2) if s_eval is None else s_eval
    device = cfg.device
    is_cuda = device.type == "cuda"
    fn = runner if runner is not None else method_runner(name)

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    cpu_mem = None
    if not is_cuda:
        cpu_mem = _CPUMemSampler()
        cpu_mem.__enter__()

    # ---- evolutionary search ----
    t0 = time.perf_counter()
    res = fn(cfg, seed, tag=f"[{name} s{seed}]")
    if is_cuda:
        torch.cuda.synchronize()
    t_search = time.perf_counter() - t0

    # ---- final archive cross-regime re-evaluation (build unified robust front) ----
    t1 = time.perf_counter()
    front, genos = res.archive.unified_robust_front(_eval_generator(cfg, seed), s_eval)
    if is_cuda:
        torch.cuda.synchronize()
    t_archive = time.perf_counter() - t1

    if not is_cuda:
        cpu_mem.__exit__()
        peak_mem = cpu_mem.peak_bytes
        mem_kind = "cpu_rss"
    else:
        torch.cuda.synchronize()
        peak_mem = int(torch.cuda.max_memory_allocated())
        mem_kind = "cuda_alloc"

    front_np = front.detach().cpu().numpy()
    if front_np.shape[0] > 1:
        front_np = front_np[np.argsort(front_np[:, 0])]

    total = t_search + t_archive
    n_eval = analytic_fitness_evals(name, cfg)
    out = {
        "method": name, "seed": int(seed),
        "front": front_np,                                  # (P,2)
        "n_front": int(front_np.shape[0]),
        "archive_size": int(len(res.archive)),
        "t_search": float(t_search), "t_archive": float(t_archive),
        "t_total": float(total),
        "archive_frac": float(t_archive / total) if total > 0 else 0.0,
        "n_fitness_evals": int(n_eval),
        "evals_per_sec": float(n_eval / t_search) if t_search > 0 else 0.0,
        "peak_mem_bytes": int(peak_mem), "mem_kind": mem_kind,
        "final_hv_curve": float(np.asarray(res.hv_history)[-1]),
    }
    if name in MFEA_METHODS:
        out["rmp_history"] = np.asarray(res.rmp_history, dtype=float)
        out["rho_history"] = np.asarray(res.rho_history, dtype=float)
        # transfer diagnostics (additive fields on MFEAResult; empty list if absent)
        out["n_transfer_history"] = np.asarray(getattr(res, "n_transfer_history", []), dtype=float)
        out["n_transfer_success_history"] = np.asarray(getattr(res, "n_transfer_success_history", []), dtype=float)
        out["n_base_history"] = np.asarray(getattr(res, "n_base_history", []), dtype=float)
        out["n_base_success_history"] = np.asarray(getattr(res, "n_base_success_history", []), dtype=float)
    return out


# ------------------------------------------------------------------ IGD helpers
def build_reference_front(fronts):
    """Common IGD reference front = non-dominated union of all supplied fronts (Nx2)."""
    allpts = np.concatenate([f for f in fronts if f is not None and f.shape[0] > 0], axis=0)
    R = non_dominated(torch.tensor(allpts, dtype=torch.float64)).cpu().numpy()
    return R


def igd_normalized(front, R, mn, rng):
    """Normalised IGD of `front` vs reference `R` (both Nx2), using shared mn/rng."""
    if front is None or front.shape[0] == 0:
        return float("nan")
    Rn = (R - mn) / rng
    Fn = (front - mn) / rng
    d = np.sqrt(((Rn[:, None, :] - Fn[None, :, :]) ** 2).sum(-1))     # (|R|,|F|)
    return float(d.min(axis=1).mean())


def robust_hv_of_front(front_np, cfg):
    """Fixed-reference HV of a robust front (Nx2)."""
    ref = hv_reference(cfg)
    return float(hypervolume_2d(torch.tensor(front_np, dtype=torch.float64), ref))
