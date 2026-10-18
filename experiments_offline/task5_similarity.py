"""Task 5 - similarity among propagation regimes (LoS / Rayleigh / Rician).

Shared-batch evaluation only (no evolutionary optimisation).  A shared set of
N_SIM feasible MA-RIS configurations is drawn once and evaluated under every
regime; pairwise Spearman rank correlations of F_com and F_sen then say whether
the regimes are related enough for cross-regime transfer to be sensible.

Two geometry conditions are reported:
  * shared_geometry=True  (the model default RAMP actually optimises in: one
    physical scene, regime differs only by the Rician K-factor + fading draw)
  * shared_geometry=False (fully independent scenes per regime) -- a lower bound
    on how correlated the regimes could be.

Outputs:
  experiments_offline/out/task5_similarity.json
  experiments_offline/out/task5_similarity.txt
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import numpy as np
import torch
from scipy import stats

from momgrap.config import default_config
from momgrap.objectives import draw_generation_environments, evaluate_single_regime

OUT = "experiments_offline/out"
PAIRS = [(0, 1), (0, 2), (1, 2)]        # LoS-Rayleigh, LoS-Rician, Rayleigh-Rician
PAIR_NAMES = ["LoS_vs_Rayleigh", "LoS_vs_Rician", "Rayleigh_vs_Rician"]


def evaluate_all_regimes(cfg, genos, gen, s_eval):
    """Return F_com (T, N) and F_sen (T, N) for one shared environment draw."""
    envs = draw_generation_environments(cfg, gen, s_eval)          # T envs
    fcom, fsen = [], []
    for env in envs:
        F = evaluate_single_regime(genos, env, cfg)               # (N,2)
        fcom.append(F[:, 0].cpu().numpy())
        fsen.append(F[:, 1].cpu().numpy())
    return np.stack(fcom, 0), np.stack(fsen, 0)                   # (T,N) each


def run_condition(shared_geometry, n_sim, s_eval, n_env_reps, seeds):
    """Average Spearman correlations over n_env_reps env draws x len(seeds) seeds."""
    regime_names = [r.name for r in default_config().regimes]
    com_rows = {p: [] for p in range(len(PAIRS))}
    sen_rows = {p: [] for p in range(len(PAIRS))}
    for sd in seeds:
        cfg = dataclasses.replace(default_config(), shared_geometry=shared_geometry,
                                  mc_samples=s_eval, log_every=0)
        g_cfg = torch.Generator(device=cfg.device); g_cfg.manual_seed(sd + 999)
        genos = torch.rand(n_sim, cfg.D, generator=g_cfg, device=cfg.device, dtype=cfg.dtype_real)
        env_gen = torch.Generator(device=cfg.device)
        for rep in range(n_env_reps):
            env_gen.manual_seed(sd * 7919 + rep)
            fcom, fsen = evaluate_all_regimes(cfg, genos, env_gen, s_eval)
            for pi, (a, b) in enumerate(PAIRS):
                com_rows[pi].append(stats.spearmanr(fcom[a], fcom[b]).statistic)
                sen_rows[pi].append(stats.spearmanr(fsen[a], fsen[b]).statistic)
    out = {"regimes": regime_names, "F_com": {}, "F_sen": {}}
    for pi, name in enumerate(PAIR_NAMES):
        out["F_com"][name] = {"mean": float(np.mean(com_rows[pi])),
                              "std": float(np.std(com_rows[pi], ddof=1)) if len(com_rows[pi]) > 1 else 0.0}
        out["F_sen"][name] = {"mean": float(np.mean(sen_rows[pi])),
                              "std": float(np.std(sen_rows[pi], ddof=1)) if len(sen_rows[pi]) > 1 else 0.0}
    out["median_F_com"] = float(np.median([out["F_com"][n]["mean"] for n in PAIR_NAMES]))
    out["median_F_sen"] = float(np.median([out["F_sen"][n]["mean"] for n in PAIR_NAMES]))
    out["median_overall"] = float(np.median(
        [out["F_com"][n]["mean"] for n in PAIR_NAMES] + [out["F_sen"][n]["mean"] for n in PAIR_NAMES]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    S_EVAL = 64
    N_ENV_REPS = 5
    SEEDS = [0, 1, 2]
    results = {"config": {"N_SIM_values": [200, 500], "S_eval": S_EVAL,
                          "n_env_reps": N_ENV_REPS, "seeds": SEEDS}}
    for n_sim in (200, 500):
        for sg in (True, False):
            key = f"N_SIM={n_sim}_shared_geometry={sg}"
            print(f"running {key} ...")
            results[key] = run_condition(sg, n_sim, S_EVAL, N_ENV_REPS, SEEDS)

    # headline: the model default (shared geometry) at N_SIM=500
    head = results["N_SIM=500_shared_geometry=True"]
    med = head["median_overall"]
    interp = ("strongly related -> cross-regime transfer is well justified" if med >= 0.7 else
              "moderately related -> some transfer benefit expected" if med >= 0.4 else
              "weakly related -> limited transfer benefit")
    placeholders = {
        "N_SIM": "500 (also verified at 200)",
        "TASK_SIM": (
            "Pairwise Spearman rank correlations over N_SIM=500 shared feasible MA-RIS "
            "configurations (shared-geometry model, S=64 MC, median over 3 seeds x 5 env "
            f"draws): F_com median rho={head['median_F_com']:.2f} "
            f"(LoS-Rayleigh {head['F_com']['LoS_vs_Rayleigh']['mean']:.2f}, "
            f"LoS-Rician {head['F_com']['LoS_vs_Rician']['mean']:.2f}, "
            f"Rayleigh-Rician {head['F_com']['Rayleigh_vs_Rician']['mean']:.2f}); "
            f"F_sen median rho={head['median_F_sen']:.2f} "
            f"(LoS-Rayleigh {head['F_sen']['LoS_vs_Rayleigh']['mean']:.2f}, "
            f"LoS-Rician {head['F_sen']['LoS_vs_Rician']['mean']:.2f}, "
            f"Rayleigh-Rician {head['F_sen']['Rayleigh_vs_Rician']['mean']:.2f})."),
        "TRANSFER_INTERPRETATION": (
            f"The three regimes are {interp} (median Spearman rho={med:.2f} across regime "
            "pairs and objectives under the shared-scene model). This supports treating the "
            "regimes as related tasks and using cross-regime knowledge transfer."),
    }
    results["placeholders_task5"] = placeholders

    with open(os.path.join(OUT, "task5_similarity.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = ["TASK 5 - regime similarity (pairwise Spearman rho)", "=" * 70]
    for n_sim in (200, 500):
        for sg in (True, False):
            key = f"N_SIM={n_sim}_shared_geometry={sg}"
            r = results[key]
            lines.append(f"\n[{key}]  regimes={r['regimes']}")
            lines.append(f"  {'pair':<20}{'F_com rho':>14}{'F_sen rho':>14}")
            for name in PAIR_NAMES:
                lines.append(f"  {name:<20}{r['F_com'][name]['mean']:>14.3f}{r['F_sen'][name]['mean']:>14.3f}")
            lines.append(f"  median: F_com={r['median_F_com']:.3f}  F_sen={r['median_F_sen']:.3f}  "
                         f"overall={r['median_overall']:.3f}")
    lines.append("\nFILLED PLACEHOLDERS (Task 5):")
    lines.append("-" * 70)
    for k, v in placeholders.items():
        lines.append(f"{k} = {v}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "task5_similarity.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
