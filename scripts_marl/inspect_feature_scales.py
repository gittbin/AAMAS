"""Inspect channel component scales on new development initial states only."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv


def main():
    torch.set_num_threads(1)
    cfg = load_evocomm_config(ROOT / "configs_marl/paper_v2.yaml")
    env = MOMGRAPParallelEnv(cfg, eval_mode=True)
    rows = []
    for seed in range(1000000, 1000030):
        obs, _ = env.reset(seed=seed)
        est = env.current_estimate()
        direct = float(est.direct.abs().square().mean())
        reflected = float(est.reflected.abs().square().mean())
        rows.append({"scenario_seed": seed, "regime": cfg.physical.regimes[env._pstate.regime_index].name,
                     "reflected_to_direct_power_db": 10 * np.log10(reflected / direct),
                     "aggregate_vs_direct_relative_error": float((est.aggregate - est.direct).norm() / est.direct.norm()),
                     "bs_ma_initial_channel_features_max_difference": float(np.max(np.abs(obs["bs_agent"][:16] - obs["ma_agent"][:16])))})
    env.close()
    out = ROOT / "outputs_marl/paper_v2/campaign/report/log_analysis/component_scales.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"scope": "30 fresh development scenario initial estimates, USPA zero-phase; not final evaluation or all possible configurations.",
              "episodes": rows,
              "reflected_to_direct_power_db_range": [min(r["reflected_to_direct_power_db"] for r in rows), max(r["reflected_to_direct_power_db"] for r in rows)],
              "regime_counts": {name: sum(r["regime"] == name for r in rows) for name in {r["regime"] for r in rows}}}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
