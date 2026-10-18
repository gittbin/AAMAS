"""Env smoke: PettingZoo API check + a short random rollout with printed stats.

Run: ``python scripts_marl/smoke_env.py [config.yaml]``
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from pettingzoo.test import parallel_api_test

from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(_REPO_ROOT / "configs_marl" / "smoke.yaml")
    evo = load_evocomm_config(cfg_path)

    print(f"--- PettingZoo parallel_api_test ({cfg_path}) ---")
    parallel_api_test(MOMGRAPParallelEnv(evo, seed=0), num_cycles=50)
    print("  parallel_api_test OK")

    env = MOMGRAPParallelEnv(evo, seed=0)
    obs, _infos = env.reset(seed=0)
    print("\n--- observation / action / state sizes ---")
    for a in env.possible_agents:
        print(f"  {a:10s} obs={obs[a].shape}  action={env.action_space(a).nvec.tolist()}")
    print(f"  centralized state dim = {env.state().shape}")

    print("\n--- random rollout ---")
    rng = np.random.default_rng(0)
    total = 0.0
    steps = 0
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        total += rewards["bs_agent"] if rewards else 0.0
        steps += 1
    info = infos["bs_agent"]
    print(f"  steps={steps}  return={total:.3f}")
    print(f"  last: regime={info['regime']} tau={info['tau']:.3f} "
          f"min_rate={info['min_user_rate']:.3f} sum_rate={info['sum_rate']:.3f} "
          f"sensing_dB={info['min_sensing_illumination_db']:.2f} outage={info['outage_count']}")
    print(f"  messages: requested={info['message_requested']} delivered={info['message_delivered']} "
          f"dropped={info['message_dropped']} graph_rejected={info['graph_rejected']}")
    print("\nENV SMOKE OK")


if __name__ == "__main__":
    main()
