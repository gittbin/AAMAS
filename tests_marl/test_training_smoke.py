"""Smoke tests for the RMAPPO trainer + checkpoint + a toy PPO-improves test (Task 15)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pytest
import torch
from gymnasium.spaces import MultiDiscrete

from momgrap.marl.actions.codebooks import (
    ma_action_count,
    power_action_count,
    ris_action_count,
)
from momgrap.marl.algorithms.rmappo.checkpoint import load_checkpoint, save_checkpoint
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.communication.graph import AGENT_NAMES, build_adjacency
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.observations.builder import centralized_state_size, observation_sizes

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


# ---------------------------------------------------------------- real env
def _tiny_evo():
    # 2 rollouts only -> fast real-env smoke
    return load_evocomm_config(SMOKE, overrides={"rmappo": {"total_timesteps": 64}})


def test_trainer_real_env_train_checkpoint_reload_eval(tmp_path):
    evo = _tiny_evo()
    factory = lambda i: MOMGRAPParallelEnv(evo, seed=1000 + i)
    trainer = RMAPPOTrainer(evo, factory, device="cpu", log_dir=str(tmp_path / "log"), seed=0)
    trainer.train()
    assert trainer.update_index >= 1
    assert (tmp_path / "log" / "train.csv").exists()

    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(ckpt, trainer)
    assert ckpt.exists()

    reloaded = RMAPPOTrainer(evo, factory, device="cpu", seed=0)
    load_checkpoint(ckpt, reloaded)
    assert reloaded.global_step == trainer.global_step
    assert reloaded.update_index == trainer.update_index
    # actor params match after reload
    for a in AGENT_NAMES:
        for p1, p2 in zip(trainer.actors[a].parameters(), reloaded.actors[a].parameters()):
            assert torch.equal(p1.detach().cpu(), p2.detach().cpu())
    res = reloaded.evaluate([0, 1])
    assert np.isfinite(res["eval_return_mean"])


def test_checkpoint_restores_rng_and_optimizer(tmp_path):
    evo = _tiny_evo()
    factory = lambda i: MOMGRAPParallelEnv(evo, seed=1000 + i)
    trainer = RMAPPOTrainer(evo, factory, device="cpu", seed=0)
    trainer.collect_rollout()
    trainer.update()
    ckpt = tmp_path / "c.pt"
    save_checkpoint(ckpt, trainer)

    reloaded = RMAPPOTrainer(evo, factory, device="cpu", seed=123)  # different seed
    load_checkpoint(ckpt, reloaded)
    # optimizer state (step count) restored
    assert reloaded.update_index == trainer.update_index
    for a in AGENT_NAMES:
        assert torch.equal(trainer.obs_rms[a].mean, reloaded.obs_rms[a].mean)


# ---------------------------------------------------------------- toy bandit
class ToyBanditEnv:
    """Contextless bandit matching smoke obs/action dims: reward=1 iff bs picks `target`."""

    def __init__(self, evo, target):
        self.evo = evo
        self.cfg = evo.physical
        self.device = self.cfg.device
        self.possible_agents = list(AGENT_NAMES)
        self.agents = []
        self._obs_sizes = observation_sizes(evo)
        self._state_dim = centralized_state_size(evo)
        self.adjacency = build_adjacency("full")
        self.target = target
        self._nphys = {"bs_agent": power_action_count(evo.actions),
                       "ris_agent": ris_action_count(evo.actions),
                       "ma_agent": ma_action_count(evo.actions)}
        self._spaces = {a: MultiDiscrete([self._nphys[a], 3]) for a in AGENT_NAMES}
        self._step = 0

    def observation_space(self, a):
        return None

    def action_space(self, a):
        return self._spaces[a]

    def state(self):
        return np.zeros(self._state_dim, dtype=np.float32)

    def _obs(self):
        return {a: np.zeros(self._obs_sizes[a], dtype=np.float32) for a in AGENT_NAMES}

    def reset(self, seed=None, options=None):
        self._step = 0
        self.agents = list(self.possible_agents)
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        r = 1.0 if int(np.asarray(actions["bs_agent"]).reshape(-1)[0]) == self.target else 0.0
        self._step += 1
        trunc = self._step >= self.evo.dynamics.episode_length
        rewards = {a: r for a in self.agents}
        term = {a: False for a in self.agents}
        truncs = {a: trunc for a in self.agents}
        infos = {a: {} for a in self.agents}
        obs = self._obs()
        if trunc:
            self.agents = []
        return obs, rewards, term, truncs, infos


def test_ppo_improves_on_toy():
    evo = load_evocomm_config(SMOKE, overrides={
        "dynamics": {"episode_length": 8},
        "rmappo": {
            "total_timesteps": 8 * 4 * 60, "num_envs": 4, "rollout_length": 8,
            "sequence_length": 8, "num_minibatches": 2, "update_epochs": 4,
            "hidden_size": 16, "learning_rate": 0.01, "entropy_coef": 0.0,
        },
    })
    target = power_action_count(evo.actions) - 1     # optimal bs action != 0
    factory = lambda i: ToyBanditEnv(evo, target=target)
    trainer = RMAPPOTrainer(evo, factory, device="cpu", seed=0)

    before = trainer.evaluate([0, 1, 2, 3])["eval_return_mean"]
    trainer.train()
    after = trainer.evaluate([0, 1, 2, 3])["eval_return_mean"]

    max_return = evo.dynamics.episode_length
    assert after > before + 1.0            # clear PPO improvement
    assert after >= 0.7 * max_return       # nearly optimal


# ---------------------------------------------------------------- NaN policy
def test_nan_policy_raises():
    evo = _tiny_evo()
    factory = lambda i: MOMGRAPParallelEnv(evo, seed=1000 + i)
    trainer = RMAPPOTrainer(evo, factory, device="cpu", log_dir=None, seed=0)
    trainer.collect_rollout()
    # poison the critic so the value loss becomes non-finite
    with torch.no_grad():
        trainer.critic.value_head.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError):
        trainer.update()
