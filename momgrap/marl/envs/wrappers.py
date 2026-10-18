"""Thin wrappers over :class:`MOMGRAPParallelEnv`.

``TorchParallelWrapper`` returns torch tensors (and a torch centralized state)
instead of numpy — a convenience for the RMAPPO trainer so it can keep everything
on-device without per-step numpy round-trips. It delegates spaces/agents to the
wrapped env and does not change the environment dynamics.
"""

from __future__ import annotations

import torch


class TorchParallelWrapper:
    def __init__(self, env, device: torch.device | str | None = None):
        self.env = env
        self.device = torch.device(device) if device is not None else env.device

    # ---- delegated attributes ----
    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def agents(self):
        return self.env.agents

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def state_size(self) -> int:
        return self.env.state_size()

    def set_tau(self, tau) -> None:
        self.env.set_tau(tau)

    # ---- torch-returning API ----
    def _to_torch(self, obs: dict) -> dict:
        return {a: torch.as_tensor(v, device=self.device) for a, v in obs.items()}

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._to_torch(obs), infos

    def step(self, actions: dict):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._to_torch(obs), rewards, terminations, truncations, infos

    def state(self) -> torch.Tensor:
        return torch.as_tensor(self.env.state(), device=self.device)

    def close(self):
        return self.env.close()
