"""Baseline (non-learned) policies for the MARL environment.

Every heuristic uses only *estimated / local* data (``env.current_estimate()`` — the
delayed estimate the actors see) and never the true channel or the true realized
reward when choosing actions.

Each ``act(observations, env)`` returns ``{agent: np.array([phys_idx, comm_idx])}``
for the currently active agents.
"""

from __future__ import annotations

import cmath

import numpy as np
import torch

from ..actions.codebooks import AGENTS


# ---------------------------------------------------------------- heuristics
def _bs_heuristic_index(est, power_codebook_size: int) -> int:
    """Equal power when user gains are balanced, inverse-gain when unbalanced."""
    agg = est.aggregate[0, 0]                       # (K,M)
    gains = agg.abs().pow(2).sum(-1)                # (K,)
    ratio = float(gains.max() / gains.min().clamp_min(1e-30))
    return min(2, power_codebook_size - 1) if ratio > 3.0 else 0   # 2 = inv-gain alpha=1


def _ris_heuristic_index(est, phase_step: float, G: int) -> int:
    """Score {no-op, global +Δ, global -Δ} by estimated min-user aggregate norm.

    A global phase shift rotates the whole reflected path by e^{jΔ}, so the candidate
    aggregate is ``direct + e^{±jΔ} reflected`` — computable from the estimate alone
    (no per-candidate physics rebuild, no true channel).
    """
    d = est.direct[0, 0]                            # (K,M)
    r = est.reflected[0, 0]                         # (K,M)

    def score(rot: complex) -> float:
        agg = d + rot * r
        return float(agg.abs().pow(2).sum(-1).sqrt().min())

    scores = [score(1.0 + 0j), score(cmath.exp(1j * phase_step)), score(cmath.exp(-1j * phase_step))]
    idx = int(np.argmax(scores))
    return (0, 2 * G + 1, 2 * G + 2)[idx]


def _ma_heuristic_index(est, step_count: int, ma_update_period: int) -> int:
    """No-op off-cycle; on update steps, expand when direct-channel spatial variation is high."""
    if step_count % ma_update_period != 0:
        return 0
    mag = est.direct[0, 0].abs()                    # (K,M)
    cv = float((mag.std(dim=-1) / mag.mean(dim=-1).clamp_min(1e-30)).mean())
    return 1 if cv > 0.5 else 0                     # 1 = expand_y


def _heuristic_phys(env) -> dict:
    est = env.current_estimate()
    G = env.evo.actions.ris_group_count
    return {
        "bs_agent": _bs_heuristic_index(est, env.evo.actions.power_codebook_size),
        "ris_agent": _ris_heuristic_index(est, env.phase_step, G),
        "ma_agent": _ma_heuristic_index(est, env.step_count, env.evo.actions.ma_update_period),
    }


def _act(phys: dict, comm: dict, agents) -> dict:
    return {a: np.array([phys[a], comm[a]], dtype=np.int64) for a in agents}


# ---------------------------------------------------------------- policies
class Policy:
    name = "policy"

    def reset(self) -> None:
        pass

    def act(self, observations: dict, env) -> dict:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self._seed)

    def act(self, observations, env) -> dict:
        out = {}
        for a in env.agents:
            nvec = env.action_space(a).nvec
            out[a] = np.array([self.rng.integers(int(nvec[0])), self.rng.integers(int(nvec[1]))],
                              dtype=np.int64)
        return out


class NoOpPolicy(Policy):
    name = "noop"

    def act(self, observations, env) -> dict:
        return {a: np.array([0, 0], dtype=np.int64) for a in env.agents}


class EqualPowerNoCommunicationPolicy(Policy):
    name = "equal_power_no_comm"

    def act(self, observations, env) -> dict:
        phys = _heuristic_phys(env)
        phys["bs_agent"] = 0                          # force equal power
        comm = {a: 0 for a in env.possible_agents}
        return _act(phys, comm, env.agents)


class HeuristicNoCommunicationPolicy(Policy):
    name = "heuristic_no_comm"

    def act(self, observations, env) -> dict:
        phys = _heuristic_phys(env)
        comm = {a: 0 for a in env.possible_agents}
        return _act(phys, comm, env.agents)


class HeuristicFullCommunicationPolicy(Policy):
    name = "heuristic_full_comm"

    def act(self, observations, env) -> dict:
        phys = _heuristic_phys(env)
        comm = {a: 1 for a in env.possible_agents}    # every agent sends (to its first other)
        return _act(phys, comm, env.agents)


class RandomCommunicationMatchedBudgetPolicy(Policy):
    name = "random_comm_matched_budget"

    def __init__(self, seed: int = 0):
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self._seed)

    def act(self, observations, env) -> dict:
        phys = _heuristic_phys(env)
        # always attempt to send (matches full-comm request rate) but to a random destination
        comm = {a: int(self.rng.integers(1, 3)) for a in env.possible_agents}
        return _act(phys, comm, env.agents)


class StaticRAMPInitializationPolicy(NoOpPolicy):
    """Static policy (no reconfiguration, no comm).

    Meant to be paired with a RAMP-initialised environment (``init_mode`` from
    Task 12): the offline-optimised MA/RIS layout is held fixed for the whole
    episode, so this measures the value of the static RAMP front point.
    """

    name = "static_ramp_init"


class TrainedActorPolicy(Policy):
    """Wrap trained recurrent actors (+ obs normalisation) as an evaluation policy.

    Bridges the RMAPPO actors to the rollout/ablation harness. Manages per-episode
    recurrent hidden state (reset() clears it) and normalises observations with the
    trainer's running statistics. Comm actions come from the actor's head — env-level
    forcing (set_communication_mode) can still override them for the ablation.
    """

    name = "trained_actor"

    def __init__(self, actors, obs_rms, device="cpu", deterministic=True, active_agents=None):
        self.actors = actors
        self.obs_rms = obs_rms
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.active_agents = set(actors if active_agents is None else active_agents)
        self._hidden = None

    def reset(self) -> None:
        self._hidden = {a: self.actors[a].zero_hidden(1, self.device) for a in self.actors}

    @torch.no_grad()
    def act(self, observations, env) -> dict:
        from ..communication.graph import valid_action_mask
        if self._hidden is None:
            self.reset()
        masks = torch.ones(1, 1, device=self.device)
        adj = env.adjacency
        actions = {}
        for i, a in enumerate(AGENTS):
            o = torch.as_tensor(observations[a], device=self.device).unsqueeze(0)
            o = self.obs_rms[a].normalize(o)
            cm = valid_action_mask(i, adj).unsqueeze(0).to(self.device)
            if not env.evo.communication.enabled:
                cm[:] = False
                cm[:, 0] = True
            pm = env.physical_action_mask(a).unsqueeze(0).to(self.device)
            if a not in self.active_agents:
                pm[:] = False
                pm[:, 0] = True
            act, _lp, self._hidden[a] = self.actors[a].act(
                o, self._hidden[a], masks, action_masks=(pm, cm),
                deterministic=self.deterministic)
            actions[a] = act[0].cpu().numpy()
        return actions


BASELINE_REGISTRY = {
    p.name: p
    for p in (
        RandomPolicy, NoOpPolicy, EqualPowerNoCommunicationPolicy,
        HeuristicNoCommunicationPolicy, HeuristicFullCommunicationPolicy,
        RandomCommunicationMatchedBudgetPolicy, StaticRAMPInitializationPolicy,
    )
}


def make_policy(name: str, seed: int = 0) -> Policy:
    if name == "online_optimizer":
        from .online_optimizer import OnlineOptimizerPolicy  # lazy: avoids import cycle
        return OnlineOptimizerPolicy(seed=seed)
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}")
    cls = BASELINE_REGISTRY[name]
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()
