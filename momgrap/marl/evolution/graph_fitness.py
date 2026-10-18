"""Fitness of a communication graph under a fixed policy, with common random numbers.

Every graph is evaluated on the *same* (seed, tau) conditions with the same fixed
policy so the comparison isolates the graph's effect (CRN). Objectives (all maximised):

  F1 =  mean min-user rate
  F2 =  mean sensing illumination (dB)
  F3 = -mean delivered messages per step
  F4 = -mean outage rate            (optional)

Results are cached by a run-level prefix (policy + config + seeds + taus) plus the
graph bitstring so repeated bitstrings within a search are not re-evaluated.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import pickle

import numpy as np
import torch

from ..evaluation.rollout import run_episode
from .graph_encoding import decode_graph


def graph_objectives(bits, env, make_policy, seeds, taus, include_outage=True) -> list[float]:
    """Evaluate one graph (bits) with the fixed policy over the (seed, tau) grid."""
    env.set_adjacency(decode_graph(bits, env.device))
    min_rates, sens, msgs, outages = [], [], [], []
    for tau in taus:
        env.set_tau(tau)
        for seed in seeds:
            policy = make_policy()
            result, _steps = run_episode(env, policy, seed)
            min_rates.append(result.min_rate_mean)
            sens.append(result.sensing_mean_db)
            msgs.append(result.messages_per_step)
            outages.append(result.outage_rate)
    objs = [float(np.mean(min_rates)), float(np.mean(sens)), -float(np.mean(msgs))]
    if include_outage:
        objs.append(-float(np.mean(outages)))
    return objs


def cache_prefix(policy_id: str, config_dict: dict, seeds, taus) -> str:
    payload = json.dumps(
        {"policy": policy_id, "config": config_dict, "seeds": list(seeds), "taus": list(taus)},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class FitnessCache:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.store: dict[str, list[float]] = {}

    @staticmethod
    def _key(bits) -> str:
        return "".join(str(int(b)) for b in torch.as_tensor(bits).reshape(-1).tolist())

    def get(self, bits):
        return self.store.get(self._key(bits))

    def put(self, bits, objs) -> None:
        self.store[self._key(bits)] = list(objs)

    def save(self, path) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"prefix": self.prefix, "store": self.store}, f)

    @classmethod
    def load(cls, path) -> "FitnessCache":
        with open(path, "rb") as f:
            data = pickle.load(f)
        c = cls(data.get("prefix", ""))
        c.store = data.get("store", {})
        return c


def make_cached_evaluator(env, make_policy, seeds, taus, cache: FitnessCache | None = None,
                          include_outage=True):
    """Return ``evaluate(bits) -> objectives`` backed by ``cache`` (created if None)."""
    cache = cache if cache is not None else FitnessCache()

    def evaluate(bits) -> list[float]:
        cached = cache.get(bits)
        if cached is not None:
            return cached
        objs = graph_objectives(bits, env, make_policy, seeds, taus, include_outage)
        cache.put(bits, objs)
        return objs

    return evaluate, cache
