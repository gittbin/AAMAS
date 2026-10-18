"""Tests for evolutionary communication-graph search (Task 17)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from momgrap.marl.communication.graph import build_adjacency
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.evolution.graph_encoding import (
    EDGES,
    GraphConstraints,
    decode_graph,
    encode_graph,
    is_weakly_connected,
    repair_graph,
)
from momgrap.marl.evolution.graph_fitness import (
    FitnessCache,
    graph_objectives,
    make_cached_evaluator,
)
from momgrap.marl.evolution.graph_search import evaluate_preset_graphs, run_graph_search
from momgrap.marl.policies.baselines import make_policy

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _env(**overrides):
    evo = load_evocomm_config(SMOKE, overrides=overrides or None)
    return MOMGRAPParallelEnv(evo, seed=0, eval_mode=True, comm_mode="learned")


# ---------------------------------------------------------------- encoding
def test_encode_decode_roundtrip():
    for name in ("none", "full", "ring", "star_bs"):
        adj = build_adjacency(name)
        bits = encode_graph(adj)
        assert bits.shape == (6,)
        back = decode_graph(bits)
        assert back.shape == (3, 3) and back.dtype == torch.bool
        assert torch.equal(back, adj)
        assert not bool(back.diagonal().any())


def test_decode_edges_mapping():
    bits = torch.zeros(6, dtype=torch.long)
    bits[0] = 1                              # bs -> ris
    A = decode_graph(bits)
    assert A[0, 1] and A.sum() == 1


# ---------------------------------------------------------------- repair
def test_repair_max_edges():
    bits = torch.ones(6, dtype=torch.long)
    out = repair_graph(bits, GraphConstraints(max_edges=2))
    assert int(out.sum()) == 2


def test_repair_non_empty_and_bs_incident():
    empty = torch.zeros(6, dtype=torch.long)
    out = repair_graph(empty, GraphConstraints(allow_empty=False))
    assert int(out.sum()) >= 1
    out2 = repair_graph(empty, GraphConstraints(require_bs_incident=True))
    incident = any(0 in EDGES[k] for k in range(6) if int(out2[k]) == 1)
    assert incident


def test_repair_weak_connectivity():
    empty = torch.zeros(6, dtype=torch.long)
    out = repair_graph(empty, GraphConstraints(require_weak_connectivity=True))
    assert is_weakly_connected(out)


def test_repair_deterministic():
    bits = torch.tensor([1, 1, 1, 1, 1, 1])
    c = GraphConstraints(max_edges=3)
    assert torch.equal(repair_graph(bits, c), repair_graph(bits, c))


# ---------------------------------------------------------------- fitness / CRN
def test_graph_objectives_shape_and_crn():
    env = _env(communication={"graph_type": "full"})
    mk = lambda: make_policy("heuristic_full_comm")
    bits = encode_graph(build_adjacency("full"))
    o1 = graph_objectives(bits, env, mk, seeds=[0], taus=[0.05])
    o2 = graph_objectives(bits, env, mk, seeds=[0], taus=[0.05])
    assert len(o1) == 4
    assert all(x == x for x in o1)           # finite
    assert o1 == o2                          # common random numbers -> deterministic


def test_graph_affects_communication():
    env = _env(communication={"graph_type": "full"})
    mk = lambda: make_policy("heuristic_full_comm")
    none_obj = graph_objectives(encode_graph(build_adjacency("none")), env, mk, [0], [0.05])
    full_obj = graph_objectives(encode_graph(build_adjacency("full")), env, mk, [0], [0.05])
    # F3 = -mean delivered messages: none delivers 0 (=0), full delivers >0 (<0)
    assert none_obj[2] == 0.0
    assert full_obj[2] < 0.0


# ---------------------------------------------------------------- cache
def test_fitness_cache_roundtrip(tmp_path):
    cache = FitnessCache("prefix")
    bits = torch.tensor([1, 0, 1, 0, 0, 1])
    cache.put(bits, [1.0, 2.0, -0.5, -0.1])
    assert cache.get(bits) == [1.0, 2.0, -0.5, -0.1]
    cache.save(tmp_path / "c.pkl")
    loaded = FitnessCache.load(tmp_path / "c.pkl")
    assert loaded.prefix == "prefix"
    assert loaded.get(bits) == [1.0, 2.0, -0.5, -0.1]


def test_cached_evaluator_reuses():
    env = _env(communication={"graph_type": "full"})
    evaluate_fn, cache = make_cached_evaluator(
        env, lambda: make_policy("heuristic_full_comm"), [0], [0.05])
    bits = encode_graph(build_adjacency("full"))
    o1 = evaluate_fn(bits)
    assert len(cache.store) == 1
    o2 = evaluate_fn(bits)                    # served from cache
    assert o1 == o2 and len(cache.store) == 1


# ---------------------------------------------------------------- search
def test_run_graph_search():
    env = _env(communication={"graph_type": "full"})
    evaluate_fn, _cache = make_cached_evaluator(
        env, lambda: make_policy("heuristic_full_comm"), [0], [0.05])
    result = run_graph_search(evaluate_fn, GraphConstraints(), pop_size=6, max_gen=2, seed=0)
    assert len(result.pareto_bits) >= 1
    assert len(result.population_bits) == 6
    assert len(result.history) == 2
    for o in result.pareto_objectives:
        assert len(o) == 4 and all(x == x for x in o)


def test_evaluate_preset_graphs():
    env = _env(communication={"graph_type": "full"})
    evaluate_fn, _c = make_cached_evaluator(
        env, lambda: make_policy("heuristic_full_comm"), [0], [0.05])
    presets = evaluate_preset_graphs(evaluate_fn, extra_random=1)
    for name in ("none", "full", "ring", "star_bs", "random_0"):
        assert name in presets
        assert len(presets[name]["objectives"]) == 4
