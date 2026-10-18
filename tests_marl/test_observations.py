"""Tests for the local observation builder + centralized state (Task 09)."""

from __future__ import annotations

import inspect
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from momgrap.channels import bs_cell_centers, sample_environment
from momgrap.genotype import decode
from momgrap.marl.actions.codebooks import build_ris_group_map
from momgrap.marl.communication.graph import AGENT_NAMES
from momgrap.marl.communication.budget import ReceivedMessage
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.normalization import FeatureProjection, RunningMeanStd
from momgrap.marl.observations.builder import (
    COMPLEX_PROJ_DIM,
    build_actor_observations,
    build_centralized_state,
    centralized_state_size,
    observation_sizes,
)
from momgrap.marl.physics.components import build_comm_channel_components
from momgrap.marl.physics.csi import estimate_channel_components

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _setup(seed: int = 0, tau: float = 0.1):
    evo = load_evocomm_config(SMOKE)
    cfg = evo.physical
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    geno = torch.rand(1, cfg.D, generator=g, device="cpu", dtype=cfg.dtype_real)
    P, theta = decode(geno, cfg)
    env = sample_environment(cfg.regimes[0], 1, g, cfg)
    true = build_comm_channel_components(P, theta, env, cfg)
    est = estimate_channel_components(true, tau, g)
    centers = bs_cell_centers(cfg).to(dtype=cfg.dtype_real)
    gmap = build_ris_group_map(cfg, evo.actions)
    proj = FeatureProjection(2 * cfg.K * cfg.M, COMPLEX_PROJ_DIM, seed=0)
    return dict(evo=evo, cfg=cfg, P=P, theta=theta, env=env, true=true, est=est,
               centers=centers, gmap=gmap, proj=proj, tau=tau)


def _actor(s, delivered=None, prev=None, step=0):
    return build_actor_observations(
        s["est"], s["theta"], s["P"], s["centers"], s["gmap"],
        prev or {}, delivered or {}, s["tau"], step, s["evo"], projection=s["proj"],
    )


# ---------------------------------------------------------------- sizes/shape
def test_actor_shapes_match_declared_sizes():
    s = _setup()
    sizes = observation_sizes(s["evo"], proj_dim=COMPLEX_PROJ_DIM)
    obs = _actor(s)
    assert set(obs.keys()) == set(AGENT_NAMES)
    for name in AGENT_NAMES:
        assert obs[name].shape == (sizes[name],), (name, obs[name].shape, sizes[name])
        assert torch.isfinite(obs[name]).all()


def test_critic_shape_matches_declared_size():
    s = _setup()
    state = build_centralized_state(
        s["true"], s["P"], s["theta"], s["centers"], s["gmap"], {}, s["tau"], 0, 0,
        s["evo"], projection=s["proj"],
    )
    assert state.shape == (centralized_state_size(s["evo"], COMPLEX_PROJ_DIM),)
    assert torch.isfinite(state).all()


def test_sizes_without_projection():
    s = _setup()
    sizes = observation_sizes(s["evo"], proj_dim=0)
    obs = build_actor_observations(
        s["est"], s["theta"], s["P"], s["centers"], s["gmap"], {}, {}, s["tau"], 0,
        s["evo"], projection=None,
    )
    for name in AGENT_NAMES:
        assert obs[name].shape == (sizes[name],)


# ---------------------------------------------------------------- no leak
def test_actor_builder_has_no_true_channel_param():
    params = set(inspect.signature(build_actor_observations).parameters)
    assert not any("true" in p for p in params)
    assert "delayed_estimated" in params


def test_changing_truth_does_not_change_actor_but_changes_critic():
    s = _setup()
    obs1 = _actor(s)

    # a second, different true channel (different genotype) with the SAME est
    g2 = torch.Generator(device="cpu")
    g2.manual_seed(999)
    geno2 = torch.rand(1, s["cfg"].D, generator=g2, device="cpu", dtype=s["cfg"].dtype_real)
    P2, theta2 = decode(geno2, s["cfg"])
    true2 = build_comm_channel_components(P2, theta2, s["env"], s["cfg"])

    # actor is rebuilt with the SAME est -> identical (it never sees truth)
    obs2 = _actor(s)
    for name in AGENT_NAMES:
        assert torch.equal(obs1[name], obs2[name])

    # critic with two different truths -> differs (critic may see truth)
    c1 = build_centralized_state(s["true"], s["P"], s["theta"], s["centers"], s["gmap"],
                                 {}, s["tau"], 0, 0, s["evo"], projection=s["proj"])
    c2 = build_centralized_state(true2, P2, theta2, s["centers"], s["gmap"],
                                 {}, s["tau"], 0, 0, s["evo"], projection=s["proj"])
    assert not torch.equal(c1, c2)


# ---------------------------------------------------------------- delayed CSI
def test_delayed_csi_respected():
    s = _setup(tau=0.2)
    obs_old = _actor(s)
    # a *different* (newer) estimate -> observation must change (uses the passed est)
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    est_new = estimate_channel_components(s["true"], 0.2, g)
    obs_new = build_actor_observations(
        est_new, s["theta"], s["P"], s["centers"], s["gmap"], {}, {}, s["tau"], 0,
        s["evo"], projection=s["proj"],
    )
    assert not torch.equal(obs_old["bs_agent"], obs_new["bs_agent"])


# ---------------------------------------------------------------- messages
def test_message_reaches_only_intended_recipient():
    s = _setup()
    mdim = s["evo"].communication.message_dim
    obs_none = _actor(s, delivered={})
    obs_ris = _actor(s, delivered={1: [ReceivedMessage(0, torch.ones(mdim))]})

    assert torch.equal(obs_none["bs_agent"], obs_ris["bs_agent"])   # bs unaffected
    assert torch.equal(obs_none["ma_agent"], obs_ris["ma_agent"])   # ma unaffected
    assert not torch.equal(obs_none["ris_agent"], obs_ris["ris_agent"])  # ris got it


def test_previous_action_changes_own_obs_only():
    s = _setup()
    obs0 = _actor(s, prev={})
    obs1 = _actor(s, prev={"bs_agent": 3})
    assert not torch.equal(obs0["bs_agent"], obs1["bs_agent"])
    assert torch.equal(obs0["ris_agent"], obs1["ris_agent"])
    assert torch.equal(obs0["ma_agent"], obs1["ma_agent"])


def test_step_time_advances():
    s = _setup()
    obs_t0 = _actor(s, step=0)
    obs_t5 = _actor(s, step=5)
    assert not torch.equal(obs_t0["bs_agent"], obs_t5["bs_agent"])


# ---------------------------------------------------------------- normalization
def test_feature_projection_deterministic_and_stable():
    p1 = FeatureProjection(16, 8, seed=0)
    p2 = FeatureProjection(16, 8, seed=0)
    assert torch.equal(p1.matrix, p2.matrix)
    x = torch.complex(torch.randn(2, 4), torch.randn(2, 4))
    a = p1.project(x)
    b = p1.project(x)
    assert a.shape == (8,) and torch.equal(a, b)


def test_running_mean_std():
    rms = RunningMeanStd(3)
    g = torch.Generator().manual_seed(0)
    data = torch.randn(5000, 3, generator=g) * torch.tensor([2.0, 5.0, 0.5]) + torch.tensor([1.0, -3.0, 10.0])
    rms.update(data)
    assert torch.allclose(rms.mean, data.mean(0), atol=0.05)
    assert torch.allclose(rms.var, data.var(0, unbiased=False), rtol=0.05)
    norm = rms.normalize(data)
    assert torch.allclose(norm.mean(0), torch.zeros(3), atol=0.05)
    assert torch.allclose(norm.std(0, unbiased=False), torch.ones(3), atol=0.05)
