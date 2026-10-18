"""Tests for recurrent MAPPO networks (Task 13)."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

from momgrap.marl.algorithms.rmappo.networks import build_actors, build_critic
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.observations.builder import centralized_state_size, observation_sizes

SMOKE = _REPO_ROOT / "configs_marl" / "smoke.yaml"


def _setup(hidden=16):
    evo = load_evocomm_config(SMOKE)
    obs_sizes = observation_sizes(evo)
    state_dim = centralized_state_size(evo)
    actors = build_actors(evo, obs_sizes, hidden_size=hidden)
    critic = build_critic(evo, state_dim, hidden_size=hidden)
    return evo, obs_sizes, state_dim, actors, critic


def _masks(B, reset=False):
    return torch.zeros(B, 1) if reset else torch.ones(B, 1)


# ---------------------------------------------------------------- shapes
def test_actor_forward_shapes():
    evo, obs_sizes, _sd, actors, _c = _setup()
    B = 5
    for name, actor in actors.items():
        obs = torch.randn(B, obs_sizes[name])
        h = actor.zero_hidden(B)
        phys_logits, comm_logits, h2 = actor.forward(obs, h, _masks(B))
        assert phys_logits.shape == (B, actor.n_phys_actions)
        assert comm_logits.shape == (B, 3)
        assert h2.shape == (B, actor.hidden_size)


def test_act_shapes_and_ranges():
    evo, obs_sizes, _sd, actors, _c = _setup()
    B = 8
    for name, actor in actors.items():
        obs = torch.randn(B, obs_sizes[name])
        actions, logp, h = actor.act(obs, actor.zero_hidden(B), _masks(B))
        assert actions.shape == (B, 2)
        assert logp.shape == (B,)
        assert torch.all(actions[:, 0] >= 0) and torch.all(actions[:, 0] < actor.n_phys_actions)
        assert torch.all(actions[:, 1] >= 0) and torch.all(actions[:, 1] < 3)
        assert torch.isfinite(logp).all()


def test_deterministic_action_is_argmax():
    _evo, obs_sizes, _sd, actors, _c = _setup()
    actor = actors["bs_agent"]
    obs = torch.randn(4, obs_sizes["bs_agent"])
    h = actor.zero_hidden(4)
    phys_logits, comm_logits, _ = actor.forward(obs, h, _masks(4))
    actions, _lp, _h = actor.act(obs, h, _masks(4), deterministic=True)
    assert torch.equal(actions[:, 0], phys_logits.argmax(-1))
    assert torch.equal(actions[:, 1], comm_logits.argmax(-1))


# ---------------------------------------------------------------- masks
def test_comm_action_mask_and_no_send_valid():
    _evo, obs_sizes, _sd, actors, _c = _setup()
    actor = actors["ris_agent"]
    B = 32
    obs = torch.randn(B, obs_sizes["ris_agent"])
    h = actor.zero_hidden(B)
    comm_mask = torch.tensor([[True, False, False]]).expand(B, 3).contiguous()  # only no-send
    actions, logp, _ = actor.act(obs, h, _masks(B), action_masks=(None, comm_mask))
    assert torch.all(actions[:, 1] == 0)          # forced to no-send
    assert torch.isfinite(logp).all()


def test_phys_action_mask():
    _evo, obs_sizes, _sd, actors, _c = _setup()
    actor = actors["bs_agent"]
    n = actor.n_phys_actions
    B = 32
    obs = torch.randn(B, obs_sizes["bs_agent"])
    phys_mask = torch.zeros(B, n, dtype=torch.bool)
    phys_mask[:, 0] = True                          # only action 0 allowed
    actions, _lp, _ = actor.act(obs, actor.zero_hidden(B), _masks(B), action_masks=(phys_mask, None))
    assert torch.all(actions[:, 0] == 0)


# ---------------------------------------------------------------- hidden reset
def test_hidden_reset_mask():
    _evo, obs_sizes, _sd, actors, _c = _setup()
    actor = actors["ma_agent"]
    obs = torch.randn(3, obs_sizes["ma_agent"])
    h_nonzero = torch.randn(3, actor.hidden_size)
    # with reset mask (0), the stale hidden is cleared -> same as starting from zeros
    _pl1, _cl1, h_reset = actor.forward(obs, h_nonzero, _masks(3, reset=True))
    _pl2, _cl2, h_zero = actor.forward(obs, actor.zero_hidden(3), _masks(3, reset=True))
    assert torch.allclose(h_reset, h_zero, atol=1e-6)


# ---------------------------------------------------------------- evaluate/seq
def test_evaluate_actions_sequence():
    _evo, obs_sizes, _sd, actors, _c = _setup()
    actor = actors["bs_agent"]
    T, N = 4, 2
    obs = torch.randn(T * N, obs_sizes["bs_agent"])
    hidden = actor.zero_hidden(N)
    masks = torch.ones(T * N, 1)
    phys_a = torch.randint(0, actor.n_phys_actions, (T * N,))
    comm_a = torch.randint(0, 3, (T * N,))
    logp, ent, h = actor.evaluate_actions(obs, hidden, masks, phys_a, comm_a)
    assert logp.shape == (T * N,) and ent.shape == (T * N,)
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all()
    assert h.shape == (N, actor.hidden_size)


# ---------------------------------------------------------------- critic
def test_critic_forward():
    _evo, _os, state_dim, _actors, critic = _setup()
    B = 6
    state = torch.randn(B, state_dim)
    value, h = critic.forward(state, critic.zero_hidden(B), _masks(B))
    assert value.shape == (B,)
    assert h.shape == (B, critic.hidden_size)
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------- backward
def test_backward_pass_finite():
    _evo, obs_sizes, state_dim, actors, critic = _setup()
    B = 5
    loss = 0.0
    for name, actor in actors.items():
        obs = torch.randn(B, obs_sizes[name])
        actions, logp, _ = actor.act(obs, actor.zero_hidden(B), _masks(B))
        _lp, ent, _ = actor.evaluate_actions(
            obs, actor.zero_hidden(B), _masks(B), actions[:, 0], actions[:, 1]
        )
        loss = loss + (-_lp.mean() - 0.01 * ent.mean())
    state = torch.randn(B, state_dim)
    value, _ = critic.forward(state, critic.zero_hidden(B), _masks(B))
    loss = loss + (value - 1.0).pow(2).mean()
    loss.backward()
    for actor in actors.values():
        for p in actor.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
    for p in critic.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# ---------------------------------------------------------------- cuda (opt)
@pytest.mark.cuda
def test_cuda_forward():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    _evo, obs_sizes, _sd, actors, _c = _setup()
    actor = actors["bs_agent"].cuda()
    obs = torch.randn(4, obs_sizes["bs_agent"], device="cuda")
    actions, logp, h = actor.act(obs, actor.zero_hidden(4, device="cuda"), torch.ones(4, 1, device="cuda"))
    assert actions.is_cuda and torch.isfinite(logp).all()
