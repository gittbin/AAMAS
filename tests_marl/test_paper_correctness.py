"""Independent numerical and information-boundary checks for paper-v2."""
import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from momgrap.comm import bd_precoding, evm_sinr_rate
from momgrap.marl.algorithms.rmappo.centralized import CentralizedPPOTrainer
from momgrap.marl.algorithms.rmappo.checkpoint import load_checkpoint
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.communication.budget import MessageRouter, ReceivedMessage, SendRequest
from momgrap.marl.communication.graph import build_adjacency
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.envs.ramp_parallel_env import MOMGRAPParallelEnv
from momgrap.marl.observations.features import inbox_features
from momgrap.marl.policies.online_optimizer import OnlineOptimizerPolicy
from momgrap.marl.envs.communication_diagnostic import CommunicationDiagnosticEnv
from momgrap.marl.physics.csi import estimate_channel_components
from momgrap.marl.algorithms.rmappo.networks import RecurrentActor


def config(**overrides):
    return load_evocomm_config("configs_marl/smoke.yaml", overrides=overrides)


@pytest.mark.parametrize("uniform", [True, False])
def test_sinr_against_scalar_interference_and_distortion_formula(uniform):
    cfg = config().physical
    h = torch.tensor([[1.0, 0.3, 0.1, 0.0], [0.4, 1.0, 0.0, 0.2]], dtype=torch.complex128)
    w = torch.eye(4, dtype=torch.complex128)[:, :2]
    p = torch.tensor([0.5, 0.5] if uniform else [0.7, 0.3], dtype=torch.float64) * cfg.p_max_watt
    expected = []
    antenna_power = [sum(float(p[j]) * abs(complex(w[m, j])) ** 2 for j in range(2)) for m in range(4)]
    for k in range(2):
        desired = (1 - cfg.evm_kappa ** 2) * float(p[k]) * abs(complex(h[k] @ w[:, k])) ** 2
        interference = sum((1 - cfg.evm_kappa ** 2) * float(p[j]) * abs(complex(h[k] @ w[:, j])) ** 2
                           for j in range(2) if j != k)
        distortion = cfg.evm_kappa ** 2 * sum(abs(complex(h[k, m])) ** 2 * antenna_power[m] for m in range(4))
        expected.append(math.log2(1 + desired / (cfg.noise_power_watt + interference + distortion)))
    result = evm_sinr_rate(h[None, None], w[None, None], cfg, None if uniform else p[None, None])
    torch.testing.assert_close(result[0, 0], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-12)
    assert all(rate < 5 for rate in expected)  # residual interference prevents an unrealistically high rate


def test_perfect_csi_nulling_and_transmit_power_trace():
    cfg = config().physical
    gen = torch.Generator().manual_seed(71)
    h = torch.randn(1, 1, 2, 4, generator=gen, dtype=torch.complex128)
    w = bd_precoding(h, cfg)
    residual = h @ w
    assert residual[..., 0, 1].abs().max() < 1e-12
    assert residual[..., 1, 0].abs().max() < 1e-12
    torch.testing.assert_close(w.norm(dim=-2), torch.ones(1, 1, 2, dtype=torch.float64))
    p = torch.tensor([0.6, 0.4], dtype=torch.float64) * cfg.p_max_watt
    nominal_cov = (w * p.sqrt()) @ (w * p.sqrt()).mH
    intended_trace = (1 - cfg.evm_kappa ** 2) * nominal_cov.diagonal(dim1=-2, dim2=-1).real.sum()
    distortion_trace = cfg.evm_kappa ** 2 * nominal_cov.diagonal(dim1=-2, dim2=-1).real.sum()
    assert float(intended_trace + distortion_trace) == pytest.approx(cfg.p_max_watt)


def test_degenerate_precoder_preserves_unit_columns():
    cfg = config().physical
    w = bd_precoding(torch.zeros(1, 1, 2, 4, dtype=torch.complex64), cfg)
    torch.testing.assert_close(w.norm(dim=-2), torch.ones(1, 1, 2))


def test_component_rms_noise_has_the_documented_aggregate_nmse():
    direct = torch.ones(1, 50000, 1, 1, dtype=torch.complex128)
    reflected = 0.5 * direct
    components = SimpleNamespace(direct=direct, reflected=reflected, aggregate=direct + reflected)
    tau = 0.2
    estimate = estimate_channel_components(components, tau, torch.Generator().manual_seed(83))
    measured = float((estimate.aggregate - components.aggregate).abs().square().mean() /
                     components.aggregate.abs().square().mean())
    expected = (math.sqrt(1 - tau * tau) - 1) ** 2 + tau * tau * 1.25 / 2.25
    assert measured == pytest.approx(expected, rel=0.025)


def test_zero_payload_preserves_sender_and_availability():
    router = MessageRouter(build_adjacency("full"), 1, 1)
    log = router.submit(3, [SendRequest(2, 0, torch.zeros(4))])
    assert log.eligible == log.delivered == 1
    assert router.deliveries_for_step(3) == {}
    messages = router.deliveries_for_step(4)[0]
    assert messages[0].sender == 2
    missing = inbox_features([], 1, 4, "cpu", torch.float32)
    received = inbox_features(messages, 1, 4, "cpu", torch.float32)
    assert received[0] == 1 and received[3] == 1
    assert torch.equal(received[4:], missing[4:])
    assert not torch.equal(received, missing)


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_delay_matches_acquisition_time(delay):
    env = MOMGRAPParallelEnv(config(csi={"actor_delay_steps": delay}), eval_mode=True)
    env.reset(seed=7)
    actions = {a: np.array([0, 0]) for a in env.agents}
    for _ in range(5):
        env.step(actions)
        assert env.current_estimate().step_index == max(0, env.step_count - delay)


def test_corruption_does_not_shift_exogenous_randomness():
    evo = config()
    reference = MOMGRAPParallelEnv(evo, eval_mode=True)
    corrupted = MOMGRAPParallelEnv(evo, eval_mode=True)
    corrupted.set_message_corruption("random")
    reference.reset(seed=9)
    corrupted.reset(seed=9)
    for _ in range(5):
        actions = {a: np.array([0, 1]) for a in reference.agents}
        ref = reference.step(actions)[4]["bs_agent"]
        alt = corrupted.step(actions)[4]["bs_agent"]
        assert ref["min_user_rate"] == alt["min_user_rate"]
        assert ref["message_delivered"] == alt["message_delivered"]
        torch.testing.assert_close(reference.current_estimate().aggregate, corrupted.current_estimate().aggregate)


def test_online_optimizer_cannot_read_hidden_simulator_state():
    env = MOMGRAPParallelEnv(config(), eval_mode=True)
    observations, _ = env.reset(seed=7)
    # Pass only permitted resources; any private-state lookup raises AttributeError.
    safe = SimpleNamespace(cfg=env.cfg, evo=env.evo, device=env.device, step_count=env.step_count,
                           group_map=env.group_map, phase_step=env.phase_step, _half=env._half,
                           agents=env.agents, current_estimate=env.current_estimate,
                           decision_configuration=env.decision_configuration)
    actions = OnlineOptimizerPolicy().act(observations, safe)
    assert set(actions) == set(env.agents)
    assert all(action[1] == 0 for action in actions.values())


def test_centralized_evaluation_has_no_true_state_input():
    evo = config(rmappo={"num_envs": 1, "rollout_length": 4, "sequence_length": 4})
    def factory(index):
        env = MOMGRAPParallelEnv(evo, eval_mode=True)
        if index == 10_000:
            def forbidden():
                raise AssertionError("centralized actor read privileged critic state")
            env.state = forbidden
        return env
    trainer = CentralizedPPOTrainer(evo, factory)
    assert trainer.obs_dim != trainer.state_dim
    assert np.isfinite(trainer.evaluate([71])["eval_return_mean"])
    trainer.collect()
    trainer.update()
    masks = trainer._roll["action_masks"]["ma"][:, 0]
    assert not masks[1:, 1:].any()


def test_frozen_configuration_retains_profile_rule_and_fresh_precoder():
    env = MOMGRAPParallelEnv(config(), eval_mode=True)
    env.reset(seed=13)
    env.step({"bs_agent": np.array([2, 0]), "ris_agent": np.array([1, 0]), "ma_agent": np.array([1, 0])})
    layout, phases = env.decision_configuration()
    env.freeze_configuration()
    for _ in range(3):
        info = env.step({a: np.array([0, 0]) for a in env.agents})[4]["bs_agent"]
        torch.testing.assert_close(env._pstate.positions, layout)
        torch.testing.assert_close(env._pstate.theta, phases)
        assert info["power_action"] == 2


def test_positive_control_bit_is_private_and_zero_payload_can_signal():
    evo = config()
    env = CommunicationDiagnosticEnv(evo)
    before, _ = env.reset(seed=3)
    env.bit = 1
    private_one = env._observations()
    env.bit = 0
    private_zero = env._observations()
    np.testing.assert_array_equal(private_one["ris_agent"], private_zero["ris_agent"])
    assert not np.array_equal(private_one["bs_agent"], private_zero["bs_agent"])
    env.bit = 1
    after, rewards, _, _, _ = env.step({a: np.array([0, 1 if a == "bs_agent" else 0]) for a in env.agents})
    assert all(value == 0 for value in rewards.values())
    assert env.inbox[1][0].sender == 0 and not env.inbox[1][0].payload.any()
    _, rewards, _, _, infos = env.step({"bs_agent": np.array([0, 0]),
                                       "ris_agent": np.array([1, 0]), "ma_agent": np.array([0, 0])})
    assert rewards["ris_agent"] == infos["ris_agent"]["accuracy"] == 1


def test_ma_clock_masks_are_stored_and_reused_in_ppo():
    evo = config(rmappo={"num_envs": 1, "rollout_length": 4, "sequence_length": 4})
    trainer = RMAPPOTrainer(evo, lambda _: MOMGRAPParallelEnv(evo))
    trainer.collect_rollout()
    masks = trainer.buffer.phys_masks["ma_agent"][:, 0]
    assert masks[0].all()
    assert masks[1:, 0].all() and not masks[1:, 1:].any()
    assert not trainer.buffer.phys_actions["ma_agent"][1:].any()
    # Re-evaluation before any parameter update must reproduce stored likelihoods.
    batch = next(trainer.buffer.recurrent_minibatches(1, 4, torch.Generator().manual_seed(0)))
    for agent in trainer.agents:
        lp, _, _ = trainer.actors[agent].evaluate_actions(
            batch["obs"][agent], batch["actor_hidden"][agent], batch["masks"],
            batch["phys_actions"][agent], batch["comm_actions"][agent],
            (batch["phys_masks"][agent], batch["comm_masks"][agent]))
        torch.testing.assert_close(lp, batch["log_probs_old"][agent])
    trainer.update()


def test_legacy_checkpoints_are_rejected_before_loading_weights(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"actors": {}}, path)
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        load_checkpoint(path, SimpleNamespace())


def test_masked_csi_preserves_hardware_clock_and_inbox():
    normal = MOMGRAPParallelEnv(config(), eval_mode=True)
    masked = MOMGRAPParallelEnv(config(csi={"feature_mode": "masked"}), eval_mode=True)
    reference, _ = normal.reset(seed=17)
    altered, _ = masked.reset(seed=17)
    for agent, length in {"bs_agent": 14, "ris_agent": 15, "ma_agent": 14}.items():
        assert not altered[agent][:length].any()
        assert reference[agent][:length].any()
        np.testing.assert_array_equal(altered[agent][length:], reference[agent][length:])


def test_feedforward_actions_do_not_depend_on_hidden_history():
    actor = RecurrentActor(7, 4, hidden_size=16, recurrent=False)
    obs = torch.randn(1, 7)
    masks = torch.ones(1, 1)
    before = actor.act(obs, torch.zeros(1, 16), masks, deterministic=True)
    after = actor.act(obs, torch.randn(1, 16) * 100, masks, deterministic=True)
    torch.testing.assert_close(before[0], after[0])
    torch.testing.assert_close(before[1], after[1])
