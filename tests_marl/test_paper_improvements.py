"""Behavioral checks for versioned ablations, held-out seed contracts and roundtrips."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch

from experiments_marl.paper_v3 import (ImprovementEnv, ImprovementTrainer, training_seed,
    rms_projection, clip_gradients, save_experiment_checkpoint, load_experiment_policy,
    restore_experiment_checkpoint)
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.normalization import FeatureProjection
from momgrap.marl.algorithms.rmappo.trainer import RMAPPOTrainer
from momgrap.marl.algorithms.rmappo.checkpoint import load_checkpoint, save_checkpoint
from momgrap.marl.config import evocomm_config_to_dict
from scripts_marl.run_paper_improvements import write_json


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny(**sections):
    overrides = {"rmappo": {"total_timesteps": 32, "num_envs": 2, "rollout_length": 8,
                             "sequence_length": 4, "hidden_size": 16, "update_epochs": 1, "num_minibatches": 2}}
    for section, values in sections.items():
        overrides.setdefault(section, {}).update(values)
    return load_evocomm_config("configs_marl/smoke.yaml", overrides=overrides)


def test_actual_resets_unique_across_policies_and_disjoint_from_development(tmp_path):
    cfg = tiny(dynamics={"episode_length": 4})
    all_seeds = []
    for seed in [0, 1]:
        folder = tmp_path / f"seed_{seed}"
        trainer = ImprovementTrainer(cfg, lambda i: ImprovementEnv(cfg), "comm_reference", seed=seed, log_dir=folder)
        trainer.collect_rollout()
        import csv
        with (folder / "episode_seeds.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        ids = [int(r["scenario_seed"]) for r in rows]
        assert len(ids) == 6 and len(set(ids)) == 6
        assert ids == [training_seed(seed, int(r["env_id"]), int(r["episode_index"]), 2) for r in rows]
        assert not set(ids) & set(range(1000000, 1000030))
        assert not set(ids) & set(range(2000000, 2000030))
        assert not set(ids) & set(range(8000000, 8000050))
        all_seeds += ids
        if trainer.writer:
            trainer.writer.close()
    assert len(set(all_seeds)) == len(all_seeds)
    with pytest.raises(ValueError, match="exhausted"):
        training_seed(0, 0, 500000, 2)


def test_rms_projection_is_scale_invariant_and_finite_for_zero():
    projection = FeatureProjection(16, 8)
    channel = torch.randn(2, 4, dtype=torch.complex64) * 1e-10
    torch.testing.assert_close(rms_projection(projection, channel), rms_projection(projection, channel * 10000), rtol=1e-5, atol=1e-6)
    assert torch.equal(rms_projection(projection, torch.zeros_like(channel)), torch.zeros(8))


def test_actor_projection_does_not_change_physics_reward_critic_or_other_features():
    cfg = tiny(csi={"actor_delay_steps": 3})
    raw, improved = ImprovementEnv(cfg), ImprovementEnv(cfg, actor_projection="component_rms")
    a, _ = raw.reset(seed=2000001)
    b, _ = improved.reset(seed=2000001)
    for step in range(6):
        np.testing.assert_array_equal(raw.state(), improved.state())
        start = 4 + cfg.physical.K
        for agent in a:
            np.testing.assert_array_equal(a[agent][:start], b[agent][:start])
            np.testing.assert_array_equal(a[agent][start + 8:], b[agent][start + 8:])
        if step > 3:
            assert improved.current_estimate().step_index < improved.step_count
        expected = rms_projection(improved.projection, improved.current_estimate().reflected[0, 0])
        np.testing.assert_allclose(b["ris_agent"][start:start + 8], expected.numpy(), rtol=1e-6)
        actions = {"bs_agent": np.array([1, 1]), "ris_agent": np.array([1, 1]), "ma_agent": np.array([0, 1])}
        a, r1, _, _, i1 = raw.step(actions)
        b, r2, _, _, i2 = improved.step(actions)
        assert r1 == r2
        np.testing.assert_equal(i1, i2)
    assert not np.allclose(a["ris_agent"][start:start + 8], b["ris_agent"][start:start + 8])


def test_separate_clip_actor_gradient_independent_of_critic_scale():
    def run(critic_scale, mode):
        actor, critic = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
        actor.grad, critic.grad = torch.tensor([3., 4.]), torch.tensor([3., 4.]) * critic_scale
        diagnostics = clip_gradients([actor], [critic], 0.5, mode)
        assert actor.grad.norm() <= 0.500001 and critic.grad.norm() <= 0.500001
        return actor.grad, diagnostics
    a, _ = run(1, "actor_critic_separate")
    b, _ = run(100, "actor_critic_separate")
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    joint, _ = run(100, "joint")
    assert joint.norm() < a.norm() / 50


def test_joint_ablation_update_matches_frozen_trainer():
    cfg = tiny()
    improved = ImprovementTrainer(cfg, lambda i: ImprovementEnv(cfg), "comm_reference", seed=0)
    improved.collect_rollout()
    original = RMAPPOTrainer(cfg, lambda i: ImprovementEnv(cfg), seed=0)
    for a in original.actors:
        original.actors[a].load_state_dict(improved.actors[a].state_dict())
    original.critic.load_state_dict(improved.critic.state_dict())
    original.buffer = copy.deepcopy(improved.buffer)
    reference_stats = original.update()
    new_stats = improved.update()
    for a in original.actors:
        for p, q in zip(original.actors[a].parameters(), improved.actors[a].parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    for p, q in zip(original.critic.parameters(), improved.critic.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    for key in reference_stats:
        assert new_stats[key] == pytest.approx(reference_stats[key], abs=1e-8)


def test_versioned_checkpoint_roundtrip_preserves_actions_and_rejects_wrong_schema(tmp_path):
    cfg = tiny()
    factory = lambda i: ImprovementEnv(cfg, actor_projection="component_rms", eval_mode=i == 10000)
    trainer = ImprovementTrainer(cfg, factory, "comm_rms_csi", seed=0)
    trainer.collect_rollout()
    trainer.update()
    path = tmp_path / "v3.pt"
    save_experiment_checkpoint(path, trainer)
    cfg2, policy, ck = load_experiment_policy(path)
    env = ImprovementEnv(cfg2, actor_projection=ck["experiment_settings"]["actor_projection"], eval_mode=True)
    obs, _ = env.reset(seed=2000000)
    actions = policy.act(obs, env)
    restored = ImprovementTrainer(cfg, factory, "comm_rms_csi", seed=0)
    restore_experiment_checkpoint(path, restored)
    for a in trainer.actors:
        torch.testing.assert_close(trainer.obs_rms[a].mean, restored.obs_rms[a].mean, rtol=0, atol=0)
    _, policy2, _ = load_experiment_policy(path)
    actions2 = policy2.act(obs, env)
    for a in actions:
        np.testing.assert_array_equal(actions[a], actions2[a])
    with pytest.raises(ValueError):
        load_checkpoint(path, restored)
    with pytest.raises(ValueError, match="mismatch"):
        load_experiment_policy(path, expected_condition="comm_reference")
    ck["observation_schema_version"] = 2
    torch.save(ck, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="mismatch"):
        load_experiment_policy(tmp_path / "wrong.pt")
    save_checkpoint(tmp_path / "v2.pt", trainer)
    with pytest.raises(ValueError, match="v3"):
        load_experiment_policy(tmp_path / "v2.pt")


def test_status_los_metadata_roundtrip_and_nonfinite_metric_rejection(tmp_path):
    import json
    cfg = tiny()
    path = tmp_path / "status.json"
    original = {"config": evocomm_config_to_dict(cfg), "return": 1.0}
    write_json(path, original)
    payload = json.loads(path.read_text())
    assert payload["config"]["physical"]["regimes"][0]["k_factor_db"] == "Infinity"
    write_json(path, payload)
    assert json.loads(path.read_text()) == payload
    with pytest.raises(ValueError, match="JSON"):
        write_json(path, {"config": original["config"], "return": float("inf")})
    assert json.loads(path.read_text()) == payload
