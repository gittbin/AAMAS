"""Protocol checks: shared prefixes and actual fixed-delivery interventions."""
import csv
import numpy as np
from scripts_marl.evaluate_paper_v2 import rollout
from momgrap.marl.config import load_evocomm_config
from momgrap.marl.policies.baselines import HeuristicNoCommunicationPolicy, HeuristicFullCommunicationPolicy
from scripts_marl.report_paper_v2 import nested_interval, exact_test, holm


def config():
    return load_evocomm_config("configs_marl/smoke.yaml", overrides={"dynamics": {"episode_length": 32}})


def test_frozen_rollout_has_exact_shared_prefix_and_post_window():
    cfg = config()
    adaptive = rollout(cfg, HeuristicNoCommunicationPolicy(), {}, [40000])
    frozen = rollout(cfg, HeuristicNoCommunicationPolicy(), {"freeze_step": 25}, [40000])
    assert adaptive[0]["prefix25_sha256"] == frozen[0]["prefix25_sha256"]
    assert frozen[0]["post25_reconfiguration"] == 0
    assert np.isfinite(adaptive[0]["post25_min_user_rate"])


def test_content_intervention_preserves_full_delivery_trace(tmp_path):
    cfg = config()
    normal_path, altered_path = tmp_path / "normal.csv", tmp_path / "altered.csv"
    rollout(cfg, HeuristicFullCommunicationPolicy(), {}, [40001], trace_path=normal_path)
    with normal_path.open(newline="") as handle:
        normal = list(csv.DictReader(handle))
    replay = {(int(row["scenario_seed"]), int(row["step"])):
              {a: int(row[f"comm_action_{a}"]) for a in ("bs_agent", "ris_agent", "ma_agent")}
              for row in normal}
    rollout(cfg, HeuristicFullCommunicationPolicy(), {"corruption": "zero"}, [40001], trace_path=altered_path, replay=replay)
    with altered_path.open(newline="") as handle:
        altered = list(csv.DictReader(handle))
    for original, changed in zip(normal, altered):
        assert all(original[key] == changed[key] for key in ("requested", "eligible", "delivered", "budget_dropped"))


def test_policy_level_permutation_and_constant_bootstrap():
    high = np.ones((5, 50))
    low = np.zeros((5, 50))
    result = nested_interval(high, low, paired=True, repetitions=100)
    assert result["mean"] == 1 and result["ci95"] == [1, 1]
    assert exact_test(high, low, paired=True) == 2 / 32
    assert exact_test(high, low, paired=False) == 2 / 252
    assert exact_test(low, low, paired=False) == 1
    assert holm([0.01, 0.04, 0.2]) == [0.03, 0.08, 0.2]
