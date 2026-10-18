"""Protect final-data gating and independence from earlier selected pilot results."""
import json
from pathlib import Path
import pytest

from scripts_marl import run_paper_confirmation as confirm
from scripts_marl.run_paper_improvements import write_json


def test_final_boundary_rejects_prior_confirmation_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, "ROOT", tmp_path)
    path = tmp_path / "outputs_marl/paper_v3/previous_confirmation/evaluation/comm_reference/seed_10/results.json"
    write_json(path, {"treatments": {"selected": {"episodes": [{"scenario_seed": 8000000}]}}})
    protocol = {"reserved_final_seed_start": 8000000, "reserved_final_scenarios": 50}
    with pytest.raises(ValueError, match="already evaluated"):
        confirm.check_final_boundary(protocol)


def test_fresh_training_rejects_reuse_of_previous_pilot_seed_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, "ROOT", tmp_path)
    path = tmp_path / "outputs_marl/paper_v3/screening/train/comm_reference/seed_0/episode_seeds.csv"
    path.parent.mkdir(parents=True)
    path.write_text("scenario_seed\n100000000\n")
    with pytest.raises(ValueError, match="overlaps previous"):
        confirm.check_training_history({"training_seeds": [0], "steps_per_run": 2048})
    assert confirm.check_training_history({"training_seeds": [10], "steps_per_run": 400384})["passed"]


def test_final_evaluation_cannot_start_before_checkpoint_selection_is_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, "verify_sources", lambda lock: None)
    write_json(tmp_path / "campaign_status.json", {"final_scenarios_opened": False})
    with pytest.raises(RuntimeError, match="pre-final checkpoint lock"):
        confirm.evaluate_worker(tmp_path, "comm_reference", 10, {})
    assert not (tmp_path / "evaluation").exists()


def test_declared_matched_design_has_identical_network_initialization_and_scenarios():
    protocol = json.loads(confirm.DEFAULT_PROTOCOL.read_text())
    assert all(h["paired_training_policies"] for h in protocol["primary_hypotheses"])
    protocol.update(training_seeds=[32], steps_per_run=2048)
    audit = confirm.audit_matched_initialization(protocol)
    assert audit["passed"] and audit["independent_training_pairs"] == 1
    assert len(audit["pairs"]["32"]["initial_scenario_seeds"]) == 8
