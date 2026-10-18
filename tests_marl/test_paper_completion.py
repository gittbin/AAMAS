"""Protect new final-data gates, central seed independence and checkpoint contracts."""
import csv
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from experiments_paper_completion.engine import (config_for, env_for, CompletionTrainer,
    CompletionCentralTrainer, save_policy, load_policy)
from experiments_marl.paper_v3 import training_seed
from scripts_marl import run_paper_completion as run
from scripts_marl.run_paper_improvements import write_json


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def tiny(condition):
    cfg=config_for(condition,490,16)
    return replace(cfg,rmappo=replace(cfg.rmappo,num_envs=2,rollout_length=8,hidden_size=16,
                   centralized_actor_hidden_size=16,sequence_length=4,update_epochs=1,num_minibatches=2),
                   dynamics=replace(cfg.dynamics,episode_length=8))


def test_central_and_decentral_actual_resets_match_without_reuse(tmp_path):
    cfg=tiny("centralized")
    central=CompletionCentralTrainer(cfg,lambda i:env_for(cfg,"centralized"),log_dir=tmp_path / "central",seed=490)
    cfg2=tiny("full_nocomm")
    other=CompletionTrainer(cfg2,lambda i:env_for(cfg2,"full_nocomm"),"full_nocomm",log_dir=tmp_path / "decentral",seed=490)
    for _ in range(2):
        central.collect(); other.collect_rollout()
    logs=[]
    for name in ("central","decentral"):
        with (tmp_path / name / "episode_seeds.csv").open(newline="") as f: entries=list(csv.DictReader(f))
        actual=[int(r["scenario_seed"]) for r in entries]
        assert len(actual)==len(set(actual))==6
        assert all(int(r["scenario_seed"])==training_seed(490,int(r["env_id"]),int(r["episode_index"]),2) for r in entries)
        logs.append(actual)
    assert logs[0]==logs[1]


def test_matched_condition_initialization_and_subset_execution():
    actors=[]; critics=[]
    for condition in ("full_nocomm","full_comm","power","ris_ma","price_free","request_price"):
        cfg=tiny(condition)
        trainer=CompletionTrainer(cfg,lambda i:env_for(cfg,condition),condition,seed=490)
        actors.append(run.initial_digest(trainer.actors)); critics.append(run.initial_digest({"critic":trainer.critic}))
        if condition=="power": assert trainer.active_agents=={"bs_agent"}
    assert len(set(actors))==len(set(critics))==1


def test_candidate_checkpoint_rejects_wrong_router_and_condition(tmp_path):
    cfg=tiny("candidate_payload")
    trainer=CompletionTrainer(cfg,lambda i:env_for(cfg,"candidate_payload"),"candidate_payload",seed=490)
    path=tmp_path / "checkpoint.pt"; save_policy(path,trainer,"candidate_payload")
    restored,policy,ck=load_policy(path,"candidate_payload")
    assert ck["observation_schema_version"]==4
    with pytest.raises(ValueError,match="compatible"): load_policy(path,"full_comm")
    ck["experiment_settings"]["router"]="rotating_ties"; torch.save(ck,path)
    with pytest.raises(ValueError,match="contract"): load_policy(path,"candidate_payload")


def test_budget_three_changes_inbox_capacity_but_prices_share_initialization():
    fingerprints=[]; counts=[]
    for condition in ("full_comm","budget3_default","budget3_free","budget3_tenfold"):
        cfg=tiny(condition)
        trainer=CompletionTrainer(cfg,lambda i:env_for(cfg,condition),condition,seed=490)
        fingerprints.append(run.initial_digest(trainer.actors))
        counts.append(sum(p.numel() for a in trainer.actors.values() for p in a.parameters()))
    assert len(set(fingerprints[1:]))==1 and fingerprints[0]!=fingerprints[1]
    assert counts[0]<counts[1] and len(set(counts[1:]))==1


def test_future_final_is_rejected_if_it_was_already_evaluated(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"ROOT",tmp_path)
    write_json(tmp_path / "outputs_marl/previous/results.json",{"nested":[{"scenario_seed":12000000}]})
    with pytest.raises(ValueError,match="already evaluated"):
        run.completion_history_audit({"training_seeds":[300],"steps_per_run":2048,"reserved_final_seed_start":12000000,"reserved_final_scenarios":150})


def test_final_cannot_open_without_immutable_protocol_and_selections(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"check_sources",lambda lock:None)
    write_json(tmp_path / "campaign_status.json",{"final_scenarios_opened":False})
    write_json(tmp_path / "selected_checkpoints_locked.json",{})
    with pytest.raises(RuntimeError,match="gated"): run.ensure_gate(tmp_path,{})
    write_json(tmp_path / "execution_protocol.json",{"original":True})
    manifest={"locked_unix":1,"development_artifacts":{"execution_protocol.json":run.sha(tmp_path / "execution_protocol.json")}}
    write_json(tmp_path / "selected_checkpoints_locked.json",manifest)
    write_json(tmp_path / "campaign_status.json",{"final_scenarios_opened":True,"final_opened_unix":2,"selection_manifest_sha256":run.sha(tmp_path / "selected_checkpoints_locked.json")})
    run.ensure_gate(tmp_path,{})
    write_json(tmp_path / "execution_protocol.json",{"original":False})
    with pytest.raises(ValueError,match="changed"): run.ensure_gate(tmp_path,{})


def test_topology_front_and_harmful_selection_are_development_only(tmp_path):
    values={str(g):{"min_user_rate":float(g),"illumination_db":float(g),"delivered":float(g>0),"outage_fraction":0.1,"net_return":float(g)} for g in range(64)}
    write_json(tmp_path / "graph_development/seed_300/results.json",{"graphs":values})
    run.select_graphs(tmp_path,{"training_seeds":[300],"topology":{"outage_limit":0.5}})
    selected=json.loads((tmp_path / "graph_selection.json").read_text())
    assert selected["pareto_front"]==[0,63]
    assert selected["selected_graph"]==63 and selected["harmful_graph"]==1
    assert not (tmp_path / "evaluation").exists()


def test_report_uses_policy_blocks_and_holm_family():
    from scripts_marl.report_paper_completion import matrix,exact_test,holm
    data={"x/normal":[(1,[{"scenario_seed":4,"rate":30.0},{"scenario_seed":5,"rate":-10.0}]),(0,[{"scenario_seed":4,"rate":0.0},{"scenario_seed":5,"rate":20.0}])]}
    value=matrix(data,"x/normal","rate")
    np.testing.assert_array_equal(value,[[0,20],[30,-10]])
    assert exact_test(value,np.zeros_like(value),paired=True)==0.5
    assert holm([.01,.04,.2])==[.03,.08,.2]
