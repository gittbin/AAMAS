"""Finish all prespecified paper studies with fresh training and an immutable final gate."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from dataclasses import replace
from experiments_paper_completion.engine import (VERSION, source_digest, config_for, env_for,
    CompletionTrainer, CompletionCentralTrainer, save_policy, load_policy)
from scripts_marl.run_paper_improvements import validate_split
from experiments_paper_completion.io import write_json
from scripts_marl.run_paper_messages import history_audit
from scripts_marl.audit_paper_messages import audit_screening
from scripts_marl.run_paper_confirmation import sha, verify_sources
from scripts_marl.evaluate_paper_v2 import (rollout as old_rollout, make_env as old_make_env,
    select_static_learned as old_static, development_features as old_features)
from momgrap.marl.config import evocomm_config_to_dict, save_resolved_config
from momgrap.marl.policies.baselines import NoOpPolicy, RandomPolicy, HeuristicNoCommunicationPolicy
from momgrap.marl.policies.online_optimizer import OnlineOptimizerPolicy
from momgrap.marl.physics.initialization import build_ramp_initializations, save_initialization, load_initialization

DEFAULT_PROTOCOL = ROOT / "configs_marl/paper_completion.json"


def completion_history_audit(protocol):
    from experiments_marl.paper_v3 import training_seed
    count=protocol["steps_per_run"]//800+2
    proposed={training_seed(s,i,e,8) for s in protocol["training_seeds"] for e in range(count) for i in range(8)}
    final=set(range(protocol["reserved_final_seed_start"],protocol["reserved_final_seed_start"]+protocol["reserved_final_scenarios"]))
    paths=list((ROOT / "outputs_marl").rglob("episode_seeds.csv"))
    for path in paths:
        with path.open(newline="") as handle: actual={int(r["scenario_seed"]) for r in csv.DictReader(handle)}
        if actual&(proposed|final): raise ValueError(f"Training/final namespace reused: {path}")
    def visit(value):
        if isinstance(value,dict):
            if value.get("scenario_seed") in final: raise ValueError("Reserved final scenario already evaluated")
            for v in value.values(): visit(v)
        elif isinstance(value,list):
            for v in value: visit(v)
    files=list((ROOT / "outputs_marl").rglob("results.json"))
    for path in files: visit(read(path))
    return {"passed":True,"training_history_files":len(paths),"result_manifests":len(files)}


def check_sources(lock):
    if lock["source_sha256"] != source_digest():
        raise RuntimeError("Completion source changed after lock; preserve this campaign")
    verify_sources(json.loads((ROOT / "outputs_marl/paper_v3/confirmation_v2/protocol_locked.json").read_text()))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def initial_digest(modules):
    digest=hashlib.sha256()
    for name,module in sorted(modules.items()):
        for key,value in sorted(module.state_dict().items()):
            digest.update((name+"/"+key).encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def rollout(cfg, policy, treatment, seeds, trace_path, protocol, condition="full_nocomm", **kwargs):
    graph = kwargs.pop("graph",None)
    def make_env(evo, settings, reference=None, initial=None, ramp=None):
        namespace = dict(old_make_env.__globals__, PROTOCOL=protocol,
                         MOMGRAPParallelEnv=lambda config, **opts: env_for(config,condition,graph=graph,**opts))
        function = types.FunctionType(old_make_env.__code__,namespace,argdefs=old_make_env.__defaults__)
        return function(evo,settings,reference,initial,ramp)
    namespace = dict(old_rollout.__globals__, PROTOCOL=protocol, make_env=make_env)
    function = types.FunctionType(old_rollout.__code__,namespace,argdefs=old_rollout.__defaults__)
    return function(cfg,policy,treatment,seeds,trace_path,**kwargs)


def graph_value(out, condition):
    return read(out / "graph_selection.json")["selected_graph"] if condition == "selected_graph" else None


def train_worker(out, condition, seed, lock):
    torch.set_num_threads(1)
    check_sources(lock)
    protocol = lock["protocol"]
    cfg = config_for(condition,seed,protocol["steps_per_run"])
    graph = graph_value(out,condition)
    if graph is not None:
        cfg = replace(cfg,communication=replace(cfg.communication,graph_type="custom"))
    folder = out / "train" / condition / f"seed_{seed}"
    if folder.exists():
        raise RuntimeError("Never reuse incomplete/completed training folder")
    folder.mkdir(parents=True)
    factory = lambda i: env_for(cfg,condition,graph=graph,eval_mode=i==10000)
    trainer = (CompletionCentralTrainer(cfg,factory,log_dir=folder,seed=seed) if condition == "centralized"
               else CompletionTrainer(cfg,factory,condition,log_dir=folder,seed=seed))
    actors = [trainer.actor] if condition == "centralized" else list(trainer.actors.values())
    status = {"condition":condition,"training_seed":seed,"source_sha256":lock["source_sha256"],
              "policy_parameters":sum(p.numel() for actor in actors for p in actor.parameters()),
              "critic_parameters":sum(p.numel() for p in trainer.critic.parameters()),
              "initial_actor_sha256":initial_digest({"central":trainer.actor} if condition=="centralized" else trainer.actors),
              "initial_critic_sha256":initial_digest({"critic":trainer.critic}),
              "complete":False,"training_complete":False,"stage":"training","pid":os.getpid(),
              "config":evocomm_config_to_dict(cfg),"actual_transitions":0,"best_development_return":None,"graph_integer":graph}
    write_json(folder / "status.json",status)
    save_resolved_config(cfg,folder / "config_resolved.yaml")
    start = time.time()
    next_eval,best = 50176,-float("inf")
    while trainer.global_step < protocol["steps_per_run"]:
        metrics = trainer.collect() if condition == "centralized" else trainer.collect_rollout()
        diagnostics = trainer.update()
        if not all(np.isfinite(v) for v in [*metrics.values(),*diagnostics.values()]):
            raise RuntimeError("Nonfinite training metrics")
        if condition == "centralized":
            with (folder / "train.csv").open("a",newline="",encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if handle.tell()==0:
                    writer.writerow(("global_step","update_index","rollout_return","learning_rate"))
                writer.writerow((trainer.global_step,trainer.update_index,metrics["return"],trainer.optimizer.param_groups[0]["lr"]))
        else:
            trainer._log(metrics,diagnostics)
        status.update(actual_transitions=trainer.global_step,updates=trainer.update_index,training_seconds=time.time()-start)
        write_json(folder / "status.json",status)
        if trainer.global_step >= next_eval or trainer.global_step >= protocol["steps_per_run"]:
            check_sources(lock)
            ev = trainer.evaluate(protocol["development_seeds"])
            if not np.isfinite(ev["eval_return_mean"]):
                raise ValueError("Nonfinite development selection")
            with (folder / "selection.csv").open("a",newline="",encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if handle.tell()==0:
                    writer.writerow(("global_step","eval_return_mean","eval_return_std"))
                writer.writerow((trainer.global_step,ev["eval_return_mean"],ev["eval_return_std"]))
            if ev["eval_return_mean"] > best:
                best = ev["eval_return_mean"]
                save_policy(folder / "checkpoint_best.pt",trainer,condition)
                status.update(best_development_return=best,best_step=trainer.global_step)
            next_eval += 50176
        if trainer.update_index % 25 == 0:
            print(f"{condition}/seed_{seed}: {trainer.global_step}/{protocol['steps_per_run']}",flush=True)
    save_policy(folder / "checkpoint_final.pt",trainer,condition)
    if getattr(trainer,"writer",None):
        trainer.writer.close()
    for env in trainer.envs:
        env.close()
    if trainer._eval_env:
        trainer._eval_env.close()
    status.update(training_complete=True,stage="awaiting_final_gate",training_seconds=time.time()-start,
                  checkpoint_best_sha256=sha(folder / "checkpoint_best.pt"),checkpoint_final_sha256=sha(folder / "checkpoint_final.pt"))
    write_json(folder / "status.json",status)


def graph_worker(out, seed, lock):
    torch.set_num_threads(1)
    check_sources(lock)
    cfg,policy,_ = load_policy(out / "train/full_comm" / f"seed_{seed}/checkpoint_best.pt","full_comm")
    ids = lock["protocol"]["development_seeds"][:lock["protocol"]["topology"]["development_scenarios"]]
    folder = out / "graph_development" / f"seed_{seed}"
    folder.mkdir(parents=True)
    values, raw_episodes = {}, {}
    for graph in range(64):
        episodes = rollout(cfg,policy,{"graph":graph},ids,None,lock["protocol"],"full_comm")
        raw_episodes[str(graph)] = episodes
        values[str(graph)] = {m:float(np.mean([e[m] for e in episodes])) for m in
                             ("net_return","min_user_rate","illumination_db","delivered","outage_fraction")}
    write_json(folder / "results.json",{"complete":True,"scope":"Development topology selection","training_seed":seed,
               "checkpoint_sha256":sha(out / "train/full_comm" / f"seed_{seed}/checkpoint_best.pt"),"scenario_ids":ids,"graphs":values,"episodes_by_graph":raw_episodes})


def select_graphs(out, protocol):
    data = [read(out / "graph_development" / f"seed_{s}/results.json")["graphs"] for s in protocol["training_seeds"]]
    values = {g:{m:float(np.mean([d[str(g)][m] for d in data])) for m in data[0][str(g)]} for g in range(64)}
    feasible = [g for g in values if values[g]["outage_fraction"] <= protocol["topology"]["outage_limit"]]
    def objectives(g):
        v = values[g]
        return np.array([v["min_user_rate"],v["illumination_db"],-v["delivered"]])
    front = [g for g in feasible if not any(np.all(objectives(h)>=objectives(g)) and np.any(objectives(h)>objectives(g)) for h in feasible)]
    if not front:
        raise ValueError("No feasible topology; cannot invent selected graph")
    selected = max(front,key=lambda g:(values[g]["net_return"],-g))
    eligible = [g for g in values if g and values[g]["delivered"]>0]
    harmful = min(eligible,key=lambda g:(values[g]["net_return"],g)) if eligible else 63
    write_json(out / "graph_selection.json",{"selected_graph":selected,"harmful_graph":harmful,"harmful_has_positive_delivery":bool(eligible),
               "pareto_front":front,"development_values":values,"scope":"Conditional frozen-policy topology; not globally optimal",
               "graphs_for_final":{"empty":0,"full":63,"ring":25,"bs_star":23,"selected":selected,"harmful":harmful}})


def static_worker(out, seed, lock):
    torch.set_num_threads(1)
    check_sources(lock)
    cfg,policy,_ = load_policy(out / "train/full_nocomm" / f"seed_{seed}/checkpoint_best.pt","full_nocomm")
    folder = out / "static_development" / f"seed_{seed}"
    folder.mkdir(parents=True)
    namespace = dict(old_static.__globals__,PROTOCOL=lock["protocol"])
    namespace["rollout"] = lambda cfg,policy,treatment,seeds,**kw: rollout(cfg,policy,treatment,seeds,None,lock["protocol"],**kw)
    function = types.FunctionType(old_static.__code__,namespace,argdefs=old_static.__defaults__)
    positions,phases,profile = function(cfg,policy,folder)
    torch.save({"positions":positions,"phases":phases,"profile":profile},folder / "selected.pt")
    write_json(folder / "results.json",{"complete":True,"scope":"Development only static configuration selection",
               "scenario_ids":lock["protocol"]["development_seeds"],"selected_sha256":sha(folder / "selected.pt")})


def archive_worker(out, lock):
    torch.set_num_threads(1)
    check_sources(lock)
    spec = lock["protocol"]["offline_archive"]
    cfg = config_for("full_nocomm",0,lock["protocol"]["steps_per_run"])
    physical = replace(cfg.physical,pop_size=spec["population"],max_gen=spec["generations"],mc_samples=spec["mc_samples"])
    archive = build_ramp_initializations(physical,seed=spec["seed"])
    save_initialization(archive,out / "offline/ramp_initialization")
    namespace = dict(old_features.__globals__, PROTOCOL=lock["protocol"])
    function = types.FunctionType(old_features.__code__,namespace,argdefs=old_features.__defaults__)
    torch.save(function(cfg),out / "offline/development_features.pt")


def lock_final_gate(out, lock):
    check_sources(lock)
    protocol = lock["protocol"]
    protected = set(protocol["development_seeds"]+protocol["assessment_seeds"])
    final = set(range(protocol["reserved_final_seed_start"],protocol["reserved_final_seed_start"]+protocol["reserved_final_scenarios"]))
    protected |= final
    # Check all historical recorded IDs again, now including other completed work.
    def visit(value):
        if isinstance(value,dict):
            if value.get("scenario_seed") in final:
                raise ValueError("Final scenario was opened before final gate")
            for item in value.values(): visit(item)
        elif isinstance(value,list):
            for item in value: visit(item)
    for path in (ROOT / "outputs_marl").rglob("results.json"):
        visit(read(path))
    selected,ids_by_seed = {},{}
    for condition in protocol["conditions"]:
        for seed in protocol["training_seeds"]:
            folder = out / "train" / condition / f"seed_{seed}"
            status = read(folder / "status.json")
            if not status["training_complete"] or status["actual_transitions"] != protocol["steps_per_run"] or status["source_sha256"] != lock["source_sha256"]:
                raise ValueError("Training incomplete/incompatible at final gate")
            with (folder / "episode_seeds.csv").open(newline="") as handle:
                ids = [int(r["scenario_seed"]) for r in csv.DictReader(handle)]
            if len(set(ids))!=len(ids) or set(ids)&protected or (seed in ids_by_seed and ids_by_seed[seed]!=set(ids)):
                raise ValueError("Actual training seed disjointness/matching failure")
            ids_by_seed[seed]=set(ids)
            selected[f"{condition}/seed_{seed}"]={"selected_checkpoint_sha256":sha(folder / "checkpoint_best.pt"),
               "final_checkpoint_sha256":sha(folder / "checkpoint_final.pt"),"selected_step":status["best_step"]}
            if selected[f"{condition}/seed_{seed}"]["selected_checkpoint_sha256"]!=status["checkpoint_best_sha256"]:
                raise ValueError("Selected checkpoint changed")
            selected[f"{condition}/seed_{seed}"]["selection_log_sha256"]=sha(folder / "selection.csv")
    values = list(ids_by_seed.values())
    if any(values[i]&values[j] for i in range(len(values)) for j in range(i)):
        raise ValueError("Independent training blocks overlap")
    for seed in protocol["training_seeds"]:
        for capacity in (1,3):
            matched=[read(out / "train" / c / f"seed_{seed}/status.json") for c in protocol["conditions"]
                     if c not in ("centralized","feedforward") and (3 if c.startswith("budget3") else 1)==capacity]
            if matched and (len({d["initial_actor_sha256"] for d in matched})!=1 or len({d["initial_critic_sha256"] for d in matched})!=1):
                raise ValueError("Same-architecture initialization did not match")
    artifacts = {"graph_selection.json":sha(out / "graph_selection.json"),"offline/ramp_initialization.pt":sha(out / "offline/ramp_initialization.pt"),
                 "protocol_locked.json":sha(out / "protocol_locked.json"),"execution_protocol.json":sha(out / "execution_protocol.json"),
                 "effective_conditions.json":sha(out / "effective_conditions.json")}
    artifacts["offline/development_features.pt"]=sha(out / "offline/development_features.pt")
    for seed in protocol["training_seeds"]:
        name=f"static_development/seed_{seed}/selected.pt"
        artifacts[name]=sha(out / name)
    pairs=("power_ris","power_ma","ris_ma")
    ranking={name:float(np.mean([read(out / "train" / name / f"seed_{s}/status.json")["best_development_return"] for s in protocol["training_seeds"]])) for name in pairs}
    best_pair=max(pairs,key=lambda name:ranking[name])
    write_json(out / "best_pair_selection.json",{"selected_pair":best_pair,"development_returns":ranking,"before_final":True})
    artifacts["best_pair_selection.json"]=sha(out / "best_pair_selection.json")
    manifest={"locked_unix":time.time(),"final_scenarios_opened_at_lock":False,"source_sha256":lock["source_sha256"],
              "selected":selected,"development_artifacts":artifacts,"actual_episode_seed_audit_passed":True}
    path=out / "selected_checkpoints_locked.json"
    if path.exists(): raise RuntimeError("Never overwrite final selections")
    write_json(path,manifest)
    return sha(path)


def ensure_gate(out, lock):
    check_sources(lock)
    state=read(out / "campaign_status.json")
    manifest=read(out / "selected_checkpoints_locked.json")
    if not state["final_scenarios_opened"] or sha(out / "selected_checkpoints_locked.json")!=state["selection_manifest_sha256"] or manifest["locked_unix"]>=state["final_opened_unix"]:
        raise RuntimeError("Final evaluation is gated by immutable complete selections")
    for name,digest in manifest["development_artifacts"].items():
        if sha(out / name)!=digest: raise ValueError("Pre-final artifact changed")
    return manifest


def eval_worker(out,condition,seed,lock):
    torch.set_num_threads(1)
    manifest=ensure_gate(out,lock)
    protocol=lock["protocol"]
    entry=manifest["selected"][f"{condition}/seed_{seed}"]
    path=out / "train" / condition / f"seed_{seed}/checkpoint_best.pt"
    if sha(path)!=entry["selected_checkpoint_sha256"]: raise ValueError("Locked checkpoint hash")
    cfg,policy,_=load_policy(path,condition)
    folder=out / "evaluation" / condition / f"seed_{seed}"
    if folder.exists(): raise RuntimeError("Never overwrite final evaluation")
    folder.mkdir(parents=True)
    ids=list(range(protocol["reserved_final_seed_start"],protocol["reserved_final_seed_start"]+protocol["reserved_final_scenarios"]))
    base={"pricing":"requested"} if condition=="request_price" else {}
    graph=graph_value(out,condition)
    treatments={"normal":base}
    if condition=="full_nocomm":
        treatments.update(frozen25={"freeze_step":25},frozen10={"freeze_step":10})
    if condition=="full_comm" or condition.startswith("candidate_"):
        treatments.update(comm_off={"comm_mode":"none"},content_zero={"corruption":"zero"})
    if condition=="full_comm":
        treatments.update(comm_random={"comm_mode":"random"},comm_forced={"comm_mode":"full"},
                          content_random={"corruption":"random"},content_permute={"corruption":"permute"})
    result={"complete":False,"condition":condition,"training_seed":seed,"checkpoint_lock":entry,"source_sha256":lock["source_sha256"],"scenario_ids":ids,"treatments":{}}
    for name,treatment in treatments.items():
        replay=None
        if name.startswith("content_"):
            with (folder / "normal.csv").open(newline="") as handle: normal=list(csv.DictReader(handle))
            replay={(int(r["scenario_seed"]),int(r["step"])):{a:int(r[f"comm_action_{a}"]) for a in cfg_agents()} for r in normal}
        episodes=rollout(cfg,policy,treatment,ids,folder / f"{name}.csv",protocol,condition,replay=replay,graph=graph)
        if replay is not None:
            from scripts_marl.audit_paper_messages import accounting_matches,rows
            accounting_matches(normal,rows(folder / f"{name}.csv"))
        result["treatments"][name]={"settings":treatment,"episodes":episodes}
        write_json(folder / "results.json",result)
    if condition=="full_nocomm":
        a=result["treatments"]["normal"]["episodes"]; b=result["treatments"]["frozen25"]["episodes"]
        if any(x["prefix25_sha256"]!=y["prefix25_sha256"] for x,y in zip(a,b)): raise ValueError("Freeze prefix mismatch")
        initial=torch.load(out / f"static_development/seed_{seed}/selected.pt",weights_only=False)
        result["treatments"]["static_learned"]={"episodes":rollout(cfg,NoOpPolicy(),{},ids,folder / "static_learned.csv",protocol,initial=(initial["positions"],initial["phases"],initial["profile"]))}
        features=torch.load(out / "offline/development_features.pt",weights_only=False)
        for mode in protocol["csi_feature_interventions"]:
            name="csi_"+mode
            reference={a:x.mean(axis=0) for a,x in features.items()} if mode=="fixed" else features
            result["treatments"][name]={"settings":{"csi_features":mode},"episodes":rollout(cfg,policy,{"csi_features":mode},ids,folder / f"{name}.csv",protocol,reference=reference)}
    if condition in ("full_nocomm","feedforward","masked_csi"):
        exploratory=ids[:protocol["exploratory_final_scenarios"]]
        settings={**{f"tau_{v}":{"tau":v} for v in protocol["tau_grid"]},
                  **{f"delay_{v}":{"delay":v} for v in protocol["delay_grid"]},
                  **{f"dynamics_{v}":{"dynamics":v} for v in ("static","slow","fast")}}
        for name,treatment in settings.items():
            result["treatments"][name]={"episodes":rollout(cfg,policy,treatment,exploratory,folder / f"{name}.csv",protocol,condition)}
            if condition=="full_nocomm" and name.startswith("dynamics_"):
                result["treatments"][name+"_frozen25"]={"episodes":rollout(cfg,policy,dict(treatment,freeze_step=25),exploratory,folder / f"{name}_frozen25.csv",protocol,condition)}
    if condition=="full_comm":
        graphs=read(out / "graph_selection.json")["graphs_for_final"]
        for name,value in graphs.items():
            result["treatments"]["graph_"+name]={"settings":{"graph":value},"episodes":rollout(cfg,policy,{"graph":value},ids,folder / f"graph_{name}.csv",protocol,condition)}
        value=graphs["harmful"]
        with (folder / "graph_harmful.csv").open(newline="") as handle: normal=list(csv.DictReader(handle))
        replay={(int(r["scenario_seed"]),int(r["step"])):{a:int(r[f"comm_action_{a}"]) for a in cfg_agents()} for r in normal}
        for mode in ("zero","random","permute"):
            name="graph_harmful_content_"+mode
            result["treatments"][name]={"episodes":rollout(cfg,policy,{"graph":value,"corruption":mode},ids,folder / f"{name}.csv",protocol,condition,replay=replay)}
            from scripts_marl.audit_paper_messages import accounting_matches,rows
            accounting_matches(normal,rows(folder / f"{name}.csv"))
    result["complete"]=True
    write_json(folder / "results.json",result)
    status=read(out / "train" / condition / f"seed_{seed}/status.json")
    status.update(complete=True,stage="complete")
    write_json(out / "train" / condition / f"seed_{seed}/status.json",status)


def cfg_agents(): return ("bs_agent","ris_agent","ma_agent")


def baseline_worker(out,lock):
    torch.set_num_threads(1)
    ensure_gate(out,lock)
    protocol=lock["protocol"]
    cfg=config_for("full_nocomm",0,protocol["steps_per_run"])
    ids=list(range(protocol["reserved_final_seed_start"],protocol["reserved_final_seed_start"]+protocol["reserved_final_scenarios"]))
    folder=out / "baselines"; folder.mkdir()
    policies={"noop":NoOpPolicy(),"random":RandomPolicy(13000001),"heuristic":HeuristicNoCommunicationPolicy(),"online_optimizer":OnlineOptimizerPolicy(rounds=protocol["online_optimizer_rounds"])}
    result={"complete":False,"references":{}}
    for name,policy in policies.items():
        result["references"][name]={"episodes":rollout(cfg,policy,{},ids,folder / f"{name}.csv",protocol)}
        write_json(folder / "results.json",result)
    archive=load_initialization(out / "offline/ramp_initialization.pt")
    for mode in ("ramp_knee","ramp_comm_extreme","ramp_sensing_extreme"):
        result["references"][mode]={"episodes":rollout(cfg,NoOpPolicy(),{"init_mode":mode},ids,folder / f"{mode}.csv",protocol,ramp=archive)}
    result["complete"]=True
    write_json(folder / "results.json",result)
    from scripts_marl.evaluate_paper_baselines import latency
    measured={"online_optimizer":latency(cfg,policies["online_optimizer"],protocol["latency"]["measurements"],protocol["latency"]["warmup"])}
    for name in ("full_nocomm","full_comm","centralized"):
        cfg,policy,_=load_policy(out / "train" / name / f"seed_{protocol['training_seeds'][0]}/checkpoint_best.pt",name)
        measured[name]=latency(cfg,policy,protocol["latency"]["measurements"],protocol["latency"]["warmup"])
    write_json(folder / "latency.json",measured)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT / "outputs_marl/paper_completion")
    parser.add_argument("--mode",choices=("controller","train","evaluate","graph","static","archive","baselines"),default="controller")
    parser.add_argument("--condition"); parser.add_argument("--seed",type=int)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--smoke-seed",type=int,default=410)
    parser.add_argument("--wait-for-v4",action="store_true")
    args=parser.parse_args(); out=args.out.resolve()
    if args.mode!="controller":
        lock=read(out / ("execution_protocol.json" if args.mode in ("evaluate","baselines") else "protocol_locked.json")); check_sources(lock)
        if args.mode=="train": train_worker(out,args.condition,args.seed,lock)
        elif args.mode=="evaluate": eval_worker(out,args.condition,args.seed,lock)
        elif args.mode=="graph": graph_worker(out,args.seed,lock)
        elif args.mode=="static": static_worker(out,args.seed,lock)
        elif args.mode=="archive": archive_worker(out,lock)
        else: baseline_worker(out,lock)
        return 0
    if out.exists(): raise RuntimeError("Preserve existing campaign; use a fresh output directory")
    out.mkdir(parents=True)
    protocol=read(DEFAULT_PROTOCOL)
    if args.smoke:
        protocol.update(stage="pipeline_smoke",training_seeds=[args.smoke_seed],steps_per_run=2048,
                        development_seeds=protocol["development_seeds"][:2],assessment_seeds=protocol["assessment_seeds"][:2],
                        reserved_final_seed_start=14000000+args.smoke_seed*100,reserved_final_scenarios=2,exploratory_final_scenarios=1)
        protocol["topology"]["development_scenarios"]=1
        protocol["offline_archive"].update(population=8,generations=2,mc_samples=2,seed=15000000+args.smoke_seed)
        protocol["latency"].update(measurements=3,warmup=1)
    else:
        dependency=ROOT / "outputs_marl/paper_v4/screening"
        while not read(dependency / "campaign_status.json")["complete"]:
            if not args.wait_for_v4: raise RuntimeError("V4 screening must finish before protocol selection")
            write_json(out / "campaign_status.json",{"pid":os.getpid(),"stage":"waiting_for_v4_screening","complete":False,"final_scenarios_opened":False})
            time.sleep(15)
        audit=audit_screening(dependency)
        write_json(dependency / "report/audit.json",audit)
        selected=read(dependency / "report/results.json")["descriptive_candidates_for_review"]
        for original,name in (("comm_payload","candidate_payload"),("comm_rotating_router","candidate_router")):
            if original in selected: protocol["conditions"].append(name)
        protocol["v4_selection_evidence_sha256"]=sha(dependency / "report/results.json")
    protocol["total_training_transitions"]=len(protocol["conditions"])*len(protocol["training_seeds"])*protocol["steps_per_run"]
    lock={"protocol":protocol,"source_sha256":source_digest(),"seed_audit":validate_split(protocol),"history_audit":completion_history_audit(protocol)}
    check_sources(lock)
    write_json(out / "protocol_locked.json",lock)
    paths=list((ROOT / "momgrap").rglob("*.py"))+list((ROOT / "experiments_marl").rglob("*.py"))+list((ROOT / "experiments_marl_v4").glob("*.py"))+list((ROOT / "experiments_paper_completion").glob("*.py"))
    paths += [DEFAULT_PROTOCOL,Path(__file__),ROOT / "scripts_marl/report_paper_completion.py",ROOT / "scripts_marl/train_paper_condition.py",ROOT / "scripts_marl/run_paper_improvements.py",ROOT / "scripts_marl/paper_seed_audit.py",ROOT / "scripts_marl/evaluate_paper_v2.py",ROOT / "scripts_marl/evaluate_paper_baselines.py",ROOT / "scripts_marl/report_paper_v2.py",ROOT / "configs_marl/paper_v2.yaml",ROOT / "configs_marl/paper_v3_screening.json",ROOT / "configs_marl/paper_v4_messages.json",ROOT / "scripts_marl/run_paper_messages.py"]
    for path in paths:
        target=out / "source_snapshot" / path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    shutil.copy2(ROOT / "tests_marl/test_paper_completion.py",out / "source_snapshot/test_paper_completion.py")
    from importlib.metadata import distributions
    packages=sorted({d.metadata["Name"]+"=="+d.version for d in distributions() if d.metadata.get("Name")})
    (out / "requirements_resolved.txt").write_text("\n".join(packages)+"\n",encoding="utf-8")
    state={"pid":os.getpid(),"experiment_version":VERSION,"stage":"training","complete":False,"start_unix":time.time(),
           "source_sha256":lock["source_sha256"],"planned_runs":len(protocol["conditions"])*len(protocol["training_seeds"]),
           "failed":[],"successful_training":[],"successful_evaluation":[],"development_jobs":[],"final_scenarios_opened":False,
           "protocol_lock_sha256":sha(out / "protocol_locked.json")}
    def run(mode,condition=None,seed=None):
        log=out / f"{mode}_{condition or 'common'}_seed_{seed if seed is not None else 'all'}.log"
        command=[sys.executable,"-u",str(Path(__file__)),"--mode",mode,"--out",str(out)]
        if condition is not None: command += ["--condition",condition]
        if seed is not None: command += ["--seed",str(seed)]
        temp=ROOT / ".tmp-paper"; temp.mkdir(exist_ok=True)
        env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TEMP=str(temp),TMP=str(temp))
        with log.open("w",encoding="utf-8") as handle:
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0).returncode
        return {"mode":mode,"condition":condition,"seed":seed,"returncode":code,"log":str(log)}
    def batch(stage,jobs,key):
        state["stage"]=stage; write_json(out / "campaign_status.json",state)
        with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
            for future in as_completed([pool.submit(run,*job) for job in jobs]):
                result=future.result(); state["failed" if result["returncode"] else key].append(result)
                state["last_update_unix"]=time.time(); write_json(out / "campaign_status.json",state)
                print(json.dumps(result),flush=True)
        if state["failed"]: raise RuntimeError("Campaign worker failures preserved in logs")
    try:
        jobs=[("train",c,s) for c in protocol["conditions"] for s in protocol["training_seeds"]]
        batch("training",jobs,"successful_training")
        batch("development_topology",[("graph",None,s) for s in protocol["training_seeds"]],"development_jobs")
        select_graphs(out,protocol)
        if protocol["topology"]["representative_graph_retraining"]:
            protocol["conditions"].append("selected_graph")
            protocol["total_training_transitions"]=len(protocol["conditions"])*len(protocol["training_seeds"])*protocol["steps_per_run"]
            # The graph choice is a locked result of the prespecified development
            # procedure, not a source/protocol rewrite after seeing final data.
            state["planned_runs"]+=len(protocol["training_seeds"])
            batch("selected_graph_training",[("train","selected_graph",s) for s in protocol["training_seeds"]],"successful_training")
        batch("development_static_and_archive",[("static",None,s) for s in protocol["training_seeds"]]+[("archive",None,None)],"development_jobs")
        write_json(out / "effective_conditions.json",{"conditions":protocol["conditions"],"selection_rule":"predeclared representative topology retraining","before_final":True})
        # Worker protocol must include selected_graph without changing the initial
        # protocol lock. Final workers receive this explicitly declared extension.
        final_lock=dict(lock,protocol=dict(protocol))
        write_json(out / "execution_protocol.json",final_lock)
        state["selection_manifest_sha256"]=lock_final_gate(out,final_lock)
        state.update(final_scenarios_opened=True,final_opened_unix=time.time())
        write_json(out / "campaign_status.json",state)
        batch("final_evaluation",[("evaluate",c,s) for c in protocol["conditions"] for s in protocol["training_seeds"]],"successful_evaluation")
        batch("final_baselines_and_latency",[("baselines",None,None)],"successful_evaluation")
        check_sources(final_lock)
        from scripts_marl.report_paper_completion import finish
        finish(out,final_lock,write_paper=not args.smoke)
        state.update(complete=True,stage="complete",end_unix=time.time())
        write_json(out / "campaign_status.json",state)
        return 0
    except Exception as error:
        state.update(complete=True,stage="failed",error=str(error),end_unix=time.time())
        write_json(out / "campaign_status.json",state)
        raise


if __name__=="__main__":
    raise SystemExit(main())
