"""Recover missing final coverage without changing training or locked selections.

All original artifacts are preserved in --origin. The new campaign is a full
independent copy. A post-final numerical correction ledger is locked before its
evaluations. All 30 tau=0 cells use that correction, retaining exact finite legacy
outputs. The other missing FF305 grid cells and all baselines use frozen code.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments_paper_completion.io import write_json
from experiments_paper_completion.engine import load_policy
from experiments_numerical_erratum.power import corrected_rollout, VERSION
from scripts_marl.run_paper_completion import read, ensure_gate, sha, rollout, baseline_worker
from scripts_marl.report_paper_recovery import finish_recovery, DISCLOSURE


def correction_sources():
    names=['experiments_numerical_erratum/__init__.py','experiments_numerical_erratum/power.py',
           'scripts_marl/run_paper_recovery.py','scripts_marl/report_paper_recovery.py',
           'tests/test_numerical_power_erratum.py']
    return {name:sha(ROOT/name) for name in names}


def grid_worker(origin,out,condition,seed,lock):
    torch.set_num_threads(1); manifest=ensure_gate(out,lock)
    ledger=read(out/'numerical_erratum_locked.json')
    if ledger['correction_source_files']!=correction_sources(): raise ValueError('Correction source changed')
    path=out/f'train/{condition}/seed_{seed}/checkpoint_best.pt'
    if sha(path)!=manifest['selected'][f'{condition}/seed_{seed}']['selected_checkpoint_sha256']:
        raise ValueError('Locked checkpoint changed')
    cfg,policy,_=load_policy(path,condition); folder=out/f'evaluation/{condition}/seed_{seed}'
    result=read(folder/'results.json'); p=lock['protocol']
    ids=result['scenario_ids'][:p['exploratory_final_scenarios']]; events=[]
    # Only this exploratory treatment uses the numerical correction.
    episodes=corrected_rollout(cfg,policy,{'tau':0.0},ids,folder/'tau_0.0.csv',p,condition,events=events)
    with (origin/f'evaluation/{condition}/seed_{seed}/tau_0.0.csv').open(newline='') as f: old=list(csv.DictReader(f))
    with (folder/'tau_0.0.csv').open(newline='') as f: new=list(csv.DictReader(f))
    cols=[c for c in old[0] if c!='decision_ms_exploratory']
    if not all(a[c]==b[c] for a,b in zip(old,new) for c in cols):
        raise ValueError('Previously finite original trace changed under erratum')
    if len(old)==len(new) and events: raise ValueError('Successful legacy treatment unexpectedly needed correction')
    result['treatments']['tau_0.0']={'episodes':episodes,'evaluator_numerical_erratum':VERSION}
    if condition=='feedforward' and seed==305:
        settings={**{f'tau_{v}':{'tau':v} for v in p['tau_grid'] if v!=0},
                  **{f'delay_{v}':{'delay':v} for v in p['delay_grid']},
                  **{f'dynamics_{v}':{'dynamics':v} for v in ('static','slow','fast')}}
        for name,treatment in settings.items():
            result['treatments'][name]={'episodes':rollout(cfg,policy,treatment,ids,folder/f'{name}.csv',p,condition)}
    result.update(complete=True,numerical_erratum={'version':VERSION,'scope':'tau_0.0 only',
                  'lock_sha256':sha(out/'numerical_erratum_locked.json')})
    write_json(folder/'results.json',result)
    status=read(out/f'train/{condition}/seed_{seed}/status.json'); status.update(complete=True,stage='complete')
    write_json(out/f'train/{condition}/seed_{seed}/status.json',status)
    evidence=out/f'erratum_evidence/{condition}/seed_{seed}.json'
    evidence.parent.mkdir(parents=True,exist_ok=True)
    write_json(evidence,{'complete':True,'condition':condition,'seed':seed,'scenario_ids':ids,
        'correction_events_count':len(events),'correction_events':events,
        'finite_original_trace_comparison_passed':True,'compared_trace_rows':len(old),
        'compared_columns':cols,'excluded_wall_clock_only':True,
        'original_csv_sha256':sha(origin/f'evaluation/{condition}/seed_{seed}/tau_0.0.csv'),
        'corrected_csv_sha256':sha(folder/'tau_0.0.csv')})
    print(f'{condition}/seed_{seed} completed; corrections={len(events)}; original rows verified={len(old)}',flush=True)


def controller(origin,out):
    lock=read(origin/'execution_protocol.json'); ensure_gate(origin,lock)
    if out.exists(): raise RuntimeError('Preserve every recovery attempt; choose a fresh directory')
    print('Copying immutable original campaign to independent recovery directory',flush=True)
    shutil.copytree(origin,out,ignore=shutil.ignore_patterns('__pycache__','report','PROGRESS.md'))
    (out/'report').mkdir(); (out/'correction_source_snapshot').mkdir()
    ledger={'version':VERSION,'locked_unix':time.time(),'after_final_opening':True,
        'original_training_source_sha256':lock['source_sha256'],
        'original_selection_manifest_sha256':sha(origin/'selected_checkpoints_locked.json'),
        'original_final_opened_unix':read(origin/'campaign_status.json')['final_opened_unix'],
        'scope':'all 30 tau=0.0 exploratory cells; no primary/nominal modification',
        'correction_source_files':correction_sources(),'disclosure':DISCLOSURE,
        'failed_original_cell':{'condition':'feedforward','seed':305,'scenario':12000003,'step':0},
        'rule':'Return legacy tensor unchanged when finite; otherwise normalize existing inverse powers in float64. No new floor/epsilon.',
        'original_artifacts_retained':True,'new_checkpoint_selection':False,'new_training':False}
    for name in ledger['correction_source_files']:
        target=out/'correction_source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,target)
    write_json(out/'numerical_erratum_locked.json',ledger)
    state=read(out/'campaign_status.json'); state.update(pid=os.getpid(),stage='numerical_recovery',complete=False,
        recovery_start_unix=time.time(),recovery_completed_jobs=[],recovery_failures=[],original_failure_preserved=True)
    write_json(out/'campaign_status.json',state)
    try:
        jobs=[(c,s) for c in ('full_nocomm','feedforward','masked_csi') for s in lock['protocol']['training_seeds']]
        with ThreadPoolExecutor(max_workers=10) as pool:
            def launch(c,s):
                log=out/f'recovery_{c}_seed_{s}.log'
                with log.open('w',encoding='utf-8') as handle:
                    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--mode','grid','--origin',str(origin),
                        '--out',str(out),'--condition',c,'--seed',str(s)],stdout=handle,stderr=subprocess.STDOUT,cwd=ROOT)
                return {'condition':c,'seed':s,'returncode':result.returncode,'log':log.name}
            futures=[pool.submit(launch,*job) for job in jobs]
            for future in as_completed(futures):
                job=future.result(); state['recovery_completed_jobs' if job['returncode']==0 else 'recovery_failures'].append(job)
                state['last_update_unix']=time.time(); write_json(out/'campaign_status.json',state); print(job,flush=True)
        if state['recovery_failures']: raise RuntimeError('Recovery failure; all raw artifacts preserved')
        state.update(stage='baselines_and_uncontended_latency'); write_json(out/'campaign_status.json',state)
        print('All grids recovered. Evaluating seven references, then uncontended latency.',flush=True)
        baseline_worker(out,lock)
        state.update(stage='strict_audit_report_manuscript'); write_json(out/'campaign_status.json',state)
        print('Baselines completed; strict audit and manuscript build.',flush=True)
        finish_recovery(out,origin,lock)
        state.update(stage='complete',complete=True,recovery_end_unix=time.time(),full_audit_passed=True)
        write_json(out/'campaign_status.json',state)
        print('Recovery, full audit, report, manuscript and supplement completed.',flush=True)
    except Exception as exc:
        state.update(stage='recovery_failed',complete=True,recovery_error=str(exc)); write_json(out/'campaign_status.json',state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=('controller','grid'),default='controller')
    parser.add_argument('--origin',type=Path,default=ROOT/'outputs_marl/paper_completion')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs_marl/paper_completion_recovery')
    parser.add_argument('--condition'); parser.add_argument('--seed',type=int)
    args=parser.parse_args(); origin=args.origin.resolve(); out=args.out.resolve()
    if args.mode=='grid': grid_worker(origin,out,args.condition,args.seed,read(out/'execution_protocol.json'))
    else: controller(origin,out)
