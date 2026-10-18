"""Read campaign progress and per-seed development scores without touching final data."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("outputs_marl/paper_completion"))
    args=parser.parse_args(); out=args.out
    state=json.loads((out / "campaign_status.json").read_text(encoding="utf-8"))
    protocol=json.loads((out / "protocol_locked.json").read_text(encoding="utf-8"))["protocol"]
    planned=(len(protocol["conditions"])+int(protocol["topology"]["representative_graph_retraining"]))*len(protocol["training_seeds"])
    statuses=[json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "train").glob("*/seed_*/status.json"))]
    total=sum(s["actual_transitions"] for s in statuses)
    print(f"Stage: {state['stage']} | complete: {state['complete']} | failed jobs: {len(state['failed'])}")
    print(f"Training: {len(state['successful_training'])}/{planned} | transitions: {total:,}/{planned*protocol['steps_per_run']:,}")
    print(f"Final opened: {state['final_scenarios_opened']} | controller PID: {state['pid']}")
    print("condition,seed,transitions,best_development_return,best_step")
    for s in statuses:
        print(f"{s['condition']},{s['training_seed']},{s['actual_transitions']},{s.get('best_development_return')},{s.get('best_step')}")
    if (out / "report/MANUSCRIPT_STATUS.json").exists():
        print("Manuscript:",json.dumps(json.loads((out / "report/MANUSCRIPT_STATUS.json").read_text(encoding="utf-8"))))


if __name__=="__main__": main()
