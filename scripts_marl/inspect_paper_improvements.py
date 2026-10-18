"""Read-only status and finite-log check; no Torch dependency."""
import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs_marl/paper_v3/screening")
    args = parser.parse_args()
    out = Path(args.out)
    campaign = json.loads((out / "campaign_status.json").read_text())
    protocol = json.loads((out / "protocol_locked.json").read_text())["protocol"]
    totals, nonfinite = 0, 0
    print("Stage:", campaign["stage"], "complete:", campaign["complete"], "failures:", len(campaign["failed"]))
    for path in sorted((out / "train").glob("*/seed_*/status.json")):
        status = json.loads(path.read_text())
        actual = status["actual_transitions"]
        totals += actual
        train = path.parent / "train.csv"
        if train.exists():
            with train.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            nonfinite += sum(not math.isfinite(float(v)) for r in rows for v in r.values())
        print(f"{status['condition']:22} seed {status['training_seed']}: {actual:7}/{protocol['steps_per_run']} "
              f"{status['stage']:22} best_dev={status['best_development_return']}")
    print("Total transitions:", totals, "/", protocol["total_training_transitions"], "nonfinite log cells:", nonfinite)
    print("Final scenarios opened:", campaign["final_scenarios_opened"])
    for failure in campaign["failed"]:
        print("FAILURE", failure)


if __name__ == "__main__":
    main()
