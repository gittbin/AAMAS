"""Atomic JSON with bounded retries for transient Windows reader file locks."""
import json
import time
from scripts_marl.run_paper_improvements import json_config


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(value)
    if "config" in payload: payload["config"]=json_config(payload["config"])
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==39: raise
            time.sleep(min(.01*(attempt+1),.25))
