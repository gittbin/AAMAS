"""Run MO-MFEA and export RAMP front initializations for the MARL environment.

Run:
  python scripts_marl/export_ramp_initializations.py --config configs_marl/smoke.yaml \
      --seed 0 --name smoke_seed0

Outputs (never touches results/ or figures/):
  outputs_marl/initializations/<name>.pt
  outputs_marl/initializations/<name>.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from momgrap.marl.config import load_evocomm_config
from momgrap.marl.physics.initialization import build_ramp_initializations, save_initialization


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO_ROOT / "configs_marl" / "smoke.yaml"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--n-spaced", type=int, default=5)
    ap.add_argument("--s-eval", type=int, default=None)
    ap.add_argument("--outdir", default=str(_REPO_ROOT / "outputs_marl" / "initializations"))
    args = ap.parse_args()

    evo = load_evocomm_config(args.config)
    print(f"running MO-MFEA (M={evo.physical.M} L={evo.physical.L} seed={args.seed}) ...")
    init = build_ramp_initializations(evo.physical, seed=args.seed, s_eval=args.s_eval,
                                      n_spaced=args.n_spaced)

    name = args.name or f"{pathlib.Path(args.config).stem}_seed{args.seed}"
    pt_path, json_path = save_initialization(init, pathlib.Path(args.outdir) / name)

    print(f"front points: {init.n_points}")
    for key, idx in init.selected_indices.items():
        com, sen = init.front[idx].tolist()
        print(f"  {key:16s} idx={idx:3d}  F_com={com:.3f}  F_sen={sen:.2f} dB")
    print(f"saved: {pt_path}\n       {json_path}")


if __name__ == "__main__":
    main()
