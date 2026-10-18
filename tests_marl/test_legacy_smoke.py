"""Legacy regression guard for the momgrap physical/optimization backend.

This is the lightweight counterpart of ``smoke_test.py``: it exercises the
single-regime objective path and a *tiny* MO-MFEA run so every core API stays
locked before any MARL code is added. It intentionally does **not** run the full
Stage-C baseline+figure pipeline (too heavy for a unit test) — that stays in
``smoke_test.py``.

Run: ``pytest -q tests_marl/test_legacy_smoke.py``
"""

from __future__ import annotations

import pathlib
import sys

# Make the repository root importable regardless of pytest's import mode / cwd.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from momgrap import smoke_config
from momgrap.channels import sample_environment
from momgrap.mfea import run_mfea
from momgrap.objectives import evaluate_single_regime


def _generator(cfg, seed: int) -> torch.Generator:
    g = torch.Generator(device=cfg.device)
    g.manual_seed(seed)
    return g


def test_single_regime_objective_sanity():
    """evaluate_single_regime on a LoS snapshot: shape/finite/positive/non-flat."""
    cfg = smoke_config()

    g = _generator(cfg, 1)
    pop = torch.rand(cfg.pop_size, cfg.D, generator=g, device=cfg.device, dtype=cfg.dtype_real)
    env = sample_environment(cfg.regimes[0], cfg.mc_samples, g, cfg)  # LoS regime

    F = evaluate_single_regime(pop, env, cfg)

    assert F.shape == (cfg.pop_size, 2), F.shape
    assert torch.isfinite(F).all(), "non-finite objectives"

    f_com, f_sen = F[:, 0], F[:, 1]
    assert float(f_com.min()) > 0.0, "worst-user rate should be positive"

    sen_spread = float(f_sen.max() - f_sen.min())
    assert sen_spread > 1e-3, "sensing objective is flat -> front would collapse"


def test_tiny_mfea_archive_nonempty():
    """A minimal MO-MFEA run still produces a non-empty rank-1 archive."""
    cfg = smoke_config(pop_size=12, max_gen=2, mc_samples=2, log_every=0)

    res = run_mfea(cfg, seed=0)

    assert len(res.archive) > 0, "archive empty after MO-MFEA"

    # Sanity: the per-regime fronts are populated (one entry per regime, >=1 point).
    assert len(res.per_regime_front) == cfg.T
    for _t, (front, _genos) in res.per_regime_front.items():
        assert front.shape[0] >= 1
