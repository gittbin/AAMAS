"""Episode result type + aggregation + run metadata for baseline evaluation."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch


@dataclass
class EpisodeResult:
    episode_return: float
    min_rate_mean: float
    min_rate_p10: float
    sum_rate_mean: float
    sensing_mean_db: float
    outage_rate: float
    messages_per_step: float
    reconfiguration_per_step: float
    violation_rate: float
    regime: str
    tau: float
    length: int

    def as_dict(self) -> dict:
        return asdict(self)


_NUMERIC_FIELDS = (
    "episode_return", "min_rate_mean", "min_rate_p10", "sum_rate_mean", "sensing_mean_db",
    "outage_rate", "messages_per_step", "reconfiguration_per_step", "violation_rate", "tau",
)


def summarize_episodes(results: list[EpisodeResult]) -> dict:
    """Mean / std / 95% CI (normal approx) / min / max per numeric metric."""
    out: dict = {"n_episodes": len(results)}
    if not results:
        return out
    for name in _NUMERIC_FIELDS:
        vals = np.array([getattr(r, name) for r in results], dtype=np.float64)
        n = len(vals)
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        out[name] = {
            "mean": float(vals.mean()),
            "std": std,
            "ci95": float(ci95),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def collect_metadata(seeds: list[int]) -> dict:
    """Reproducibility metadata: git, versions, device, seed list."""
    cuda = torch.cuda.is_available()
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "platform": platform.platform(),
        "seeds": list(seeds),
    }
