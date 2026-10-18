"""Normalisation utilities: a fixed random feature projection + a running mean/std.

Both are checkpointable (``state_dict`` / ``load_state_dict``) — the trainer stores
the projection matrix and normalisation statistics so evaluation reproduces training
observations exactly. The projection matrix is drawn once from a fixed seed and is
never redrawn per episode.
"""

from __future__ import annotations

import math

import torch


class FeatureProjection:
    """Fixed (seeded) linear projection of a flattened complex tensor's [re; im]."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        self.matrix = torch.randn(
            self.in_dim, self.out_dim, generator=g, device=device, dtype=dtype
        ) / math.sqrt(self.in_dim)

    def project(self, x_complex: torch.Tensor) -> torch.Tensor:
        """Project a complex tensor: concat(real.flatten, imag.flatten) @ matrix -> (out_dim,)."""
        flat = torch.cat([x_complex.real.reshape(-1), x_complex.imag.reshape(-1)])
        if flat.shape[0] != self.in_dim:
            raise ValueError(f"FeatureProjection expected in_dim={self.in_dim}, got {flat.shape[0]}")
        return flat.to(self.matrix.dtype) @ self.matrix

    def to(self, device: torch.device | str) -> "FeatureProjection":
        self.matrix = self.matrix.to(device)
        return self

    def state_dict(self) -> dict:
        return {"matrix": self.matrix, "in_dim": self.in_dim, "out_dim": self.out_dim}

    def load_state_dict(self, sd: dict) -> None:
        self.matrix = sd["matrix"]
        self.in_dim = int(sd["in_dim"])
        self.out_dim = int(sd["out_dim"])


class RunningMeanStd:
    """Online mean/variance (Chan's parallel algorithm) for observation normalisation."""

    def __init__(self, shape, epsilon: float = 1e-4,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.var = torch.ones(shape, device=device, dtype=dtype)
        self.count = float(epsilon)

    def update(self, x: torch.Tensor) -> None:
        x = x.to(self.mean.dtype)
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / tot)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot)
        self.var = m2 / tot
        self.count = tot

    def normalize(self, x: torch.Tensor, clip: float | None = None) -> torch.Tensor:
        out = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        if clip is not None:
            out = out.clamp(-clip, clip)
        return out

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = sd["mean"]
        self.var = sd["var"]
        self.count = float(sd["count"])
