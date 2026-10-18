"""Checkpoint save / load for the RMAPPO trainer (full reproducible state)."""

from __future__ import annotations

import random

import numpy as np
import torch

from ...config import evocomm_config_to_dict
from ...protocol import EVALUATOR_VERSION, OBSERVATION_SCHEMA_VERSION


def save_checkpoint(path, trainer) -> None:
    ckpt = {
        "evaluator_version": EVALUATOR_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "active_agents": sorted(trainer.active_agents),
        "actors": {a: trainer.actors[a].state_dict() for a in trainer.actors},
        "critic": trainer.critic.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
        "global_step": trainer.global_step,
        "update_index": trainer.update_index,
        "config": evocomm_config_to_dict(trainer.evo),
        "obs_rms": {a: trainer.obs_rms[a].state_dict() for a in trainer.obs_rms},
        "state_rms": trainer.state_rms.state_dict(),
        "projection": trainer.projection.state_dict() if trainer.projection is not None else None,
        "py_rng": random.getstate(),
        "np_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, trainer, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if (ckpt.get("evaluator_version") != EVALUATOR_VERSION or
            ckpt.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION):
        raise ValueError("Legacy checkpoint uses a different evaluator/inbox schema; retrain for paper-v2.")
    trainer.active_agents = set(ckpt["active_agents"])
    for a in trainer.actors:
        trainer.actors[a].load_state_dict(ckpt["actors"][a])
    trainer.critic.load_state_dict(ckpt["critic"])
    trainer.optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scheduler") is not None and trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(ckpt["scheduler"])
    trainer.global_step = ckpt["global_step"]
    trainer.update_index = ckpt["update_index"]
    for a in trainer.obs_rms:
        trainer.obs_rms[a].load_state_dict(ckpt["obs_rms"][a])
    trainer.state_rms.load_state_dict(ckpt["state_rms"])
    if ckpt.get("projection") is not None and trainer.projection is not None:
        trainer.projection.load_state_dict(ckpt["projection"])
    random.setstate(ckpt["py_rng"])
    np.random.set_state(ckpt["np_rng"])
    torch.set_rng_state(_as_uint8(ckpt["torch_rng"]))
    if ckpt.get("torch_cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_uint8(s) for s in ckpt["torch_cuda_rng"]])
    return ckpt


def _as_uint8(t: torch.Tensor) -> torch.Tensor:
    # RNG states must be CPU ByteTensors regardless of checkpoint map_location.
    return t.detach().to("cpu", torch.uint8)
