from __future__ import annotations

from typing import Any
import importlib

from .curriculum_phases import get_phase


def _import_reward_module(env: Any):
    """
    Dynamically import the reward module for the current phase.

    Expects: env.cfg.phase_id exists. If not, defaults to 0.
    """
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        raise RuntimeError("Env has no cfg yet; cannot select reward module.")

    module_name = getattr(cfg, "reward_module", None)
    if not module_name:
        raise RuntimeError("cfg.reward_module is missing/empty; check __post_init__ in Go2HybridEnvCfg.")

    # Same package as this file
    pkg = __package__
    module_path = f"{pkg}.{module_name}"

    return importlib.import_module(module_path)


def compute_all_rewards(env: Any, *args, **kwargs):
    """
    Stable entrypoint used by the environment.

    Requirement:
      Each reward file (e.g. rewards_plane_p0.py, rewards_ascent_p1.py, rewards_UD_icra_p2.py)
      must implement: compute_all_rewards(env, *args, **kwargs)
    """
    mod = _import_reward_module(env)

    fn = getattr(mod, "compute_all_rewards", None)
    if fn is None:
        raise AttributeError(f"Reward module '{mod.__name__}' missing compute_all_rewards(...).")
    return fn(env, *args, **kwargs)

