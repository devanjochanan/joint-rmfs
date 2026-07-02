"""
PPO training script for PPS (Pick Pod Selection) in RMFS.

SB3 controls the full training loop through a single DummyVecEnv.
n_steps is set large enough to contain one full episode, so PPO
updates happen approximately once per episode.

Uses Stable-Baselines3 PPO with:
  - Custom Gymnasium env (PPSEnv)
  - TensorBoard logging (pps_reward/ and pps_metrics/ sections)
  - CSV data recording
  - Model save/load + checkpoints

Usage:
    python scripts/training/train_pps_rl.py --episodes 500 --max-ticks 5000
    python scripts/training/train_pps_rl.py --resume
    python scripts/training/train_pps_rl.py --eval
    python -m tensorboard.main --logdir data/models/pps/runs
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)
os.environ.setdefault("RMFS_RUN_PROFILE", "training")
os.environ.setdefault("RMFS_DETAIL_DB", "0")

# Keep Matplotlib/SB3 font-cache writes inside the project folder on Windows.
_MPLCONFIGDIR = _REPO_ROOT / "data" / "models" / "pps" / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import numpy as np
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from src.rmfs.rl.pps.env import PPSEnv
from src.rmfs.decisions.pps import DEFAULT_PPS_MODEL_PATH
from src.rmfs.runtime_io.run_profiles import available_profiles, resolve_run_profile

try:
    from src.rmfs.rl.pps.env import (
        ALPHA_OCT,
        FAST_TRAIN_MODE,
        PICKED_QTY_WEIGHT,
        POD_VISIT_PENALTY,
    )
except ImportError:
    PICKED_QTY_WEIGHT = 0.0
    ALPHA_OCT = 1.0
    POD_VISIT_PENALTY = 0.0
    FAST_TRAIN_MODE = False


_PPS_ENV_INIT_ARGS = set(inspect.signature(PPSEnv.__init__).parameters)


def ensure_numpy_pickle_compat() -> None:
    """Allow NumPy 2.x SB3 archives to load in older NumPy 1.x environments."""
    try:
        import numpy.core as numpy_core
    except ImportError:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for submodule in (
        "multiarray",
        "umath",
        "numeric",
        "fromnumeric",
        "_multiarray_umath",
        "_methods",
        "overrides",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{submodule}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{submodule}", module)


def make_pps_env_kwargs(
    max_ticks: int,
    picked_qty_weight: float,
    reward_alpha: float,
    visit_penalty: float,
    base_seed: int | None = None,
    run_profile: str = "training",
    bootstrap_n_orders: int | None = None,
    demand_horizon_ticks: int | None = None,
    demand_buffer_ticks: int | None = None,
    pod_location_mode: str | None = None,
    pod_location_seed: int | None = None,
    full_raw_order_replay: bool = False,
    runtime_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Build env kwargs compatible with both newer and older PPSEnv versions."""
    kwargs: Dict[str, Any] = {}
    if "max_episode_ticks" in _PPS_ENV_INIT_ARGS:
        kwargs["max_episode_ticks"] = max_ticks
    if "reward_picked_qty_weight" in _PPS_ENV_INIT_ARGS:
        kwargs["reward_picked_qty_weight"] = picked_qty_weight
    if "reward_alpha" in _PPS_ENV_INIT_ARGS:
        kwargs["reward_alpha"] = reward_alpha
    if "reward_visit_penalty" in _PPS_ENV_INIT_ARGS:
        kwargs["reward_visit_penalty"] = visit_penalty
    if base_seed is not None and "base_seed" in _PPS_ENV_INIT_ARGS:
        kwargs["base_seed"] = base_seed
    optional = {
        "run_profile": run_profile,
        "bootstrap_n_orders": bootstrap_n_orders,
        "demand_horizon_ticks": demand_horizon_ticks,
        "demand_buffer_ticks": demand_buffer_ticks,
        "pod_location_mode": pod_location_mode,
        "pod_location_seed": pod_location_seed,
        "full_raw_order_replay": full_raw_order_replay,
        "runtime_root": runtime_root,
    }
    for key, value in optional.items():
        if key in _PPS_ENV_INIT_ARGS and value is not None:
            kwargs[key] = value
    return kwargs


def generate_training_seed() -> int:
    """Generate a lightweight reproducible-run seed when --seed is omitted."""
    return int.from_bytes(os.urandom(4), "little")

# ---------------------------------------------------------------------------
# Paths (defaults, overridable via CLI)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join("data", "models", "pps")
LOG_DIR = os.path.join(MODEL_DIR, "runs")
METRICS_DIR = os.path.join(MODEL_DIR, "metrics")
BEST_MODEL = str(DEFAULT_PPS_MODEL_PATH.with_suffix(""))
BEST_THROUGHPUT_MODEL = os.path.join(MODEL_DIR, "pps_rl_best_throughput")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
TRAIN_STATE_FILE = os.path.join(MODEL_DIR, "train_state.json")
VALIDATION_DIR = os.path.join(MODEL_DIR, "validation")
DEFAULT_VALIDATION_SEEDS = [20260701, 20260702, 20260703, 20260704, 20260705]


def parse_seed_list(seed_text: str | None) -> List[int]:
    """Parse a comma-separated seed list."""
    if not seed_text:
        return list(DEFAULT_VALIDATION_SEEDS)
    seeds: List[int] = []
    for part in seed_text.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("At least one validation seed is required.")
    return seeds


def _model_zip_path(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")
    return model_path


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# CSV Metrics Recorder
# ---------------------------------------------------------------------------
class MetricsRecorder:
    """Records per-episode metrics to a CSV file for later analysis."""

    HEADER = [
        "episode", "timestamp",
        "training_seed", "env_base_seed", "episode_seed",
        "throughput", "cumulative_path_cost", "total_energy",
        "total_flow_time_cost", "completed_orders_time_sum",
        "unfinished_orders_age_sum", "reward_flow_time_cost_delta",
        "pile_on_rate", "pile_on_items", "picked_quantity", "pile_on_visits",
        "reward_picked_qty_delta", "reward_pod_visits_delta",
        "reward_avg_completion_time_delta",
        "avg_order_completion_time",
        "total_reward", "episode_steps", "sim_ticks",
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def record(self, episode: int, info: Dict, total_reward: float, steps: int):
        row = [
            episode,
            datetime.now().isoformat(),
            info.get("training_seed", ""),
            info.get("env_base_seed", ""),
            info.get("episode_seed", ""),
            info.get("throughput", 0),
            info.get("cumulative_path_cost", 0.0),
            info.get("total_energy", info.get("cumulative_path_cost", 0.0)),
            info.get("total_flow_time_cost", 0.0),
            info.get("completed_orders_time_sum", 0.0),
            info.get("unfinished_orders_age_sum", 0.0),
            info.get("reward_flow_time_cost_delta", 0.0),
            info.get("pile_on_rate", 0.0),
            info.get("pile_on_items", 0),
            info.get("picked_quantity", info.get("pile_on_items", 0)),
            info.get("pile_on_visits", 0),
            info.get("reward_picked_qty_delta", 0.0),
            info.get("reward_pod_visits_delta", 0.0),
            info.get("reward_avg_completion_time_delta", 0.0),
            info.get("avg_order_completion_time", 0.0),
            total_reward,
            steps,
            info.get("tick", 0),
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)


class ValidationRecorder:
    """Records fixed-seed validation summaries to CSV."""

    HEADER = [
        "policy_update", "timestamp", "model_timesteps", "validation_seeds",
        "mean_total_flow_time_cost", "mean_total_reward", "mean_throughput",
        "mean_avg_order_completion_time", "mean_total_energy",
        "mean_pod_visits", "mean_pile_on_rate",
    ]

    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def record(self, summary: Dict[str, Any]) -> None:
        row = [
            summary.get("policy_update", 0),
            datetime.now().isoformat(),
            summary.get("model_timesteps", 0),
            ",".join(str(seed) for seed in summary.get("seeds", [])),
            summary.get("mean_total_flow_time_cost", 0.0),
            summary.get("mean_total_reward", 0.0),
            summary.get("mean_throughput", 0.0),
            summary.get("mean_avg_order_completion_time", 0.0),
            summary.get("mean_total_energy", 0.0),
            summary.get("mean_pod_visits", 0.0),
            summary.get("mean_pile_on_rate", 0.0),
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)


def run_fixed_seed_validation(
    model_path: str,
    seeds: List[int],
    max_episode_ticks: int,
    picked_qty_weight: float,
    reward_alpha: float,
    visit_penalty: float,
    run_profile: str,
    bootstrap_n_orders: int | None,
    demand_horizon_ticks: int | None,
    demand_buffer_ticks: int | None,
    pod_location_mode: str | None,
    pod_location_seed: int | None,
    full_raw_order_replay: bool,
) -> Dict[str, Any]:
    """Evaluate one saved policy on fixed seeds in the current process.

    This function is called from a child process during training so the
    validation environments cannot disturb the active PPO rollout state.
    """
    ensure_numpy_pickle_compat()
    model = PPO.load(model_path, device="cpu")
    env_kwargs = make_pps_env_kwargs(
        max_episode_ticks,
        picked_qty_weight,
        reward_alpha,
        visit_penalty,
        run_profile=run_profile,
        bootstrap_n_orders=bootstrap_n_orders,
        demand_horizon_ticks=demand_horizon_ticks,
        demand_buffer_ticks=demand_buffer_ticks,
        pod_location_mode=pod_location_mode,
        pod_location_seed=pod_location_seed,
        full_raw_order_replay=full_raw_order_replay,
    )
    env = PPSEnv(**env_kwargs)
    results: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=int(seed))
            done = False
            total_reward = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                done = bool(terminated or truncated)
                steps += 1
            episode_result = dict(info)
            episode_result["seed"] = int(seed)
            episode_result["total_reward"] = float(total_reward)
            episode_result["episode_steps"] = int(steps)
            results.append(episode_result)
    finally:
        env.close()

    return {
        "seeds": [int(seed) for seed in seeds],
        "episodes": results,
        "mean_total_flow_time_cost": _mean([float(r.get("total_flow_time_cost", 0.0)) for r in results]),
        "mean_total_reward": _mean([float(r.get("total_reward", 0.0)) for r in results]),
        "mean_throughput": _mean([float(r.get("throughput", 0.0)) for r in results]),
        "mean_avg_order_completion_time": _mean([float(r.get("avg_order_completion_time", 0.0)) for r in results]),
        "mean_total_energy": _mean([float(r.get("total_energy", 0.0)) for r in results]),
        "mean_pod_visits": _mean([float(r.get("pile_on_visits", 0.0)) for r in results]),
        "mean_pile_on_rate": _mean([float(r.get("pile_on_rate", 0.0)) for r in results]),
    }


# ---------------------------------------------------------------------------
# Custom Callback — handles TensorBoard, CSV, console, checkpoints
# ---------------------------------------------------------------------------
class PPSMetricsCallback(BaseCallback):
    """
    Tracks per-episode metrics inside SB3's training loop.

    TensorBoard sections:
      pps_reward/   : total_reward, total_flow_time_cost, completed_orders_time_sum
      pps_metrics/  : throughput, total_energy, cumulative_path_cost, episode_steps, pile_on_rate
    """

    def __init__(
        self,
        tb_writer: SummaryWriter | None = None,
        recorder: MetricsRecorder | None = None,
        verbose: int = 0,
        episode_offset: int = 0,
        best_throughput: int = 0,
        n_envs: int = 1,
        pbar: tqdm | None = None,
        training_seed: int | None = None,
    ):
        super().__init__(verbose)
        self._tb_writer = tb_writer
        self._recorder = recorder
        self._best_throughput = best_throughput
        self._episode_count = episode_offset
        self._n_envs = n_envs
        self._training_seed = training_seed
        # Per-env accumulators (support multi-env rollouts)
        self._env_reward = np.zeros(n_envs, dtype=np.float64)
        self._env_steps = np.zeros(n_envs, dtype=np.int64)
        self._env_start = [time.time()] * n_envs
        self._pbar = pbar

    def _on_step(self) -> bool:
        # Accumulate per-env reward and step counts
        rewards = self.locals.get("rewards", [])
        for i, r in enumerate(rewards):
            self._env_reward[i] += float(r)
            self._env_steps[i] += 1

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, info in enumerate(infos):
            if i < len(dones) and dones[i]:
                self._episode_count += 1
                ep = self._episode_count
                ep_reward = float(self._env_reward[i])
                ep_steps = int(self._env_steps[i])
                ep_elapsed = time.time() - self._env_start[i]

                throughput = info.get("throughput", 0)
                pile_on = info.get("pile_on_rate", 0.0)
                pod_visits = info.get("pile_on_visits", 0)
                avg_oct = info.get("avg_order_completion_time", 0.0)
                total_flow_time_cost = info.get("total_flow_time_cost", 0.0)
                completed_orders_time_sum = info.get("completed_orders_time_sum", 0.0)
                cpc = info.get("cumulative_path_cost", 0.0)
                total_energy = info.get("total_energy", cpc)
                if self._training_seed is not None:
                    info["training_seed"] = self._training_seed

                # TensorBoard
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar("pps_reward/total_reward", ep_reward, ep)
                    self._tb_writer.add_scalar("pps_reward/total_flow_time_cost", total_flow_time_cost, ep)
                    self._tb_writer.add_scalar("pps_reward/completed_orders_time_sum", completed_orders_time_sum, ep)
                    self._tb_writer.add_scalar("pps_metrics/throughput", throughput, ep)
                    self._tb_writer.add_scalar("pps_metrics/pile_on_rate", pile_on, ep)
                    self._tb_writer.add_scalar("pps_metrics/total_energy", total_energy, ep)
                    self._tb_writer.add_scalar("pps_metrics/cumulative_path_cost", cpc, ep)
                    self._tb_writer.add_scalar("pps_metrics/episode_steps", ep_steps, ep)
                    self._tb_writer.flush()

                # CSV
                if self._recorder is not None:
                    self._recorder.record(ep, info, ep_reward, ep_steps)

                # tqdm progress bar — one line, auto-updating
                if self._pbar is not None:
                    self._pbar.update(1)
                    self._pbar.set_postfix({
                        "tp": throughput,
                        "po": f"{pile_on:.2f}",
                        "vis": pod_visits,
                        "oct": f"{avg_oct:.0f}",
                        "rew": f"{ep_reward:.1f}",
                        "seed": info.get("episode_seed", ""),
                        "t": f"{ep_elapsed:.1f}s",
                    })

                # Periodic checkpoint
                if ep % 10 == 0:
                    cp_path = os.path.join(CHECKPOINT_DIR, f"pps_ppo_ep{ep}")
                    self.model.save(cp_path)

                # Reset this env's accumulators
                self._env_reward[i] = 0.0
                self._env_steps[i] = 0
                self._env_start[i] = time.time()

        return True


class FixedSeedValidationCallback(BaseCallback):
    """Evaluate the current policy on fixed seeds before PPO updates it."""

    def __init__(
        self,
        validation_seeds: List[int],
        validation_every_updates: int,
        validation_recorder: ValidationRecorder | None,
        tb_writer: SummaryWriter | None,
        max_episode_ticks: int,
        picked_qty_weight: float,
        reward_alpha: float,
        visit_penalty: float,
        run_profile: str,
        bootstrap_n_orders: int | None,
        demand_horizon_ticks: int | None,
        demand_buffer_ticks: int | None,
        pod_location_mode: str | None,
        pod_location_seed: int | None,
        full_raw_order_replay: bool,
        order_cycle_time: int,
        best_flow_time_cost: float,
        best_validation_throughput: float,
        policy_update_offset: int = 0,
        pbar: tqdm | None = None,
        validation_enabled: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.validation_seeds = validation_seeds
        self.validation_every_updates = max(1, int(validation_every_updates))
        self.validation_recorder = validation_recorder
        self.tb_writer = tb_writer
        self.max_episode_ticks = max_episode_ticks
        self.picked_qty_weight = picked_qty_weight
        self.reward_alpha = reward_alpha
        self.visit_penalty = visit_penalty
        self.run_profile = run_profile
        self.bootstrap_n_orders = bootstrap_n_orders
        self.demand_horizon_ticks = demand_horizon_ticks
        self.demand_buffer_ticks = demand_buffer_ticks
        self.pod_location_mode = pod_location_mode
        self.pod_location_seed = pod_location_seed
        self.full_raw_order_replay = full_raw_order_replay
        self.order_cycle_time = order_cycle_time
        self.best_flow_time_cost = best_flow_time_cost
        self.best_validation_throughput = best_validation_throughput
        self.policy_update_count = int(policy_update_offset)
        self.pbar = pbar
        self.validation_enabled = validation_enabled

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # SB3 calls this after collecting rollout data and before PPO.train().
        self.policy_update_count += 1
        if self.pbar is not None:
            self.pbar.update(1)

        if not self.validation_enabled:
            return
        if self.policy_update_count % self.validation_every_updates != 0:
            return

        summary = self._run_validation()
        if not summary:
            return

        summary["policy_update"] = self.policy_update_count
        summary["model_timesteps"] = int(self.model.num_timesteps)
        if self.validation_recorder is not None:
            self.validation_recorder.record(summary)

        mean_flow = float(summary.get("mean_total_flow_time_cost", 0.0))
        mean_reward = float(summary.get("mean_total_reward", 0.0))
        mean_tp = float(summary.get("mean_throughput", 0.0))
        mean_oct = float(summary.get("mean_avg_order_completion_time", 0.0))
        mean_energy = float(summary.get("mean_total_energy", 0.0))
        mean_visits = float(summary.get("mean_pod_visits", 0.0))
        mean_pile_on = float(summary.get("mean_pile_on_rate", 0.0))

        if self.tb_writer is not None:
            step = self.policy_update_count
            self.tb_writer.add_scalar("pps_validation/mean_total_flow_time_cost", mean_flow, step)
            self.tb_writer.add_scalar("pps_validation/mean_total_reward", mean_reward, step)
            self.tb_writer.add_scalar("pps_validation/mean_throughput", mean_tp, step)
            self.tb_writer.add_scalar("pps_validation/mean_avg_order_completion_time", mean_oct, step)
            self.tb_writer.add_scalar("pps_validation/mean_total_energy", mean_energy, step)
            self.tb_writer.add_scalar("pps_validation/mean_pod_visits", mean_visits, step)
            self.tb_writer.add_scalar("pps_validation/mean_pile_on_rate", mean_pile_on, step)
            self.tb_writer.flush()

        if self.pbar is not None:
            self.pbar.set_postfix({
                "val_flow": f"{mean_flow:.0f}",
                "val_tp": f"{mean_tp:.1f}",
            })
            self.pbar.write(
                f"  >> update {self.policy_update_count}: "
                f"validation flow={mean_flow:.1f}, tp={mean_tp:.1f}"
            )

        candidate_zip = self._candidate_zip_path()
        if mean_flow < self.best_flow_time_cost:
            self.best_flow_time_cost = mean_flow
            shutil.copyfile(candidate_zip, _model_zip_path(BEST_MODEL))
            if self.pbar is not None:
                self.pbar.write(
                    f"  >> update {self.policy_update_count}: "
                    f"new best validation flow {mean_flow:.1f} - {BEST_MODEL}.zip saved"
                )

        if mean_tp > self.best_validation_throughput:
            self.best_validation_throughput = mean_tp
            shutil.copyfile(candidate_zip, _model_zip_path(BEST_THROUGHPUT_MODEL))
            if self.pbar is not None:
                self.pbar.write(
                    f"  >> update {self.policy_update_count}: "
                    f"new best validation throughput {mean_tp:.1f} - "
                    f"{BEST_THROUGHPUT_MODEL}.zip saved"
                )

    def _candidate_model_path(self) -> Path:
        os.makedirs(VALIDATION_DIR, exist_ok=True)
        return Path(VALIDATION_DIR) / "pps_validation_candidate"

    def _candidate_zip_path(self) -> Path:
        return _model_zip_path(self._candidate_model_path())

    def _run_validation(self) -> Dict[str, Any] | None:
        candidate_path = self._candidate_model_path()
        self.model.save(str(candidate_path))
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_validation-only",
            "--validation-model-path",
            str(self._candidate_zip_path()),
            "--validation-seeds",
            ",".join(str(seed) for seed in self.validation_seeds),
            "--max-ticks",
            str(self.max_episode_ticks),
            "--order-cycle-time",
            str(self.order_cycle_time),
            "--profile",
            self.run_profile,
            "--picked-qty-weight",
            str(self.picked_qty_weight),
            "--reward-alpha",
            str(self.reward_alpha),
            "--visit-penalty",
            str(self.visit_penalty),
        ]
        if self.bootstrap_n_orders is not None:
            cmd.extend(["--bootstrap-n-orders", str(self.bootstrap_n_orders)])
        if self.demand_horizon_ticks is not None:
            cmd.extend(["--demand-horizon-ticks", str(self.demand_horizon_ticks)])
        if self.demand_buffer_ticks is not None:
            cmd.extend(["--demand-buffer-ticks", str(self.demand_buffer_ticks)])
        if self.pod_location_mode is not None:
            cmd.extend(["--pod-location-mode", self.pod_location_mode])
        if self.pod_location_seed is not None:
            cmd.extend(["--pod-location-seed", str(self.pod_location_seed)])
        if self.full_raw_order_replay:
            cmd.append("--full-raw-order-replay")

        env = os.environ.copy()
        env["RMFS_ORDER_CYCLE_TIME"] = str(self.order_cycle_time)
        env.setdefault("RMFS_DETAIL_DB", "0")
        result = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if self.pbar is not None:
                self.pbar.write("  >> validation failed:")
                self.pbar.write((result.stderr or result.stdout)[-2000:])
            return None
        marker = "VALIDATION_JSON:"
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(marker):
                return json.loads(line[len(marker):])
        if self.pbar is not None:
            self.pbar.write("  >> validation output did not include JSON summary")
        return None


# ---------------------------------------------------------------------------
# Stop training after a fixed number of session episodes
# ---------------------------------------------------------------------------
class StopAfterEpisodesCallback(BaseCallback):
    """Stops `model.learn()` after a target number of episodes in this session.

    Lets us pass a large total_timesteps to learn() (so `_total_timesteps`
    matches the LR decay plan), while still limiting how much we train
    per invocation.
    """

    def __init__(self, session_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.session_episodes = session_episodes
        self._session_ep_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self._session_ep_count += 1
                if self._session_ep_count >= self.session_episodes:
                    print(
                        f"\n[StopCallback] Session target reached: "
                        f"{self._session_ep_count} episodes. Stopping."
                    )
                    return False
        return True


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_pps_env(**kwargs):
    """Factory for creating PPSEnv instances (single-process / DummyVecEnv)."""
    def _init():
        return PPSEnv(**kwargs)
    return _init


def make_worker_env(
    worker_id: int,
    base_dir: str,
    max_ticks: int,
    picked_qty_weight: float,
    reward_alpha: float,
    visit_penalty: float,
    base_seed: int | None,
    run_profile: str,
    bootstrap_n_orders: int | None,
    demand_horizon_ticks: int | None,
    demand_buffer_ticks: int | None,
    pod_location_mode: str | None,
    pod_location_seed: int | None,
    full_raw_order_replay: bool,
):
    """Factory for SubprocVecEnv workers.

    Each worker gets a private runtime root under data/runtime/tmp and lets
    RunContext resolve canonical inputs. The worker CWD is isolated only as a
    temporary guard for legacy CWD-relative output helpers.
    """
    def _init():
        workdir = Path(base_dir) / "data" / "runtime" / "tmp" / f"pps_worker_{worker_id}_{os.getpid()}"
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "output").mkdir(exist_ok=True)
        os.chdir(workdir)

        return PPSEnv(**make_pps_env_kwargs(
            max_ticks,
            picked_qty_weight,
            reward_alpha,
            visit_penalty,
            None if base_seed is None else base_seed + worker_id * 1_000_000,
            run_profile=run_profile,
            bootstrap_n_orders=bootstrap_n_orders,
            demand_horizon_ticks=demand_horizon_ticks,
            demand_buffer_ticks=demand_buffer_ticks,
            pod_location_mode=pod_location_mode,
            pod_location_seed=pod_location_seed,
            full_raw_order_replay=full_raw_order_replay,
            runtime_root=workdir,
        ))
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    total_episodes: int = 50,
    resume: bool = False,
    learning_rate: float = 1e-4,
    lr_end: float | None = None,
    lr_plan_episodes: int | None = None,
    batch_size: int = 256,
    n_epochs: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    target_kl: float | None = 0.03,
    ent_coef: float = 0.001,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    max_episode_ticks: int | None = None,
    picked_qty_weight: float = PICKED_QTY_WEIGHT,
    reward_alpha: float = ALPHA_OCT,
    visit_penalty: float = POD_VISIT_PENALTY,
    n_steps: int = 8192,
    n_envs: int = 1,
    seed: int | None = None,
    policy_updates: int | None = None,
    validation_seeds: List[int] | None = None,
    validation_every_updates: int = 1,
    disable_validation: bool = False,
    run_profile: str = "training",
    bootstrap_n_orders: int | None = None,
    demand_horizon_ticks: int | None = None,
    demand_buffer_ticks: int | None = None,
    pod_location_mode: str | None = None,
    pod_location_seed: int | None = None,
    full_raw_order_replay: bool = False,
):
    """
    Train PPO for PPS using SB3's standard training loop.

    n_steps controls how many PPS decision steps SB3 collects before
    running a PPO update. Setting n_steps large enough to contain one
    full episode means the policy updates approximately once per episode.

    total_timesteps = n_steps * total_episodes, so the model runs
    for roughly total_episodes episodes before stopping.

    Args:
        total_episodes: Approximate number of episodes to train.
        resume: Load existing model checkpoint.
        learning_rate: PPO learning rate.
        batch_size: Minibatch size for SGD within each update.
        n_epochs: PPO epochs (passes over rollout data) per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping parameter.
        target_kl: Early-stop PPO epochs when the policy update moves too far.
        ent_coef: Entropy bonus coefficient (encourages exploration).
        vf_coef: Value function loss coefficient.
        max_grad_norm: Gradient clipping norm.
        max_episode_ticks: Max simulation ticks per episode.
        picked_qty_weight: Reward per picked unit.
        reward_alpha: Weight for assigned-order flow-time cost penalty.
        visit_penalty: Penalty per successful pod-station visit.
        n_steps: Steps per PPO rollout (set >= typical episode length).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    # Load previous training state before env creation so resumed runs can keep
    # the original seed sequence.
    prev_episode_count = 0
    prev_best_throughput = 0
    prev_best_validation_flow = float("inf")
    prev_best_validation_throughput = 0.0
    prev_policy_update_count = 0
    state_validation_seeds = None
    run_name = None
    prev_plan_total_timesteps = None
    state_training_seed = None

    if resume and os.path.exists(TRAIN_STATE_FILE):
        with open(TRAIN_STATE_FILE, "r") as f:
            state = json.load(f)
        prev_episode_count = state.get("episode_count", 0)
        prev_best_throughput = state.get("best_throughput", 0)
        prev_best_validation_flow = state.get("best_validation_total_flow_time_cost", float("inf"))
        prev_best_validation_throughput = state.get(
            "best_validation_throughput",
            float(prev_best_throughput),
        )
        prev_policy_update_count = state.get("policy_update_count", 0)
        state_validation_seeds = state.get("validation_seeds", None)
        run_name = state.get("run_name", None)
        prev_plan_total_timesteps = state.get("plan_total_timesteps", None)
        state_training_seed = state.get("training_seed", None)
        print(
            f"Restoring training state: {prev_episode_count} episodes, "
            f"{prev_policy_update_count} policy updates, "
            f"best validation flow {prev_best_validation_flow}"
        )
        if prev_plan_total_timesteps is not None:
            print(f"  Restored LR decay plan: {prev_plan_total_timesteps} timesteps")

    training_seed = int(seed if seed is not None else (state_training_seed if state_training_seed is not None else generate_training_seed()))
    session_base_seed = training_seed + prev_episode_count
    if validation_seeds is None:
        validation_seeds = (
            [int(seed_value) for seed_value in state_validation_seeds]
            if state_validation_seeds
            else list(DEFAULT_VALIDATION_SEEDS)
        )
    random.seed(training_seed)
    np.random.seed(training_seed)
    set_random_seed(training_seed)
    profile_cfg = resolve_run_profile(
        run_profile,
        run_horizon_ticks=max_episode_ticks,
        bootstrap_n_orders=bootstrap_n_orders,
        demand_horizon_ticks=demand_horizon_ticks,
        demand_buffer_ticks=demand_buffer_ticks,
        full_raw_order_replay=full_raw_order_replay,
        pod_location_mode=pod_location_mode,
        pod_location_seed=pod_location_seed,
        seed=training_seed,
    )
    max_episode_ticks = int(profile_cfg.run_horizon_ticks or 0)
    os.environ.update(profile_cfg.env())

    # Vec env: Dummy (serial) for n_envs=1, Subproc (parallel) otherwise
    if n_envs > 1:
        base_dir = os.getcwd()
        env_fns = [
            make_worker_env(
                i,
                base_dir,
                max_episode_ticks,
                picked_qty_weight,
                reward_alpha,
                visit_penalty,
                session_base_seed,
                profile_cfg.profile,
                profile_cfg.bootstrap_n_orders,
                profile_cfg.demand_horizon_ticks,
                profile_cfg.demand_buffer_ticks,
                profile_cfg.pod_location_mode,
                profile_cfg.pod_location_seed,
                profile_cfg.full_raw_order_replay,
            )
            for i in range(n_envs)
        ]
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        print(f"Using SubprocVecEnv with {n_envs} parallel workers.")
    else:
        vec_env = DummyVecEnv([
            make_pps_env(**make_pps_env_kwargs(
                max_episode_ticks,
                picked_qty_weight,
                reward_alpha,
                visit_penalty,
                session_base_seed,
                run_profile=profile_cfg.profile,
                bootstrap_n_orders=profile_cfg.bootstrap_n_orders,
                demand_horizon_ticks=profile_cfg.demand_horizon_ticks,
                demand_buffer_ticks=profile_cfg.demand_buffer_ticks,
                pod_location_mode=profile_cfg.pod_location_mode,
                pod_location_seed=profile_cfg.pod_location_seed,
                full_raw_order_replay=profile_cfg.full_raw_order_replay,
            ))
        ])

    if run_name is None:
        run_name = f"pps_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # LR schedule plan (persisted across resumes so decay curve is continuous)
    if prev_plan_total_timesteps is not None:
        plan_total_timesteps = prev_plan_total_timesteps
    else:
        plan_episodes = (
            lr_plan_episodes
            if lr_plan_episodes is not None
            else (policy_updates if policy_updates is not None else total_episodes)
        )
        plan_total_timesteps = n_steps * plan_episodes

    # Final LR for linear decay (default: 10% of start LR)
    if lr_end is None:
        lr_end = learning_rate * 0.1

    lr_schedule = get_linear_fn(learning_rate, lr_end, end_fraction=1.0)

    # TensorBoard + CSV (reuse same folder on resume for continuous graph)
    tb_writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))
    csv_path = os.path.join(METRICS_DIR, f"metrics_{run_name}.csv")
    validation_csv_path = os.path.join(METRICS_DIR, f"validation_{run_name}.csv")
    recorder = MetricsRecorder(csv_path)
    validation_recorder = ValidationRecorder(validation_csv_path)

    if resume and os.path.exists(BEST_MODEL + ".zip"):
        print(f"Resuming from {BEST_MODEL}")
        ensure_numpy_pickle_compat()
        model = PPO.load(BEST_MODEL, env=vec_env, device="cpu")
        model.tensorboard_log = os.path.join(LOG_DIR, run_name)
        # PPO.load restores lr_schedule (the linear decay fn) and optimizer state.
        current_progress = 1.0 - float(model.num_timesteps) / float(plan_total_timesteps)
        current_progress = max(0.0, min(1.0, current_progress))
        current_lr = lr_schedule(current_progress)
        print(f"  Restored num_timesteps: {model.num_timesteps}")
        print(f"  Current scheduled LR:   {current_lr:.2e}")
        model.lr_schedule = lr_schedule
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = current_lr
        model.n_steps = n_steps
        model.batch_size = batch_size
        model.n_epochs = n_epochs
        model.target_kl = target_kl
        # Recreate rollout buffer to match new n_steps
        from stable_baselines3.common.buffers import DictRolloutBuffer
        model.rollout_buffer = DictRolloutBuffer(
            n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=n_envs,
        )
    else:
        print("Training from scratch")
        # Linear LR schedule from learning_rate -> lr_end over plan_total_timesteps.
        # get_linear_fn is picklable, so it persists through model.save/load.
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=lr_schedule,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            target_kl=target_kl,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            normalize_advantage=True,
            verbose=1,
            tensorboard_log=os.path.join(LOG_DIR, run_name),
            device="cpu",
            seed=training_seed,
        )

    # Pass total_timesteps so SB3's _total_timesteps equals plan_total_timesteps
    # (needed for the linear LR schedule to decay over the full plan horizon).
    # The StopAfterEpisodesCallback limits actual run length to `total_episodes`.
    if policy_updates is not None:
        learn_total_timesteps = n_steps * int(policy_updates)
    elif resume:
        learn_total_timesteps = max(1, plan_total_timesteps - model.num_timesteps)
    else:
        learn_total_timesteps = plan_total_timesteps

    print(f"\n{'='*60}")
    print(f"PPS RL Training")
    print(f"{'='*60}")
    print(f"  Session episodes         : {total_episodes}")
    if policy_updates is not None:
        print(f"  Session policy updates   : {policy_updates}")
    print(f"  Parallel envs            : {n_envs}")
    print(f"  LR decay plan (ts total) : {plan_total_timesteps}")
    print(f"  LR decay plan (episodes) : {plan_total_timesteps // n_steps}")
    print(f"  n_steps per rollout      : {n_steps}")
    print(f"  Initial learning rate    : {learning_rate}")
    print(f"  Final learning rate      : {lr_end}")
    print(f"  Batch size               : {batch_size}")
    print(f"  PPO epochs/update        : {n_epochs}")
    print(f"  Gamma                    : {gamma}")
    print(f"  GAE lambda               : {gae_lambda}")
    print(f"  Clip range               : {clip_range}")
    print(f"  Target KL                : {target_kl}")
    print(f"  Entropy coef             : {ent_coef}")
    print(f"  Training seed            : {training_seed}")
    print(f"  Run profile              : {profile_cfg.profile}")
    print(f"  Order stream             : full GUI history, shuffled each episode")
    print(f"  Order cycle rate         : {int(os.environ.get('RMFS_ORDER_CYCLE_TIME', '500'))} orders/hour")
    print(f"  Demand horizon ticks     : {profile_cfg.demand_horizon_ticks}")
    print(f"  Pod location mode        : {profile_cfg.pod_location_mode}")
    print(f"  First episode seed       : {session_base_seed}")
    print(f"  Validation seeds         : {', '.join(str(s) for s in validation_seeds)}")
    print(
        "  Validation cadence       : "
        f"{'off' if disable_validation else f'every {validation_every_updates} policy update(s)'}"
    )
    print(f"  Reward objective         : minimize assigned-order flow-time cost")
    print(f"  Fast training I/O        : {'on' if FAST_TRAIN_MODE else 'off'}")
    print(
        "  Dynamic pod-job update   : "
        f"{'off' if os.environ.get('RMFS_DYNAMIC_JOB_UPDATE', '1').strip().lower() in {'0', 'false', 'no', 'off'} else 'on'}"
    )
    print(f"  Reward picked qty weight : {picked_qty_weight} (inactive)")
    print(f"  Reward flow alpha        : {reward_alpha}")
    print(f"  Reward visit penalty     : {visit_penalty} (inactive)")
    print(f"  Max episode ticks        : {max_episode_ticks}")
    print(f"  TensorBoard              : python -m tensorboard.main --logdir \"{os.path.abspath(LOG_DIR)}\"")
    print(f"  CSV metrics              : {csv_path}")
    print(f"  CSV validation           : {validation_csv_path}")
    print(f"{'='*60}\n")

    # tqdm progress bar: one line per episode or policy update.
    progress_by_updates = policy_updates is not None
    pbar = tqdm(
        total=policy_updates if progress_by_updates else total_episodes,
        desc="Policy updates" if progress_by_updates else "Training",
        unit="upd" if progress_by_updates else "ep",
        dynamic_ncols=True,
    )

    # Custom callback handles all logging
    metrics_cb = PPSMetricsCallback(
        tb_writer=tb_writer,
        recorder=recorder,
        verbose=1,
        episode_offset=prev_episode_count,
        best_throughput=prev_best_throughput,
        n_envs=n_envs,
        pbar=None if progress_by_updates else pbar,
        training_seed=training_seed,
    )

    validation_cb = FixedSeedValidationCallback(
        validation_seeds=validation_seeds,
        validation_every_updates=validation_every_updates,
        validation_recorder=validation_recorder,
        tb_writer=tb_writer,
        max_episode_ticks=max_episode_ticks,
        picked_qty_weight=picked_qty_weight,
        reward_alpha=reward_alpha,
        visit_penalty=visit_penalty,
        run_profile=profile_cfg.profile,
        bootstrap_n_orders=profile_cfg.bootstrap_n_orders,
        demand_horizon_ticks=profile_cfg.demand_horizon_ticks,
        demand_buffer_ticks=profile_cfg.demand_buffer_ticks,
        pod_location_mode=profile_cfg.pod_location_mode,
        pod_location_seed=profile_cfg.pod_location_seed,
        full_raw_order_replay=profile_cfg.full_raw_order_replay,
        order_cycle_time=int(os.environ.get("RMFS_ORDER_CYCLE_TIME", "500")),
        best_flow_time_cost=prev_best_validation_flow,
        best_validation_throughput=prev_best_validation_throughput,
        policy_update_offset=prev_policy_update_count,
        pbar=pbar if progress_by_updates else None,
        validation_enabled=not disable_validation,
    )

    # Stop after `total_episodes` episodes in THIS session unless update count controls training.
    stop_cb = StopAfterEpisodesCallback(session_episodes=total_episodes, verbose=1)
    callbacks = [metrics_cb, validation_cb]
    if policy_updates is None:
        callbacks.append(stop_cb)

    try:
        model.learn(
            total_timesteps=learn_total_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=not resume,
            tb_log_name="sb3",
        )
    finally:
        pbar.close()

    # Save final model
    final_path = os.path.join(MODEL_DIR, "pps_rl_final")
    model.save(final_path)

    # Save training state for resume continuity
    with open(TRAIN_STATE_FILE, "w") as f:
        json.dump({
            "episode_count": metrics_cb._episode_count,
            "best_throughput": validation_cb.best_validation_throughput,
            "best_validation_total_flow_time_cost": validation_cb.best_flow_time_cost,
            "best_validation_throughput": validation_cb.best_validation_throughput,
            "policy_update_count": validation_cb.policy_update_count,
            "validation_seeds": validation_seeds,
            "run_name": run_name,
            "plan_total_timesteps": plan_total_timesteps,
            "training_seed": training_seed,
            "next_episode_seed": training_seed + metrics_cb._episode_count,
        }, f)

    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Best validation flow-time cost: {validation_cb.best_flow_time_cost}")
    print(f"Best validation throughput: {validation_cb.best_validation_throughput}")
    print(f"Total policy updates so far: {validation_cb.policy_update_count}")
    print(f"Total episodes so far: {metrics_cb._episode_count}")

    tb_writer.close()
    vec_env.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    n_episodes: int = 1,
    max_episode_ticks: int = 32400,
    picked_qty_weight: float = PICKED_QTY_WEIGHT,
    reward_alpha: float = ALPHA_OCT,
    visit_penalty: float = POD_VISIT_PENALTY,
):
    """Evaluate trained PPS model and print per-episode results."""
    if not os.path.exists(BEST_MODEL + ".zip"):
        print(f"No model found at {BEST_MODEL}. Train first.")
        return

    env_kwargs = make_pps_env_kwargs(
        max_episode_ticks,
        picked_qty_weight,
        reward_alpha,
        visit_penalty,
    )
    env = PPSEnv(**env_kwargs)
    ensure_numpy_pickle_compat()
    model = PPO.load(BEST_MODEL, device="cpu")
    print(
        f"Loaded model: {BEST_MODEL}.zip "
        f"(episodes={n_episodes}, max_ticks={max_episode_ticks})"
    )

    results = []
    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        results.append(info)
        print(
            f"Episode {ep}/{n_episodes}: "
            f"reward={total_reward:.2f}, "
            f"steps={steps}, "
            f"throughput={info.get('throughput', 0)}, "
            f"pile_on={info.get('pile_on_rate', 0):.2f}, "
            f"avg_oct={info.get('avg_order_completion_time', 0):.1f}, "
            f"path_cost={info.get('cumulative_path_cost', 0):.1f}, "
            f"total_energy={info.get('total_energy', info.get('cumulative_path_cost', 0)):.1f}"
        )

    # Summary
    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({n_episodes} episodes)")
    avg_tp = np.mean([r.get("throughput", 0) for r in results])
    avg_po = np.mean([r.get("pile_on_rate", 0) for r in results])
    avg_oct = np.mean([r.get("avg_order_completion_time", 0) for r in results])
    avg_cpc = np.mean([r.get("cumulative_path_cost", 0) for r in results])
    avg_energy = np.mean([r.get("total_energy", r.get("cumulative_path_cost", 0)) for r in results])
    print(f"  Avg throughput          : {avg_tp:.1f}")
    print(f"  Avg pile-on rate        : {avg_po:.2f}")
    print(f"  Avg order completion    : {avg_oct:.1f} ticks")
    print(f"  Avg cumulative path cost: {avg_cpc:.1f}")
    print(f"  Avg total energy        : {avg_energy:.1f}")
    print(f"{'='*50}")
    env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPS RL agent")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate trained model")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="Number of episodes to run during --eval")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Approximate number of episodes to train")
    parser.add_argument("--policy-updates", type=int, default=None,
                        help="Train for this many PPO rollout/policy-update cycles. "
                             "When set, training is controlled by update count instead "
                             "of completed episode count.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (start of linear decay)")
    parser.add_argument("--lr-end", type=float, default=None,
                        help="Final learning rate (default: lr * 0.1)")
    parser.add_argument("--lr-plan-episodes", type=int, default=None,
                        help="LR decay horizon in episodes. Persisted across "
                             "resumes. Default on first run = --episodes.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="Entropy coefficient")
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="Early-stop PPO updates above this approximate KL")
    parser.add_argument("--picked-qty-weight", type=float, default=PICKED_QTY_WEIGHT,
                        help="Inactive compatibility option; picked quantity is logged only")
    parser.add_argument("--reward-alpha", type=float, default=ALPHA_OCT,
                        help="Weight for assigned-order flow-time cost penalty")
    parser.add_argument("--visit-penalty", type=float, default=POD_VISIT_PENALTY,
                        help="Inactive compatibility option; pod visits are logged only")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Override model save directory")
    parser.add_argument("--profile", choices=available_profiles(), default="training",
                        help="Run profile for horizon, demand, detail DB, and pod-location defaults")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="Max simulation ticks per episode. Defaults to the selected profile.")
    parser.add_argument("--bootstrap-n-orders", type=int, default=None,
                        help="Override profile bootstrap order count")
    parser.add_argument("--demand-horizon-ticks", type=int, default=None,
                        help="Override generated demand horizon")
    parser.add_argument("--demand-buffer-ticks", type=int, default=None,
                        help="Override generated demand buffer beyond run horizon")
    parser.add_argument("--order-cycle-time", type=int, default=500,
                        help="PPS training order rate in orders per simulated hour")
    parser.add_argument("--pod-location-mode", choices=("fixed", "randomize_slots"), default=None,
                        help="Override profile pod-location mode")
    parser.add_argument("--pod-location-seed", type=int, default=None,
                        help="Override pod-location randomization seed")
    parser.add_argument("--full-raw-order-replay", action="store_true", default=False,
                        help="Opt in to replaying all unique raw orders")
    parser.add_argument("--n-steps", type=int, default=8192,
                        help="Steps per PPO rollout (set >= typical episode length)")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel envs (SubprocVecEnv). 1 = serial.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed. If omitted, one is generated and saved.")
    parser.add_argument("--validation-seeds", type=str,
                        default=",".join(str(seed) for seed in DEFAULT_VALIDATION_SEEDS),
                        help="Comma-separated fixed seeds used to select best models")
    parser.add_argument("--validation-every-updates", type=int, default=1,
                        help="Run fixed-seed validation every N policy updates")
    parser.add_argument("--disable-validation", action="store_true", default=False,
                        help="Disable fixed-seed validation during training")
    parser.add_argument("--_validation-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validation-model-path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--detail-db", action="store_true", default=False,
                        help="Enable detail SQLite DB writes during PPS training/eval.")

    args = parser.parse_args()
    if args.order_cycle_time <= 0:
        parser.error("--order-cycle-time must be a positive orders-per-hour value")
    if args.policy_updates is not None and args.policy_updates <= 0:
        parser.error("--policy-updates must be positive when provided")
    if args.validation_every_updates <= 0:
        parser.error("--validation-every-updates must be positive")
    validation_seeds = parse_seed_list(args.validation_seeds)
    os.environ["RMFS_ORDER_CYCLE_TIME"] = str(args.order_cycle_time)
    os.environ["RMFS_DETAIL_DB"] = "1" if args.detail_db else "0"

    # Override paths if --save-path provided
    if args.save_path:
        save_dir = os.path.dirname(args.save_path) or MODEL_DIR
        MODEL_DIR = save_dir
        LOG_DIR = os.path.join(save_dir, "runs")
        METRICS_DIR = os.path.join(save_dir, "metrics")
        BEST_MODEL = os.path.splitext(args.save_path)[0] + "_best"
        BEST_THROUGHPUT_MODEL = os.path.splitext(args.save_path)[0] + "_best_throughput"
        CHECKPOINT_DIR = os.path.join(save_dir, "checkpoints")
        VALIDATION_DIR = os.path.join(save_dir, "validation")

    if args._validation_only:
        if not args.validation_model_path:
            parser.error("--validation-model-path is required for internal validation")
        validation_profile = resolve_run_profile(
            args.profile,
            run_horizon_ticks=args.max_ticks,
            bootstrap_n_orders=args.bootstrap_n_orders,
            demand_horizon_ticks=args.demand_horizon_ticks,
            demand_buffer_ticks=args.demand_buffer_ticks,
            full_raw_order_replay=args.full_raw_order_replay,
            pod_location_mode=args.pod_location_mode,
            pod_location_seed=args.pod_location_seed,
            seed=validation_seeds[0] if validation_seeds else args.seed,
        )
        os.environ.update(validation_profile.env())
        summary = run_fixed_seed_validation(
            model_path=args.validation_model_path,
            seeds=validation_seeds,
            max_episode_ticks=int(validation_profile.run_horizon_ticks or 0),
            picked_qty_weight=args.picked_qty_weight,
            reward_alpha=args.reward_alpha,
            visit_penalty=args.visit_penalty,
            run_profile=validation_profile.profile,
            bootstrap_n_orders=validation_profile.bootstrap_n_orders,
            demand_horizon_ticks=validation_profile.demand_horizon_ticks,
            demand_buffer_ticks=validation_profile.demand_buffer_ticks,
            pod_location_mode=validation_profile.pod_location_mode,
            pod_location_seed=validation_profile.pod_location_seed,
            full_raw_order_replay=validation_profile.full_raw_order_replay,
        )
        print("VALIDATION_JSON:" + json.dumps(summary, sort_keys=True))
        raise SystemExit(0)

    if args.eval:
        eval_profile = resolve_run_profile(
            args.profile,
            run_horizon_ticks=args.max_ticks,
            bootstrap_n_orders=args.bootstrap_n_orders,
            demand_horizon_ticks=args.demand_horizon_ticks,
            demand_buffer_ticks=args.demand_buffer_ticks,
            full_raw_order_replay=args.full_raw_order_replay,
            pod_location_mode=args.pod_location_mode,
            pod_location_seed=args.pod_location_seed,
            seed=args.seed,
        )
        os.environ.update(eval_profile.env())
        evaluate(
            n_episodes=args.eval_episodes,
            max_episode_ticks=int(eval_profile.run_horizon_ticks or 0),
            picked_qty_weight=args.picked_qty_weight,
            reward_alpha=args.reward_alpha,
            visit_penalty=args.visit_penalty,
        )
    else:
        train(
            total_episodes=args.episodes,
            resume=args.resume,
            learning_rate=args.lr,
            lr_end=args.lr_end,
            lr_plan_episodes=args.lr_plan_episodes,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            picked_qty_weight=args.picked_qty_weight,
            reward_alpha=args.reward_alpha,
            visit_penalty=args.visit_penalty,
            max_episode_ticks=args.max_ticks,
            n_steps=args.n_steps,
            n_envs=args.n_envs,
            seed=args.seed,
            policy_updates=args.policy_updates,
            validation_seeds=validation_seeds,
            validation_every_updates=args.validation_every_updates,
            disable_validation=args.disable_validation,
            run_profile=args.profile,
            bootstrap_n_orders=args.bootstrap_n_orders,
            demand_horizon_ticks=args.demand_horizon_ticks,
            demand_buffer_ticks=args.demand_buffer_ticks,
            pod_location_mode=args.pod_location_mode,
            pod_location_seed=args.pod_location_seed,
            full_raw_order_replay=args.full_raw_order_replay,
        )
