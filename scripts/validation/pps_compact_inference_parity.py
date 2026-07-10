#!/usr/bin/env python3
"""Verify EXACT deterministic action parity between the full PPS PPO archive and
the compact policy-only inference artifact used by the campaign.

The full archive is read from the external preservation path (or the repo copy
if still present). Fails nonzero on any mismatch.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ["PPS_MODE"] = "ppo"

from src.rmfs.decisions.pps.runtime import (
    load_pps_rl_model_strict, load_pps_rl_inference_policy,
    PPS_RL_MAX_PODS, PPS_RL_POD_FEATURE_DIM, PPS_RL_NUM_STATIONS,
    PPS_RL_TOP_K_SKUS, PPS_RL_NUM_TRAFFIC_ZONES,
)

COMPACT = REPO / "data/models/pps/pps_rl_policy_inference.zip"
META = REPO / "data/models/pps/pps_rl_policy_inference.metadata.json"
REPO_FULL = REPO / "data/models/pps/pps_rl_best.zip"

def full_checkpoint_path() -> Path:
    meta = json.loads(META.read_text())
    ext = Path(meta["source_full_checkpoint"]["external_archive_path"])
    if ext.exists():
        return ext
    if REPO_FULL.exists():
        return REPO_FULL
    raise SystemExit("full checkpoint unavailable (neither external archive nor repo copy present)")

def rand_obs(rng, n_cand=None):
    if n_cand is None:
        n_cand = int(rng.integers(0, PPS_RL_MAX_PODS + 1))
    pod = np.zeros((PPS_RL_MAX_PODS, PPS_RL_POD_FEATURE_DIM), dtype=np.float32)
    pod[:n_cand] = rng.random((n_cand, PPS_RL_POD_FEATURE_DIM), dtype=np.float32)
    return {
        "pod_features": pod,
        "station_features": rng.random((PPS_RL_NUM_STATIONS, PPS_RL_TOP_K_SKUS), dtype=np.float32),
        "num_candidates": np.array([n_cand], dtype=np.int32),
        "zone_robot_counts": rng.random((PPS_RL_NUM_TRAFFIC_ZONES,), dtype=np.float32) * 10.0,
    }

def main():
    full = load_pps_rl_model_strict(str(full_checkpoint_path()))
    compact = load_pps_rl_inference_policy(str(COMPACT))
    obs_list = []
    try:
        from src.rmfs.rl.pps.env import PPSEnv
        for s in (42, 43, 44):
            env = PPSEnv(base_seed=s, run_profile="training")
            o, _ = env.reset(seed=s)
            obs_list.append(("env_%d" % s, o))
    except Exception as e:
        print("[warn] env observations unavailable:", e)
    rng = np.random.default_rng(0)
    obs_list += [(f"rand_{i}", rand_obs(rng)) for i in range(40)]
    obs_list += [(f"pad_{nc}", rand_obs(rng, n_cand=nc)) for nc in (0, 1, 2, 30, 59, 60)]
    base = rand_obs(rng, n_cand=12)
    for i in range(10):
        o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        perm = rng.permutation(12)
        o["pod_features"][:12] = base["pod_features"][perm]
        obs_list.append((f"perm_{i}", o))

    mism = 0
    for name, o in obs_list:
        a_full, _ = full.predict(o, deterministic=True)
        a_comp, _ = compact.predict(o, deterministic=True)
        if not np.array_equal(np.asarray(a_full), np.asarray(a_comp)):
            mism += 1
            print(f"  MISMATCH [{name}]")
    print(f"pps compact inference parity: {len(obs_list)-mism}/{len(obs_list)} exact matches")
    if mism:
        print("PARITY FAILED")
        raise SystemExit(1)
    print("ALL PPS COMPACT INFERENCE PARITY CHECKS PASSED")

if __name__ == "__main__":
    main()
