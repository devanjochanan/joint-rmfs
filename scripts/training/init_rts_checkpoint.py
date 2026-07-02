#!/usr/bin/env python3
"""CLI tool to bootstrap an initial RTS-RL policy checkpoint."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.features import build_action_feature_names, build_stock_feature_names
from src.rmfs.rl.rts.ablation import resolve_ablation
from src.rmfs.rl.rts.training.checkpoint import atomic_torch_save, optimizer_state_fingerprint, write_feature_schema
from src.rmfs.rl.rts.training.policy_loader import load_policy_from_checkpoint
from src.rmfs.rl.rts.training.reward_normalizer import default_reward_normalizer_metadata
from src.rmfs.orchestration.local_executor import git_value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Initialize a bootstrap RTS checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory to save the checkpoint.")
    parser.add_argument("--zone-ids", default="auto", help="Comma-separated list of zone IDs, or 'auto' to use canonical zones from layout configuration.")
    parser.add_argument("--policy-checkpoint-id", default="bootstrap_000000", help="Policy checkpoint ID.")
    parser.add_argument("--seed", type=int, default=42, help="Torch initialization seed.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--hidden-sizes", default="64,64", help="Hidden sizes of the MLP row encoder.")
    parser.add_argument("--stock-hidden-sizes", default="32,32", help="Stock encoder hidden sizes.")
    parser.add_argument("--stock-embedding-dim", type=int, default=16, help="Stock embedding dimension.")
    parser.add_argument("--feature-ablation", default="full", help="Name-based feature/branch ablation identity.")
    parser.add_argument("--charging-enabled", action="store_const", const="enabled", dest="charging_mode")
    parser.add_argument("--charging-disabled", action="store_const", const="disabled", dest="charging_mode")
    parser.add_argument("--charging-inherit-default", action="store_const", const="inherit", dest="charging_mode")
    parser.add_argument("--write-legacy-cycle-reference", action="store_true", help="Write optional synthetic cycle_reference.json for legacy compatibility.")
    parser.set_defaults(charging_mode="inherit")
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.zone_ids == "auto":
        from model.layout import Layout
        layout = Layout()
        zone_ids = tuple(
            f"rts_z_r{r:02d}_c{c:02d}"
            for r in range(layout.pod_batch_vertical_max)
            for c in range(layout.pod_batch_horizontal_max)
        )
    else:
        zone_ids = tuple(z.strip() for z in args.zone_ids.split(",") if z.strip())

    if not zone_ids:
        parser.error("At least one zone ID is required.")
    policy_checkpoint_id = str(args.policy_checkpoint_id).strip()
    if not policy_checkpoint_id:
        parser.error("--policy-checkpoint-id must be nonblank")

    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip())
    stock_hidden_sizes = tuple(int(x.strip()) for x in args.stock_hidden_sizes.split(",") if x.strip())
    if float(args.learning_rate) <= 0.0:
        parser.error("--learning-rate must be positive")
    torch.manual_seed(int(args.seed))
    ablation = resolve_ablation(args.feature_ablation)

    action_feature_names = build_action_feature_names(zone_ids)
    stock_feature_names = build_stock_feature_names()

    model = RTSMaskedActorCritic(
        action_feature_dim=len(action_feature_names),
        stock_feature_dim=len(stock_feature_names),
        hidden_sizes=hidden_sizes,
        stock_hidden_sizes=stock_hidden_sizes,
        stock_embedding_dim=args.stock_embedding_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    atomic_torch_save(model.state_dict(), checkpoint_dir / "model.pt")
    atomic_torch_save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    feature_schema = write_feature_schema(
        checkpoint_dir / "feature_schema.json",
        action_feature_names=action_feature_names,
        stock_feature_names=stock_feature_names,
    )

    if args.write_legacy_cycle_reference:
        from src.rmfs.rl.rts.cycle_reference import write_cycle_reference
        from src.rmfs.rl.rts.training.references import create_synthetic_cycle_reference

        cycle_ref = create_synthetic_cycle_reference()
        write_cycle_reference(checkpoint_dir / "cycle_reference.json", cycle_ref)

    # Save zone_ids as a json file
    with (checkpoint_dir / "zone_ids").open("w") as f:
        json.dump(list(zone_ids), f)

    # Save policy_checkpoint_id
    with (checkpoint_dir / "policy_checkpoint_id").open("w") as f:
        f.write(policy_checkpoint_id + "\n")

    repo_commit = git_value(REPO_ROOT, "rev-parse", "HEAD")
    repo_branch = git_value(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    model_architecture = {
        "class": "RTSMaskedActorCritic",
        "action_feature_dim": len(action_feature_names),
        "stock_feature_dim": len(stock_feature_names),
        "hidden_sizes": list(hidden_sizes),
        "stock_hidden_sizes": list(stock_hidden_sizes),
        "stock_embedding_dim": args.stock_embedding_dim,
    }

    metadata = {
        "checkpoint_kind": "initial_untrained",
        "policy_checkpoint_id": policy_checkpoint_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "initialization_seed": int(args.seed),
        "repo_branch": repo_branch,
        "repo_commit": repo_commit,
        "model_architecture": model_architecture,
        "optimizer": {
            "type": "Adam",
            "learning_rate": float(args.learning_rate),
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "optimizer_state_fingerprint": optimizer_state_fingerprint(optimizer.state_dict()),
        "training_zone_provenance": {
            "source": "layout_auto" if args.zone_ids == "auto" else "cli",
            "zone_count": len(zone_ids),
            "zone_ids": list(zone_ids),
            "informational_only": True,
        },
        "charging_mode": args.charging_mode,
        "committed_next_proposal_policy": "nearest_guaranteed",
        "committed_next_reservations_enabled": True,
        "feature_ablation": ablation.name,
        "feature_ablation_hash": ablation.hash,
        "feature_ablation_specification": ablation.to_json_dict(),
        "training_config": {
            "hidden_sizes": list(hidden_sizes),
            "stock_hidden_sizes": list(stock_hidden_sizes),
            "stock_embedding_dim": args.stock_embedding_dim,
            "learning_rate": float(args.learning_rate),
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "zone_ids": list(zone_ids),
            "feature_ablation": ablation.name,
        },
        "feature_schema": feature_schema,
        "feature_schema_id": feature_schema["feature_schema_id"],
        "reward_normalizer": default_reward_normalizer_metadata(),
    }
    with (checkpoint_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    loaded = load_policy_from_checkpoint(checkpoint_dir, device="cpu")
    if loaded.policy_checkpoint_id != policy_checkpoint_id:
        raise RuntimeError("production loader returned a different policy checkpoint ID")

    print(f"Bootstrapped checkpoint successfully in {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
