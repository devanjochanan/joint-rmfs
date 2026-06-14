#!/usr/bin/env python3
"""CLI tool to bootstrap an initial RTS-RL policy checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.rmfs.rl.rts.model import RTSMaskedActorCritic
from src.rmfs.rl.rts.features import build_action_feature_names, build_stock_feature_names
from src.rmfs.rl.rts.cycle_reference import write_cycle_reference
from src.rmfs.rl.rts.training.references import create_synthetic_cycle_reference
from src.rmfs.rl.rts.training.checkpoint import atomic_torch_save, write_feature_schema


def main(argv=None):
    parser = argparse.ArgumentParser(description="Initialize a bootstrap RTS checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory to save the checkpoint.")
    parser.add_argument("--zone-ids", default="A,B", help="Comma-separated list of zone IDs.")
    parser.add_argument("--policy-checkpoint-id", default="bootstrap_000000", help="Policy checkpoint ID.")
    parser.add_argument("--hidden-sizes", default="64,64", help="Hidden sizes of the MLP row encoder.")
    parser.add_argument("--stock-hidden-sizes", default="32,32", help="Stock encoder hidden sizes.")
    parser.add_argument("--stock-embedding-dim", type=int, default=16, help="Stock embedding dimension.")
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    zone_ids = tuple(z.strip() for z in args.zone_ids.split(",") if z.strip())
    if not zone_ids:
        parser.error("At least one zone ID is required.")

    hidden_sizes = tuple(int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip())
    stock_hidden_sizes = tuple(int(x.strip()) for x in args.stock_hidden_sizes.split(",") if x.strip())

    action_feature_names = build_action_feature_names(zone_ids)
    stock_feature_names = build_stock_feature_names()

    model = RTSMaskedActorCritic(
        action_feature_dim=len(action_feature_names),
        stock_feature_dim=len(stock_feature_names),
        hidden_sizes=hidden_sizes,
        stock_hidden_sizes=stock_hidden_sizes,
        stock_embedding_dim=args.stock_embedding_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    atomic_torch_save(model.state_dict(), checkpoint_dir / "model.pt")
    atomic_torch_save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

    feature_schema = write_feature_schema(
        checkpoint_dir / "feature_schema.json",
        action_feature_names=action_feature_names,
        stock_feature_names=stock_feature_names,
    )

    cycle_ref = create_synthetic_cycle_reference()
    write_cycle_reference(checkpoint_dir / "cycle_reference.json", cycle_ref)

    # Save zone_ids as a json file
    with (checkpoint_dir / "zone_ids").open("w") as f:
        json.dump(list(zone_ids), f)

    # Save policy_checkpoint_id
    with (checkpoint_dir / "policy_checkpoint_id").open("w") as f:
        f.write(args.policy_checkpoint_id.strip() + "\n")

    metadata = {
        "policy_checkpoint_id": args.policy_checkpoint_id,
        "training_config": {
            "hidden_sizes": list(hidden_sizes),
            "stock_hidden_sizes": list(stock_hidden_sizes),
            "stock_embedding_dim": args.stock_embedding_dim,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "zone_ids": list(zone_ids),
        },
        "feature_schema": feature_schema,
    }
    with (checkpoint_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Bootstrapped checkpoint successfully in {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
