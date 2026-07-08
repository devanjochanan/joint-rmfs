#!/usr/bin/env bash
set -euo pipefail
cd "/home/citi/Documents/Dewa's Sandbox/netlogo-rmfs"
exec "/home/citi/Documents/Dewa's Sandbox/torch-gpu/bin/python" scripts/experiments/distributed_sensitivity_campaign.py --manifest "data/runtime/distributed_sensitivity/sensitivity_full_kpi_v2_e7c1a5bb20e3/manifest.json" --machine-id "citi_angiebow" --run-continuously --resume --progress
