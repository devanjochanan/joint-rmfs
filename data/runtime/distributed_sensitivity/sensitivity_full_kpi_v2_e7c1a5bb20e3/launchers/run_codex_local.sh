#!/usr/bin/env bash
set -euo pipefail
cd "/home/dewan/Project Ta/Fresh Start Structure V1/Rika's Version"
exec "/home/dewan/torch-gpu/bin/python" scripts/experiments/distributed_sensitivity_campaign.py --manifest "data/runtime/distributed_sensitivity/sensitivity_full_kpi_v2_e7c1a5bb20e3/manifest.json" --machine-id "codex_local" --run-continuously --resume --progress
