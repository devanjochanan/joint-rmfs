#!/bin/bash
set -e

source ~/torch-gpu/bin/activate
cd ~/Project\ Ta/Fresh\ Start\ Structure\ V1/Rika\'s\ Version

SEED_PACK="data/runtime/eval_seed_packs/eval_pack_8561a587acb9.json"
TRAIN_DIR="data/runtime/rts_training/dewa_rts_train_20260706T155814_877ab31"
BATCH20_CKPT="${TRAIN_DIR}/batch_000020/checkpoint"
BATCH05_CKPT="${TRAIN_DIR}/batch_000005/checkpoint"

echo "=== Eval 1/2: Batch 5 (best orders+cycle, 30 reps, 8 workers) ==="
echo "Started at $(date)"
python scripts/experiments/evaluate_rts_checkpoint.py \
    --checkpoint-dir "$BATCH05_CKPT" \
    --policy-mode rts_rl_explicit \
    --zone-ids auto \
    --seed-pack "$SEED_PACK" \
    --max-workers 8 \
    --execute

echo ""
echo "=== Eval 2/2: Batch 20 (latest, 30 reps, 8 workers) ==="
echo "Started at $(date)"
python scripts/experiments/evaluate_rts_checkpoint.py \
    --checkpoint-dir "$BATCH20_CKPT" \
    --policy-mode rts_rl_explicit \
    --zone-ids auto \
    --seed-pack "$SEED_PACK" \
    --max-workers 8 \
    --execute

echo ""
echo "=== Both evaluations complete at $(date) ==="
