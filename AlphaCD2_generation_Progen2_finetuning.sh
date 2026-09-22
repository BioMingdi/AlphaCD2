#!/usr/bin/env bash
set -euo pipefail

PROGEN2_DIR="/path/to/ProGen2-finetuning"
TRAIN_FILE="./CD_train.json"
TEST_FILE="./CD_test.json"

python "${PROGEN2_DIR}/src/finetune.py" \
    --model=hugohrban/progen2-base \
    --train_file="${TRAIN_FILE}" \
    --test_file="${TEST_FILE}" \
    --device=cuda \
    --epochs=10 \
    --batch_size=2 \
    --accumulation_steps=4 \
    --lr=1e-4 \
    --decay=cosine \
    --warmup_steps=200 \
    --eval_before_train \
    --seed=2025
