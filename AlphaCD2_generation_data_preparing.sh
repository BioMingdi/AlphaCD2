#!/usr/bin/env bash
set -euo pipefail

PROGEN2_DIR="/path/to/ProGen2-finetuning"
INPUT_FASTA="/path/to/CD_sequences.fasta"
TRAIN_FILE="./CD_train.json"
TEST_FILE="./CD_test.json"

python3 "${PROGEN2_DIR}/src/prepare_data.py" \
    --input_files "${INPUT_FASTA}" \
    --output_file_train "${TRAIN_FILE}" \
    --output_file_test "${TEST_FILE}" \
    --train_split_ratio 0.8 \
    --bidirectional \
    --seed 2025
