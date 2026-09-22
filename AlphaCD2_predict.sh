#!/usr/bin/env bash
set -euo pipefail

ESM_CHECKPOINT="${ESM_CHECKPOINT:-/path/to/esmc_600m_2024_12_v0.pth}"

python AlphaCD2_predict.py \
    --input-txt test.txt \
    --output-tsv test_result.tsv \
    --ontarget-script AlphaCD2_ontarget.py \
    --manifest bilstm_manifest.json \
    --esm-checkpoint "${ESM_CHECKPOINT}" \
    --specificity-checkpoint auxiliary_metrics.pt
