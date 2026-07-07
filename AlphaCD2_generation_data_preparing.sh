python3 $Path/ProGen2-finetuning/src/prepare_data.py \
    --input_files "${INPUT_FASTA}" \
    --output_file_train="${TRAIN_FILE}" \
    --output_file_test="${TEST_FILE}" \
    --train_split_ratio=0.8 \
    --bidirectional \
    --seed 2025
