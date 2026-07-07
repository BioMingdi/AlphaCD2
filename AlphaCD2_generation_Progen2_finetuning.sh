export HF_ENDPOINT="https://hf-mirror.com"
python $Path/ProGen2-finetuning/src/finetune.py \
    --model=hugohrban/progen2-base \
    --train_file=${TRAIN_FILE} \
    --test_file=${TEST_FILE} \
    --device=cuda \
    --epochs=10 \
    --batch_size=2 \
    --accumulation_steps=4 \
    --lr=1e-4 \
    --decay=cosine \
    --warmup_steps=200 \
    --eval_before_train \
    --seed 2025
