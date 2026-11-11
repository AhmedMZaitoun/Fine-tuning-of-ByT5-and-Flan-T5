#!/bin/bash
python peft_t5_finetune1.py \
    --model_path "pretrained_models/byt5-small-prod-sep-model" \
    --train_path "data/c-h/train_prod.csv" \
    --val_path "data/c-h/val_prod.csv" \
    --max_new_tokens 96 \
    --max_length 192 \
    --max_source_len 96 \
    --max_target_len 96 \
    --seed 42 \
    --tokenization "none" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --learning_rate 0.03 \
    --num_epochs 50