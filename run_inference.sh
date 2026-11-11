#!/bin/bash
python peft_t5_inference.py \
    --base_model_path "pretrained_models/byt5-small-prod-sep-model" \
    --adapter_path "output/byt5-small-prod-sep-model-peft-lr0.03-r8-e100" \
    --test_path "data/c-h/test_prod.csv" \
    --output_path "byt5-small-prod-sep-model-peft-lr0.03-r8-e100.csv" \
    --tokenization "none" \
    --max_source_len 96 \
    --max_new_tokens 96 \
    --num_beams 5 \
    --batch_size 32 \
    --seed 42
