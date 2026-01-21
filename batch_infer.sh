#! /bin/bash

MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
LORA_PATH="./result15-rank64_cache_all_balanced/checkpoint-4200"
TASK_FILE="./tasks.json"
OUTPUT_DIR="./outputs_batch-0121"

python scripts/quick_infer_batch.py \
    --pretrained_model "$MODEL_PATH" \
    --lora_weight "$LORA_PATH" \
    --task_file "$TASK_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --cfg_scale 6.0 \
    --infer_steps 50 \
    --target_area $((512*512))