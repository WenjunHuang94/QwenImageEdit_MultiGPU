#! /bin/bash

MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
LORA_PATH="./result/checkpoint-500"
INPUT_IMG="./control_images2/001.jpg"
OUTPUT_DIR="./outputs"

RESOLUTION=$((512*512))

python scripts/quick_infer.py \
    --pretrained_model "$MODEL_PATH" \
    --lora_weight "$LORA_PATH" \
    --ctrl_img "$INPUT_IMG" \
    --output_img "$OUTPUT_DIR/result_$(date +%Y%m%d_%H%M%S).png" \
    --prompt "Put a necklace on him" \
    --cfg_scale 6.0 \
    --infer_steps 50 \
    --target_area $RESOLUTION \
    "$@"