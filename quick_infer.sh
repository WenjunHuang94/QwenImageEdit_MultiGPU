#! /bin/bash

MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
LORA_PATH="./result4/checkpoint-3000"
INPUT_IMG="./test2.jpg"
OUTPUT_DIR="./outputs"

RESOLUTION=$((512*512))

# 提取LORA路径的basename用于输出文件名
LORA_NAME=$(basename "$LORA_PATH")

python scripts/quick_infer.py \
    --pretrained_model "$MODEL_PATH" \
    --lora_weight "$LORA_PATH" \
    --ctrl_img "$INPUT_IMG" \
    --output_img "$OUTPUT_DIR/result_${LORA_NAME}_$(date +%Y%m%d_%H%M%S).png" \
    --prompt "Edit the image according to the text instruction in the image" \
    --cfg_scale 6.0 \
    --infer_steps 50 \
    --target_area $RESOLUTION \
    "$@"