#!/bin/bash

# ==========================================
# 批量推理脚本：对比 Qwen-Base 和 Qwen+LoRA
# ==========================================

set -e

# --- 路径配置 ---
INPUT_DIR="./evaluation_samples_vismarker_mixed-0125V1/input"
OUTPUT_BASE="./evaluation_samples_vismarker_mixed-0125V1/qwen_base"
OUTPUT_OURS="./evaluation_samples_vismarker_mixed-0125V1/ours"

MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
LORA_PATH="./result15-rank64_cache_all_balanced/checkpoint-4200"

# --- 推理参数 ---
PROMPT="Follow the visual pointer and text to edit the image"
RESOLUTION=$((512*512))

# 创建输出目录
mkdir -p "$OUTPUT_BASE"
mkdir -p "$OUTPUT_OURS"

# 获取文件列表
shopt -s nullglob
img_files=("$INPUT_DIR"/*.{png,jpg,jpeg,PNG,JPG})
IMAGE_COUNT=${#img_files[@]}

echo "Found $IMAGE_COUNT images in $INPUT_DIR"

# ========================================
# Step 1: Qwen-Base 推理 (不加载 LoRA)
# ========================================
echo "------------------------------------------"
echo "Running Step 1: Qwen-Base"
echo "------------------------------------------"
#
#for ((i=0; i<$IMAGE_COUNT; i++)); do
#    img_path="${img_files[$i]}"
#    filename=$(basename "$img_path")
#    output_path="$OUTPUT_BASE/$filename"
#
#    echo "[$(($i+1))/$IMAGE_COUNT] Base Processing: $filename"
#
#    # 注意：不传入 --lora_weight 即可运行 Base 模型
#    python scripts/quick_infer.py \
#        --pretrained_model "$MODEL_PATH" \
#        --ctrl_img "$img_path" \
#        --output_img "$output_path" \
#        --prompt "$PROMPT" \
#        --cfg_scale 6.0 \
#        --infer_steps 50 \
#        --target_area $RESOLUTION \
#        --lora_weight ""
#done

# ========================================
# Step 2: Qwen+LoRA 推理 (加载 LoRA)
# ========================================
echo "------------------------------------------"
echo "Running Step 2: Qwen+LoRA (Ours)"
echo "------------------------------------------"

for ((i=0; i<$IMAGE_COUNT; i++)); do
    img_path="${img_files[$i]}"
    filename=$(basename "$img_path")
    output_path="$OUTPUT_OURS/$filename"

    echo "[$(($i+1))/$IMAGE_COUNT] LoRA Processing: $filename"

    python scripts/quick_infer.py \
        --pretrained_model "$MODEL_PATH" \
        --lora_weight "$LORA_PATH" \
        --ctrl_img "$img_path" \
        --output_img "$output_path" \
        --prompt "$PROMPT" \
        --cfg_scale 6.0 \
        --infer_steps 50 \
        --target_area $RESOLUTION
done

echo "=========================================="
echo "All tasks completed!"
echo "Base results: $OUTPUT_BASE"
echo "Ours results: $OUTPUT_OURS"
echo "=========================================="