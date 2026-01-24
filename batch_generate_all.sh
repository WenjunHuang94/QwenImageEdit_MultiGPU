#!/bin/bash

# 批量推理脚本：用Qwen-Base和Qwen+LoRA生成所有结果
# 输入：evaluation_samples_with_textbox_0124V1/input/ 中的所有图片
# 输出：qwen_base/ 和 ours/ 目录

set -e

echo "=========================================="
echo "Batch Inference: Qwen-Base vs Qwen+LoRA"
echo "=========================================="

# 配置
INPUT_DIR="./evaluation_samples_with_textbox_0124V1/input"
OUTPUT_BASE="./evaluation_samples_with_textbox_0124V1/qwen_base"
OUTPUT_OURS="./evaluation_samples_with_textbox_0124V1/ours"
PROMPT="Modify the image at the annotated location according to the text instruction"

# 模型路径（根据你的实际路径修改）
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"  # 修改这里
LORA_PATH="./result15-rank64_cache_all_balanced/checkpoint-4200"   # 修改这里

# 创建输出目录
mkdir -p "$OUTPUT_BASE"
mkdir -p "$OUTPUT_OURS"

# 统计图片数量
IMAGE_COUNT=$(ls "$INPUT_DIR"/*.{png,jpg,jpeg,JPEG} 2>/dev/null | wc -l)
echo ""
echo "Found $IMAGE_COUNT images in $INPUT_DIR"
echo "Prompt: $PROMPT"
echo ""

# ========================================
# Step 1: 用Qwen-Base生成
# ========================================
echo "=========================================="
echo "Step 1: Generating with Qwen-Base"
echo "=========================================="

COUNTER=0
for img_path in "$INPUT_DIR"/*.{png,jpg,jpeg,JPEG}; do
    # 检查文件是否存在
    [ -f "$img_path" ] || continue
    
    COUNTER=$((COUNTER + 1))
    filename=$(basename "$img_path")
    output_path="$OUTPUT_BASE/$filename"
    
    echo "[$COUNTER/$IMAGE_COUNT] Processing: $filename"
    
    python batch_local_image_edit.py \
        --image_path "$img_path" \
        --prompt "$PROMPT" \
        --output_path "$output_path" \
        --model_path "$MODEL_PATH"
    
    echo "  ✓ Saved to: $output_path"
done

echo ""
echo "✓ Qwen-Base: Generated $COUNTER images"

# ========================================
# Step 2: 用Qwen+LoRA生成
# ========================================
echo ""
echo "=========================================="
echo "Step 2: Generating with Qwen+LoRA"
echo "=========================================="

COUNTER=0
for img_path in "$INPUT_DIR"/*.{png,jpg,jpeg,JPEG}; do
    # 检查文件是否存在
    [ -f "$img_path" ] || continue
    
    COUNTER=$((COUNTER + 1))
    filename=$(basename "$img_path")
    output_path="$OUTPUT_OURS/$filename"
    
    echo "[$COUNTER/$IMAGE_COUNT] Processing: $filename"
    
    python batch_local_image_edit.py \
        --image_path "$img_path" \
        --prompt "$PROMPT" \
        --output_path "$output_path" \
        --model_path "$MODEL_PATH" \
        --lora_path "$LORA_PATH"
    
    echo "  ✓ Saved to: $output_path"
done

echo ""
echo "✓ Qwen+LoRA: Generated $COUNTER images"

# ========================================
# 完成
# ========================================
echo ""
echo "=========================================="
echo "✓ All Done!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Qwen-Base: $OUTPUT_BASE/ ($COUNTER images)"
echo "  Qwen+LoRA: $OUTPUT_OURS/ ($COUNTER images)"
echo ""
echo "Next step: Run evaluation"
echo "  python evaluate_simple.py \\"
echo "      --test_data ./evaluation_samples_with_textbox_0124V1/test_set.json \\"
echo "      --results_dir ./evaluation_samples_with_textbox_0124V1 \\"
echo "      --clip_model ./models/ViT-B-32.pt"
