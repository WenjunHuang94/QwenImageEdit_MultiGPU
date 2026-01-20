#!/bin/bash

# ================= 统一批量生成 Prompt 脚本 =================
# 支持三种标注类型：text（纯文本）、arrow（箭头+文本）、box（框+文本）

# API Key（可选，默认使用脚本中的）
API_KEY="sk-1649650cd15847b685cd57def55a7a56"

# API 调用间隔（秒），避免限流
RETRY_DELAY=0.5

# ================= 使用示例 =================

# 示例 1: 处理纯文本指令数据
#echo "📝 处理纯文本指令数据..."
#python batch_generate_prompts_unified.py \
#    --annotation_type text \
#    --input_folder "./images_with_text_instructions" \
#    --output_folder "./images_with_text_instructions_result" \
#    --api_key "$API_KEY" \
#    --retry_delay $RETRY_DELAY

# 示例 2: 处理箭头+文本指令数据
echo "🎯 处理箭头+文本指令数据..."
python batch_generate_prompts_unified.py \
    --annotation_type arrow \
    --input_folder "./images_with_arrow_instructions" \
    --output_folder "./images_with_arrow_instructions_result" \
    --api_key "$API_KEY" \
    --retry_delay $RETRY_DELAY

# 示例 3: 处理框+文本指令数据
echo "📦 处理框+文本指令数据..."
python batch_generate_prompts_unified.py \
    --annotation_type box \
    --input_folder "./images_with_box_instructions" \
    --output_folder "./images_with_box_instructions_result" \
    --api_key "$API_KEY" \
    --retry_delay $RETRY_DELAY

echo "✅ 全部处理完成！"
