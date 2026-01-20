#!/bin/bash

# ================= 配置区域 =================
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
LORA_PATH=""  # 如果不需要 LoRA，设置为 "none"

# 输入输出路径
ORIGINAL_IMAGES="./images_with_text_instructions"
PROMPTS_FOLDER="./images_with_text_instructions_result"
OUTPUT_DIR="./edited_images_local_output"

# 推理参数
TARGET_AREA=$((512*512))
INFER_STEPS=50
CFG_SCALE=6.0
SEED=42

# 负向提示词
NEG_PROMPT=""

# ================= 运行脚本 =================
python batch_local_image_edit.py \
    --pretrained_model "$MODEL_PATH" \
    --lora_weight "$LORA_PATH" \
    --original_images "$ORIGINAL_IMAGES" \
    --prompts_folder "$PROMPTS_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --neg_prompt "$NEG_PROMPT" \
    --target_area $TARGET_AREA \
    --infer_steps $INFER_STEPS \
    --cfg_scale $CFG_SCALE \
    --seed $SEED \
    --skip_existing \
    "$@"
