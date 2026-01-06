#! /bin/bash
# 重新混合所有数据训练（推荐方案）
# 包括：generate + edit + annotated_edit + pointer_edit

MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"

# 使用merge_all_datasets.sh生成的混合数据cache
TXT="cache_all_combined/text_embs/"
IMG="cache_all_combined/img_embs/"
CTRL="cache_all_combined/img_embs_control/"
OUTPUT="result_all_retrain/"

LORA_RANK=128  # 推荐128：4个任务需要更多容量，pointer_edit任务复杂需要精细学习
LR=3e-4      # 从头训练可以使用正常学习率

# 训练参数（从头训练）：
# - EPOCH: 2-3个epoch通常足够
# - MAX_STEP: 根据总数据量调整
# - WARM_STEP: 总步数的5-10%
# - CKP: 每500-1000步保存一次检查点
EPOCH=2
WARM_STEP=500
MAX_STEP=30000  # 根据实际数据量调整（假设总共15000样本）
CKP=500

# 从头训练，不使用resume
# RESUME_FROM=""  # 不设置，从头开始

# 可选：启用保存最佳模型
SAVE_BEST=true

# 可选：设置随机种子（用于可复现性）
SEED=42

echo "=========================================="
echo "重新混合所有数据训练（从头开始）"
echo "=========================================="
echo "学习率: $LR"
echo "训练步数: $MAX_STEP"
echo "Rank: $LORA_RANK"
echo "=========================================="
echo ""

python scripts/pp_consumer.py \
    --output_dir "$OUTPUT" \
    --logging_dir "./logger" \
    --pretrained_model "$MODEL_PATH" \
    --rank $LORA_RANK \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --epochs $EPOCH \
    --max_train_steps $MAX_STEP \
    --lr_warmup_steps $WARM_STEP \
    --lr_scheduler constant_with_warmup \
    --learning_rate $LR \
    --num_workers 4 \
    --max_grad_norm 1.0 \
    --checkpointing_steps $CKP \
    --txt_cache_dir "$TXT" \
    --img_cache_dir "$IMG" \
    --control_img_cache_dir "$CTRL" \
    ${SAVE_BEST:+--save_best_model} \
    ${SEED:+--seed $SEED} \
    "$@"

