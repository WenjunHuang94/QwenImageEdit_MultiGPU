#! /bin/bash
# 使用合并后的vismarker数据集训练LoRA

MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
TXT="cache_vismarker_combined/text_embs/"
IMG="cache_vismarker_combined/img_embs/"
CTRL="cache_vismarker_combined/img_embs_control/"
OUTPUT="result_vismarker/"

LORA_RANK=64
LR=3e-4

# 训练参数建议（针对合并后的数据集）：
# - EPOCH: 1-2个epoch通常足够，LoRA训练收敛快
# - MAX_STEP: 根据合并后的数据量调整
#   如果8个数据集，每个2000样本，总共16000样本
#   假设batch_size=1，1个epoch需要16000步
# - WARM_STEP: 总步数的5-10%，用于学习率预热
# - CKP: 每500-1000步保存一次检查点
EPOCH=2
WARM_STEP=500
MAX_STEP=32000  # 16000样本 * 2 epochs，根据实际数据量调整
CKP=500

# 可选：从检查点恢复训练（取消注释并设置路径）
# RESUME_FROM="./result_vismarker/checkpoint-5000"

# 可选：启用保存最佳模型
SAVE_BEST=true

# 可选：设置随机种子（用于可复现性）
SEED=42

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
    ${RESUME_FROM:+--resume_from_checkpoint "$RESUME_FROM"} \
    ${SAVE_BEST:+--save_best_model} \
    ${SEED:+--seed $SEED} \
    "$@"

