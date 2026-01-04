#! /bin/bash
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
TXT="cache2/text_embs/"
IMG="cache2/img_embs/"
CTRL="cache2/img_embs_control/"
OUTPUT="result10/"

LORA_RANK=64
LR=3e-4

# 训练参数建议（针对10000个样本，batch_size=1）：
# - EPOCH: 1-2个epoch通常足够，LoRA训练收敛快
# - MAX_STEP: 10000-20000步（1-2个epoch）
# - WARM_STEP: 总步数的5-10%，用于学习率预热
# - CKP: 每500-1000步保存一次检查点
EPOCH=3
WARM_STEP=200
MAX_STEP=6000
CKP=300

# 可选：从检查点恢复训练（取消注释并设置路径）
# 例如：RESUME_FROM="result/checkpoint-250"
# 如果未设置或注释掉，则从头开始训练
RESUME_FROM="./result4/checkpoint-3000"

# 可选：启用保存最佳模型（设置为 true 或 false，或留空）
# 如果启用，会在每次发现更好的损失时自动保存到 output_dir/best/
SAVE_BEST=true

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
    "$@"