#! /bin/bash
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
TXT="cache_all_balanced/text_embs/"
IMG="cache_all_balanced/img_embs/"
CTRL="cache_all_balanced/img_embs_control/"
OUTPUT="result15-rank64_cache_all_balanced/"

LORA_RANK=64
LR=1e-4  # 降低学习率！基于checkpoint-3000训练，需要保护已有的优秀权重

# 训练参数建议（基于checkpoint-3000继续训练混合数据集）：
# - LR: 1e-4（比从零训练的3e-4低，避免破坏已有的生成能力）
# - EPOCH: 1-2个epoch通常足够，LoRA训练收敛快
# - MAX_STEP: 10000-20000步（1-2个epoch）
# - WARM_STEP: 1000-2000步，让模型平滑适应混合数据（包含新的Edit任务）
# - CKP: 每200-500步保存一次检查点，方便观察能力变化
EPOCH=2
WARM_STEP=500  # 增加warmup，让模型适应新的数据分布
MAX_STEP=40000
CKP=200

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