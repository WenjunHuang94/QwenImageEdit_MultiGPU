#! /bin/bash
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
TXT="cache/text_embs/"
IMG="cache/img_embs/"
CTRL="cache/img_embs_control/"
OUTPUT="result/"

LORA_RANK=64
LR=3e-4

EPOCH=2
WARM_STEP=3
MAX_STEP=10
CKP=1

# 可选：从检查点恢复训练（取消注释并设置路径）
# 例如：RESUME_FROM="result/checkpoint-250"
# 如果未设置或注释掉，则从头开始训练
RESUME_FROM=""

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