#! /bin/bash
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
IMG="./extracted_samples/shard_000000/output"
CTRL="./extracted_samples/shard_000000/input/"
CACHE="cache2/"

RESOLUTION=$((512*512))

python scripts/producer.py \
    --pretrained_model "$MODEL_PATH" \
    --img_dir "$IMG" \
    --control_dir "$CTRL" \
    --target_area $RESOLUTION \
    --output_dir "$CACHE" \
    --prompt_with_image \
    "$@"