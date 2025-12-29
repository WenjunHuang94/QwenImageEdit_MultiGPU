#! /bin/bash
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
IMG="images2/"
CTRL="control_images2"
CACHE="cache/"

RESOLUTION=$((512*512))

python scripts/producer.py \
    --pretrained_model "$MODEL_PATH" \
    --img_dir "$IMG" \
    --control_dir "$CTRL" \
    --target_area $RESOLUTION \
    --output_dir "$CACHE" \
    --prompt_with_image \
    "$@"