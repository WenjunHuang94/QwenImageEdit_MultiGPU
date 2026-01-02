#! /bin/bash
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
IMG="/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/output/"
CTRL="/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/ultraedit/input/"
CACHE="cache3/"

RESOLUTION=$((512*512))

python scripts/producer.py \
    --pretrained_model "$MODEL_PATH" \
    --img_dir "$IMG" \
    --control_dir "$CTRL" \
    --target_area $RESOLUTION \
    --output_dir "$CACHE" \
    --prompt_with_image \
    "$@"