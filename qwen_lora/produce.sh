#! /bin/bash
accelerate launch producer.py \
    --pretrained_model "/mnt/output/v-jinpewang/az_workspace/rico_model/qwen_image_edit" \
    --img_dir "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_Accgen/with_textbox/output" \
    --control_dir "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_Accgen/with_textbox/input" \
    # --empty_img "" \
    --target_area 512*512 \
    --output_dir "/storage/v-jinpewang/az_workspace/rico_model/cache" \
    --prompt_with_image