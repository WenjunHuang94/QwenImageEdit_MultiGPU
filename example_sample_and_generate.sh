#!/bin/bash

# ================= 批量抽样并生成专业提示 =================
# 从大数据集中随机抽取指定数量的图片，复制到新文件夹，并生成专业提示

# ================= 使用示例 =================

## 示例 1: 从数据集中抽取 100 张图片并生成提示
#echo "📊 示例 1: 抽取 100 张图片"
#python batch_sample_and_generate.py \
#    --source_folder "./large_dataset" \
#    --num_samples 100 \
#    --seed 42

# 示例 2: 抽取 500 张图片，指定输出路径
echo "📊 示例 2: 抽取 500 张图片，自定义路径"
python batch_sample_and_generate.py \
    --source_folder "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/omniedit/input/" \
    --num_samples 500 \
    --target_folder "./sampled_images_500" \
    --output_folder "./sampled_prompts_500" \
    --seed 42

## 示例 3: 只复制图片，不生成提示（先看看抽样结果）
#echo "📊 示例 3: 只复制图片"
#python batch_sample_and_generate.py \
#    --source_folder "./large_dataset" \
#    --num_samples 200 \
#    --skip_generate
#
## 示例 4: 对已有文件夹生成提示（跳过复制步骤）
#echo "📊 示例 4: 对已有图片生成提示"
#python batch_sample_and_generate.py \
#    --source_folder "./large_dataset" \
#    --num_samples 200 \
#    --target_folder "./existing_images" \
#    --skip_copy

echo "✅ 完成！"
