"""
Step 1: 随机选择10张图片并复制到本地 (修复文件名匹配逻辑)
"""

import os
import shutil
import random
from pathlib import Path

select_num = 10

# 源路径
INPUT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_Accgen/with_textbox/input"
OUTPUT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_Accgen/with_textbox/output"

# 目标路径
LOCAL_INPUT = "./evaluation_samples_with_textbox_0124V1/input"
LOCAL_GT = "./evaluation_samples_with_textbox_0124V1/output"

print("="*80)
print("Step 1: Randomly selecting and copying 10 images")
print("="*80)

# 创建目录
Path(LOCAL_INPUT).mkdir(parents=True, exist_ok=True)
Path(LOCAL_GT).mkdir(parents=True, exist_ok=True)

# 获取所有输入图片 (匹配 _textbox.png 或 .jpg)
input_files = list(Path(INPUT_DIR).glob("*_textbox.png")) + list(Path(INPUT_DIR).glob("*_textbox.jpg"))
print(f"\nTotal images in dataset: {len(input_files)}")

# 随机选择10张
selected_files = random.sample(input_files, min(select_num, len(input_files)))
print(f"Selected {len(selected_files)} images\n")

# 复制文件
copied_count = 0
for input_file in selected_files:
    filename = input_file.name # 例如: 001_textbox.png

    # 1. 复制输入图片
    input_dest = Path(LOCAL_INPUT) / filename
    shutil.copy2(input_file, input_dest)
    print(f"✓ Input: {filename}")

    # 2. 构造对应的 GT 文件名 (将 _textbox 替换为 _edited)
    # 假设输入是 xxxx_textbox.png -> 输出对应 xxxx_edited.png
    gt_filename = filename.replace("_textbox", "_edited")
    gt_file = Path(OUTPUT_DIR) / gt_filename

    if gt_file.exists():
        # 3. 复制并重命名：保持和 Input 的名字一模一样
        gt_dest = Path(LOCAL_GT) / filename  # 这里依然使用 input 的 filename
        shutil.copy2(gt_file, gt_dest)
        print(f"  ✓ GT Found ({gt_filename}) -> Saved as: {filename}")
        copied_count += 1
    else:
        print(f"  ⚠ GT not found: {gt_filename}")

print(f"\n{'='*80}")
print(f"✓ Copied {copied_count} pairs of images")
print(f"{'='*80}")