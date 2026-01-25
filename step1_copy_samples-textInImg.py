"""
Step 1: 随机选择10张图片并复制到本地
(修复版：文件名一致模式 + 自动清理旧数据)
"""

import os
import shutil
import random
from pathlib import Path

select_num = 10

# ================= 配置区域 =================
# 源路径
INPUT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/omniedit/input"
OUTPUT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/omniedit/output"

# 目标路径
LOCAL_INPUT = "./evaluation_samples_textInImg-0125V1/input"
LOCAL_GT = "./evaluation_samples_textInImg-0125V1/output"
# ===========================================

print("="*80)
print("Step 1: Randomly selecting and copying images")
print("="*80)

# 0. 自动清理旧目录 (防止文件积压)
def clean_dir(path):
    p = Path(path)
    if p.exists():
        shutil.rmtree(path)
    p.mkdir(parents=True, exist_ok=True)

clean_dir(LOCAL_INPUT)
clean_dir(LOCAL_GT)
print("✓ Cleaned old directories")

# 1. 获取所有输入图片
# 修改：不再强制匹配 *_textbox，而是匹配所有图片
input_path_obj = Path(INPUT_DIR)
if not input_path_obj.exists():
    print(f"❌ Error: Input directory not found: {INPUT_DIR}")
    exit()

# 搜索常见的图片格式
input_files = (
    list(input_path_obj.glob("*.png")) +
    list(input_path_obj.glob("*.jpg")) +
    list(input_path_obj.glob("*.jpeg"))
)
print(f"\nTotal images in dataset: {len(input_files)}")

if len(input_files) == 0:
    print("❌ No images found!")
    exit()

# 2. 随机选择
selected_files = random.sample(input_files, min(select_num, len(input_files)))
print(f"Selected {len(selected_files)} images\n")

# 3. 复制文件
copied_count = 0
for input_file in selected_files:
    filename = input_file.name

    # --- A. 复制输入图片 ---
    input_dest = Path(LOCAL_INPUT) / filename
    shutil.copy2(input_file, input_dest)
    print(f"✓ Input: {filename}")

    # --- B. 复制对应的 GT 图片 ---
    # 【关键修改】GT 文件名和 Input 文件名完全一致，不需要 replace
    gt_filename = filename

    gt_file = Path(OUTPUT_DIR) / gt_filename

    if gt_file.exists():
        gt_dest = Path(LOCAL_GT) / filename # 保持同名
        shutil.copy2(gt_file, gt_dest)
        print(f"  ✓ GT Found -> Saved as: {filename}")
        copied_count += 1
    else:
        print(f"  ⚠ GT not found: {gt_filename}")

print(f"\n{'='*80}")
print(f"✓ Successfully copied {copied_count} pairs of images")
print(f"{'='*80}")