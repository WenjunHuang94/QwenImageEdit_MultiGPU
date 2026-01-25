"""
Step 1: 随机选择图片并复制到本地
(极速优化版：限制扫描数量，防止在大文件夹中卡死)
"""

import os
import shutil
import random
from pathlib import Path

# ================= 配置区域 =================
select_num = 10
PRE_SCAN_LIMIT = 1000  # 【优化】只预读取前1000个文件，避免遍历几十万文件卡死

# 源路径
INPUT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/omniedit/input"
OUTPUT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_wo_textbox/addtion/omniedit/output"

# 目标路径
LOCAL_INPUT = "./evaluation_samples_textInImg-0125V1/input"
LOCAL_GT = "./evaluation_samples_textInImg-0125V1/output"
# ===========================================

print("="*80)
print("Step 1: Randomly selecting and copying images (Lazy Mode)")
print("="*80)

# 0. 自动清理旧目录
def clean_dir(path):
    p = Path(path)
    if p.exists():
        shutil.rmtree(path)
    p.mkdir(parents=True, exist_ok=True)

clean_dir(LOCAL_INPUT)
clean_dir(LOCAL_GT)
print("✓ Cleaned old directories")

input_path_obj = Path(INPUT_DIR)
if not input_path_obj.exists():
    print(f"❌ Error: Input directory not found: {INPUT_DIR}")
    exit()

# 1. 快速获取候选文件名 (限制数量)
candidates = []
valid_exts = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

print(f"Scanning directory (limit: first {PRE_SCAN_LIMIT} files)...")

try:
    # 使用 scandir 迭代器，读够了就停
    with os.scandir(INPUT_DIR) as entries:
        for entry in entries:
            if entry.is_file() and os.path.splitext(entry.name)[1] in valid_exts:
                candidates.append(entry.name)

            # 【核心优化】读够数量立刻停止扫描
            if len(candidates) >= PRE_SCAN_LIMIT:
                print(f"  ⚡ Reached scan limit ({PRE_SCAN_LIMIT}), stopping scan.")
                break
except Exception as e:
    print(f"❌ Error scanning directory: {e}")
    exit()

if not candidates:
    print("❌ No images found in the scanned range!")
    exit()

# 2. 打乱顺序
random.shuffle(candidates)
print(f"Found {len(candidates)} candidates. Selecting valid pairs...\n")

# 3. 寻找有效配并复制 (找到即停)
copied_count = 0

for filename in candidates:
    input_file = Path(INPUT_DIR) / filename
    gt_file = Path(OUTPUT_DIR) / filename  # GT文件名一致

    # 检查 GT 是否存在 (只有这一步会有 IO 开销)
    if gt_file.exists():
        # 复制 Input
        shutil.copy2(input_file, Path(LOCAL_INPUT) / filename)

        # 复制 GT
        shutil.copy2(gt_file, Path(LOCAL_GT) / filename)

        copied_count += 1
        print(f"  [{copied_count}/{select_num}] Copied: {filename}")

    # 【核心优化】凑够了立马退出循环
    if copied_count >= select_num:
        break

print(f"\n{'='*80}")
if copied_count < select_num:
    print(f"⚠ Warning: Only found {copied_count} valid pairs (Target: {select_num})")
else:
    print(f"✓ Successfully copied {copied_count} pairs of images")
print(f"{'='*80}")