import os
import shutil
import random
from pathlib import Path

# ================= 配置区域 =================
select_num = 10
PRE_SCAN_LIMIT = 1000  # 只预读取前1000个文件
INPUT_EXT = ".JPEG"  # 你的输入后缀
OUTPUT_EXT = ".png"  # 你的输出后缀（实际存储的）

# 源路径
INPUT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text-to-image-2M/input"
OUTPUT_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text-to-image-2M/output"

# 目标路径
LOCAL_INPUT = "./evaluation_samples_text-0125V1/input"
LOCAL_GT = "./evaluation_samples_text-0125V1/output"
# ===========================================

print("=" * 80)
print("Step 1: Randomly selecting and copying images (Extension Mapping Mode)")
print("=" * 80)


def clean_dir(path):
    p = Path(path)
    if p.exists():
        shutil.rmtree(path)
    p.mkdir(parents=True, exist_ok=True)


clean_dir(LOCAL_INPUT)
clean_dir(LOCAL_GT)
print("✓ Cleaned old directories")

# 1. 快速获取候选文件名
candidates = []
valid_exts = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}

print(f"Scanning directory (limit: first {PRE_SCAN_LIMIT} files)...")
try:
    with os.scandir(INPUT_DIR) as entries:
        for entry in entries:
            if entry.is_file() and os.path.splitext(entry.name)[1] in valid_exts:
                candidates.append(entry.name)
            if len(candidates) >= PRE_SCAN_LIMIT:
                break
except Exception as e:
    print(f"❌ Error scanning directory: {e}")
    exit()

if not candidates:
    print("❌ No images found!")
    exit()

random.shuffle(candidates)

# 2. 寻找对应后缀的文件并复制
copied_count = 0
for filename in candidates:
    input_file = Path(INPUT_DIR) / filename

    # 【核心逻辑修改】：
    # 1. 获取文件名（不含后缀），例如 "001"
    file_stem = Path(filename).stem
    # 2. 拼接出 output 文件夹中对应的真实文件名，例如 "001.png"
    real_output_name = file_stem + OUTPUT_EXT
    gt_file = Path(OUTPUT_DIR) / real_output_name

    # 检查对应的 PNG 是否存在
    if gt_file.exists():
        # 复制 Input (.JPEG)
        shutil.copy2(input_file, Path(LOCAL_INPUT) / filename)

        # 复制 GT (把 .png 复制过去，并重命名为和 input 一样的名字)
        # 如果你想保持 output 还是 .png，就把下行的 filename 换成 real_output_name
        shutil.copy2(gt_file, Path(LOCAL_GT) / filename)

        copied_count += 1
        print(f"  [{copied_count}/{select_num}] Matched: {filename} <-> {real_output_name}")

    if copied_count >= select_num:
        break

print(f"\n{'=' * 80}")
print(f"✓ Task Finished: {copied_count} pairs copied.")
print(f"Note: Output files have been renamed to match input extensions for consistency.")
print(f"{'=' * 80}")