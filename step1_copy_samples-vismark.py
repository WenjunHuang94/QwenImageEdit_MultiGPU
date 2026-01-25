import os
import shutil
import random
from pathlib import Path

# ================= 配置区域 =================

# 1. 源数据根目录
BASE_DATA_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_vismarker"

# 2. 目标本地目录
LOCAL_BASE_DIR = "./evaluation_samples_vismarker_mixed-0125V1"
LOCAL_INPUT = os.path.join(LOCAL_BASE_DIR, "input")
LOCAL_OUTPUT = os.path.join(LOCAL_BASE_DIR, "output")

# 3. 每个数据集采样的数量
SAMPLES_PER_DATASET = 5

# 【关键优化】为了不卡死，只需找到这么多组有效配对，就停止扫描该文件夹
# 比如设为 100，只要找到 100 对能用的，就不往后找了，直接从中抽 5 个
# 这样避免了遍历几十万个文件
SEARCH_LIMIT = 200

# 4. 数据集列表
datasets = [
    "omniedit_attribute_modification",
    "omniedit_object_swap",
    "omniedit_removal",
    "omniedit_swap",
    "ultraedit_change_color",
    "ultraedit_change_local",
    "ultraedit_replace",
    "ultraedit_turn"
]

# 支持的图片扩展名
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}


# ===========================================

def main():
    print("=" * 80)
    print(f"开始处理 Vismarker 数据集采样 (极速模式)")
    print(f"目标路径: {LOCAL_BASE_DIR}")
    print("=" * 80)

    # 创建目标目录
    Path(LOCAL_INPUT).mkdir(parents=True, exist_ok=True)
    Path(LOCAL_OUTPUT).mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for dataset_name in datasets:
        print(f"\n正在处理数据集: [ {dataset_name} ]")

        src_input_dir = Path(BASE_DATA_DIR) / dataset_name / "input"
        src_output_dir = Path(BASE_DATA_DIR) / dataset_name / "output"

        if not src_input_dir.exists():
            print(f"  ⚠ 目录不存在跳过: {src_input_dir}")
            continue

        valid_pairs = []
        scanned_count = 0

        # 【优化点】使用 os.scandir (比 Path.glob 快)，并且找到够了就停
        try:
            with os.scandir(src_input_dir) as entries:
                for entry in entries:
                    # 1. 基础过滤
                    if not entry.is_file(): continue
                    ext = os.path.splitext(entry.name)[1]
                    if ext not in VALID_EXTS: continue

                    # 2. 检查 output 是否存在 (这是最耗时的步骤)
                    out_file_path = src_output_dir / entry.name

                    if out_file_path.exists():
                        valid_pairs.append((Path(entry.path), out_file_path))
                        scanned_count += 1

                        # 显示进度 (可选)
                        if scanned_count % 50 == 0:
                            print(f"  ...已找到 {scanned_count} 组匹配...")

                    # 【核心优化】只要找到 200 个匹配的，就强制停止，不找了
                    if len(valid_pairs) >= SEARCH_LIMIT:
                        print(f"  ⚡ 达到搜索上限 ({SEARCH_LIMIT})，停止扫描以节省时间。")
                        break

        except Exception as e:
            print(f"  ❌ 扫描出错: {e}")
            continue

        print(f"  最终可用候选池: {len(valid_pairs)} 组")

        if len(valid_pairs) == 0:
            print("  ⚠ 未找到任何匹配图片")
            continue

        # 3. 随机采样
        sample_count = min(SAMPLES_PER_DATASET, len(valid_pairs))
        selected_pairs = random.sample(valid_pairs, sample_count)

        # 4. 复制并重命名
        for inp_path, out_path in selected_pairs:
            new_filename = f"{dataset_name}_{inp_path.name}"

            dest_input_path = Path(LOCAL_INPUT) / new_filename
            dest_output_path = Path(LOCAL_OUTPUT) / new_filename

            try:
                shutil.copy2(inp_path, dest_input_path)
                shutil.copy2(out_path, dest_output_path)
                total_copied += 1
            except Exception as e:
                print(f"    ❌ 复制失败: {e}")

    print("\n" + "=" * 80)
    print(f"所有任务完成！")
    print(f"共复制图片对: {total_copied} 对")
    print("=" * 80)


if __name__ == "__main__":
    main()