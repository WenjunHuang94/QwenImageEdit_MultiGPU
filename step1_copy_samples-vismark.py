import os
import shutil
import random
from pathlib import Path

# ================= 配置区域 =================

# 1. 源数据根目录
BASE_DATA_DIR = "/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_vismarker"

# 2. 目标本地目录 (会自动创建)
LOCAL_BASE_DIR = "./evaluation_samples_vismarker_mixed-0125V1"
LOCAL_INPUT = os.path.join(LOCAL_BASE_DIR, "input")
LOCAL_OUTPUT = os.path.join(LOCAL_BASE_DIR, "output")

# 3. 每个数据集采样的数量
SAMPLES_PER_DATASET = 5

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
    print(f"开始处理 Vismarker 数据集采样")
    print(f"源路径: {BASE_DATA_DIR}")
    print(f"目标路径: {LOCAL_BASE_DIR}")
    print("=" * 80)

    # 创建目标目录
    Path(LOCAL_INPUT).mkdir(parents=True, exist_ok=True)
    Path(LOCAL_OUTPUT).mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for dataset_name in datasets:
        print(f"\n正在处理数据集: [ {dataset_name} ]")

        # 构建当前子数据集的路径
        src_input_dir = Path(BASE_DATA_DIR) / dataset_name / "input"
        src_output_dir = Path(BASE_DATA_DIR) / dataset_name / "output"

        # 检查目录是否存在
        if not src_input_dir.exists() or not src_output_dir.exists():
            print(f"  ⚠ 警告: 目录不存在，跳过: {src_input_dir}")
            return

        # 1. 寻找有效的图片对
        # 假设 input 和 output 的文件名是一致的 (例如 input/a.jpg 和 output/a.jpg)
        input_files = [f for f in src_input_dir.iterdir() if f.suffix in VALID_EXTS]

        valid_pairs = []
        for inp_file in input_files:
            # 对应的 output 文件路径
            out_file = src_output_dir / inp_file.name

            if out_file.exists():
                valid_pairs.append((inp_file, out_file))

        print(f"  找到有效图片对: {len(valid_pairs)} 组")

        if len(valid_pairs) == 0:
            continue

        # 2. 随机采样
        sample_count = min(SAMPLES_PER_DATASET, len(valid_pairs))
        selected_pairs = random.sample(valid_pairs, sample_count)

        print(f"  随机抽取: {sample_count} 组")

        # 3. 复制并重命名
        for inp_path, out_path in selected_pairs:
            # 为了防止不同数据集文件名冲突 (比如不同文件夹里都有 001.jpg)
            # 我们给文件名加上数据集前缀
            # 新文件名: omniedit_swap_001.jpg
            new_filename = f"{dataset_name}_{inp_path.name}"

            dest_input_path = Path(LOCAL_INPUT) / new_filename
            dest_output_path = Path(LOCAL_OUTPUT) / new_filename

            try:
                # 复制文件
                shutil.copy2(inp_path, dest_input_path)
                shutil.copy2(out_path, dest_output_path)

                # print(f"    已复制: {new_filename}")
                total_copied += 1
            except Exception as e:
                print(f"    ❌ 复制失败: {new_filename} - {e}")

    print("\n" + "=" * 80)
    print(f"所有任务完成！")
    print(f"共复制图片对: {total_copied} 对")
    print(f"Input 目录: {os.path.abspath(LOCAL_INPUT)}")
    print(f"Output 目录: {os.path.abspath(LOCAL_OUTPUT)}")
    print("=" * 80)

if __name__ == "__main__":
    main()