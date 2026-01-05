#!/usr/bin/env python3
"""
合并两个cache目录的数据，用于组合训练
- cache2: 数据集（1）- 纯文本描述+真实图片
- cache5: 数据集（2）- 控制图片+编辑后图片

对于数据集（1），control图像使用原始图片本身（复制img_embs作为control）
"""

import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def merge_caches(cache1_dir, cache2_dir, output_dir):
    """
    合并两个cache目录的数据

    Args:
        cache1_dir: 第一个cache目录（数据集1，如cache2）
        cache2_dir: 第二个cache目录（数据集2，如cache5）
        output_dir: 输出目录（合并后的cache）
        dataset1_use_img_as_control: 对于数据集1，是否使用img_embs作为control（默认True）
    """
    cache1 = Path(cache1_dir)
    cache2 = Path(cache2_dir)
    output = Path(output_dir)

    # 创建输出目录
    txt_output = output / "text_embs"
    img_output = output / "img_embs"
    ctrl_output = output / "img_embs_control"

    txt_output.mkdir(parents=True, exist_ok=True)
    img_output.mkdir(parents=True, exist_ok=True)
    ctrl_output.mkdir(parents=True, exist_ok=True)

    # 获取所有文件
    cache1_txt = sorted(cache1.glob("text_embs/*.pt"))
    cache1_img = sorted(cache1.glob("img_embs/*.pt"))
    cache1_ctrl = sorted(cache1.glob("img_embs_control/*.pt")) if (cache1 / "img_embs_control").exists() else []

    cache2_txt = sorted(cache2.glob("text_embs/*.pt"))
    cache2_img = sorted(cache2.glob("img_embs/*.pt"))
    cache2_ctrl = sorted(cache2.glob("img_embs_control/*.pt")) if (cache2 / "img_embs_control").exists() else []

    print(f"数据集1: {len(cache1_txt)} 个text, {len(cache1_img)} 个img, {len(cache1_ctrl)} 个control")
    print(f"数据集2: {len(cache2_txt)} 个text, {len(cache2_img)} 个img, {len(cache2_ctrl)} 个control")

    # 用于跟踪文件名，避免冲突
    file_counter = 0

    # 处理数据集1
    print("\n处理数据集1...")
    for txt_file, img_file, ctrl_file in tqdm(zip(cache1_txt, cache1_img, cache1_ctrl), total=len(cache1_txt)):
        # 生成新的文件名（添加前缀避免冲突）
        new_name = f"dataset1_{file_counter:06d}.pt"

        # 复制text和img
        shutil.copy2(txt_file, txt_output / new_name)
        shutil.copy2(img_file, img_output / new_name)
        shutil.copy2(ctrl_file, ctrl_output / new_name)

        file_counter += 1

    # 处理数据集2
    print("\n处理数据集2...")
    for txt_file, img_file, ctrl_file in tqdm(zip(cache2_txt, cache2_img, cache2_ctrl), total=len(cache2_txt)):
        # 生成新的文件名
        new_name = f"dataset2_{file_counter:06d}.pt"

        # 复制所有文件
        shutil.copy2(txt_file, txt_output / new_name)
        shutil.copy2(img_file, img_output / new_name)
        shutil.copy2(ctrl_file, ctrl_output / new_name)

        file_counter += 1

    print(f"\n合并完成！")
    print(f"输出目录: {output}")
    print(f"总共: {len(list(txt_output.glob('*.pt')))} 个样本")
    print(f"  - text_embs: {len(list(txt_output.glob('*.pt')))}")
    print(f"  - img_embs: {len(list(img_output.glob('*.pt')))}")
    print(f"  - img_embs_control: {len(list(ctrl_output.glob('*.pt')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并两个cache目录的数据")
    parser.add_argument("--cache1", required=True, help="第一个cache目录（数据集1，如cache2）")
    parser.add_argument("--cache2", required=True, help="第二个cache目录（数据集2，如cache5）")
    parser.add_argument("--output", required=True, help="输出目录（合并后的cache）")
    parser.add_argument("--dataset1_use_img_as_control", action="store_true", default=True,
                        help="对于数据集1，使用img_embs作为control（默认True，适用于文本到图像任务）")

    args = parser.parse_args()

    merge_caches(
        args.cache1,
        args.cache2,
        args.output
    )

