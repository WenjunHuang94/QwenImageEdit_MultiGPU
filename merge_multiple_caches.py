#!/usr/bin/env python3
"""
合并多个cache目录的数据
用于将多个数据集的cache合并成一个用于训练
"""

import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def merge_multiple_caches(cache_dirs, output_dir, max_samples_per_dataset=None):
    """
    合并多个cache目录到输出目录
    
    Args:
        cache_dirs: cache目录列表
        output_dir: 输出目录
        max_samples_per_dataset: 每个数据集的最大样本数（None表示处理全部）
    """
    output = Path(output_dir)
    
    # 创建输出目录结构
    txt_output = output / "text_embs"
    img_output = output / "img_embs"
    ctrl_output = output / "img_embs_control"
    
    txt_output.mkdir(parents=True, exist_ok=True)
    img_output.mkdir(parents=True, exist_ok=True)
    ctrl_output.mkdir(parents=True, exist_ok=True)
    
    file_counter = 0
    total_samples = 0
    
    # 处理每个cache目录
    for idx, cache_dir in enumerate(cache_dirs):
        cache = Path(cache_dir)
        dataset_name = cache.name
        
        print(f"\n处理数据集 {idx+1}/{len(cache_dirs)}: {dataset_name}")
        
        # 获取所有文件
        txt_files = sorted(cache.glob("text_embs/*.pt"))
        img_files = sorted(cache.glob("img_embs/*.pt"))
        ctrl_files = sorted(cache.glob("img_embs_control/*.pt")) if (cache / "img_embs_control").exists() else []
        
        # 限制样本数量
        if max_samples_per_dataset is not None:
            txt_files = txt_files[:max_samples_per_dataset]
            img_files = img_files[:max_samples_per_dataset]
            if len(ctrl_files) > 0:
                ctrl_files = ctrl_files[:max_samples_per_dataset]
        
        print(f"  样本数: {len(txt_files)}")
        
        # 如果ctrl_files为空，使用img_files作为control（适用于某些数据集）
        if len(ctrl_files) == 0:
            print(f"  注意: 没有control文件，将使用img_embs作为control")
            ctrl_files = img_files
        
        # 确保文件数量一致
        min_count = min(len(txt_files), len(img_files), len(ctrl_files))
        txt_files = txt_files[:min_count]
        img_files = img_files[:min_count]
        ctrl_files = ctrl_files[:min_count]
        
        # 复制文件
        for txt_file, img_file, ctrl_file in tqdm(zip(txt_files, img_files, ctrl_files), 
                                                    total=min_count, 
                                                    desc=f"  复制 {dataset_name}"):
            new_name = f"dataset{idx+1}_{file_counter:06d}.pt"
            
            shutil.copy2(txt_file, txt_output / new_name)
            shutil.copy2(img_file, img_output / new_name)
            shutil.copy2(ctrl_file, ctrl_output / new_name)
            
            file_counter += 1
            total_samples += 1
    
    print(f"\n合并完成！")
    print(f"输出目录: {output}")
    print(f"总共: {total_samples} 个样本")
    print(f"  - text_embs: {len(list(txt_output.glob('*.pt')))}")
    print(f"  - img_embs: {len(list(img_output.glob('*.pt')))}")
    print(f"  - img_embs_control: {len(list(ctrl_output.glob('*.pt')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个cache目录的数据")
    parser.add_argument("--cache_dirs", nargs="+", required=True,
                       help="cache目录列表（用空格分隔）")
    parser.add_argument("--output", required=True,
                       help="输出目录（合并后的cache）")
    parser.add_argument("--max_samples_per_dataset", type=int, default=None,
                       help="每个数据集的最大样本数（默认None，处理全部）")
    
    args = parser.parse_args()
    
    merge_multiple_caches(
        args.cache_dirs,
        args.output,
        args.max_samples_per_dataset
    )

