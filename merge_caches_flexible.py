#!/usr/bin/env python3
"""
灵活合并多个cache目录的数据
支持指定每个目录的前缀和提取数量

用法示例:
    python merge_caches_flexible.py \
        --cache cache2:generate:4000 \
        --cache cache5:edit:4000 \
        --cache cache6:annotated_edit:4000 \
        --cache cache_vismarker/omniedit_attribute_modification:attr_mod:1000 \
        --cache cache_vismarker/omniedit_object_swap:obj_swap:1000 \
        --output cache_all_balanced
"""

import argparse
from pathlib import Path
import shutil
import random
from tqdm import tqdm


def parse_cache_entry(entry):
    """
    解析cache条目：cache_path:prefix:max_samples
    例如：cache2:generate:4000
    """
    parts = entry.split(':')
    cache_path = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    max_samples = int(parts[2]) if len(parts) > 2 else None
    return cache_path, prefix, max_samples


def check_files_consistency(txt_files, img_files, ctrl_files, cache_path):
    """
    检查txt_files、img_files、ctrl_files的长度和文件名是否一致
    
    Args:
        txt_files: text_embs文件列表
        img_files: img_embs文件列表
        ctrl_files: img_embs_control文件列表
        cache_path: cache目录路径（用于错误提示）
    
    Returns:
        bool: 是否一致
    """
    # 检查长度
    txt_count = len(txt_files)
    img_count = len(img_files)
    ctrl_count = len(ctrl_files)
    
    if txt_count != img_count or txt_count != ctrl_count:
        print(f"  ❌ 错误: 文件数量不一致！")
        print(f"     text_embs: {txt_count} 个文件")
        print(f"     img_embs: {img_count} 个文件")
        print(f"     img_embs_control: {ctrl_count} 个文件")
        return False
    
    # 检查文件名是否一致（去掉路径和扩展名，只比较文件名）
    for i, (txt_file, img_file, ctrl_file) in enumerate(zip(txt_files, img_files, ctrl_files)):
        txt_name = txt_file.stem  # 文件名（不含扩展名）
        img_name = img_file.stem
        ctrl_name = ctrl_file.stem
        
        if txt_name != img_name or txt_name != ctrl_name:
            print(f"  ❌ 错误: 第 {i+1} 个文件的名字不一致！")
            print(f"     text_embs: {txt_name}")
            print(f"     img_embs: {img_name}")
            print(f"     img_embs_control: {ctrl_name}")
            return False
    
    print(f"  ✓  检查通过: {txt_count} 个文件，文件名一致")
    return True


def merge_caches_flexible(cache_entries, output_dir, seed=42):
    """
    灵活合并多个cache目录
    
    Args:
        cache_entries: cache条目列表，格式为 "cache_path:prefix:max_samples"
        output_dir: 输出目录
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    output = Path(output_dir)
    
    # 创建输出目录
    txt_output = output / "text_embs"
    img_output = output / "img_embs"
    ctrl_output = output / "img_embs_control"
    
    txt_output.mkdir(parents=True, exist_ok=True)
    img_output.mkdir(parents=True, exist_ok=True)
    ctrl_output.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    file_counter = 0

    # 处理每个cache条目
    for idx, entry in enumerate(cache_entries):
        cache_path, prefix, max_samples = parse_cache_entry(entry)
        cache = Path(cache_path)

        print(f"\n[{idx + 1}/{len(cache_entries)}] 处理: {cache_path}")
        print(f"  前缀: {prefix if prefix else '(无)'}")
        print(f"  最大样本数: {max_samples if max_samples else '全部'}")

        if not cache.exists():
            print(f"  ⚠️  警告: cache目录不存在，跳过")
            continue

        # 获取所有文件
        txt_files = sorted(cache.glob("text_embs/*.pt"))
        img_files = sorted(cache.glob("img_embs/*.pt"))
        ctrl_files = sorted(cache.glob("img_embs_control/*.pt")) if (cache / "img_embs_control").exists() else []

        # 检查文件一致性问题
        if not check_files_consistency(txt_files, img_files, ctrl_files, cache_path):
            print(f"  ⚠️  警告: {cache_path} 的文件不一致!!!")
            raise Exception(f"{cache_path} 的文件不一致")
    
    print("==========================================")
    print("开始合并多个cache目录")
    print("==========================================")
    
    # 处理每个cache条目
    for idx, entry in enumerate(cache_entries):
        cache_path, prefix, max_samples = parse_cache_entry(entry)
        cache = Path(cache_path)
        
        print(f"\n[{idx+1}/{len(cache_entries)}] 处理: {cache_path}")
        print(f"  前缀: {prefix if prefix else '(无)'}")
        print(f"  最大样本数: {max_samples if max_samples else '全部'}")
        
        if not cache.exists():
            print(f"  ⚠️  警告: cache目录不存在，跳过")
            continue
        
        # 获取所有文件
        txt_files = sorted(cache.glob("text_embs/*.pt"))
        img_files = sorted(cache.glob("img_embs/*.pt"))
        ctrl_files = sorted(cache.glob("img_embs_control/*.pt")) if (cache / "img_embs_control").exists() else []
        
        # 确保文件数量一致（虽然已经检查过，但保留此逻辑以防万一）
        min_count = min(len(txt_files), len(img_files), len(ctrl_files))
        txt_files = txt_files[:min_count]
        img_files = img_files[:min_count]
        ctrl_files = ctrl_files[:min_count]
        
        print(f"  📊 原始样本数: {min_count}")
        
        # 应用max_samples限制
        if max_samples is not None and min_count > max_samples:
            # 随机采样
            indices = random.sample(range(min_count), max_samples)
            txt_files = [txt_files[i] for i in indices]
            img_files = [img_files[i] for i in indices]
            ctrl_files = [ctrl_files[i] for i in indices]
            actual_count = max_samples
            print(f"  ✂️  随机采样后: {actual_count} 个样本")
        else:
            actual_count = min_count
            print(f"  ✓  使用全部: {actual_count} 个样本")
        
        # 复制文件
        for txt_file, img_file, ctrl_file in tqdm(zip(txt_files, img_files, ctrl_files),
                                                    total=len(txt_files),
                                                    desc=f"  复制 {prefix if prefix else cache.name}"):
            # 生成新的文件名（使用前缀避免冲突）
            if prefix:
                new_name = f"{prefix}_{file_counter:06d}.pt"
            else:
                new_name = f"dataset{idx+1}_{file_counter:06d}.pt"
            
            shutil.copy2(txt_file, txt_output / new_name)
            shutil.copy2(img_file, img_output / new_name)
            shutil.copy2(ctrl_file, ctrl_output / new_name)
            
            file_counter += 1
            total_samples += 1
    
    print(f"\n==========================================")
    print(f"合并完成！")
    print(f"==========================================")
    print(f"输出目录: {output}")
    print(f"总共: {total_samples} 个样本")
    print(f"  - text_embs: {len(list(txt_output.glob('*.pt')))} 个文件")
    print(f"  - img_embs: {len(list(img_output.glob('*.pt')))} 个文件")
    print(f"  - img_embs_control: {len(list(ctrl_output.glob('*.pt')))} 个文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="灵活合并多个cache目录的数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python merge_caches_flexible.py \\
      --cache cache2:generate:4000 \\
      --cache cache5:edit:4000 \\
      --output cache_combined

  # 复杂用法（多个cache，不同前缀和数量）
  python merge_caches_flexible.py \\
      --cache cache2:generate:4000 \\
      --cache cache5:edit:4000 \\
      --cache cache6:annotated:4000 \\
      --cache cache_vismarker/dataset1:ds1:1000 \\
      --cache cache_vismarker/dataset2:ds2:1000 \\
      --output cache_all
        """
    )
    
    parser.add_argument("--cache", action="append", required=True,
                       help="cache条目，格式：cache_path:prefix:max_samples (可多次指定)")
    parser.add_argument("--output", required=True,
                       help="输出目录（合并后的cache）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（用于采样，默认42）")
    
    args = parser.parse_args()
    
    merge_caches_flexible(
        args.cache,
        args.output,
        args.seed
    )

