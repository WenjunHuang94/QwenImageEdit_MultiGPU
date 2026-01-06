#!/usr/bin/env python3
"""
重命名cache6/img_embs目录下的文件
将 xxx_edited.pt 重命名为 xxx_textbox.pt

用法:
    python rename_img_embs.py --cache_dir cache6
    python rename_img_embs.py --cache_dir cache6 --dry_run  # 只查看，不实际重命名
"""

import argparse
from pathlib import Path


def rename_img_embs(cache_dir, dry_run=False, suffix_from="_edited", suffix_to="_textbox"):
    """
    重命名img_embs目录下的文件
    
    Args:
        cache_dir: cache目录路径
        dry_run: 如果为True，只显示将要重命名的文件，不实际重命名
        suffix_from: 源后缀（如"_edited"）
        suffix_to: 目标后缀（如"_textbox"）
    """
    cache = Path(cache_dir)
    img_cache_dir = cache / "img_embs"
    
    if not img_cache_dir.exists():
        raise ValueError(f"img_embs目录不存在: {img_cache_dir}")
    
    # 获取所有以suffix_from结尾的文件
    files_to_rename = []
    for img_file in img_cache_dir.glob("*.pt"):
        if img_file.stem.endswith(suffix_from):
            base_name = img_file.stem[:-len(suffix_from)]  # 去掉suffix_from
            new_name = base_name + suffix_to + ".pt"
            new_file = img_cache_dir / new_name
            
            files_to_rename.append({
                'old_file': img_file,
                'new_file': new_file,
                'old_name': img_file.name,
                'new_name': new_name
            })
    
    if len(files_to_rename) == 0:
        print(f"没有找到以 '{suffix_from}' 结尾的文件")
        return
    
    print(f"找到 {len(files_to_rename)} 个需要重命名的文件")
    
    if dry_run:
        print("\n=== 干运行模式：只显示将要重命名的文件 ===")
        for item in files_to_rename[:10]:  # 只显示前10个
            print(f"  {item['old_name']} -> {item['new_name']}")
        if len(files_to_rename) > 10:
            print(f"  ... 还有 {len(files_to_rename) - 10} 个文件")
        print(f"\n总共将重命名 {len(files_to_rename)} 个文件")
        print("使用不带 --dry_run 参数来实际执行重命名")
        return
    
    # 实际重命名
    print("\n开始重命名...")
    renamed_count = 0
    skipped_count = 0
    
    for item in files_to_rename:
        old_file = item['old_file']
        new_file = item['new_file']
        
        # 检查目标文件是否已存在
        if new_file.exists():
            print(f"警告: 目标文件已存在，跳过: {new_file.name}")
            skipped_count += 1
            continue
        
        try:
            old_file.rename(new_file)
            renamed_count += 1
            if renamed_count % 100 == 0:
                print(f"  已重命名 {renamed_count} 个文件...")
        except Exception as e:
            print(f"错误: 重命名失败 {old_file.name} -> {new_file.name}: {e}")
            skipped_count += 1
    
    print(f"\n重命名完成!")
    print(f"  成功: {renamed_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重命名img_embs目录下的文件")
    parser.add_argument("--cache_dir", required=True,
                       help="cache目录路径（如cache6）")
    parser.add_argument("--dry_run", action="store_true",
                       help="干运行模式：只显示将要重命名的文件，不实际重命名")
    parser.add_argument("--suffix_from", type=str, default="_edited",
                       help="源后缀（默认：_edited）")
    parser.add_argument("--suffix_to", type=str, default="_textbox",
                       help="目标后缀（默认：_textbox）")
    
    args = parser.parse_args()
    
    rename_img_embs(
        args.cache_dir,
        dry_run=args.dry_run,
        suffix_from=args.suffix_from,
        suffix_to=args.suffix_to
    )

