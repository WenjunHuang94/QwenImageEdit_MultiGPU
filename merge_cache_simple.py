#!/usr/bin/env python3
"""
简单合并cache目录：将源cache目录的所有文件复制到目标cache目录
自动处理文件名冲突（通过添加序号）
"""

import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def merge_cache_simple(source_cache, target_cache, prefix=None):
    """
    将源cache目录的文件合并到目标cache目录
    
    Args:
        source_cache: 源cache目录路径
        target_cache: 目标cache目录路径
        prefix: 必需的文件名前缀（用于区分来源，避免文件名冲突）
    """
    source = Path(source_cache)
    target = Path(target_cache)
    
    # 检查前缀是否提供
    if not prefix or prefix.strip() == "":
        raise ValueError("错误: 必须提供 --prefix 参数（文件名前缀）")
    
    if not source.exists():
        raise ValueError(f"源目录不存在: {source}")
    
    # 定义三个子目录
    subdirs = ["text_embs", "img_embs", "img_embs_control"]
    
    # 创建目标目录
    for subdir in subdirs:
        (target / subdir).mkdir(parents=True, exist_ok=True)
    
    # 获取源目录的文件列表
    source_txt_files = sorted((source / "text_embs").glob("*.pt"))
    source_img_files = sorted((source / "img_embs").glob("*.pt"))
    source_ctrl_files = sorted((source / "img_embs_control").glob("*.pt"))
    
    # 检查文件数量是否一致
    if len(source_txt_files) != len(source_img_files) or len(source_txt_files) != len(source_ctrl_files):
        error_msg = f"⚠️ 错误: 文件数量不一致\n" \
                    f"   text_embs: {len(source_txt_files)}\n" \
                    f"   img_embs: {len(source_img_files)}\n" \
                    f"   img_embs_control: {len(source_ctrl_files)}"

        # 直接抛出异常，停止程序执行
        raise ValueError(error_msg)
    
    # 检查文件名是否一致（确保三个目录中的文件能正确匹配）
    for txt_file, img_file, ctrl_file in zip(source_txt_files, source_img_files, source_ctrl_files):
        if txt_file.stem != img_file.stem or txt_file.stem != ctrl_file.stem:
            raise ValueError(f"文件名不一致: text={txt_file.stem}, img={img_file.stem}, ctrl={ctrl_file.stem}")
    
    # 使用前缀 + 原始文件名格式
    print(f"使用前缀 '{prefix}'，文件名格式: {prefix}_原始文件名.pt（不会与已有文件冲突）")
    
    print(f"源目录: {source}")
    print(f"目标目录: {target}")
    print(f"文件数量: {len(source_txt_files)}")
    
    # 复制文件
    for i, (txt_file, img_file, ctrl_file) in enumerate(tqdm(
        zip(source_txt_files, source_img_files, source_ctrl_files),
        total=len(source_txt_files),
        desc="复制文件"
    )):
        # 使用前缀 + 原始文件名
        original_name = txt_file.stem  # 三个文件的stem应该相同
        new_name = f"{prefix}_{original_name}.pt"
        
        # 复制文件
        shutil.copy2(txt_file, target / "text_embs" / new_name)
        shutil.copy2(img_file, target / "img_embs" / new_name)
        shutil.copy2(ctrl_file, target / "img_embs_control" / new_name)
    
    print(f"\n合并完成！")
    print(f"已复制 {len(source_txt_files)} 个样本到 {target}")
    print(f"  - text_embs: {len(list((target / 'text_embs').glob('*.pt')))} 个文件")
    print(f"  - img_embs: {len(list((target / 'img_embs').glob('*.pt')))} 个文件")
    print(f"  - img_embs_control: {len(list((target / 'img_embs_control').glob('*.pt')))} 个文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="简单合并cache目录：将源cache目录的所有文件复制到目标cache目录"
    )
    parser.add_argument("--source", required=True,
                       help="源cache目录路径（如 cache6-2）")
    parser.add_argument("--target", required=True,
                       help="目标cache目录路径（如 cache_all_balanced）")
    parser.add_argument("--prefix", type=str, required=True,
                       help="必需的文件名前缀（如 annotated_edit2），用于避免文件名冲突")
    
    args = parser.parse_args()
    
    merge_cache_simple(args.source, args.target, args.prefix)

