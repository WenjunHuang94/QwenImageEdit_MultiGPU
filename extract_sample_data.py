#!/usr/bin/env python3
"""
从大型数据集中提取一小部分样本数据
从 input, output, text_save 三个文件夹中提取对应的文件
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple


def get_base_name(filename: str) -> str:
    """获取文件名的基础名称（不含扩展名）"""
    return os.path.splitext(filename)[0]


def find_matching_files(input_dir: str, output_dir: str, text_dir: str, 
                       num_samples: int = 100, random_seed: int = 42) -> List[Tuple[str, str, str]]:
    """
    找到三个文件夹中对应的文件
    
    Args:
        input_dir: input文件夹路径（包含.JPEG文件）
        output_dir: output文件夹路径（包含.png文件）
        text_dir: text_save文件夹路径（包含文本文件）
        num_samples: 要提取的样本数量
        random_seed: 随机种子
    
    Returns:
        匹配的文件三元组列表 [(input_path, output_path, text_path), ...]
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 读取input文件夹中的所有.JPEG文件
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input目录不存在: {input_dir}")
    
    input_files = list(input_path.glob("*.JPEG")) + list(input_path.glob("*.jpeg"))
    print(f"找到 {len(input_files)} 个input文件")
    
    # 读取output和text_save文件夹
    output_path = Path(output_dir)
    text_path = Path(text_dir)
    
    if not output_path.exists():
        raise ValueError(f"Output目录不存在: {output_dir}")
    if not text_path.exists():
        raise ValueError(f"Text目录不存在: {text_dir}")
    
    # 创建文件名到路径的映射
    output_map = {get_base_name(f.name): f for f in output_path.glob("*.png")}
    text_map = {get_base_name(f.name): f for f in text_path.glob("*")}
    
    print(f"找到 {len(output_map)} 个output文件")
    print(f"找到 {len(text_map)} 个text文件")
    
    # 找到三个文件夹中都存在的文件
    matched_files = []
    for input_file in input_files:
        base_name = get_base_name(input_file.name)
        
        if base_name in output_map and base_name in text_map:
            matched_files.append((
                str(input_file),
                str(output_map[base_name]),
                str(text_map[base_name])
            ))
    
    print(f"找到 {len(matched_files)} 个完全匹配的文件对")
    
    # 随机选择指定数量的样本
    if len(matched_files) < num_samples:
        print(f"警告: 只找到 {len(matched_files)} 个匹配文件，少于请求的 {num_samples} 个")
        selected_files = matched_files
    else:
        selected_files = random.sample(matched_files, num_samples)
    
    print(f"将提取 {len(selected_files)} 个样本")
    return selected_files


def copy_files(selected_files: List[Tuple[str, str, str]], 
               target_dir: str = "."):
    """
    将选中的文件复制到目标目录
    
    Args:
        selected_files: 文件三元组列表
        target_dir: 目标目录（当前目录）
    """
    target_path = Path(target_dir)
    
    # 创建目标文件夹结构
    input_target = target_path / "input"
    output_target = target_path / "output"
    text_target = target_path / "text_save"
    
    input_target.mkdir(parents=True, exist_ok=True)
    output_target.mkdir(parents=True, exist_ok=True)
    text_target.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始复制文件到: {target_path.absolute()}")
    
    copied_count = 0
    for input_file, output_file, text_file in selected_files:
        try:
            # 复制input文件（保持.JPEG扩展名）
            shutil.copy2(input_file, input_target / Path(input_file).name)
            
            # 复制output文件（保持.png扩展名）
            shutil.copy2(output_file, output_target / Path(output_file).name)
            
            # 复制text文件
            shutil.copy2(text_file, text_target / Path(text_file).name)
            
            copied_count += 1
            if copied_count % 10 == 0:
                print(f"已复制 {copied_count}/{len(selected_files)} 个样本...")
                
        except Exception as e:
            print(f"复制文件时出错 {input_file}: {e}")
    
    print(f"\n完成! 成功复制 {copied_count} 个样本")
    print(f"文件保存在:")
    print(f"  - {input_target.absolute()}")
    print(f"  - {output_target.absolute()}")
    print(f"  - {text_target.absolute()}")


def main():
    # 源数据目录
    source_base = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text-to-image-2M"
    
    input_dir = os.path.join(source_base, "input")
    output_dir = os.path.join(source_base, "output")
    text_dir = os.path.join(source_base, "text_save")
    
    # 要提取的样本数量
    num_samples = 100
    
    print("=" * 60)
    print("数据集样本提取工具")
    print("=" * 60)
    print(f"源目录: {source_base}")
    print(f"目标目录: 当前目录")
    print(f"提取数量: {num_samples} 个样本")
    print("=" * 60)
    
    try:
        # 查找匹配的文件
        matched_files = find_matching_files(
            input_dir, output_dir, text_dir, 
            num_samples=num_samples
        )
        
        if not matched_files:
            print("错误: 没有找到匹配的文件!")
            return
        
        # 复制文件到当前目录
        copy_files(matched_files)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

