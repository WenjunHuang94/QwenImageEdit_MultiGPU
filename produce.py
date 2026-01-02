#!/usr/bin/env python3
"""
从 WebDataset 读取数据并进行编码的 producer
支持从 tar 文件中读取 input、output、text_save 三个数据流
"""

import argparse
from pathlib import Path
import torch
from tqdm.auto import tqdm
import random
import os

import webdataset as wds
from diffusers import AutoencoderKLQwenImage
from PIL import Image
import numpy as np
from diffusers import QwenImageEditPipeline
from QwenEdit import calculate_dimensions
import gc
import math


# > tools -----------------------------------------------------------------------------

def calculate_dimensions(target_area, ratio):
    """计算易于被32整除的尺寸"""
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


def get_prompt(use_random=True):
    """
    获取 instruction prompt，支持多种变体以增加训练数据的多样性
    
    针对任务：根据原始图片和文字指令编辑图像（在现有图片上添加/修改内容）
    
    Args:
        use_random: 是否随机选择 instruction（推荐 True，增加数据多样性）
    
    Returns:
        instruction 字符串
    """
    instructions = [
        # 中文变体 - 强调编辑图像
        "根据图片中的文字指令编辑图像",
        "按照文字描述修改图片",
        "根据文字提示在图片上添加内容",
        "按照图片中的文字指令编辑图像",
        "根据文字描述编辑图片",
        "按照文字提示修改图像",
        "根据图片中的文字编辑图像",
        "按照文字指令在图片上添加内容",
        "根据文字描述在图片上进行编辑",
        "按照文字提示编辑图片",
        "根据图片中的文字指令修改图像",
        "按照文字描述在图片上添加元素",
        "根据文字提示编辑图像",
        "按照图片中的文字修改图像",
        "根据文字指令编辑图片",
        
        # 英文变体（如果希望模型支持英文）
        "Edit the image according to the text instruction in the image",
        "Modify the image based on the text description in the image",
        "Edit the image according to the text prompt",
        "Modify the image based on the text instruction",
        "Edit the image following the text description",
        "Apply the text instruction to edit the image",
        "Edit the image according to the text in the image",
        "Modify the image based on the text prompt in the image",
    ]
    
    if use_random:
        return random.choice(instructions)
    else:
        return instructions[0]


def get_shard_files(root_dir, shard_ids=None, max_shards=None):
    """
    获取要处理的 shard 文件列表
    
    Args:
        root_dir: 数据根目录
        shard_ids: 指定的 shard ID 列表（如 ["0", "1", "2"]），如果为 None 则处理所有
        max_shards: 最大处理 shard 数量（如果 shard_ids 为 None）
    
    Returns:
        shard 文件路径列表 [(input_path, output_path, text_path), ...]
    """
    root_path = Path(root_dir)
    input_dir = root_path / "input"
    output_dir = root_path / "output"
    text_dir = root_path / "text_save"
    
    if not input_dir.exists() or not output_dir.exists() or not text_dir.exists():
        raise ValueError(f"数据目录不存在: {root_dir}")
    
    # 获取所有可用的 shard 文件
    all_shard_files = {}
    for shard_file in sorted(input_dir.glob("data_*.tar")):
        # 从文件名提取 shard ID，例如 data_000000.tar -> 000000
        shard_id = shard_file.stem.split("_")[-1]
        
        input_path = shard_file
        output_path = output_dir / shard_file.name
        text_path = text_dir / shard_file.name
        
        # 检查三个文件是否都存在
        if output_path.exists() and text_path.exists():
            all_shard_files[shard_id] = (input_path, output_path, text_path)
    
    print(f"找到 {len(all_shard_files)} 个完整的 shard 文件")
    
    # 根据参数选择要处理的 shard
    if shard_ids is not None:
        # 处理指定的 shard
        selected_shards = []
        for sid in shard_ids:
            sid_str = str(sid).zfill(6)  # 格式化为 6 位数字
            if sid_str in all_shard_files:
                selected_shards.append(all_shard_files[sid_str])
            else:
                print(f"警告: shard {sid_str} 不存在或文件不完整，跳过")
        return selected_shards
    else:
        # 处理所有或前 N 个 shard
        shard_list = list(all_shard_files.values())
        if max_shards is not None:
            shard_list = shard_list[:max_shards]
        return shard_list


def process_webdataset_shard(input_path, output_path, text_path, 
                              text_encoding_pipeline, vae, resizer,
                              txt_cache_dir, img_cache_dir, ctrl_cache_dir,
                              target_area, device, weight_dtype,
                              prompt_with_image, use_random_prompt,
                              processed_keys=None):
    """
    处理单个 shard 文件中的所有样本
    
    Args:
        input_path: input tar 文件路径
        output_path: output tar 文件路径
        text_path: text_save tar 文件路径
        text_encoding_pipeline: 文本编码 pipeline（如果 prompt_with_image）
        vae: VAE 编码器
        resizer: 图像处理器
        txt_cache_dir: 文本 embedding 缓存目录
        img_cache_dir: 图像 embedding 缓存目录
        ctrl_cache_dir: 控制图像 embedding 缓存目录
        target_area: 目标区域大小
        device: 设备
        weight_dtype: 权重数据类型
        prompt_with_image: 是否使用图像进行 prompt 编码
        use_random_prompt: 是否使用随机 prompt
        processed_keys: 已处理的 key 集合（用于去重）
    
    Returns:
        处理的样本数量
    """
    if processed_keys is None:
        processed_keys = set()
    
    # 加载 WebDataset
    ds_input = wds.WebDataset(str(input_path)).decode("pil")
    ds_output = wds.WebDataset(str(output_path)).decode("pil")
    ds_text = wds.WebDataset(str(text_path)).decode()
    
    processed_count = 0
    skipped_count = 0
    
    # 遍历三个数据流，匹配对应的样本
    for sample_input, sample_output, sample_text in zip(ds_input, ds_output, ds_text):
        key = "unknown"  # 初始化 key，用于错误处理
        try:
            # 获取 key（应该相同）
            key_input = sample_input.get("__key__")
            key_output = sample_output.get("__key__")
            key_text = sample_text.get("__key__")
            
            # 检查 key 是否匹配
            if not (key_input == key_output == key_text):
                print(f"警告: key 不匹配 - input:{key_input}, output:{key_output}, text:{key_text}")
                skipped_count += 1
                continue
            
            key = key_input
            
            # 检查是否已处理
            if key in processed_keys:
                skipped_count += 1
                continue
            
            # 提取数据
            # input: 控制图（可能是 jpg, jpeg, png）
            img_ctrl = sample_input.get("jpg") or sample_input.get("jpeg") or sample_input.get("png")
            # output: 结果图（可能是 png, jpg）
            img_output = sample_output.get("png") or sample_output.get("jpg")
            # text: 文本内容
            text_content = sample_text.get("txt")
            
            if img_ctrl is None or img_output is None:
                print(f"警告: {key} 缺少图像数据，跳过")
                skipped_count += 1
                continue
            
            # 转换为 RGB
            if not isinstance(img_ctrl, Image.Image):
                img_ctrl = Image.fromarray(img_ctrl).convert('RGB')
            else:
                img_ctrl = img_ctrl.convert('RGB')
            
            if not isinstance(img_output, Image.Image):
                img_output = Image.fromarray(img_output).convert('RGB')
            else:
                img_output = img_output.convert('RGB')
            
            # 文本编码（如果启用）
            if prompt_with_image and text_encoding_pipeline is not None:
                calculated_width, calculated_height = calculate_dimensions(
                    target_area, img_ctrl.size[0] / img_ctrl.size[1]
                )
                prompt_image = resizer.resize(img_ctrl, calculated_height, calculated_width)
                
                prompt = get_prompt(use_random=use_random_prompt)
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    image=prompt_image,
                    prompt=[prompt],
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                
                txt_cache_path = txt_cache_dir / f"{key}.pt"
                torch.save({
                    'prompt_embeds': prompt_embeds[0].to('cpu'),
                    'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')
                }, txt_cache_path)
            
            # 输出图像编码
            calculated_width, calculated_height = calculate_dimensions(
                target_area, img_output.size[0] / img_output.size[1]
            )
            img_output_resized = resizer.resize(img_output, calculated_height, calculated_width)
            
            img_array = np.array(img_output_resized)
            img_tensor = torch.from_numpy((img_array / 127.5) - 1)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            pixel_values = img_tensor.unsqueeze(2)
            pixel_values = pixel_values.to(dtype=weight_dtype, device=device)
            
            with torch.inference_mode():
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
            
            img_cache_path = img_cache_dir / f"{key}.pt"
            torch.save(pixel_latents, img_cache_path)
            del pixel_latents
            
            # 控制图像编码
            calculated_width, calculated_height = calculate_dimensions(
                target_area, img_ctrl.size[0] / img_ctrl.size[1]
            )
            img_ctrl_resized = resizer.resize(img_ctrl, calculated_height, calculated_width)
            
            img_array = np.array(img_ctrl_resized)
            img_tensor = torch.from_numpy((img_array / 127.5) - 1)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            pixel_values = img_tensor.unsqueeze(2)
            pixel_values = pixel_values.to(dtype=weight_dtype, device=device)
            
            with torch.inference_mode():
                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
            
            ctrl_cache_path = ctrl_cache_dir / f"{key}.pt"
            torch.save(pixel_latents, ctrl_cache_path)
            del pixel_latents
            
            processed_keys.add(key)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"  已处理 {processed_count} 个样本...")
                
        except Exception as e:
            print(f"处理样本 {key} 时出错: {e}")
            skipped_count += 1
            continue
    
    return processed_count, skipped_count


# > main -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="从 WebDataset 读取数据并进行编码")
    parser.add_argument("--pretrained_model", required=True, 
                       help="HuggingFace repo id or local path for Qwen-Image-Edit")
    parser.add_argument("--data_dir", required=True,
                       help="WebDataset 数据根目录（包含 input/, output/, text_save/ 三个子目录）")
    parser.add_argument("--target_area", type=int, default=512*512,
                       help="Approximate target area (H*W) for 32-aligned resize")
    parser.add_argument("--output_dir", required=True,
                       help="Root output directory; caches will be saved under output-dir/cache/")
    parser.add_argument("--prompt_with_image", action="store_true",
                       help="load VLM to rephrase prompt but need to be set to True")
    parser.add_argument("--fixed_prompt", action="store_true",
                       help="Use fixed prompt instead of random (default: random for diversity)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for prompt selection (default: 42 for reproducibility)")
    parser.add_argument("--shard_ids", type=str, nargs="+", default=None,
                       help="要处理的 shard ID 列表（如 0 1 2），如果不指定则处理所有")
    parser.add_argument("--max_shards", type=int, default=None,
                       help="最大处理 shard 数量（仅在未指定 shard_ids 时有效）")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="使用的 GPU 设备")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    weight_dtype = torch.bfloat16
    device = torch.device(args.device)
    
    # 输出目录
    cache_dir = Path(args.output_dir)
    txt_cache_dir = cache_dir / "text_embs"
    img_cache_dir = cache_dir / "img_embs"
    ctrl_cache_dir = cache_dir / "img_embs_control"
    
    cache_dir.mkdir(exist_ok=True)
    txt_cache_dir.mkdir(exist_ok=True)
    img_cache_dir.mkdir(exist_ok=True)
    ctrl_cache_dir.mkdir(exist_ok=True)
    
    # 获取要处理的 shard 文件列表
    shard_files = get_shard_files(args.data_dir, args.shard_ids, args.max_shards)
    
    if not shard_files:
        print("错误: 没有找到可处理的 shard 文件!")
        return
    
    print(f"将处理 {len(shard_files)} 个 shard 文件")
    
    # 初始化模型
    # 总是需要加载 pipeline 来获取 image_processor（用于图像 resize）
    print("加载 pipeline（用于获取 image_processor）...")
    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    resizer = text_encoding_pipeline.image_processor
    
    if args.prompt_with_image:
        # 如果启用 prompt_with_image，将 pipeline 移到 GPU 用于文本编码
        text_encoding_pipeline.to(device)
    else:
        # 如果不需要文本编码，只保留 resizer，释放 pipeline
        text_encoding_pipeline = None
    
    print("加载 VAE...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
    )
    vae.to(device, dtype=weight_dtype)
    
    # 处理所有 shard
    total_processed = 0
    total_skipped = 0
    processed_keys = set()  # 用于去重
    
    for idx, (input_path, output_path, text_path) in enumerate(shard_files):
        print(f"\n处理 shard {idx+1}/{len(shard_files)}: {input_path.name}")
        
        processed_count, skipped_count = process_webdataset_shard(
            input_path, output_path, text_path,
            text_encoding_pipeline, vae, resizer,
            txt_cache_dir, img_cache_dir, ctrl_cache_dir,
            args.target_area, device, weight_dtype,
            args.prompt_with_image, not args.fixed_prompt,
            processed_keys
        )
        
        total_processed += processed_count
        total_skipped += skipped_count
        
        print(f"  Shard {idx+1} 完成: 处理 {processed_count} 个，跳过 {skipped_count} 个")
    
    # 清理
    if text_encoding_pipeline is not None:
        text_encoding_pipeline.to("cpu")
        del text_encoding_pipeline
    
    vae.to("cpu")
    del vae
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n全部完成!")
    print(f"总共处理: {total_processed} 个样本")
    print(f"总共跳过: {total_skipped} 个样本")
    print(f"缓存保存在: {cache_dir.absolute()}")


if __name__ == "__main__":
    main()

