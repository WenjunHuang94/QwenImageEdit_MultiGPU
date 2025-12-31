import argparse
from pathlib import Path
import torch
from tqdm.auto import tqdm
import random

from diffusers import AutoencoderKLQwenImage
from PIL import Image
import numpy as np
from diffusers import QwenImageEditPipeline
from QwenEdit import calculate_dimensions
import gc
import math


# > tools -----------------------------------------------------------------------------

def get_image_files(directory):
    """Get all image files (.png and .jpg) from directory recursively."""
    png_files = list(directory.rglob("*.png"))
    jpg_files = list(directory.rglob("*.jpg"))
    return png_files + jpg_files

def get_prompt(use_random=True):
    """
    获取 instruction prompt，支持多种变体以增加训练数据的多样性
    
    针对任务：根据纯文字图片生成真实实景图片
    
    Args:
        use_random: 是否随机选择 instruction（推荐 True，增加数据多样性）
    
    Returns:
        instruction 字符串
    """
    # 文字到图像生成任务相关的 instruction 变体（中英文混合）
    # 强调从文字描述生成真实图像，而非编辑
    instructions = [
        # 中文变体 - 强调生成真实图像
        "根据图片中的文字描述生成真实图像",
        "按照文字描述生成真实实景图片",
        "根据文字描述创建真实图像",
        "按照图片中的文字生成真实场景图片",
        "根据文字内容生成真实图像",
        "按照文字描述生成真实的场景图片",
        "根据图片中的文字生成真实图像",
        "按照文字指令生成真实实景图片",
        "根据文字描述生成图像",
        "按照文字内容创建真实图像",
        
        # 英文变体（如果希望模型支持英文）
        "Generate a realistic image based on the text description in the image",
        "Create a realistic scene image according to the text in the image",
        "Generate a real image from the text description",
        "Generate realistic image based on text instructions",
        "Create realistic image according to text description",
        "Generate real scene image from text description",
    ]
    
    if use_random:
        # 随机选择，增加数据多样性，提高模型泛化能力
        return random.choice(instructions)
    else:
        # 固定使用第一个（用于调试或特定需求）
        return instructions[0]


# > main -----------------------------------------------------------------------------

def main():
    # > config

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", required=True, help="HuggingFace repo id or local path for Qwen-Image-Edit")
    parser.add_argument("--img_dir", required=True, help="Directory containing edited images (e.g., *_edit.png)")
    parser.add_argument("--control_dir", required=True, help="Directory containing control images (e.g., *_textbox.png)")
    parser.add_argument("--target_area", type=int, default=512*512, help="Approximate target area (H*W) for 32-aligned resize")
    parser.add_argument("--output_dir", required=True, help="Root output directory; caches will be saved under output-dir/cache/")
    parser.add_argument("--prompt_with_image", action="store_true", help="load VLM to rephrase prompt but need to be set to True")
    parser.add_argument("--fixed_prompt", action="store_true", help="Use fixed prompt instead of random (default: random for diversity)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt selection (default: 42 for reproducibility)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for quick testing, e.g., 500 or 1000)")
    args = parser.parse_args()
    
    # 设置固定的随机种子（默认 42，保证可复现性）
    random.seed(args.seed)

    weight_dtype = torch.bfloat16  # TODO: 注意原来是float16
    device = torch.device("cuda:1")

    
    # > input----------------------------------------------------------------------------
    img_dir = Path(args.img_dir)
    ctrl_dir = Path(args.control_dir) if args.control_dir else None

    # > output----------------------------------------------------------------------------
    cache_dir = Path(args.output_dir)
    txt_cache_dir = cache_dir /  "text_embs"
    img_cache_dir = cache_dir /  "img_embs"
    ctrl_cache_dir = cache_dir /  "img_embs_control"
    
    cache_dir.mkdir(exist_ok=True)
    txt_cache_dir.mkdir(exist_ok=True)
    img_cache_dir.mkdir(exist_ok=True)
    ctrl_cache_dir.mkdir(exist_ok=True)
    
    # 预先获取并排序文件列表，确保一致性（特别是当使用max_samples时）
    if ctrl_dir is not None:
        all_ctrl_files = sorted(get_image_files(ctrl_dir))
        all_img_files = sorted(get_image_files(img_dir))
        # 如果设置了最大样本数，只处理前N个（用于快速测试）
        if args.max_samples is not None:
            if len(all_ctrl_files) > args.max_samples:
                print(f"限制处理数量：从 {len(all_ctrl_files)} 个样本中选择前 {args.max_samples} 个")
                all_ctrl_files = all_ctrl_files[:args.max_samples]
            if len(all_img_files) > args.max_samples:
                all_img_files = all_img_files[:args.max_samples]
    else:
        all_ctrl_files = []
        all_img_files = sorted(get_image_files(img_dir))
        if args.max_samples is not None and len(all_img_files) > args.max_samples:
            print(f"限制处理数量：从 {len(all_img_files)} 个样本中选择前 {args.max_samples} 个")
            all_img_files = all_img_files[:args.max_samples]

    # > pre-process -----------------------------------------------------------------------------
    
    # > define text_encoding_pipeline VL
    
    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(device)

    # > text encoding
    with torch.inference_mode():

        if args.prompt_with_image:
            for img_name in tqdm(all_ctrl_files):
                img = Image.open(img_name).convert('RGB')
                calculated_width, calculated_height = calculate_dimensions(args.target_area, img.size[0] / img.size[1])
                prompt_image = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)

                prompt = get_prompt(use_random=not args.fixed_prompt)
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    image=prompt_image,
                    prompt=[prompt],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                stem = img_name.stem
                temp = txt_cache_dir / f"{stem}.pt"
                torch.save({'prompt_embeds': prompt_embeds[0].to('cpu'), 'prompt_embeds_mask': prompt_embeds_mask[0].to('cpu')}, temp)

    
    # > image_encoding_pipeline VAE

    resizer = text_encoding_pipeline.image_processor
    text_encoding_pipeline.to("cpu")
    del text_encoding_pipeline
    torch.cuda.empty_cache()
    gc.collect()

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
    )
    vae.to(device, dtype=weight_dtype)

    # > image encoding
    with torch.inference_mode():
        for img_name in tqdm(all_img_files):
            img = Image.open(img_name).convert('RGB')
            calculated_width, calculated_height = calculate_dimensions(args.target_area, img.size[0] / img.size[1])
            img = resizer.resize(img, calculated_height, calculated_width)

            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1).unsqueeze(0)
            pixel_values = img.unsqueeze(2)
            pixel_values = pixel_values.to(dtype=weight_dtype,device=device)

            pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
            stem = img_name.stem
            temp = img_cache_dir / f"{stem}.pt"
            torch.save(pixel_latents, temp)
            del pixel_latents

    # > contorl image encoding
    if ctrl_dir is not None:
        with torch.inference_mode():
            for img_name in tqdm(all_ctrl_files):
                img = Image.open(img_name).convert('RGB')
                calculated_width, calculated_height = calculate_dimensions(args.target_area, img.size[0] / img.size[1])
                img = resizer.resize(img, calculated_height, calculated_width)

                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1).unsqueeze(0)
                pixel_values = img.unsqueeze(2)
                pixel_values = pixel_values.to(dtype=weight_dtype,device=device)

                pixel_latents = vae.encode(pixel_values).latent_dist.sample().to('cpu')[0]
                stem = img_name.stem
                temp = ctrl_cache_dir / f"{stem}.pt"
                torch.save(pixel_latents, temp)
                del pixel_latents

    vae.to('cpu')
    del vae
    
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
