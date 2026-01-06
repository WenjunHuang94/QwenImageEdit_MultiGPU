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

def get_prompt(use_random=True, prompt_type="edit"):
    """
    获取 instruction prompt，支持多种变体以增加训练数据的多样性
    
    Args:
        use_random: 是否随机选择 instruction（推荐 True，增加数据多样性）
        prompt_type: prompt类型
            - "edit": 图像编辑任务（根据原始图片和文字指令编辑图像）
            - "generate": 文本描述生成图片任务（根据文字描述绘画出真实图片）
    
    Returns:
        instruction 字符串
    """
    if prompt_type == "generate":
        # 文本描述生成图片任务相关的 instruction 变体
        instructions = [
            # 中文变体 - 强调根据文字描述生成图片
            "根据图片文字描述绘画出真实图片",
            "根据文字描述生成真实图片",
            "按照文字描述绘制真实图片",
            "根据图片中的文字描述生成图片",
            "按照文字提示绘画出真实图片",
            "根据文字描述创作真实图片",
            "按照图片文字描述生成真实图像",
            "根据文字提示绘制真实图片",
            "按照文字描述生成图片",
            "根据图片中的文字描述绘画图片",
            "按照文字提示生成真实图片",
            "根据文字描述绘制图片",
            "按照图片文字描述创作真实图片",
            "根据文字提示生成图片",
            "按照文字描述绘画图片",
            
            # 英文变体
            "Generate a realistic image based on the text description in the image",
            "Draw a realistic image according to the text description",
            "Create a realistic image from the text description",
            "Generate an image based on the text prompt in the image",
            "Draw a realistic picture according to the text description",
            "Create a picture from the text description in the image",
        ]
    elif prompt_type == "annotated_edit":
        # 针对带框/标注的图像编辑指令
        instructions = [
            # 中文变体 - 强调标注位置和内容
            "根据图片中的框标注和文字指令修改图像",
            "在图片中标注的指定位置添加文字描述的内容",
            "参考图中的颜色框标注，在对应位置生成目标物体",
            "按照标注框旁边的文字提示，修改图片中的指定区域",
            "根据标注指示，在图片对应位置进行绘画",
            "按照图中的框选区域和文字描述编辑图像",
            "根据图片中的标注框位置，绘画出文字描述的实景内容",
            "在图中框出的位置，按照文字指令进行修改",
            "结合图中的位置标注和文字提示，生成真实的场景",

            # 英文变体
            "Modify the image at the annotated location according to the text instruction",
            "Edit the specified area in the image based on the colored box and text prompt",
            "Add the object described by the text at the position indicated by the box",
            "Based on the annotations in the image, edit the specific region following the text",
            "Generate the content in the boxed area as described by the text prompt",
            "Follow the visual markers and text instructions to modify the image",
        ]
    elif prompt_type == "doodle_edit":
        # 针对涂鸦引导的任务
        instructions = [
            # 中文变体 - 强调将涂鸦转化为真实感
            "将图片中的涂鸦内容绘画成真实的物体",
            "根据图中的手绘涂鸦生成对应的真实图像内容",
            "参考图片中的涂鸦形状，在对应位置绘制出真实的物体",
            "按照图中的涂鸦痕迹，将其修改为具有真实质感的实景",
            "根据图片中的手绘提示，在对应位置生成真实的景象",
            "根据图片中的涂鸦引导，绘画出文字描述的实物",
            "将图中的简单涂鸦转化为细腻的真实效果",
            "参考涂鸦的轮廓，在图片上生成真实的装饰或物体",

            # 英文变体
            "Convert the doodle in the image into a realistic object",
            "Generate realistic content based on the hand-drawn sketches in the image",
            "Turn the simple drawings in the image into realistic photographic elements",
            "Refine the doodles in the image into real-life objects following the text prompt",
            "Translate the visual sketches into realistic textures and shapes",
            "Use the provided doodles as a guide to paint realistic objects on the image"
        ]
    elif prompt_type == "pointer_edit":
        instructions = [
            # 类别 1：原始通用 Edit 指令（完全不提箭头，迫使模型自动关联视觉信号）
            "根据图片中的文字指令编辑图像",
            "按照文字描述修改图片内容",
            "根据文字提示在图片上进行编辑",
            "Edit the image following the text description",
            "Modify the image based on the text prompt",

            # 类别 2：显式空间引导指令（强力建立坐标感）
            "请识别图中的箭头指向，按照旁边的文字要求修改对应区域",
            "根据指示箭头和文字操作描述，对图片进行实景化修改",
            "根据图中箭头标记的位置，执行文字描述的编辑任务",
            "Follow the visual pointer and text to edit the image",
            "Execute the instruction written next to the arrow",

            # 类别 3：中性标注引导指令（平衡态）
            "根据图片里的标注信息，把对应的物体换成文字描述的样子",
            "参考图中的提示文字和指向，完成图像编辑",
            "按照图片中的手写文字指令，对指定物体进行修改",
            "Look at the handwritten instructions in the image to perform the edit",
            "Based on the annotations, update the pointed part of the image"
        ]
    else:
        # 图像编辑任务相关的 instruction 变体（中英文混合）
        # 强调根据文字指令编辑/修改现有图像，而非从零生成
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
    parser.add_argument("--prompt_type", type=str, default="edit", choices=["edit", "generate", "annotated_edit", "doodle_edit", "pointer_edit"],
                       help="Prompt type: 'edit' for image editing, 'generate' for text-to-image generation (default: edit)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt selection (default: 42 for reproducibility)")
    parser.add_argument("--max_samples", type=int, default=4000, help="Maximum number of samples to process (for quick testing, e.g., 500 or 1000)")
    parser.add_argument("--shuffle_input", action="store_true", help="Shuffle input files before selection")
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
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    txt_cache_dir.mkdir(parents=True, exist_ok=True)
    img_cache_dir.mkdir(parents=True, exist_ok=True)
    ctrl_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 预先获取并排序文件列表，确保一致性（特别是当使用max_samples时）
    if ctrl_dir is not None:
        all_ctrl_files = sorted(get_image_files(ctrl_dir))
        all_img_files = sorted(get_image_files(img_dir))

        # >>>>> 新增代码开始 >>>>>
        if args.shuffle_input:
            print(f"🔀 正在根据种子 {args.seed} 随机打乱文件列表...")
            # 必须保证 image 和 control 使用相同的随机顺序，否则图片和控制图会对不上！
            # 这种写法利用 zip 绑定打乱，再解压
            combined = list(zip(all_ctrl_files, all_img_files))
            random.shuffle(combined)
            all_ctrl_files, all_img_files = zip(*combined)
            all_ctrl_files = list(all_ctrl_files)
            all_img_files = list(all_img_files)
        # <<<<< 新增代码结束 <<<<<

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

        # >>>>> 新增代码开始 (针对无 control 的情况) >>>>>
        if args.shuffle_input:
            print(f"🔀 正在根据种子 {args.seed} 随机打乱文件列表...")
            random.shuffle(all_img_files)
        # <<<<< 新增代码结束 <<<<<

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

                prompt = get_prompt(use_random=not args.fixed_prompt, prompt_type=args.prompt_type)
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
