import torch
from PIL import Image
import os
import argparse
from peft import PeftModel
from QwenEdit import VanillaPipeline, MultiGPUTransformer, get_image
from tqdm import tqdm
import time

# ================= 配置区域 =================
# 模型路径
MODEL_PATH = "/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
LORA_PATH = ""  # 如果不需要 LoRA，设置为 None

# 输入和输出路径
ORIGINAL_IMAGES = "./images_with_text_instructions"  # 原始图片文件夹
PROMPTS_FOLDER = "./images_with_text_instructions_result"  # Prompt 文本文件夹
OUTPUT_FOLDER = "./edited_images_local_output"  # 输出文件夹

# 推理参数
TARGET_AREA = 512 * 512  # 目标分辨率
INFER_STEPS = 50  # 推理步数
CFG_SCALE = 6.0  # CFG 引导强度
SEED = 42  # 随机种子

# 负向提示词（用于去除文字标注等）
NEG_PROMPT = ""


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Local Inference for Qwen Image Edit")
    
    # 路径配置
    parser.add_argument("--pretrained_model", type=str, default=MODEL_PATH)
    parser.add_argument("--lora_weight", type=str, default=LORA_PATH, help="LoRA权重路径，不需要则设为None")
    parser.add_argument("--original_images", type=str, default=ORIGINAL_IMAGES)
    parser.add_argument("--prompts_folder", type=str, default=PROMPTS_FOLDER)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER)
    
    # 推理参数
    parser.add_argument("--neg_prompt", type=str, default=NEG_PROMPT)
    parser.add_argument("--target_area", type=int, default=TARGET_AREA)
    parser.add_argument("--infer_steps", type=int, default=INFER_STEPS)
    parser.add_argument("--cfg_scale", type=float, default=CFG_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    
    # 其他选项
    parser.add_argument("--skip_existing", action="store_true", default=True, help="跳过已存在的输出文件")
    
    return parser.parse_args()


def load_model(args):
    """加载模型（只加载一次）"""
    print("🔧 正在加载模型...")
    dtype = torch.bfloat16
    
    # 加载基础模型
    pipe = VanillaPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=dtype
    ).to("cpu")
    
    pipe.vae.to("cuda:0")
    pipe.text_encoder.to("cuda:0")
    
    # 加载 Transformer 并分配到多 GPU
    flux_transformer = MultiGPUTransformer(pipe.transformer).auto_split()
    
    # 加载 LoRA 权重（如果有）
    if args.lora_weight and args.lora_weight.lower() != "none":
        print(f"📦 加载 LoRA 权重: {args.lora_weight}")
        
        def _unwrap(m):
            return m._orig_mod if hasattr(m, "_orig_mod") else m
        
        flux_transformer = PeftModel.from_pretrained(
            _unwrap(flux_transformer),
            args.lora_weight,
            low_cpu_mem_usage=False
        )
    
    flux_transformer.eval()
    pipe.transformer = flux_transformer
    pipe.set_progress_bar_config(disable=True)  # 禁用内部进度条
    
    print("✅ 模型加载完成！")
    return pipe


def process_single_image(pipe, generator, img_file, original_folder, prompts_folder, output_folder, args):
    """处理单张图片"""
    
    # 构建路径
    img_path = os.path.join(original_folder, img_file)
    txt_filename = os.path.splitext(img_file)[0] + ".txt"
    prompt_path = os.path.join(prompts_folder, txt_filename)
    
    # 输出文件名
    output_filename = os.path.splitext(img_file)[0] + "_edited.png"
    output_path = os.path.join(output_folder, output_filename)
    
    # 检查是否已处理
    if args.skip_existing and os.path.exists(output_path):
        return "skipped"
    
    # 检查 prompt 文件是否存在
    if not os.path.exists(prompt_path):
        print(f"⚠️ 未找到对应的 prompt 文件: {txt_filename}")
        return "no_prompt"
    
    # 读取 prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()
    
    # 如果 prompt 包含 "【Editing Instruction】:" 标记，提取实际指令
    if "【Editing Instruction】:" in prompt_text:
        prompt_text = prompt_text.split("【Editing Instruction】:", 1)[1].strip()
    
    try:
        # 加载图片
        image = get_image(img_path)
        
        # 准备输入
        inputs = {
            "image": image,
            "prompt": prompt_text,
            "generator": generator,
            "true_cfg_scale": args.cfg_scale,
            "negative_prompt": args.neg_prompt,
            "num_inference_steps": args.infer_steps,
            "target_area": args.target_area,
            "max_sequence_length": 1024
        }
        
        # 推理
        with torch.inference_mode():
            output = pipe(**inputs)
            output.images[0].save(output_path)
        
        return "success"
    
    except Exception as e:
        print(f"❌ 处理失败 {img_file}: {e}")
        return "failed"


def main():
    args = parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(args.original_images)
                   if f.lower().endswith(valid_extensions)]
    
    print("=" * 60)
    print("🚀 批量本地图片编辑")
    print("=" * 60)
    print(f"📁 原始图片: {args.original_images}")
    print(f"📄 Prompt 文件: {args.prompts_folder}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"🖼️  发现 {len(image_files)} 张图片")
    print(f"🎯 推理步数: {args.infer_steps}")
    print(f"🎨 CFG Scale: {args.cfg_scale}")
    print(f"🌱 随机种子: {args.seed}")
    print("=" * 60)
    
    # 加载模型（只加载一次）
    pipe = load_model(args)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    
    # 统计信息
    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "no_prompt": 0
    }
    
    # 批量处理
    print("\n🎬 开始批量处理...\n")
    start_time = time.time()
    
    for img_file in tqdm(image_files, desc="Processing Images"):
        result = process_single_image(
            pipe,
            generator,
            img_file,
            args.original_images,
            args.prompts_folder,
            args.output_dir,
            args
        )
        
        stats[result] += 1
    
    elapsed_time = time.time() - start_time
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("📊 处理完成统计:")
    print(f"  ✅ 成功: {stats['success']}")
    print(f"  ❌ 失败: {stats['failed']}")
    print(f"  ⏭️  跳过（已存在）: {stats['skipped']}")
    print(f"  ⚠️  无 Prompt: {stats['no_prompt']}")
    print(f"\n⏱️  总耗时: {elapsed_time:.2f} 秒")
    if stats['success'] > 0:
        print(f"⚡ 平均每张: {elapsed_time / stats['success']:.2f} 秒")
    print(f"\n💾 编辑后的图片保存在: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
