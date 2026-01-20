import os
import base64
import shutil
import random
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation
from tqdm import tqdm
import argparse
import time

# ================= 配置区域 =================
dashscope.api_key = "sk-1649650cd15847b685cd57def55a7a56"

# ================= 系统提示词 =================
SYSTEM_PROMPT = """
You are a professional image editing assistant. Based on the following input, generate a detailed, precise, and actionable text instruction that can be directly used by an image editing model (e.g., Qwen-Image-Edit).

Input:
- Image content: [Describe the main elements in the image, including subjects, background, lighting, weather, time of day, etc.]
- In-image text: [Extract and describe any visible text written within the image, such as "Add a white horse with a saddle in the background."]

Requirements:
1. Clearly identify what object(s) should be added, removed, or modified based on the in-image text.
2. Determine the most plausible location for the new object(s), using relative spatial references (e.g., "in the background behind the right character", "to the left of the tree") or depth cues.
3. Describe visual attributes in detail: color, size, type, orientation, material, style (realistic, cinematic, etc.), and whether it should appear partially occluded or fully visible.
4. Ensure environmental consistency: match lighting direction, shadow casting, perspective, scale, and texture with the original scene.
5. Explicitly instruct the model to remove any visible text annotations from the final image.
6. Use fluent natural language but be highly specific—avoid ambiguity so the editing model can execute accurately.

Output format:
【Editing Instruction】: [A single, coherent, and comprehensive sentence or paragraph describing the edit.]
"""


def encode_image_to_base64(image_path):
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def call_qwen_vl_max(image_path):
    """调用 Qwen-VL-Max 生成编辑指令"""
    messages = [
        {
            "role": "system",
            "content": [{"text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"image": encode_image_to_base64(image_path)},
                {"text": "Analyze this image. Identify the text instruction written on the image, and generate the professional Editing Instruction as requested."}
            ]
        }
    ]

    try:
        response = MultiModalConversation.call(
            model='qwen-vl-max-latest',
            messages=messages,
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content[0]['text']
        else:
            print(f"API Error: {response.code} - {response.message}")
            return None

    except Exception as e:
        print(f"Request Failed: {e}")
        return None


def sample_images(source_folder, num_samples, seed=None):
    """从源文件夹中随机抽取指定数量的图片"""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [f for f in os.listdir(source_folder) 
                  if f.lower().endswith(valid_extensions)]
    
    if num_samples > len(all_images):
        print(f"⚠️ 警告: 请求 {num_samples} 张图片，但只有 {len(all_images)} 张可用")
        num_samples = len(all_images)
    
    # 设置随机种子以保证可复现
    if seed is not None:
        random.seed(seed)
    
    sampled_images = random.sample(all_images, num_samples)
    return sampled_images


def copy_images(image_files, source_folder, target_folder):
    """复制图片到目标文件夹"""
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    print(f"📋 正在复制 {len(image_files)} 张图片到 {target_folder}...")
    
    for img_file in tqdm(image_files, desc="Copying Images"):
        src_path = os.path.join(source_folder, img_file)
        dst_path = os.path.join(target_folder, img_file)
        
        # 如果目标文件已存在，跳过
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)


def generate_prompts(image_folder, output_folder, retry_delay=0.5):
    """为图片生成专业提示"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(valid_extensions)]
    
    print(f"\n📝 正在为 {len(image_files)} 张图片生成专业提示...")
    
    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0
    }
    
    for img_file in tqdm(image_files, desc="Generating Prompts"):
        img_path = os.path.join(image_folder, img_file)
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        save_path = os.path.join(output_folder, txt_filename)
        
        # 断点续传：如果文件已存在，跳过
        if os.path.exists(save_path):
            stats["skipped"] += 1
            continue
        
        # 调用 API
        generated_prompt = call_qwen_vl_max(img_path)
        
        if generated_prompt:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(generated_prompt)
            stats["success"] += 1
        else:
            stats["failed"] += 1
            print(f"❌ 处理失败: {img_file}")
        
        # 避免 API 限流
        if generated_prompt:
            time.sleep(retry_delay)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="从数据集中抽取指定数量的图片，复制到指定文件夹，并生成专业提示"
    )
    
    # 必选参数
    parser.add_argument(
        "--source_folder",
        type=str,
        required=True,
        help="源数据集文件夹路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="要抽取的图片数量"
    )
    
    # 可选参数
    parser.add_argument(
        "--target_folder",
        type=str,
        default=None,
        help="目标文件夹（复制图片到这里），默认为 source_folder_sampled_N"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="输出 Prompt 文件夹，默认为 target_folder_prompts"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，用于可复现的抽样（可选）"
    )
    parser.add_argument(
        "--skip_copy",
        action="store_true",
        help="跳过复制步骤，直接对 target_folder 中的图片生成提示"
    )
    parser.add_argument(
        "--skip_generate",
        action="store_true",
        help="只复制图片，不生成提示"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="阿里云 API Key（可选）"
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=0.5,
        help="API 调用间隔（秒），避免限流"
    )
    
    args = parser.parse_args()
    
    # 设置 API Key
    if args.api_key:
        dashscope.api_key = args.api_key
    
    # 设置默认路径
    if args.target_folder is None:
        source_basename = os.path.basename(args.source_folder.rstrip('/'))
        args.target_folder = f"{source_basename}_sampled_{args.num_samples}"
    
    if args.output_folder is None:
        args.output_folder = f"{args.target_folder}_prompts"
    
    print("=" * 60)
    print("🎲 批量抽样并生成专业提示")
    print("=" * 60)
    print(f"📂 源文件夹: {args.source_folder}")
    print(f"🎯 抽样数量: {args.num_samples}")
    print(f"📁 目标文件夹: {args.target_folder}")
    print(f"💾 Prompt 输出: {args.output_folder}")
    if args.seed is not None:
        print(f"🌱 随机种子: {args.seed}")
    print("=" * 60)
    
    # 步骤 1: 抽样并复制图片
    if not args.skip_copy:
        print("\n📊 步骤 1/2: 抽样并复制图片")
        sampled_images = sample_images(args.source_folder, args.num_samples, args.seed)
        print(f"✅ 已抽取 {len(sampled_images)} 张图片")
        copy_images(sampled_images, args.source_folder, args.target_folder)
        print(f"✅ 图片已复制到: {args.target_folder}")
    else:
        print("\n⏭️  跳过复制步骤，使用现有的目标文件夹")
    
    # 步骤 2: 生成专业提示
    if not args.skip_generate:
        print("\n📝 步骤 2/2: 生成专业提示")
        stats = generate_prompts(args.target_folder, args.output_folder, args.retry_delay)
        
        print("\n" + "=" * 60)
        print("📊 处理完成统计:")
        print(f"  ✅ 成功: {stats['success']}")
        print(f"  ❌ 失败: {stats['failed']}")
        print(f"  ⏭️  跳过（已存在）: {stats['skipped']}")
        print(f"\n💾 生成的 Prompt 保存在: {args.output_folder}")
        print("=" * 60)
    else:
        print("\n⏭️  跳过生成提示步骤")
        print(f"✅ 图片已准备好在: {args.target_folder}")


if __name__ == "__main__":
    main()
