import os
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# 1. 路径设置
# [输入] 原始图片文件夹
INPUT_IMAGE_FOLDER = "./images_with_box_instructions"

# [输入] VLM 生成的 Prompt 文本文件夹
INPUT_PROMPT_FOLDER = "./images_with_box_instructions_result"

# [输出] 结果保存文件夹
OUTPUT_IMAGE_FOLDER = "./images_with_box_instructions_local_base_model_results"

# 2. 模型设置
BASE_MODEL_PATH = "Qwen/Qwen-Image-Edit"
# 注意：这里删除了 LoRA 相关的配置路径

# 3. 推理参数
SEED = 42  # 随机种子
GUIDANCE_SCALE = 6.0  # 基础模型通常可以用高一点的 CFG (默认为 7.5)
NUM_STEPS = 50  # 基础模型 30-50 步通常足够


# ===================================================================

def load_base_model():
    """仅加载基础模型"""
    print("=" * 50)
    print(f"正在加载基础模型: {BASE_MODEL_PATH} ...")

    try:
        # 直接加载 Pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        print("✅ 基础模型加载完毕 (无 LoRA)。")

        return pipe
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit(1)


def clean_prompt(prompt_text):
    """
    清洗 VLM 返回的 prompt。
    去掉 '【Editing Instruction】:' 前缀，保留纯指令。
    """
    cleaned = prompt_text.replace("【Editing Instruction】:", "")
    cleaned = cleaned.replace("【Editing Instruction】", "")
    return cleaned.strip()


def main():
    # 1. 准备目录
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)

    # 2. 加载模型
    pipe = load_base_model()

    # 3. 获取所有图片
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.lower().endswith(valid_exts)]

    print("=" * 50)
    print(f"📂 发现 {len(image_files)} 张待处理图片")
    print(f"📂 Prompt 来源: {INPUT_PROMPT_FOLDER}")
    print(f"💾 输出目录: {OUTPUT_IMAGE_FOLDER}")
    print("=" * 50)

    # 4. 开始批量处理
    success_count = 0

    for img_file in tqdm(image_files, desc="Base Model Inference"):
        # --- A. 路径准备 ---
        img_path = os.path.join(INPUT_IMAGE_FOLDER, img_file)
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        prompt_path = os.path.join(INPUT_PROMPT_FOLDER, txt_filename)

        output_filename = os.path.splitext(img_file)[0] + "_edited.png"
        output_path = os.path.join(OUTPUT_IMAGE_FOLDER, output_filename)

        # --- B. 检查 ---
        if os.path.exists(output_path):
            continue

        if not os.path.exists(prompt_path):
            tqdm.write(f"⚠️ 跳过: 找不到 Prompt 文件 {txt_filename}")
            continue

        try:
            # --- C. 读取数据 ---
            original_image = Image.open(img_path).convert("RGB")

            with open(prompt_path, 'r', encoding='utf-8') as f:
                raw_prompt = f.read()
                # final_prompt = clean_prompt(raw_prompt)
                final_prompt = raw_prompt

            # --- D. 推理 ---
            inputs = {
                "image": original_image,
                "prompt": final_prompt,
                "generator": torch.Generator(device="cuda").manual_seed(SEED),
                "true_cfg_scale": GUIDANCE_SCALE,  # 这里用基础模型的 CFG
                "negative_prompt": "low quality, blurry, distortion, text, watermark",  # 加一些负面词去噪
                "num_inference_steps": NUM_STEPS,
            }

            with torch.inference_mode():
                output = pipe(**inputs)
                output_image = output.images[0]

            # --- E. 保存 ---
            output_image.save(output_path)
            success_count += 1

        except Exception as e:
            tqdm.write(f"❌ 处理 {img_file} 失败: {e}")

    print("\n" + "=" * 50)
    print(f"✅ 批量处理完成！成功生成 {success_count} 张图片。")
    print(f"查看结果: {os.path.abspath(OUTPUT_IMAGE_FOLDER)}")


if __name__ == "__main__":
    main()