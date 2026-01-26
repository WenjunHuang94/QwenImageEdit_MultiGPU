import os
import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel
# 假设 QwenEdit 就在当前目录下，或者已在 PYTHONPATH 中
from QwenEdit import VanillaPipeline, MultiGPUTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="GEdit-Bench Inference with Qwen Image Edit")

    # === 模型相关参数 ===
    parser.add_argument("--pretrained_model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_weight", type=str, default=None, help="LoRA weight path")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name used for output directory structure (e.g., Step1X-Edit)")

    # === 数据集与输出 ===
    parser.add_argument("--dataset_path", type=str, default="./mydata", help="Path to local GEdit-Bench dataset")
    parser.add_argument("--output_root", type=str, default="results", help="Root directory for results")

    # === 推理参数 ===
    parser.add_argument("--neg_prompt", type=str,
                        default="bounding box, text, labels, artifacts, bad quality, low quality, distortion")
    parser.add_argument("--target_area", type=int, default=512 * 512)
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16

    # ==========================================
    # 1. 初始化模型 (保留你原本的逻辑)
    # ==========================================
    print(f"Loading model from: {args.pretrained_model}")
    pipe = VanillaPipeline.from_pretrained(args.pretrained_model, torch_dtype=dtype).to("cpu")

    # 显存分配策略
    pipe.vae.to("cuda:0")
    pipe.text_encoder.to("cuda:0")

    # 处理 Transformer (支持多卡或自动拆分)
    flux_transformer = MultiGPUTransformer(pipe.transformer).auto_split()

    # 加载 LoRA
    if args.lora_weight:
        print(f"Loading LoRA weights: {args.lora_weight}")

        def _unwrap(m): return m._orig_mod if hasattr(m, "_orig_mod") else m

        flux_transformer = PeftModel.from_pretrained(_unwrap(flux_transformer), args.lora_weight)

    flux_transformer.eval()
    pipe.transformer = flux_transformer

    # 设置随机数生成器
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ==========================================
    # 2. 加载 GEdit-Bench 数据集
    # ==========================================
    print(f"Loading dataset from: {args.dataset_path}")
    try:
        dataset = load_dataset(args.dataset_path)
        data_split = dataset['train']  # GEdit-Bench 默认数据在 train split
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Start Inference. Total images: {len(data_split)}")

    # ==========================================
    # 3. 循环推理
    # ==========================================
    for item in tqdm(data_split, desc="Inferencing"):
        # --- 提取字段 ---
        image_input = item['input_image']  # PIL Image
        prompt = item['instruction']  # 编辑指令
        task_type = item['task_type']  # 任务类型
        lang = item['instruction_language']  # 语言
        file_key = item['key']  # 文件ID

        # --- 路径构建 (results/{model}/{fullset}/{task}/{lang}/{key}.png) ---
        if lang in ['zh', 'chinese', 'cn']:
            lang_folder = 'cn'
        else:
            lang_folder = 'en'

        save_dir = os.path.join(
            args.output_root,
            args.model_name,
            "fullset",
            task_type,
            lang_folder
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_key}.png")

        # 如果已存在则跳过 (可选，方便断点续跑)
        if os.path.exists(save_path):
            continue

        # --- 准备输入 ---
        # 注意: dataset 中的 image 已经是 PIL 对象，无需 get_image()
        # 建议转为 RGB 防止原本是 RGBA 或 L 模式导致报错
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        inputs = {
            "image": image_input,
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": args.cfg_scale,
            "negative_prompt": args.neg_prompt,
            "num_inference_steps": args.infer_steps,
            "target_area": args.target_area,
            "max_sequence_length": 1024
        }

        # --- 执行推理 ---
        with torch.inference_mode():
            output = pipe(**inputs)
            # 保存结果
            output.images[0].save(save_path)

    print(f"\nAll tasks completed. Results saved in {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()