import os
import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm
from peft import PeftModel
import sys

# 尝试导入 QwenEdit
try:
    from QwenEdit import VanillaPipeline, MultiGPUTransformer
except ImportError:
    print("Error: 无法导入 QwenEdit。请确保 'QwenEdit.py' 在当前目录下，或者已添加到 PYTHONPATH 中。")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="GEdit-Bench Inference with Qwen Image Edit")

    # === 模型路径参数 ===
    parser.add_argument("--pretrained_model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_weight", type=str, default=None, help="LoRA weight path")

    # === 任务与输出参数 ===
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local GEdit-Bench dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for directory structure")
    parser.add_argument("--output_root", type=str, default="results", help="Root directory for saving results")

    # === ✨ 新增：控制推理数量 ===
    parser.add_argument("--num_images", type=int, default=None,
                        help="Debug: Number of images to process (e.g. 1). Default is all.")

    # === 推理超参数 ===
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
    # 1. 初始化模型
    # ==========================================
    print(f"Loading base model from: {args.pretrained_model}")
    pipe = VanillaPipeline.from_pretrained(args.pretrained_model, torch_dtype=dtype).to("cpu")
    pipe.vae.to("cuda:0")
    pipe.text_encoder.to("cuda:0")

    flux_transformer = MultiGPUTransformer(pipe.transformer).auto_split()

    if args.lora_weight:
        print(f"Loading LoRA weights: {args.lora_weight}")

        def _unwrap(m): return m._orig_mod if hasattr(m, "_orig_mod") else m

        flux_transformer = PeftModel.from_pretrained(_unwrap(flux_transformer), args.lora_weight)

    flux_transformer.eval()
    pipe.transformer = flux_transformer
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ==========================================
    # 2. 加载数据集
    # ==========================================
    print(f"Loading dataset from: {args.dataset_path}")
    try:
        dataset = load_dataset(args.dataset_path)
        data_split = dataset['train']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # === ✨ 新增：如果有参数，截取数据集 ===
    if args.num_images is not None and args.num_images > 0:
        # 确保不越界
        limit = min(args.num_images, len(data_split))
        print(f"👀 Debug Mode: 只运行前 {limit} 张图片")
        # === 修改为随机抽取 ===
        # 先打乱顺序 (shuffle)，再取前 N 张
        # seed=args.seed 保证每次随机的结果是一样的，方便复现
        data_split = data_split.shuffle(seed=args.seed).select(range(limit))

    print(f"Start Inference. Total images: {len(data_split)}")

    # ==========================================
    # 3. 循环推理
    # ==========================================
    for item in tqdm(data_split, desc="Inferencing"):
        image_input = item['input_image']
        prompt = item['instruction']
        task_type = item['task_type']
        lang = item['instruction_language']
        file_key = item['key']

        # 路径构建
        lang_folder = 'cn' if lang in ['zh', 'chinese', 'cn'] else 'en'
        save_dir = os.path.join(args.output_root, args.model_name, "fullset", task_type, lang_folder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_key}.png")

        if os.path.exists(save_path):
            continue

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

        with torch.inference_mode():
            output = pipe(**inputs)
            output.images[0].save(save_path)

    print(f"\nAll tasks completed. Results saved in: {os.path.abspath(args.output_root)}")


if __name__ == "__main__":
    main()