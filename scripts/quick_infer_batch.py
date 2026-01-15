import torch
from PIL import Image
import json
import os
import argparse
from peft import PeftModel
from QwenEdit import VanillaPipeline, MultiGPUTransformer, get_image


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Inference for Qwen Image Edit")
    parser.add_argument("--task_file", type=str, required=True, help="Path to JSON task file")
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--lora_weight", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--neg_prompt", type=str, default="bounding box, text, labels, artifacts...")  # 保持你之前的长负向词
    parser.add_argument("--target_area", type=int, default=512 * 512)
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. 初始化模型 (只加载一次)
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

    # 2. 读取批量任务
    with open(args.task_file, 'r') as f:
        tasks = json.load(f)

    # 3. 循环处理
    for i, task in enumerate(tasks):
        img_path = task['img_path']
        prompt = task['prompt']
        save_path = os.path.join(args.output_dir, task.get('output_name', f"out_{i}.png"))

        print(f"[{i + 1}/{len(tasks)}] Processing: {img_path} with prompt: {prompt}")

        image = get_image(img_path)

        inputs = {
            "image": image,
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

    print(f"Batch processing completed. Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()