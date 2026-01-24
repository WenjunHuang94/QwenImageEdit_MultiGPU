"""
批量推理：用Qwen-Base和Qwen+LoRA生成所有结果
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse


def batch_inference(
    input_dir: str,
    output_dir: str,
    model_path: str,
    lora_path: str = None,
    prompt: str = "Modify the image at the annotated location according to the text instruction"
):
    """批量推理"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.JPG', '*.PNG']:
        image_files.extend(input_dir.glob(ext))
    
    image_files = sorted(image_files)
    
    print(f"\nFound {len(image_files)} images")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_dir}\n")
    
    # 批量处理
    for img_path in tqdm(image_files, desc="Generating"):
        filename = img_path.name
        output_path = output_dir / filename
        
        # 构建命令
        cmd = f"python batch_local_image_edit.py " \
              f"--image_path \"{img_path}\" " \
              f"--prompt \"{prompt}\" " \
              f"--output_path \"{output_path}\" " \
              f"--model_path \"{model_path}\""
        
        if lora_path:
            cmd += f" --lora_path \"{lora_path}\""
        
        # 执行
        ret = os.system(cmd)
        
        if ret != 0:
            print(f"\n⚠ Failed: {filename}")
    
    print(f"\n✓ Generated {len(image_files)} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch Inference for Evaluation")
    
    parser.add_argument('--input_dir', type=str, 
                       default="./evaluation_samples_with_textbox_0124V1/input",
                       help='Input directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Qwen-VL-Chat model')
    parser.add_argument('--lora_path', type=str, default=None,
                       help='Path to LoRA weights (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--prompt', type=str,
                       default="Modify the image at the annotated location according to the text instruction",
                       help='Prompt for generation')
    
    args = parser.parse_args()
    
    print("="*80)
    if args.lora_path:
        print("Batch Inference: Qwen+LoRA")
    else:
        print("Batch Inference: Qwen-Base")
    print("="*80)
    
    batch_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        lora_path=args.lora_path,
        prompt=args.prompt
    )


if __name__ == "__main__":
    main()
