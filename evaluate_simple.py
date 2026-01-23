"""
简化版评估脚本：对比 Qwen-Base vs Qwen+LoRA
使用Ground Truth作为参考
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import clip
import cv2
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class SimpleEvaluator:
    """简化评估器：对比两个模型与Ground Truth"""
    
    def __init__(self, device='cuda', clip_model_path=None, blip2_model_path=None):
        self.device = device
        
        # 加载CLIP（必需）
        print("="*60)
        print("Loading CLIP...")
        if clip_model_path and Path(clip_model_path).exists():
            print(f"  ✓ Using local CLIP model: {clip_model_path}")
            self.clip_model, self.clip_preprocess = clip.load(clip_model_path, device=device)
            self.has_clip = True
        else:
            raise FileNotFoundError(f"CLIP model is required: {clip_model_path}")
        
        # 加载BLIP-2（可选）
        print("="*60)
        print("Loading BLIP-2...")
        if blip2_model_path and Path(blip2_model_path).exists():
            try:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                self.blip_processor = Blip2Processor.from_pretrained(
                    blip2_model_path, local_files_only=True
                )
                self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                    blip2_model_path, torch_dtype=torch.float16, local_files_only=True
                ).to(device)
                self.has_blip2 = True
                print("  ✓ BLIP-2 loaded")
            except Exception as e:
                print(f"  ⚠ BLIP-2 skipped: {e}")
                self.has_blip2 = False
        else:
            print("  ⚠ BLIP-2 not provided")
            self.has_blip2 = False
        
        print("="*60)
    
    def compute_clip_image_similarity(self, image1_path: str, image2_path: str) -> float:
        """计算两张图片的CLIP相似度"""
        img1 = self.clip_preprocess(Image.open(image1_path)).unsqueeze(0).to(self.device)
        img2 = self.clip_preprocess(Image.open(image2_path)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat1 = self.clip_model.encode_image(img1)
            feat2 = self.clip_model.encode_image(img2)
            
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            
            similarity = (feat1 @ feat2.T).item()
        
        return similarity * 100
    
    def compute_clip_text_similarity(self, image_path: str, text: str) -> float:
        """计算图片与文本的CLIP相似度"""
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text_tokens)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
        
        return similarity * 100
    
    def compute_psnr(self, image1_path: str, image2_path: str) -> float:
        """计算PSNR"""
        try:
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            return psnr(img1, img2)
        except Exception as e:
            print(f"  ⚠ PSNR error: {e}")
            return None
    
    def compute_ssim(self, image1_path: str, image2_path: str) -> float:
        """计算SSIM"""
        try:
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            return ssim(gray1, gray2)
        except Exception as e:
            print(f"  ⚠ SSIM error: {e}")
            return None
    
    def compute_mrr(self, image_path: str) -> float:
        """计算标记去除率（Marker Removal Rate）"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 检测红色标记像素（方框、箭头通常是红色）
            red_mask = (img_rgb[:, :, 0] > 200) & (img_rgb[:, :, 1] < 100) & (img_rgb[:, :, 2] < 100)
            marker_pixels = np.sum(red_mask)
            
            total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
            mrr = 1.0 - (marker_pixels / total_pixels)
            
            return max(0.0, min(1.0, mrr))
        except Exception as e:
            print(f"  ⚠ MRR error: {e}")
            return None
    
    def evaluate_sample(self, sample: dict, generated_path: str) -> dict:
        """评估单个样本"""
        ground_truth_path = sample.get('ground_truth')
        text_instruction = sample.get('text_instruction', '')
        
        results = {
            'sample_id': sample['id'],
            'type': sample['type']
        }
        
        # 1. CLIP Score (生成图 vs Ground Truth) - 最重要！
        if ground_truth_path and Path(ground_truth_path).exists():
            results['clip_score_gt'] = self.compute_clip_image_similarity(
                generated_path, ground_truth_path
            )
        else:
            results['clip_score_gt'] = None
        
        # 2. CLIP Score (生成图 vs 文本指令) - 辅助
        if text_instruction:
            results['clip_score_text'] = self.compute_clip_text_similarity(
                generated_path, text_instruction
            )
        else:
            results['clip_score_text'] = None
        
        # 3. PSNR (生成图 vs Ground Truth)
        if ground_truth_path and Path(ground_truth_path).exists():
            results['psnr'] = self.compute_psnr(generated_path, ground_truth_path)
        else:
            results['psnr'] = None
        
        # 4. SSIM (生成图 vs Ground Truth)
        if ground_truth_path and Path(ground_truth_path).exists():
            results['ssim'] = self.compute_ssim(generated_path, ground_truth_path)
        else:
            results['ssim'] = None
        
        # 5. MRR (标记去除率)
        results['mrr'] = self.compute_mrr(generated_path)
        
        return results
    
    def evaluate_all(self, test_data_path: str, results_dir: str, output_csv: str):
        """评估所有结果"""
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        results_dir = Path(results_dir)
        all_results = []
        
        print("\n" + "="*60)
        print("Starting Evaluation...")
        print("="*60)
        
        for sample in tqdm(test_data, desc="Evaluating"):
            sample_id = sample['id']
            
            # 评估2个方法
            for method in ['qwen_base', 'ours']:
                output_path = results_dir / method / f"{sample_id}.png"
                
                if output_path.exists():
                    result = self.evaluate_sample(sample, str(output_path))
                    result['method'] = method
                    all_results.append(result)
                else:
                    print(f"  ⚠ Missing: {output_path}")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存详细结果
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        # 生成对比表格
        comparison_table = self.generate_comparison_table(df)
        comparison_table.to_csv(output_path.parent / "comparison_table.csv", index=False)
        
        # 按类型分组
        type_table = self.generate_type_table(df)
        type_table.to_csv(output_path.parent / "type_comparison.csv", index=False)
        
        # 打印结果
        print("\n" + "="*80)
        print("COMPARISON: Qwen-Base vs Ours (Qwen+LoRA)")
        print("="*80)
        print(comparison_table.to_string(index=False))
        
        print("\n" + "="*80)
        print("PERFORMANCE BY TYPE")
        print("="*80)
        print(type_table.to_string(index=False))
        
        return comparison_table, type_table
    
    def generate_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成对比表格"""
        stats = []
        
        for method in ['qwen_base', 'ours']:
            method_df = df[df['method'] == method]
            
            row = {'Method': 'Qwen-Base' if method == 'qwen_base' else 'Ours (Qwen+LoRA)'}
            
            # CLIP Score (vs GT)
            if method_df['clip_score_gt'].notna().any():
                row['CLIP-GT'] = f"{method_df['clip_score_gt'].mean():.2f}"
            else:
                row['CLIP-GT'] = "N/A"
            
            # CLIP Score (vs Text)
            if method_df['clip_score_text'].notna().any():
                row['CLIP-Text'] = f"{method_df['clip_score_text'].mean():.2f}"
            else:
                row['CLIP-Text'] = "N/A"
            
            # PSNR
            if method_df['psnr'].notna().any():
                row['PSNR'] = f"{method_df['psnr'].mean():.2f}"
            else:
                row['PSNR'] = "N/A"
            
            # SSIM
            if method_df['ssim'].notna().any():
                row['SSIM'] = f"{method_df['ssim'].mean():.3f}"
            else:
                row['SSIM'] = "N/A"
            
            # MRR
            if method_df['mrr'].notna().any():
                row['MRR'] = f"{method_df['mrr'].mean():.1%}"
            else:
                row['MRR'] = "N/A"
            
            stats.append(row)
        
        return pd.DataFrame(stats)
    
    def generate_type_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """按类型生成对比表格"""
        stats = []
        
        type_names = {
            'text_only': 'Type I (Pure Text)',
            'in_image_text': 'Type II (In-Image Text)',
            'arrow': 'Type III (Arrow)',
            'box': 'Type IV (Box)'
        }
        
        for method in ['qwen_base', 'ours']:
            for type_key, type_name in type_names.items():
                type_df = df[(df['method'] == method) & (df['type'] == type_key)]
                
                if len(type_df) > 0:
                    row = {
                        'Method': 'Qwen-Base' if method == 'qwen_base' else 'Ours',
                        'Type': type_name,
                        'Count': len(type_df)
                    }
                    
                    if type_df['clip_score_gt'].notna().any():
                        row['CLIP-GT'] = f"{type_df['clip_score_gt'].mean():.2f}"
                    else:
                        row['CLIP-GT'] = "N/A"
                    
                    if type_df['mrr'].notna().any():
                        row['MRR'] = f"{type_df['mrr'].mean():.1%}"
                    else:
                        row['MRR'] = "N/A"
                    
                    stats.append(row)
        
        return pd.DataFrame(stats)


def main():
    parser = argparse.ArgumentParser(description="Simple Evaluation: Qwen-Base vs Qwen+LoRA")
    
    parser.add_argument('--test_data', type=str, required=True,
                       help='Test data JSON path')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Results directory (contains qwen_base/ and ours/)')
    parser.add_argument('--output_csv', type=str, default="./experiments/evaluation_results.csv",
                       help='Output CSV path')
    parser.add_argument('--clip_model', type=str, required=True,
                       help='Local CLIP model path (REQUIRED)')
    parser.add_argument('--blip2_model', type=str, default=None,
                       help='Local BLIP-2 model directory (OPTIONAL)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Simple Evaluation: Qwen-Base vs Qwen+LoRA")
    print("="*80)
    print(f"Test data: {args.test_data}")
    print(f"Results dir: {args.results_dir}")
    print(f"CLIP model: {args.clip_model}")
    print("="*80)
    
    evaluator = SimpleEvaluator(
        device=args.device,
        clip_model_path=args.clip_model,
        blip2_model_path=args.blip2_model
    )
    
    comparison, type_comparison = evaluator.evaluate_all(
        test_data_path=args.test_data,
        results_dir=args.results_dir,
        output_csv=args.output_csv
    )
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
