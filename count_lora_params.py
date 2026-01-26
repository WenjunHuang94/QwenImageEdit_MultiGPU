"""
计算LoRA实际参数量
"""

import torch
from pathlib import Path


def count_lora_parameters(lora_path: str):
    """计算LoRA权重的实际参数量"""
    
    print("="*80)
    print(f"Analyzing LoRA: {lora_path}")
    print("="*80)
    
    # 加载checkpoint
    checkpoint = torch.load(lora_path, map_location='cpu')
    
    # 统计参数
    total_params = 0
    trainable_params = 0
    
    print("\nParameter breakdown:")
    print("-"*80)
    
    for name, param in checkpoint.items():
        num_params = param.numel()
        total_params += num_params
        
        # LoRA参数通常包含 lora_A 和 lora_B
        if 'lora' in name.lower():
            trainable_params += num_params
            print(f"{name:60s} {num_params:>15,} ({param.shape})")
    
    print("-"*80)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable (LoRA) parameters: {trainable_params:,}")
    print(f"Parameter size: {trainable_params / 1e6:.2f}M")
    print(f"Parameter size: {trainable_params / 1e9:.3f}B")
    
    # 文件大小
    file_size = Path(lora_path).stat().st_size
    print(f"\nFile size: {file_size / 1024 / 1024:.2f} MB")
    print(f"File size: {file_size / 1024 / 1024 / 1024:.3f} GB")
    
    # 计算存储效率
    bytes_per_param = file_size / trainable_params if trainable_params > 0 else 0
    print(f"\nBytes per parameter: {bytes_per_param:.2f}")
    print(f"Data type: {'float32 (4 bytes)' if bytes_per_param > 3 else 'float16 (2 bytes)'}")
    
    print("="*80)
    
    return trainable_params


def compare_lora_ranks(base_dir: str = "./lora_weights"):
    """对比不同rank的LoRA参数量"""
    
    base_dir = Path(base_dir)
    
    print("\n" + "="*80)
    print("Comparing LoRA Ranks")
    print("="*80)
    
    ranks = [32, 64, 128]
    results = []
    
    for rank in ranks:
        lora_path = base_dir / f"rank_{rank}" / "adapter_model.bin"
        
        if not lora_path.exists():
            print(f"\n⚠ Not found: {lora_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Rank {rank}")
        print(f"{'='*80}")
        
        params = count_lora_parameters(str(lora_path))
        
        results.append({
            'rank': rank,
            'params': params,
            'params_M': params / 1e6,
            'params_B': params / 1e9
        })
    
    # 打印对比表格
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"{'Rank':<10} {'Parameters':<20} {'Size (M)':<15} {'Size (B)':<15}")
    print("-"*80)
    
    for r in results:
        print(f"{r['rank']:<10} {r['params']:>15,}    {r['params_M']:>10.2f}M    {r['params_B']:>10.3f}B")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type=str, default=None,
                       help='Path to single LoRA checkpoint')
    parser.add_argument('--compare_ranks', action='store_true',
                       help='Compare different ranks')
    parser.add_argument('--base_dir', type=str, default='./lora_weights',
                       help='Base directory for rank comparison')
    
    args = parser.parse_args()
    
    if args.compare_ranks:
        compare_lora_ranks(args.base_dir)
    elif args.lora_path:
        count_lora_parameters(args.lora_path)
    else:
        print("Usage:")
        print("  Single LoRA: python count_lora_params.py --lora_path ./lora_weights/rank_64/adapter_model.bin")
        print("  Compare: python count_lora_params.py --compare_ranks --base_dir ./lora_weights")
