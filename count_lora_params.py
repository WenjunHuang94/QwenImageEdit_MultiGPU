"""
计算LoRA实际参数量（支持 .bin 和 .safetensors）
"""

import torch
from pathlib import Path
from safetensors import safe_open


def count_lora_parameters(lora_path: str):
    """计算LoRA权重的实际参数量"""
    
    print("="*80)
    print(f"Analyzing LoRA: {lora_path}")
    print("="*80)
    
    lora_path = Path(lora_path)
    
    # 判断文件格式
    if lora_path.suffix == '.safetensors':
        # 加载 safetensors
        tensors = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    elif lora_path.suffix in ['.bin', '.pt']:
        # 加载 PyTorch checkpoint
        tensors = torch.load(lora_path, map_location='cpu')
    else:
        raise ValueError(f"Unsupported file format: {lora_path.suffix}")
    
    # 统计参数
    total_params = 0
    lora_params = 0
    
    print("\nParameter breakdown:")
    print("-"*80)
    
    for name, param in tensors.items():
        if isinstance(param, torch.Tensor):
            num_params = param.numel()
            total_params += num_params
            
            # LoRA参数通常包含 lora_A 和 lora_B
            if 'lora' in name.lower():
                lora_params += num_params
                print(f"{name:60s} {num_params:>15,} ({list(param.shape)})")
    
    print("-"*80)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Parameter size: {lora_params / 1e6:.2f}M")
    print(f"Parameter size: {lora_params / 1e9:.3f}B")
    
    # 文件大小
    file_size = lora_path.stat().st_size
    print(f"\nFile size: {file_size / 1024 / 1024:.2f} MB")
    print(f"File size: {file_size / 1024 / 1024 / 1024:.3f} GB")
    
    # 计算存储效率
    if lora_params > 0:
        bytes_per_param = file_size / lora_params
        print(f"\nBytes per parameter: {bytes_per_param:.2f}")
        if bytes_per_param > 3:
            print(f"Data type: float32 (4 bytes)")
        elif bytes_per_param > 1.5:
            print(f"Data type: float16/bfloat16 (2 bytes)")
        else:
            print(f"Data type: int8 or compressed")
    
    print("="*80)
    
    return lora_params


def find_lora_file(checkpoint_dir: str):
    """在checkpoint目录中查找LoRA文件"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # 优先查找 safetensors
    safetensors_file = checkpoint_dir / "adapter_model.safetensors"
    if safetensors_file.exists():
        return safetensors_file
    
    # 然后查找 .bin
    bin_file = checkpoint_dir / "adapter_model.bin"
    if bin_file.exists():
        return bin_file
    
    # 查找其他可能的文件
    for pattern in ["*.safetensors", "*.bin"]:
        files = list(checkpoint_dir.glob(pattern))
        if files:
            # 排除 optimizer 和 scheduler
            for f in files:
                if 'optimizer' not in f.name and 'scheduler' not in f.name:
                    return f
    
    return None


def compare_lora_ranks(base_dir: str = "./"):
    """对比不同rank的LoRA参数量"""
    
    base_dir = Path(base_dir)
    
    print("\n" + "="*80)
    print("Comparing LoRA Ranks")
    print("="*80)
    
    # 查找所有包含 rank 的目录
    rank_dirs = []
    for d in base_dir.glob("*rank*"):
        if d.is_dir():
            rank_dirs.append(d)
    
    if not rank_dirs:
        print("No rank directories found!")
        return
    
    results = []
    
    for rank_dir in sorted(rank_dirs):
        print(f"\n{'='*80}")
        print(f"Analyzing: {rank_dir.name}")
        print(f"{'='*80}")
        
        # 查找LoRA文件
        lora_file = find_lora_file(rank_dir)
        
        if not lora_file:
            print(f"⚠ No LoRA file found in {rank_dir}")
            continue
        
        params = count_lora_parameters(str(lora_file))
        
        # 尝试从目录名提取rank
        import re
        match = re.search(r'rank[_-]?(\d+)', rank_dir.name, re.IGNORECASE)
        rank = int(match.group(1)) if match else "Unknown"
        
        results.append({
            'name': rank_dir.name,
            'rank': rank,
            'params': params,
            'params_M': params / 1e6,
            'params_B': params / 1e9
        })
    
    # 打印对比表格
    if results:
        print("\n" + "="*80)
        print("Summary Table")
        print("="*80)
        print(f"{'Directory':<40} {'Rank':<10} {'Parameters':<20} {'Size (M)':<15}")
        print("-"*80)
        
        for r in sorted(results, key=lambda x: x['rank'] if isinstance(x['rank'], int) else 999):
            print(f"{r['name']:<40} {str(r['rank']):<10} {r['params']:>15,}    {r['params_M']:>10.2f}M")
        
        print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type=str, default=None,
                       help='Path to LoRA file or checkpoint directory')
    parser.add_argument('--compare_ranks', action='store_true',
                       help='Compare different ranks in current directory')
    parser.add_argument('--base_dir', type=str, default='./',
                       help='Base directory for rank comparison')
    
    args = parser.parse_args()
    
    if args.compare_ranks:
        compare_lora_ranks(args.base_dir)
    elif args.lora_path:
        lora_path = Path(args.lora_path)
        
        # 如果是目录，查找LoRA文件
        if lora_path.is_dir():
            lora_file = find_lora_file(lora_path)
            if lora_file:
                print(f"Found LoRA file: {lora_file}")
                count_lora_parameters(str(lora_file))
            else:
                print(f"❌ No LoRA file found in {lora_path}")
        else:
            count_lora_parameters(str(lora_path))
    else:
        print("Usage:")
        print("  Single LoRA file:")
        print("    python count_lora_params.py --lora_path ./checkpoint/adapter_model.safetensors")
        print("  Single checkpoint directory:")
        print("    python count_lora_params.py --lora_path ./checkpoint/")
        print("  Compare all ranks:")
        print("    python count_lora_params.py --compare_ranks --base_dir ./")
