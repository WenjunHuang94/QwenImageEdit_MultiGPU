#!/bin/bash
# 合并cache2和cache5，然后训练合并后的数据集
# 用于避免lora_2忘记lora_1的能力

# 配置
CACHE1="./cache2"  # 类型（1）：文本描述+真实图片
CACHE2="./cache5"  # 类型（2）：控制图片+编辑结果
CACHE_COMBINED="./cache_combined"  # 合并后的cache目录

# 合并cache
echo "=========================================="
echo "步骤1: 合并cache2和cache5到cache_combined"
echo "=========================================="
python merge_caches.py \
    --cache1 "$CACHE1" \
    --cache2 "$CACHE2" \
    --output "$CACHE_COMBINED"