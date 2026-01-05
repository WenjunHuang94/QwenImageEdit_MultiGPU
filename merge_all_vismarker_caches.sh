#!/bin/bash
# 合并所有vismarker数据集的cache

BASE_CACHE_DIR="cache_vismarker"
OUTPUT_CACHE="cache_vismarker_combined"

datasets=(
    "omniedit_attribute_modification"
    "omniedit_object_swap"
    "omniedit_removal"
    "omniedit_swap"
    "ultraedit_change_color"
    "ultraedit_change_local"
    "ultraedit_replace"
    "ultraedit_turn"
)

echo "=========================================="
echo "合并所有vismarker数据集的cache"
echo "=========================================="

# 收集所有cache目录
cache_dirs=()
for dataset in "${datasets[@]}"; do
    cache_dir="${BASE_CACHE_DIR}/${dataset}"
    if [ -d "$cache_dir" ]; then
        cache_dirs+=("$cache_dir")
        echo "找到cache: $cache_dir"
    else
        echo "警告: cache目录不存在: $cache_dir，跳过"
    fi
done

if [ ${#cache_dirs[@]} -eq 0 ]; then
    echo "错误: 没有找到任何cache目录"
    exit 1
fi

echo ""
echo "找到 ${#cache_dirs[@]} 个cache目录，开始合并..."

# 使用merge_multiple_caches.py一次性合并所有cache
python merge_multiple_caches.py \
    --cache_dirs "${cache_dirs[@]}" \
    --output "$OUTPUT_CACHE" \
    --max_samples_per_dataset 2000

if [ $? -ne 0 ]; then
    echo "错误: 合并失败"
    exit 1
fi

echo ""
echo "合并后的cache统计:"
echo "  - text_embs: $(find $OUTPUT_CACHE/text_embs -name "*.pt" 2>/dev/null | wc -l) 个文件"
echo "  - img_embs: $(find $OUTPUT_CACHE/img_embs -name "*.pt" 2>/dev/null | wc -l) 个文件"
echo "  - img_embs_control: $(find $OUTPUT_CACHE/img_embs_control -name "*.pt" 2>/dev/null | wc -l) 个文件"

