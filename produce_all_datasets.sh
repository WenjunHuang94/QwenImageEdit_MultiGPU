#!/bin/bash
# 批量处理8个数据集的预处理
# 数据集路径：/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_vismarker/

MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
BASE_DATA_DIR="/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_vismarker"
BASE_CACHE_DIR="cache_vismarker"

RESOLUTION=$((512*512))

# 定义所有数据集
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

# 每个数据集的最大样本数（可以根据实际情况调整）
# 如果某个数据集样本较少，可以设置更大的值或使用None处理全部
MAX_SAMPLES=2000  # 可以根据实际情况调整，或设置为None处理全部

echo "=========================================="
echo "开始批量处理 ${#datasets[@]} 个数据集"
echo "=========================================="

for dataset in "${datasets[@]}"; do
    echo ""
    echo "=========================================="
    echo "处理数据集: $dataset"
    echo "=========================================="
    
    # 设置路径
    IMG_DIR="${BASE_DATA_DIR}/${dataset}/output"
    CTRL_DIR="${BASE_DATA_DIR}/${dataset}/input"
    CACHE_DIR="${BASE_CACHE_DIR}/${dataset}"
    
    # 检查目录是否存在
    if [ ! -d "$IMG_DIR" ]; then
        echo "警告: 输出目录不存在: $IMG_DIR，跳过"
        continue
    fi
    
    if [ ! -d "$CTRL_DIR" ]; then
        echo "警告: 控制目录不存在: $CTRL_DIR，跳过"
        continue
    fi
    
    # 统计文件数量
    img_count=$(find "$IMG_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
    ctrl_count=$(find "$CTRL_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
    
    echo "  图片数量: $img_count"
    echo "  控制图数量: $ctrl_count"
    
    if [ "$img_count" -eq 0 ] || [ "$ctrl_count" -eq 0 ]; then
        echo "警告: 数据集 $dataset 没有找到图片文件，跳过"
        continue
    fi
    
    # 运行producer
    python scripts/producer.py \
        --pretrained_model "$MODEL_PATH" \
        --img_dir "$IMG_DIR" \
        --control_dir "$CTRL_DIR" \
        --target_area $RESOLUTION \
        --output_dir "$CACHE_DIR" \
        --prompt_type "pointer_edit" \
        --prompt_with_image \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
        "$@"
    
    if [ $? -ne 0 ]; then
        echo "错误: 处理数据集 $dataset 失败"
        exit 1
    fi
    
    echo "✓ 数据集 $dataset 处理完成，缓存保存在: $CACHE_DIR"
done

echo ""
echo "=========================================="
echo "所有数据集处理完成！"
echo "=========================================="
echo "缓存目录: $BASE_CACHE_DIR"
echo ""
echo "下一步：使用 merge_all_vismarker_caches.sh 合并所有cache"
echo "然后使用 consume_vismarker.sh 进行训练"

