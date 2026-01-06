#!/bin/bash
# 灵活合并所有数据集
# 根据您的需求：cache2(4000) + cache5(4000) + cache6(4000) + 8个vismarker子数据集(各1000)

OUTPUT_CACHE="cache_all_balanced"

echo "=========================================="
echo "灵活合并所有数据集"
echo "=========================================="
echo ""

python merge_caches_flexible.py \
    --cache "cache2:generate:4000" \
    --cache "cache5:edit:4000" \
    --cache "cache6:annotated_edit:4000" \
    --cache "cache_vismarker/omniedit_attribute_modification:att_mod:1000" \
    --cache "cache_vismarker/omniedit_object_swap:obj_swap:1000" \
    --cache "cache_vismarker/omniedit_removal:removal:1000" \
    --cache "cache_vismarker/omniedit_swap:swap:1000" \
    --cache "cache_vismarker/ultraedit_change_color:change_color:1000" \
    --cache "cache_vismarker/ultraedit_change_local:change_local:1000" \
    --cache "cache_vismarker/ultraedit_replace:replace:1000" \
    --cache "cache_vismarker/ultraedit_turn:turn:1000" \
    --output "$OUTPUT_CACHE" \
    --seed 42

if [ $? -ne 0 ]; then
    echo "错误: 合并失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "合并完成！"
echo "=========================================="
echo "输出目录: $OUTPUT_CACHE"
echo ""
echo "数据统计:"
echo "  - generate: 4000 样本"
echo "  - edit: 4000 样本"
echo "  - annotated_edit: 4000 样本"
echo "  - pointer_edit (8个子数据集): 8000 样本 (各1000)"
echo "  - 总计: 20000 样本"
echo ""
echo "下一步：使用 consume_retrain_all.sh 进行训练"

