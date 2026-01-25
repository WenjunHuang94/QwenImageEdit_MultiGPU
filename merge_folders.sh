#!/bin/bash

# 定义源文件夹列表（根据你的实际路径修改！）
# 注意：如果文件夹路径包含空格，需要用引号包裹
SOURCE_FOLDERS=(
    "evaluation_samples_text-0125V1"
    "evaluation_samples_textInImg-0125V1"
    "evaluation_samples_vismarker_mixed-0125V2"
    "evaluation_samples_with_textbox_0124V2"
)

# 定义目标主文件夹
TARGET_ROOT="evaluation_samples_all_0125"

# 定义需要合并的子文件夹名称
SUB_FOLDERS=("input" "ours" "output" "qwen_base")

# 遍历每个需要合并的子文件夹
for sub_folder in "${SUB_FOLDERS[@]}"; do
    # 创建目标子文件夹（-p 确保父文件夹也被创建，且已存在时不报错）
    mkdir -p "${TARGET_ROOT}/${sub_folder}"

    # 遍历每个源文件夹，复制对应子文件夹的内容
    for src_folder in "${SOURCE_FOLDERS[@]}"; do
        # 检查源子文件夹是否存在，避免报错
        if [ -d "${src_folder}/${sub_folder}" ]; then
            echo "正在复制 ${src_folder}/${sub_folder} 到 ${TARGET_ROOT}/${sub_folder}..."
            # -r：递归复制文件夹
            # -n：不覆盖已存在的文件（如需覆盖，将 -n 改为 -f）
            # --preserve=all：保留文件权限、时间戳等属性
            cp -r -n --preserve=all "${src_folder}/${sub_folder}/"* "${TARGET_ROOT}/${sub_folder}/"
        else
            echo "警告：${src_folder}/${sub_folder} 不存在，跳过..."
        fi
    done
done

echo "所有文件合并完成！目标文件夹：${TARGET_ROOT}"