import json
import os
from pathlib import Path

# 配置路径
INPUT_DIR = "./evaluation_samples_with_textbox_0124V1/input"
GT_DIR = "./evaluation_samples_with_textbox_0124V1/output"
OUTPUT_JSON = "./evaluation_samples_with_textbox_0124V1/test_set_auto.json"

# 默认 Prompt
DEFAULT_PROMPT = "Modify the image at the annotated location according to the text instruction"

test_data = []

# 遍历所有输入图片
img_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.PNG')
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(img_extensions)]

for filename in sorted(files):
    sample_id = Path(filename).stem  # 获取不带后缀的文件名作为 ID

    # 构建单个条目
    item = {
        "id": sample_id,
        "type": "in_image_text",  # 默认类型，评估脚本会按此分类
        "input_image": str(Path(INPUT_DIR) / filename),
        "ground_truth": str(Path(GT_DIR) / filename),
        "text_instruction": DEFAULT_PROMPT,
        "marker_box": None,
        "arrow_region": None,
        "object_name": None
    }

    # 检查 GT 是否存在，如果不存在则跳过该样本的评估
    if os.path.exists(item["ground_truth"]):
        test_data.append(item)
    else:
        print(f"⚠ Warning: GT not found for {filename}, skipping...")

# 写入 JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"Successfully generated {len(test_data)} samples to {OUTPUT_JSON}")