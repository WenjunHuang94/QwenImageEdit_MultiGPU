import os
import base64
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation
from tqdm import tqdm  # 进度条库，如果没有请运行 pip install tqdm

# ================= 配置区域 =================
# 1. 填入你的阿里云 API Key
dashscope.api_key = "sk-1649650cd15847b685cd57def55a7a56"

# 2. 输入和输出路径
# INPUT_FOLDER: 存放那些"写了文字指令的图片"的文件夹
INPUT_FOLDER = "./images_with_text_instructions"
# OUTPUT_FOLDER: 存放生成好的 Prompt 文本文件的文件夹
OUTPUT_FOLDER = "./images_with_text_instructions_result"

# ================= 系统提示词 (你提供的版本) =================
SYSTEM_PROMPT = """
You are a professional image editing assistant. Based on the following input, generate a detailed, precise, and actionable text instruction that can be directly used by an image editing model (e.g., Qwen-Image-Edit).

Input:
- Image content: [Describe the main elements in the image, including subjects, background, lighting, weather, time of day, etc.]
- In-image text: [Extract and describe any visible text written within the image, such as "Add a white horse with a saddle in the background."]

Requirements:
1. Clearly identify what object(s) should be added, removed, or modified based on the in-image text.
2. Determine the most plausible location for the new object(s), using relative spatial references (e.g., "in the background behind the right character", "to the left of the tree") or depth cues.
3. Describe visual attributes in detail: color, size, type, orientation, material, style (realistic, cinematic, etc.), and whether it should appear partially occluded or fully visible.
4. Ensure environmental consistency: match lighting direction, shadow casting, perspective, scale, and texture with the original scene.
5. Explicitly instruct the model to remove any visible text annotations from the final image.
6. Use fluent natural language but be highly specific—avoid ambiguity so the editing model can execute accurately.

Output format:
【Editing Instruction】: [A single, coherent, and comprehensive sentence or paragraph describing the edit.]
"""


def encode_image_to_base64(image_path):
    """将本地图片转换为 Base64 编码，以便通过 API 发送"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def call_qwen_vl_max(image_path):
    """调用 Qwen-VL-Max 生成编辑指令"""

    # 构建消息
    # 注意：这里 User Prompt 不需要再传具体的 "User Intent" 变量了
    # 而是告诉模型：去读图片里的字！
    messages = [
        {
            "role": "system",
            "content": [{"text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"image": encode_image_to_base64(image_path)},  # 传入包含文字指令的图片
                {
                    "text": "Analyze this image. Identify the text instruction written on the image, and generate the professional Editing Instruction as requested."}
            ]
        }
    ]

    try:
        # 必须使用 VL 系列模型，因为它需要“看”图片
        response = MultiModalConversation.call(
            model='qwen-vl-max-latest',
            messages=messages,
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content[0]['text']
        else:
            print(f"API Error: {response.code} - {response.message}")
            return None

    except Exception as e:
        print(f"Request Failed: {e}")
        return None


def main():
    # 1. 确保输出目录存在
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]

    print(f"📂 发现 {len(image_files)} 张图片，开始批量生成 Prompt...")

    # 3. 使用进度条遍历处理
    for img_file in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(INPUT_FOLDER, img_file)

        # 构造对应的输出 txt 文件名
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        save_path = os.path.join(OUTPUT_FOLDER, txt_filename)

        # 断点续传：如果文件已存在，跳过
        if os.path.exists(save_path):
            continue

        # 4. 调用 API
        generated_prompt = call_qwen_vl_max(img_path)

        if generated_prompt:
            # 5. 保存结果到 txt
            # 这里保存的内容就是类似 "【Editing Instruction】: Add a white horse..." 的文本
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(generated_prompt)
        else:
            print(f"❌ 处理失败: {img_file}")

    print(f"\n✅ 全部完成！结果保存在: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()