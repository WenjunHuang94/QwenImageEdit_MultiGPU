import os
import base64
import requests  # <--- 新增：用于下载 URL 图片
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation
from tqdm import tqdm
import time

# ================= 配置区域 =================
dashscope.api_key = "sk-1649650cd15847b685cd57def55a7a56"  # 注意保护你的 Key

# ... (路径配置保持不变) ...
ORIGINAL_IMAGES = "./images_with_box_instructions"
PROMPTS_FOLDER = "./images_with_box_instructions_result"
OUTPUT_FOLDER = "./images_with_box_instructions_edited_images_output"

RETRY_TIMES = 3
RETRY_DELAY = 2


def encode_image_to_base64(image_path):
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


# ==================== 修改核心：智能保存函数 ====================
def save_image_content(content_string, output_path):
    """
    智能保存图片：
    1. 如果是 URL (http开头)，则下载保存。
    2. 如果是 Base64，则解码保存（包含自动补全 padding 的逻辑）。
    """
    try:
        # 情况 A: 返回的是 URL 链接
        if content_string.startswith("http://") or content_string.startswith("https://"):
            # print("  ⬇️ 检测到图片 URL，正在下载...")
            response = requests.get(content_string, timeout=30)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                f.write(response.content)

        # 情况 B: 返回的是 Base64 字符串
        else:
            # 1. 移除可能存在的前缀
            if ',' in content_string:
                content_string = content_string.split(',', 1)[1]

            # 2. 修复 Padding 问题 (Base64长度必须是4的倍数)
            missing_padding = len(content_string) % 4
            if missing_padding:
                content_string += '=' * (4 - missing_padding)

            # 3. 解码并保存
            image_data = base64.b64decode(content_string)
            with open(output_path, 'wb') as f:
                f.write(image_data)

    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
        # 打印出 content_string 的前50个字符帮助调试
        print(f"   Content String snippet: {content_string[:50]}...")
        raise e  # 抛出异常让外层重试逻辑捕获


# ==============================================================

def call_qwen_image_edit(image_path, prompt_text):
    """调用 Qwen-Image-Edit API 进行图片编辑"""
    messages = [
        {
            "role": "user",
            "content": [
                {"image": encode_image_to_base64(image_path)},
                {"text": prompt_text}
            ]
        }
    ]

    try:
        response = MultiModalConversation.call(
            model='qwen-image-edit',  # 确保这是正确的模型名称
            messages=messages,
            result_format='message'
        )

        if response.status_code == HTTPStatus.OK:
            result = response.output.choices[0].message.content
            # 查找图片数据
            for item in result:
                if 'image' in item:
                    # 注意：这里返回的可能是 URL，也可能是 Base64
                    return item['image']

            print(f"⚠️ 响应中未找到图片数据")
            return None
        else:
            print(f"❌ API Error: {response.code} - {response.message}")
            return None

    except Exception as e:
        print(f"❌ Request Failed: {e}")
        return None


def process_single_image(img_file, original_folder, prompts_folder, output_folder):
    """处理单张图片"""
    img_path = os.path.join(original_folder, img_file)
    txt_filename = os.path.splitext(img_file)[0] + ".txt"
    prompt_path = os.path.join(prompts_folder, txt_filename)

    output_filename = os.path.splitext(img_file)[0] + "_edited" + os.path.splitext(img_file)[1]
    output_path = os.path.join(output_folder, output_filename)

    if os.path.exists(output_path):
        return "skipped"

    if not os.path.exists(prompt_path):
        print(f"⚠️ 未找到对应的 prompt 文件: {txt_filename}")
        return "no_prompt"

    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()

    for attempt in range(RETRY_TIMES):
        # 这里拿到的可能是 Base64 也可能是 URL
        image_content = call_qwen_image_edit(img_path, prompt_text)

        if image_content:
            try:
                # 使用新的智能保存函数
                save_image_content(image_content, output_path)
                return "success"
            except Exception:
                # 如果保存失败（比如下载超时），这也算作一次失败尝试，进入重试循环
                pass

        if attempt < RETRY_TIMES - 1:
            print(f"🔄 重试 {attempt + 1}/{RETRY_TIMES - 1}...")
            time.sleep(RETRY_DELAY)

    return "failed"


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(ORIGINAL_IMAGES)
                   if f.lower().endswith(valid_extensions)]

    print(f"📂 发现 {len(image_files)} 张图片，开始批量编辑...")
    print(f"📁 原始图片: {ORIGINAL_IMAGES}")
    print(f"📄 Prompt 文件: {PROMPTS_FOLDER}")
    print(f"💾 输出目录: {OUTPUT_FOLDER}")
    print("-" * 60)

    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "no_prompt": 0
    }

    for img_file in tqdm(image_files, desc="Editing Images"):
        result = process_single_image(
            img_file,
            ORIGINAL_IMAGES,
            PROMPTS_FOLDER,
            OUTPUT_FOLDER
        )

        stats[result] += 1
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("📊 处理完成统计:")
    print(f"  ✅ 成功: {stats['success']}")
    print(f"  ❌ 失败: {stats['failed']}")
    print(f"  ⏭️  跳过（已存在）: {stats['skipped']}")
    print(f"  ⚠️  无 Prompt: {stats['no_prompt']}")
    print(f"\n💾 编辑后的图片保存在: {OUTPUT_FOLDER}")
    print("=" * 60)


if __name__ == "__main__":
    main()