import os
import base64
from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation
from tqdm import tqdm
import argparse
import time

# ================= 配置区域 =================
dashscope.api_key = "sk-1649650cd15847b685cd57def55a7a56"

# ================= 三种系统提示词 =================

# 1. 纯文本指令（原版）
SYSTEM_PROMPT_TEXT_ONLY = """
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

# 2. 箭头 + 文本指令
SYSTEM_PROMPT_ARROW = """
You are a professional image editing assistant. Based on the following input, generate a detailed, precise, and actionable text instruction that can be directly used by an image editing model (e.g., Qwen-Image-Edit).

Input:  
- Image content: [Describe the main elements in the image, including subjects, background, lighting, weather, time of day, etc.]  
- Visual annotations: [Describe any arrows, bounding boxes, or text labels present. Specify the object pointed to by the arrow and its approximate location. Also quote the exact textual instruction written in the image.]

Requirements:  
1. Clearly identify the target object indicated by the arrow (e.g., "the red hoodie").  
2. Determine the exact edit type based on the quoted text: swap, replace, change color, delete, add, etc.  
3. Describe the new object in detail: color, material, style, size, orientation, and how it should fit into the scene (e.g., "light blue denim button-up shirt with two chest pockets").  
4. Ensure environmental consistency: match lighting direction, shadows, fabric texture, perspective, and scale with the original image.  
5. If the edit involves replacing an object, explicitly instruct the model to remove the original object and seamlessly integrate the new one without visible seams, distortions, or artifacts.  
6. Explicitly and unambiguously instruct the model to first erase all artificial visual annotations before performing the main edit. This includes:  
   - The arrow (specify its color and direction if known),  
   - The adjacent text label (quote the exact words),  
   - Any other non-original markings such as boxes, lines, or highlights.  
   Ensure these elements are completely removed with no residual pixels, halos, blurring, or background mismatch—the final image must appear as if no annotations were ever present.  
7. Use fluent natural language but be highly specific—avoid ambiguity so the editing model can execute accurately.

Output format:  
【Editing Instruction】: [A single, coherent, and comprehensive sentence or paragraph describing the edit. Begin by instructing the removal of all annotations, then describe the main edit in full detail.]
"""

# 3. 框 + 文本指令
SYSTEM_PROMPT_BOX = """
You are a professional image editing assistant. Based on the following input, generate a detailed, precise, and actionable text instruction that can be directly used by an image editing model (e.g., Qwen-Image-Edit).

Input:
- Image content: [Describe the main elements in the image, including subjects, background, lighting, weather, time of day, etc.]
- Visual annotations: [Describe any human-added markings such as red bounding boxes, arrows, handwritten or typed text labels, and their spatial relationship to objects in the scene]

Requirements:
1. Clearly specify what object should be added, removed, or modified (e.g., "a red toy car").
2. Precisely indicate its location using relative spatial references (e.g., "5 cm in front of the elephant's left front foot") or approximate coordinates if appropriate.
3. Describe visual attributes in detail: color, size, type, orientation, material, style (e.g., realistic vs. cartoon), etc.
4. Ensure environmental consistency: match lighting direction, shadow casting, ground texture, perspective, and scale with the original scene.
5. Use fluent natural language, but be highly specific—avoid ambiguity so the editing model can execute the change accurately.
6. If the image contains artificial annotations (e.g., red bounding boxes, text labels like 'create a violin'), explicitly include instructions to remove or erase them from the final edited image.

Output format:
【Editing Instruction】: [A single, coherent, and comprehensive sentence or paragraph describing the edit.]
"""

# 提示词映射
PROMPT_TYPES = {
    "text": SYSTEM_PROMPT_TEXT_ONLY,
    "arrow": SYSTEM_PROMPT_ARROW,
    "box": SYSTEM_PROMPT_BOX,
}


def encode_image_to_base64(image_path):
    """将本地图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def call_qwen_vl_max(image_path, system_prompt, user_prompt_hint):
    """调用 Qwen-VL-Max 生成编辑指令"""
    messages = [
        {
            "role": "system",
            "content": [{"text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"image": encode_image_to_base64(image_path)},
                {"text": user_prompt_hint}
            ]
        }
    ]

    try:
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


def process_single_image(img_file, input_folder, output_folder, system_prompt, user_prompt_hint):
    """处理单张图片"""
    img_path = os.path.join(input_folder, img_file)
    txt_filename = os.path.splitext(img_file)[0] + ".txt"
    save_path = os.path.join(output_folder, txt_filename)

    # 断点续传：如果文件已存在，跳过
    if os.path.exists(save_path):
        return "skipped"

    # 调用 API
    generated_prompt = call_qwen_vl_max(img_path, system_prompt, user_prompt_hint)

    if generated_prompt:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(generated_prompt)
        return "success"
    else:
        return "failed"


def main():
    parser = argparse.ArgumentParser(description="统一批量生成图片编辑 Prompt")
    
    # 必选参数：标注类型
    parser.add_argument(
        "--annotation_type",
        type=str,
        required=True,
        choices=["text", "arrow", "box"],
        help="标注类型: text(纯文本), arrow(箭头+文本), box(框+文本)"
    )
    
    # 输入输出路径
    parser.add_argument("--input_folder", type=str, required=True, help="输入图片文件夹")
    parser.add_argument("--output_folder", type=str, required=True, help="输出 Prompt 文件夹")
    
    # 可选参数
    parser.add_argument("--api_key", type=str, default=None, help="阿里云 API Key（可选，默认使用脚本中的）")
    parser.add_argument("--retry_delay", type=float, default=0.5, help="API 调用间隔（秒），避免限流")
    
    args = parser.parse_args()
    
    # 设置 API Key
    if args.api_key:
        dashscope.api_key = args.api_key
    
    # 确保输出目录存在
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # 选择对应的系统提示词
    system_prompt = PROMPT_TYPES[args.annotation_type]
    
    # 根据类型定制用户提示
    user_prompt_hints = {
        "text": "Analyze this image. Identify the text instruction written on the image, and generate the professional Editing Instruction as requested.",
        "arrow": "Analyze this image. Identify the arrow pointing to an object and the text instruction next to it. Generate a professional Editing Instruction that first removes all annotations, then performs the requested edit.",
        "box": "Analyze this image. Identify the bounding box and the text instruction describing what to add/remove/modify. Generate a professional Editing Instruction that removes all annotations and performs the edit."
    }
    user_prompt_hint = user_prompt_hints[args.annotation_type]
    
    # 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(args.input_folder) 
                   if f.lower().endswith(valid_extensions)]
    
    print("=" * 60)
    print(f"🎯 标注类型: {args.annotation_type}")
    print(f"📁 输入文件夹: {args.input_folder}")
    print(f"💾 输出文件夹: {args.output_folder}")
    print(f"🖼️  发现 {len(image_files)} 张图片")
    print("=" * 60)
    
    # 统计信息
    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0
    }
    
    # 批量处理
    for img_file in tqdm(image_files, desc="Generating Prompts"):
        result = process_single_image(
            img_file,
            args.input_folder,
            args.output_folder,
            system_prompt,
            user_prompt_hint
        )
        
        stats[result] += 1
        
        # 避免 API 限流
        if result == "success":
            time.sleep(args.retry_delay)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("📊 处理完成统计:")
    print(f"  ✅ 成功: {stats['success']}")
    print(f"  ❌ 失败: {stats['failed']}")
    print(f"  ⏭️  跳过（已存在）: {stats['skipped']}")
    print(f"\n💾 生成的 Prompt 保存在: {args.output_folder}")
    print("=" * 60)


if __name__ == "__main__":
    main()
