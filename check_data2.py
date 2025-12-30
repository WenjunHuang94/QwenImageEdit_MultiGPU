import webdataset as wds
import os
from PIL import Image


def extract_and_save_samples(root_dir, shard_input, save_base_dir="./extracted_samples"):
    # --- 核心修改：支持整数输入 ---
    if isinstance(shard_input, int):
        # 如果输入 100，则 shard_ids 为 ["0", "1", ..., "99"]
        shard_ids = [str(i) for i in range(shard_input)]
        print(f"📦 模式：自动提取前 {shard_input} 个分片的数据")
    else:
        shard_ids = shard_input
        print(f"📦 模式：提取指定分片列表 {shard_ids}")

    for sid in shard_ids:
        shard_name = f"data_{int(sid):06d}.tar"

        p_in = os.path.join(root_dir, "input", shard_name)
        p_out = os.path.join(root_dir, "output", shard_name)
        p_txt = os.path.join(root_dir, "text_save", shard_name)

        # 检查文件是否存在，不存在则跳过
        if not all(os.path.exists(p) for p in [p_in, p_out, p_txt]):
            print(f"⚠️ 跳过 {shard_name}: 文件不全")
            continue

        print(f"\n🚀 正在导出分片内容: {shard_name}")

        # 准备该分片的存储目录
        shard_save_path = os.path.join(save_base_dir, f"shard_{int(sid):06d}")
        for sub in ["input", "output", "text"]:
            os.makedirs(os.path.join(shard_save_path, sub), exist_ok=True)

        # 加载流
        try:
            ds_in = wds.WebDataset(p_in).decode("pil")
            ds_out = wds.WebDataset(p_out).decode("pil")
            ds_txt = wds.WebDataset(p_txt).decode()

            for i, (s_in, s_out, s_txt) in enumerate(zip(ds_in, ds_out, ds_txt)):
                if i >= 10: break  # 每个 shard 只提取前 10 个用于检查

                key = s_in["__key__"]

                # 提取图片和文本
                img_in = s_in.get("jpg") or s_in.get("png") or s_in.get("jpeg")
                img_out = s_out.get("png") or s_out.get("jpg")
                text_content = s_txt.get("txt")

                # 保存到本地
                if img_in:
                    img_in.save(os.path.join(shard_save_path, "input", f"{key}.jpg"))
                if img_out:
                    img_out.save(os.path.join(shard_save_path, "output", f"{key}.png"))
                if text_content:
                    with open(os.path.join(shard_save_path, "text", f"{key}.txt"), "w", encoding="utf-8") as f:
                        f.write(str(text_content))

                print(f"  💾 已还原样本: {key}")
        except Exception as e:
            print(f"❌ 提取 {shard_name} 时出错: {e}")


# --- 运行示例 ---
root = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text2image_refine_new_2/"

# 示例 1: 提取 0-99 号分片（每个分片提 10 个样本）
extract_and_save_samples(root, 100)

# 示例 2: 提取特定的分片
# extract_and_save_samples(root, ["0", "42"])