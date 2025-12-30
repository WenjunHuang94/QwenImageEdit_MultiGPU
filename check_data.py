import webdataset as wds
import os


def check_shards_with_stats(root_dir, shard_input, save_dir="./check_results"):
    os.makedirs(save_dir, exist_ok=True)
    error_log = os.path.join(save_dir, "mismatch_errors.txt")

    if isinstance(shard_input, int):
        shard_ids = [str(i) for i in range(shard_input)]
        print(f"🔍 模式：自动检查并统计前 {shard_input} 个分片 (0 到 {shard_input - 1})")
    else:
        shard_ids = shard_input
        print(f"🔍 模式：统计指定分片列表 {shard_ids}")

    for sid in shard_ids:
        shard_name = f"data_{int(sid):06d}.tar"

        p_in = os.path.join(root_dir, "input", shard_name)
        p_out = os.path.join(root_dir, "output", shard_name)
        p_txt = os.path.join(root_dir, "text_save", shard_name)

        if not all(os.path.exists(p) for p in [p_in, p_out, p_txt]):
            print(f"⚠️ 跳过 {shard_name}: 文件不全")
            continue

        print(f"\n🚀 正在校验并统计: {shard_name}")

        try:
            # 注意：统计数据量时不需要 decode("pil")，直接 decode() 会快很多
            ds_in = wds.WebDataset(p_in).decode()
            ds_out = wds.WebDataset(p_out).decode()
            ds_txt = wds.WebDataset(p_txt).decode()

            sample_count = 0
            mismatch_count = 0

            # 遍历整个分片以获取准确总数
            for i, (s_in, s_out, s_txt) in enumerate(zip(ds_in, ds_out, ds_txt)):
                k1, k2, k3 = s_in["__key__"], s_out["__key__"], s_txt["__key__"]

                # 仅对前 3 个进行 Match 打印，避免刷屏
                if i < 5:
                    if k1 == k2 == k3:
                        print(f"  ✅ [Match] {k1}")
                    else:
                        mismatch_count += 1
                        msg = f"❌ [MISMATCH] Shard: {shard_name}, Index: {i}, Keys: In:{k1}, Out:{k2}, Txt:{k3}"
                        print(f"  {msg}")
                        with open(error_log, "a") as f:
                            f.write(msg + "\n")

                # 如果发现 Key 不一致，增加计数
                elif k1 != k2 or k1 != k3:
                    mismatch_count += 1

                sample_count += 1

            print(f"📊 统计结果: {shard_name} 共有 {sample_count} 个样本" +
                  (f" (⚠️ 发现 {mismatch_count} 个错误)" if mismatch_count > 0 else " (全部对齐)"))

        except Exception as e:
            print(f"❌ 读取分片 {shard_name} 时发生错误: {e}")


# --- 运行 ---
root = "/storage/v-jinpewang/lab_folder/junchao/data/large_scale/text_image/text2image_refine_new_2/"
check_shards_with_stats(root, 100)