# 合并数据集训练指南

## 问题背景

您有两种数据集：
1. **类型（1）**：纯文本图片描述 + 真实图片（训练了lora_1，结果在cache2）
2. **类型（2）**：控制图片（带文字）+ 编辑后结果图片（训练了lora_2，结果在cache5）

担心lora_2可能会忘记lora_1的能力，需要将两种数据集合并训练。

## 解决方案

### 方案1：直接合并已编码的cache（推荐）

如果cache2和cache5的数据已经编码好了，可以直接合并：

```bash
# 步骤1: 合并cache2和cache5到cache_combined
python merge_caches.py \
    --cache1 cache2 \
    --cache2 cache5 \
    --output cache_combined \
    --prefix1 "type1_" \
    --prefix2 "type2_"

# 步骤2: 使用合并后的数据集训练
bash consume_combined.sh
```

或者使用一键脚本：

```bash
bash merge_and_train.sh
```

### 方案2：重新生成cache（如果需要统一prompt）

如果您想重新生成cache以确保prompt的一致性，可以：

#### 2.1 为类型（1）数据生成cache（使用generate类型的prompt）

```bash
# 假设类型（1）的数据在 images_type1/ 和 descriptions_type1/
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
IMG="images_type1/"  # 真实图片目录
CTRL="descriptions_type1/"  # 文本描述图片目录（如果有）
CACHE="cache_type1/"

RESOLUTION=$((512*512))

python scripts/producer.py \
    --pretrained_model "$MODEL_PATH" \
    --img_dir "$IMG" \
    --control_dir "$CTRL" \
    --target_area $RESOLUTION \
    --output_dir "$CACHE" \
    --prompt_with_image \
    --prompt_type generate  # 使用generate类型的prompt
```

#### 2.2 为类型（2）数据生成cache（使用edit类型的prompt）

```bash
# 假设类型（2）的数据在 images_type2/ 和 control_type2/
MODEL_PATH="/storage/v-jinpewang/az_workspace/wenjun/Qwen-Image2/my_hf_cache/Qwen-Image-Edit"
IMG="images_type2/"  # 编辑后的结果图片
CTRL="control_type2/"  # 控制图片（带文字）
CACHE="cache_type2/"

RESOLUTION=$((512*512))

python scripts/producer.py \
    --pretrained_model "$MODEL_PATH" \
    --img_dir "$IMG" \
    --control_dir "$CTRL" \
    --target_area $RESOLUTION \
    --output_dir "$CACHE" \
    --prompt_with_image \
    --prompt_type edit  # 使用edit类型的prompt（默认）
```

#### 2.3 合并两个cache并训练

```bash
# 合并
python merge_caches.py \
    --cache1 cache_type1 \
    --cache2 cache_type2 \
    --output cache_combined \
    --prefix1 "type1_" \
    --prefix2 "type2_"

# 训练
bash consume_combined.sh
```

## Prompt说明

### 类型（1）- generate类型prompt
用于"根据文字描述生成图片"任务，prompt示例：
- "根据图片文字描述绘画出真实图片"
- "根据文字描述生成真实图片"
- "按照文字描述绘制真实图片"

### 类型（2）- edit类型prompt
用于"根据图片和文字指令编辑图像"任务，prompt示例：
- "根据图片中的文字指令编辑图像"
- "按照文字描述修改图片"
- "根据文字提示在图片上添加内容"

## 训练建议

1. **数据平衡**：确保两种类型的数据量大致平衡，或者根据重要性调整比例
2. **学习率**：合并训练时可以使用较小的学习率（如3e-4），避免破坏已有能力
3. **训练步数**：根据合并后的总数据量调整MAX_STEP
   - 如果cache2有N1个样本，cache5有N2个样本
   - 总步数 ≈ (N1 + N2) × epochs
4. **检查点**：定期保存检查点，观察loss变化，确保两种能力都得到保持

## 文件说明

- `merge_caches.py`: 合并两个cache目录的脚本
- `merge_and_train.sh`: 一键合并和训练的脚本
- `consume_combined.sh`: 使用合并后数据集训练的配置
- `scripts/producer.py`: 已更新支持两种prompt类型（--prompt_type参数）

## 注意事项

1. **文件名冲突**：如果cache2和cache5中有相同文件名的文件，建议使用--prefix1和--prefix2参数添加前缀
2. **数据一致性**：确保cache2和cache5中的文件是对应的（text_embs、img_embs、img_embs_control三个目录中的文件名应该匹配）
3. **验证合并结果**：合并后检查文件数量是否正确

