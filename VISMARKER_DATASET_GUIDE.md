# Vismarker数据集处理指南

## 数据集概述

您有8个不同的图像编辑数据集，每个数据集都包含：
- `input/`: 控制图（包含箭头指向和文字说明）
- `output/`: 结果图（编辑后的真实图像）

数据集列表：
1. `omniedit_attribute_modification` - 属性修改
2. `omniedit_object_swap` - 物体交换
3. `omniedit_removal` - 物体移除
4. `omniedit_swap` - 交换
5. `ultraedit_change_color` - 改变颜色
6. `ultraedit_change_local` - 局部改变
7. `ultraedit_replace` - 替换
8. `ultraedit_turn` - 转向/旋转

## 处理流程

### 步骤1: 批量生成预处理数据

```bash
bash produce_all_datasets.sh
```

这个脚本会：
- 自动处理所有8个数据集
- 为每个数据集生成独立的cache（保存在 `cache_vismarker/` 下）
- 默认每个数据集最多处理2000个样本（可在脚本中修改 `MAX_SAMPLES`）

**自定义参数**：
- 修改 `MAX_SAMPLES` 变量来调整每个数据集的样本数
- 如果某个数据集样本较少，可以设置更大的值或注释掉该参数处理全部

### 步骤2: 合并所有cache

```bash
bash merge_all_vismarker_caches.sh
```

这个脚本会：
- 合并所有8个数据集的cache
- 生成统一的cache保存在 `cache_vismarker_combined/`
- 默认每个数据集最多2000个样本（可在脚本中修改）

### 步骤3: 训练LoRA

```bash
bash consume_vismarker.sh
```

## 训练策略建议

### 方案1: 整合训练（推荐）

**优点**：
- 模型可以同时学习所有8种编辑任务
- 避免任务间的遗忘问题
- 一个模型支持多种编辑能力
- 数据多样性更好，泛化能力更强

**缺点**：
- 如果某个任务样本很少，可能学习不充分
- 需要确保数据平衡

**建议**：
- 如果每个数据集样本数相近（500-2000），推荐整合训练
- 如果某个数据集样本特别少（<100），可以考虑单独训练或增加该数据集的权重

### 方案2: 分开训练

**优点**：
- 每个任务可以独立优化
- 可以针对不同任务使用不同的超参数

**缺点**：
- 需要训练8个不同的LoRA模型
- 推理时需要根据任务选择不同的模型
- 任务间可能无法共享知识

**建议**：
- 如果某个任务非常特殊，或者样本数差异很大，可以考虑分开训练
- 可以先整合训练，如果效果不好再考虑分开

## 样本数量建议

### LoRA训练样本数量参考

1. **最少样本数**：
   - 每个任务至少 **500-1000** 个样本
   - 少于500个样本可能导致过拟合或学习不充分

2. **推荐样本数**：
   - 每个任务 **1000-2000** 个样本
   - 这个数量通常能获得较好的效果

3. **理想样本数**：
   - 每个任务 **2000-5000** 个样本
   - 更多样本通常能提升泛化能力

4. **整合训练总样本数**：
   - 如果8个数据集，每个2000样本，总共 **16000** 个样本
   - 这个数量对于LoRA训练是充足的

### 训练步数计算

假设：
- 总样本数：16000（8个数据集 × 2000样本）
- Batch size：1
- Epochs：2

训练步数 = 总样本数 × Epochs = 16000 × 2 = 32000 步

在 `consume_vismarker.sh` 中已经设置了 `MAX_STEP=32000`

## 使用示例

### 完整流程

```bash
# 1. 生成所有数据集的预处理cache
bash produce_all_datasets.sh

# 2. 合并所有cache
bash merge_all_vismarker_caches.sh

# 3. 训练LoRA
bash consume_vismarker.sh
```

### 自定义处理

如果某个数据集样本特别多，想限制数量：

```bash
# 修改 produce_all_datasets.sh 中的 MAX_SAMPLES
MAX_SAMPLES=1000  # 每个数据集只处理1000个样本
```

如果想处理全部样本：

```bash
# 注释掉 MAX_SAMPLES 参数
# MAX_SAMPLES=2000
```

然后在脚本中移除 `${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}` 这一行。

## 检查数据

处理完成后，可以检查cache：

```bash
# 检查每个数据集的cache
ls -lh cache_vismarker/*/text_embs/*.pt | wc -l
ls -lh cache_vismarker/*/img_embs/*.pt | wc -l
ls -lh cache_vismarker/*/img_embs_control/*.pt | wc -l

# 检查合并后的cache
ls -lh cache_vismarker_combined/text_embs/*.pt | wc -l
ls -lh cache_vismarker_combined/img_embs/*.pt | wc -l
ls -lh cache_vismarker_combined/img_embs_control/*.pt | wc -l
```

## 注意事项

1. **数据平衡**：确保8个数据集的样本数不要差异太大（建议比例在1:3以内）
2. **prompt类型**：使用 `--prompt_type edit`，已经包含了各种编辑任务的prompt变体
3. **随机打乱**：训练时数据会自动随机打乱，确保模型学习到所有任务
4. **检查点**：建议设置 `CKP=500`，定期保存检查点
5. **最佳模型**：启用 `SAVE_BEST=true`，自动保存最佳模型

