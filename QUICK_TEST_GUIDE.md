# 快速测试指南

## 为什么先用小样本测试？

1. **快速验证流程**：确保数据预处理和训练流程没有错误
2. **节省时间**：10000张样本预处理可能需要数小时，小样本只需几分钟
3. **快速迭代**：可以快速调整超参数和训练策略
4. **节省资源**：避免在错误配置上浪费大量计算资源

## 推荐的测试流程

### 阶段1：超小样本测试（500张）
- **目的**：验证整个流程是否正常工作
- **时间**：预处理约10-20分钟，训练约30分钟-1小时
- **命令**：
```bash
# 修改 produce.sh，添加 --max_samples 500
bash produce.sh --max_samples 500
```

### 阶段2：小样本测试（1000-2000张）
- **目的**：观察模型是否能够学习，loss是否正常下降
- **时间**：预处理约20-40分钟，训练约1-2小时
- **命令**：
```bash
bash produce.sh --max_samples 1000
```

### 阶段3：中等样本训练（5000张）
- **目的**：评估模型效果，决定是否需要调整策略
- **时间**：预处理约1-2小时，训练约3-5小时

### 阶段4：全量训练（10000张或更多）
- **目的**：最终模型训练
- **时间**：预处理约2-4小时，训练约5-10小时

## 使用方法

### 方法1：直接在命令行添加参数
```bash
bash produce.sh --max_samples 500
```

### 方法2：修改 produce.sh 文件
在 `produce.sh` 中添加 `--max_samples` 参数：
```bash
python scripts/producer.py \
    --pretrained_model "$MODEL_PATH" \
    --img_dir "$IMG" \
    --control_dir "$CTRL" \
    --target_area $RESOLUTION \
    --output_dir "$CACHE" \
    --prompt_with_image \
    --max_samples 500 \  # 添加这一行
    "$@"
```

## 训练步数估算

假设 `batch_size=1`，`gradient_accumulation_steps=1`：

- **500张样本**：1个epoch = 500步，训练1-2个epoch即可测试
- **1000张样本**：1个epoch = 1000步，训练1-2个epoch
- **10000张样本**：1个epoch = 10000步，训练2-3个epoch

## 注意事项

1. **文件对应关系**：确保 `img_dir` 和 `control_dir` 中的文件名对应（通过stem匹配）
2. **缓存目录**：每次测试使用不同的 `output_dir`，避免覆盖之前的缓存
3. **训练配置**：在 `consume.sh` 中，`MAX_STEP` 应该设置为至少一个epoch的步数

## 快速测试检查清单

- [ ] 数据预处理完成，没有错误
- [ ] 缓存文件数量正确（text_embs、img_embs、img_embs_control数量一致）
- [ ] 训练可以正常启动，loss正常下降
- [ ] 生成效果符合预期（至少能看到一些学习迹象）
- [ ] 没有OOM（显存溢出）错误

如果以上都通过，再扩展到更多样本！

