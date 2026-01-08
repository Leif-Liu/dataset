# 数据处理优化指南 | Data Processing Optimization Guide

本文档总结了SFT训练中数据处理的优化措施和最佳实践。

## 🚀 已实现的优化措施

### 1. 并发处理优化 (3-5x 性能提升)

**实现方式:**
- 启用多进程并行处理 (`num_proc=8`)
- 自动检测CPU核数并设置合理默认值
- 批量tokenization利用tokenizer内部并发

**配置参数:**
```yaml
data:
  processing:
    num_proc: auto  # 或具体数值如 8
```

**性能提升:**
- 单进程处理: ~100 样本/秒
- 8进程处理: ~400-500 样本/秒
- **预期加速比: 3-5x**

### 2. 智能缓存机制

**实现方式:**
- 基于配置生成MD5缓存键
- 避免重复计算相同配置的数据
- 缓存文件采用高效的Arrow格式

**缓存内容包括:**
- tokenizer配置 (`name_or_path`)
- 模板 (`template`)
- 序列长度 (`max_length`)
- 训练设置 (`train_on_prompt`, `add_eos_token`)

**配置参数:**
```yaml
data:
  processing:
    enable_cache: true
    cache_dir: data/cache
```

### 3. 批量处理优化

**实现方式:**
- 批量文本构建和tokenization
- 延迟padding到DataCollator阶段
- 批量数据验证和清洗

**配置参数:**
```yaml
data:
  processing:
    batch_size: 1000  # 可根据内存调整
```

**内存优化:**
- 避免提前padding，节省内存
- 批量处理减少函数调用开销
- 支持大数据集的流式处理

### 4. 数据验证和清洗

**验证规则:**
- 基础长度检查（最小5字符）
- 序列长度预估（4字符≈1token）
- 内容清洗（去除多余换行）

**配置参数:**
```yaml
data:
  processing:
    enable_validation: true
```

## 📊 性能基准测试

### 运行性能测试

```bash
cd /home/liufeng/sdk-ragflow/sft
python benchmark_data_processing.py --samples 1000 --output benchmark_results.json
```

**测试参数:**
- 样本数量: 1000
- 进程数: [1, 2, 4, 8]
- 批次大小: [100, 500, 1000, 2000]

### 预期性能提升

| 配置 | 处理时间 | 样本/秒 | 加速比 |
|------|----------|---------|---------|
| 单进程基准 | 10.0s | 100.0 | 1.0x |
| 8进程+1000批次 | 2.5s | 400.0 | 4.0x |
| 最优配置 | 2.0s | 500.0 | 5.0x |

## 🔧 配置优化建议

### 1. 并发进程数优化

```yaml
data:
  processing:
    # 推荐设置: CPU核数的50-100%
    num_proc: 8        # 8核CPU推荐值
    # num_proc: auto   # 自动检测 (默认)
```

**调优原则:**
- CPU密集型任务: `num_proc = CPU核数`
- 内存受限环境: 减少到 `CPU核数/2`
- 大数据集: 可增加到 `CPU核数 * 1.5`

### 2. 批次大小优化

```yaml
data:
  processing:
    batch_size: 1000   # 标准配置
    # batch_size: 500  # 内存较小时
    # batch_size: 2000 # 内存充足时
```

**调优原则:**
- 内存充足: 增大批次减少overhead
- 内存受限: 减小批次避免OOM
- 观察内存使用率调整

### 3. 缓存策略

```yaml
data:
  processing:
    enable_cache: true
    cache_dir: data/cache  # 使用SSD存储
```

**最佳实践:**
- 开发阶段: 启用缓存避免重复处理
- 生产环境: 定期清理过期缓存
- 存储选择: 使用SSD提高I/O性能

## 🛠️ 使用方法

### 1. 在训练脚本中启用优化

优化已自动集成到 `training/sft.py` 中，无需额外配置。

### 2. 自定义配置

修改 `configs/sft.yaml` 中的处理参数:

```yaml
data:
  processing:
    num_proc: 8        # 并发进程数
    batch_size: 1000   # 批处理大小
    enable_cache: true # 启用缓存
    enable_validation: true # 启用验证
```

### 3. 使用优化的数据处理器

```python
from data.processors import DataProcessor

processor = DataProcessor(
    tokenizer=tokenizer,
    template=template,
    max_length=2048,
    num_proc=8,          # 并发处理
    batch_size=1000,     # 批量大小
    enable_cache=True,   # 启用缓存
)

dataset = processor.process_dataset(dataset)
```

## 📈 监控和调优

### 1. 性能监控

训练时观察日志输出:
```
🚀 Creating optimized data processor...
   - num_proc: 8
   - batch_size: 1000
   - enable_cache: true
   - enable_validation: true

📋 Validating data... (samples: 1000)
✅ Validation complete. Valid samples: 987

🚀 Starting batch processing...
   - Batch size: 1000
   - Processes: 8
   - Cache: Enabled
   - Cache file: data/cache/train_a1b2c3d4.arrow

✅ Processing complete! Processed samples: 987
📊 Dataset statistics:
   - Average sequence length: 456.7
   - Max sequence length: 2048
```

### 2. 性能调优指标

**吞吐量指标:**
- 目标: >300 样本/秒
- 基准: ~100 样本/秒 (单进程)
- 优化后: 400-500 样本/秒

**资源利用率:**
- CPU使用率: 70-90%
- 内存使用: <80% 系统总内存
- 磁盘I/O: 缓存命中率 >80%

### 3. 故障排除

**常见问题:**

1. **内存不足 (OOM)**
   - 解决方案: 减少 `batch_size` 或 `num_proc`

2. **进程数过多导致性能下降**
   - 解决方案: 设置 `num_proc = CPU核数`

3. **缓存文件过大**
   - 解决方案: 定期清理 `data/cache/` 目录

4. **tokenizer加载失败**
   - 解决方案: 确保网络连接和模型访问权限

## 🎯 预期收益

通过实施这些优化措施，您可以期待:

1. **处理速度提升:** 3-5倍性能改进
2. **资源利用率:** CPU多核心充分利用
3. **开发效率:** 缓存机制避免重复计算
4. **数据质量:** 自动验证和清洗确保数据可靠性
5. **可扩展性:** 支持大数据集的流式处理

这些优化特别适用于:
- 大规模数据集训练 (>10K样本)
- 多次实验迭代的开发环境
- 计算资源受限的环境
- 需要快速原型验证的场景