from ultralytics import YOLO
import torch

# 检查CUDA可用性
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset with improved configuration
train_results = model.train(
    data="coco8.yaml",          # Path to dataset configuration file
    epochs=100,                 # Number of training epochs
    imgsz=640,                  # Image size for training
    device="0",                 # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    
    # === 性能优化参数 ===
    batch=16,                   # 批次大小，根据GPU内存调整
    cache=True,                 # 缓存数据集到磁盘，显著提升训练速度
    workers=8,                  # 数据加载器工作进程数
    amp=True,                   # 自动混合精度训练，节省显存
    
    # === 训练控制参数 ===
    patience=30,                # 早停耐心值，30个epoch无改善后停止
    save_period=25,             # 每25个epoch保存一次模型
    val=True,                   # 训练期间进行验证
    plots=True,                 # 生成训练图表和可视化
    verbose=True,               # 详细输出训练信息
    
    # === 学习率和优化器 ===
    lr0=0.01,                   # 初始学习率
    lrf=0.01,                   # 最终学习率 = lr0 * lrf
    momentum=0.937,             # SGD动量
    weight_decay=0.0005,        # 权重衰减，防止过拟合
    warmup_epochs=3.0,          # 学习率预热轮数
    cos_lr=False,               # 是否使用余弦学习率调度
    
    # === 数据增强参数 ===
    hsv_h=0.015,                # 色调增强范围
    hsv_s=0.7,                  # 饱和度增强范围
    hsv_v=0.4,                  # 明度增强范围
    degrees=0.0,                # 旋转角度范围 (+/-deg)
    translate=0.1,              # 平移范围 (+/-fraction)
    scale=0.5,                  # 缩放范围 (+/-gain)
    fliplr=0.5,                 # 水平翻转概率
    mosaic=1.0,                 # 马赛克增强概率
    mixup=0.0,                  # 混合增强概率
    
    # === 损失函数权重 ===
    box=7.5,                    # 边界框损失权重
    cls=0.5,                    # 分类损失权重
    dfl=1.5,                    # DFL损失权重 # 可能在train中不起作用，因为输出84通道没有DFL数据
    
    # === 项目管理 ===
    project='yolo_training',    # 项目目录名称
    name='coco8_run',           # 实验名称
    
    # === 高级设置 ===
    seed=42,                    # 随机种子，保证可重复性
    deterministic=True,         # 确定性训练
    resume=False,               # 是否从上次检查点恢复训练
)

print(f"\n训练完成！最佳模型保存在: {train_results}")

# Evaluate the model's performance on the validation set
print("\n开始验证模型性能...")
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# Perform object detection on an image
print("\n测试图像预测...")
try:
    results = model("test_image.jpg")  # Predict on an image
    results[0].show()  # Display results
    print("预测完成，结果已显示")
    
    # === 查看模型的直接输出数据 ===
    print("\n=== 模型直接输出数据分析 ===")
    
    # 获取原始PyTorch模型
    raw_model = model.model
    
    # 准备输入数据（模拟YOLO预处理）
    import cv2
    img = cv2.imread("test_image.jpg")
    if img is not None:
        print(f"原始图像尺寸: {img.shape}")
        
        # YOLO预处理
        img_resized = cv2.resize(img, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # 添加batch维度
        
        # 确保张量在正确的设备上（GPU或CPU）
        device = next(raw_model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        print(f"预处理后张量形状: {img_tensor.shape}")
        print(f"张量设备: {img_tensor.device}")
        print(f"模型设备: {device}")
        
        # 获取模型原始输出
        with torch.no_grad():
            raw_outputs = raw_model(img_tensor)
        
        print(f"\n模型原始输出信息:")
        print(f"输出数量: {len(raw_outputs)}")
        
        for i, output in enumerate(raw_outputs):
            # 检查输出类型
            if isinstance(output, torch.Tensor):
                print(f"输出 {i}: 张量，形状 {output.shape}")
                print(f"  数据类型: {output.dtype}")
                print(f"  设备: {output.device}")
                print(f"  数值范围: [{output.min().item():.6f}, {output.max().item():.6f}]")
                print(f"  平均值: {output.mean().item():.6f}")
            elif isinstance(output, list):
                print(f"输出 {i}: 列表，包含 {len(output)} 个元素")
                for j, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        print(f"  列表元素 {j}: 张量，形状 {item.shape}")
                        print(f"    数据类型: {item.dtype}")
                        print(f"    设备: {item.device}")
                        print(f"    数值范围: [{item.min().item():.6f}, {item.max().item():.6f}]")
                        print(f"    平均值: {item.mean().item():.6f}")
                    else:
                        print(f"  列表元素 {j}: 类型 {type(item)}")
            else:
                print(f"输出 {i}: 未知类型 {type(output)}")
            
            # 详细分析主要输出张量
            if i == 0 and isinstance(output, torch.Tensor):  # 主要检测输出
                try:
                    if len(output.shape) == 3:
                        batch_size, channels, num_anchors = output.shape
                        print(f"\n=== 主要输出张量详细分析 ===")
                        print(f"批次大小: {batch_size}")
                        print(f"特征通道数: {channels} (包含坐标+类别+其他特征)")
                        print(f"锚点总数: {num_anchors}")
                        
                        # 分析不同部分的数据
                        output_data = output[0]  # 去除batch维度
                        
                        # 坐标信息 (前4个通道)
                        if channels >= 4:
                            coords = output_data[:4, :]
                            print(f"\n坐标信息 (前4个通道):")
                            print(f"  X坐标范围: [{coords[0].min().item():.3f}, {coords[0].max().item():.3f}]")
                            print(f"  Y坐标范围: [{coords[1].min().item():.3f}, {coords[1].max().item():.3f}]")
                            print(f"  宽度范围: [{coords[2].min().item():.3f}, {coords[2].max().item():.3f}]")
                            print(f"  高度范围: [{coords[3].min().item():.3f}, {coords[3].max().item():.3f}]")
                        
                        # 类别概率信息 (COCO有80个类别)
                        if channels >= 84:  # 4个坐标 + 80个类别
                            class_probs = output_data[4:84, :]
                            print(f"\n类别概率信息 (80个COCO类别):")
                            print(f"  概率范围: [{class_probs.min().item():.6f}, {class_probs.max().item():.6f}]")
                            print(f"  平均概率: {class_probs.mean().item():.6f}")
                            
                            # 找到置信度最高的检测
                            max_class_scores, max_class_indices = torch.max(class_probs, dim=0)
                            top_conf_idx = torch.argmax(max_class_scores)
                            
                            print(f"\n置信度最高的检测:")
                            print(f"  锚点索引: {top_conf_idx.item()}")
                            print(f"  最高置信度: {max_class_scores[top_conf_idx].item():.6f}")
                            print(f"  预测类别ID: {max_class_indices[top_conf_idx].item()}")
                            if channels >= 4:
                                print(f"  对应坐标: x={coords[0, top_conf_idx].item():.3f}, "
                                      f"y={coords[1, top_conf_idx].item():.3f}, "
                                      f"w={coords[2, top_conf_idx].item():.3f}, "
                                      f"h={coords[3, top_conf_idx].item():.3f}")
                        
                        # 锚点分布分析
                        print(f"\n锚点分布分析:")
                        print(f"8400个锚点来自不同尺度的特征图:")
                        print(f"  - 80×80 特征图: 6400个锚点 (细粒度检测)")
                        print(f"  - 40×40 特征图: 1600个锚点 (中等尺度检测)")
                        print(f"  - 20×20 特征图: 400个锚点 (大目标检测)")
                        print(f"  总计: 6400 + 1600 + 400 = 8400 个锚点")
                    else:
                        print(f"\n⚠️  输出张量维度不符合预期: {output.shape}")
                        print("   预期: 3维张量 [batch_size, channels, num_anchors]")
                        
                except Exception as e:
                    print(f"\n❌ 分析主要输出张量时出错: {e}")
                    print(f"   输出张量形状: {output.shape}")
                    print(f"   输出张量类型: {type(output)}")
        
        print(f"\n=== YOLO11双输出结构说明 ===")
        print("🎯 输出0: 主要检测结果")
        print("  - 形状: [1, 84, 8400]")
        print("  - 84个通道 = 4个坐标特征 + 80个类别概率")
        print("  - 8400个锚点来自3个尺度特征图的合并")
        print("  - 用户友好的简化输出")
        print()
        print("🎯 输出1: 原始特征图列表")
        print("  - 包含3个张量的列表")
        print("  - 张量0: [1, 144, 80, 80] - 高分辨率特征")
        print("  - 张量1: [1, 144, 40, 40] - 中分辨率特征") 
        print("  - 张量2: [1, 144, 20, 20] - 低分辨率特征")
        print("  - 144个通道包含完整训练特征(含DFL)")
        print("  - 用于调试和高级应用")
        print()
        print("=== 后处理算法说明 ===")
        print("1. 置信度过滤: 保留置信度 > threshold 的检测")
        print("2. 坐标解码: 将相对坐标转换为绝对像素坐标")
        print("3. NMS(非极大值抑制): 移除重叠的检测框")
        print("4. 坐标缩放: 将640×640尺寸的坐标缩放到原图尺寸")
        
    else:
        print("无法读取测试图像，跳过直接输出分析")
        
except Exception as e:
    print(f"预测失败: {e}")
    print("提示: 请确保test_image.jpg文件存在")

# Export the model to ONNX format for deployment
print("\n导出ONNX模型...")
try:
    path = model.export(format="onnx")  # Returns the path to the exported model
    print(f"ONNX模型已导出到: {path}")
except Exception as e:
    print(f"导出失败: {e}")

print("\n=== 训练流程完成 ===")
print("检查以下目录获取训练结果:")
print("- yolo_training/coco8_run/weights/ (模型权重)")
print("- yolo_training/coco8_run/ (训练图表和日志)")