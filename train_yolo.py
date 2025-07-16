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
    dfl=1.5,                    # DFL损失权重
    
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