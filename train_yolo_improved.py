#!/usr/bin/env python3
"""
改进的YOLO训练脚本 - 包含更优的超参数配置
"""

from ultralytics import YOLO
import torch

def main():
    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n.pt")
    
    # 检查可用设备
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # ===== 基础配置（适合初学者） =====
    print("\n=== 开始基础训练配置 ===")
    basic_results = model.train(
        data="coco8.yaml",
        epochs=50,              # 减少轮数用于快速测试
        imgsz=640,
        batch=16,               # 明确指定批次大小
        device="0",
        
        # 性能优化
        cache=True,             # 缓存数据集到磁盘，加速训练
        workers=8,              # 数据加载器工作进程
        amp=True,               # 自动混合精度，节省显存
        
        # 训练监控
        patience=20,            # 早停耐心值
        save_period=10,         # 每10个epoch保存一次
        plots=True,             # 生成训练图表
        verbose=True,           # 详细输出
        
        # 项目管理
        project='yolo_training', # 项目目录
        name='basic_run',       # 实验名称
    )
    
    # ===== 高级配置（适合生产环境） =====
    print("\n=== 开始高级训练配置 ===")
    
    # 重新加载模型（如果需要）
    model = YOLO("yolo11n.pt")
    
    advanced_results = model.train(
        data="coco8.yaml",
        epochs=200,             # 更多轮数
        imgsz=640,
        batch=32,               # 更大批次（根据GPU内存调整）
        device="0",             # 如果有多GPU: [0,1,2,3]
        
        # 优化器和学习率
        optimizer='AdamW',      # 使用AdamW优化器
        lr0=0.001,             # 初始学习率
        lrf=0.01,              # 最终学习率比例
        momentum=0.937,        # 动量
        weight_decay=0.0005,   # 权重衰减
        warmup_epochs=3.0,     # 预热轮数
        cos_lr=True,           # 余弦学习率调度
        
        # 数据增强（增强训练数据多样性）
        hsv_h=0.015,           # 色调增强
        hsv_s=0.7,             # 饱和度增强
        hsv_v=0.4,             # 明度增强
        degrees=10.0,          # 旋转角度范围
        translate=0.1,         # 平移范围
        scale=0.5,             # 缩放范围
        shear=2.0,             # 剪切角度
        perspective=0.0001,    # 透视变换
        flipud=0.0,            # 垂直翻转概率
        fliplr=0.5,            # 水平翻转概率
        mosaic=1.0,            # 马赛克增强
        mixup=0.15,            # 混合增强
        copy_paste=0.3,        # 复制粘贴增强
        
        # 损失函数权重调整
        box=7.5,               # 边界框损失权重
        cls=0.5,               # 分类损失权重
        dfl=1.5,               # DFL损失权重
        label_smoothing=0.1,   # 标签平滑
        
        # 性能优化
        cache='ram',           # 缓存到内存（需要足够内存）
        workers=16,            # 更多工作进程
        amp=True,              # 混合精度训练
        multi_scale=True,      # 多尺度训练
        
        # 训练控制
        patience=50,           # 早停耐心值
        save_period=25,        # 保存周期
        val=True,              # 训练期间验证
        plots=True,            # 生成图表
        verbose=True,
        
        # 项目管理
        project='yolo_training',
        name='advanced_run',
        
        # 高级设置
        seed=42,               # 随机种子，保证可重复性
        deterministic=True,    # 确定性训练
        resume=False,          # 是否从检查点恢复
    )
    
    # ===== 小数据集专用配置 =====
    print("\n=== 小数据集训练配置示例 ===")
    
    # 重新加载模型
    model = YOLO("yolo11n.pt")
    
    small_dataset_results = model.train(
        data="coco8.yaml",
        epochs=300,            # 小数据集需要更多轮数
        imgsz=416,             # 较小图像尺寸，减少计算量
        batch=8,               # 较小批次
        device="0",
        
        # 学习率调整（小数据集）
        lr0=0.0001,            # 更小的学习率
        lrf=0.001,             # 更小的最终学习率
        warmup_epochs=5.0,     # 更长的预热
        
        # 减少过拟合
        dropout=0.2,           # 添加dropout
        weight_decay=0.001,    # 增加权重衰减
        
        # 轻度数据增强（避免过度增强）
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,           # 较小的旋转角度
        translate=0.05,        # 较小的平移
        scale=0.3,             # 较小的缩放
        mixup=0.0,             # 关闭mixup
        copy_paste=0.0,        # 关闭copy_paste
        
        # 训练控制
        patience=100,          # 更大的耐心值
        save_period=50,
        
        project='yolo_training',
        name='small_dataset_run',
    )
    
    print("\n=== 训练完成 ===")
    
    # 评估模型性能
    print("评估模型性能...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 测试预测
    if len(advanced_results) > 0:
        print("测试预测...")
        try:
            results = model("test_image.jpg")
            results[0].show()
        except Exception as e:
            print(f"预测测试失败: {e}")
    
    # 导出模型
    print("导出ONNX模型...")
    try:
        onnx_path = model.export(format="onnx")
        print(f"ONNX模型已导出到: {onnx_path}")
    except Exception as e:
        print(f"导出失败: {e}")

def show_hyperparameter_recommendations():
    """
    显示超参数调整建议
    """
    print("\n" + "="*60)
    print("超参数调整建议")
    print("="*60)
    
    recommendations = {
        "GPU内存不足": [
            "减少batch size: batch=8 或 batch=4",
            "减少图像尺寸: imgsz=416 或 imgsz=320",
            "关闭缓存: cache=False",
            "减少workers: workers=4"
        ],
        "训练速度慢": [
            "增加batch size: batch=32 或 batch=64",
            "开启缓存: cache='ram' 或 cache='disk'",
            "增加workers: workers=16",
            "开启混合精度: amp=True",
            "使用多GPU: device=[0,1,2,3]"
        ],
        "过拟合问题": [
            "增加数据增强强度",
            "增加dropout: dropout=0.2",
            "增加权重衰减: weight_decay=0.001",
            "减少学习率: lr0=0.0001",
            "增加label_smoothing: label_smoothing=0.1"
        ],
        "欠拟合问题": [
            "增加模型复杂度: 使用yolo11s.pt或yolo11m.pt",
            "增加训练轮数: epochs=500",
            "增加学习率: lr0=0.01",
            "减少数据增强强度",
            "检查数据标注质量"
        ],
        "小数据集(<1000张)": [
            "使用更小的学习率: lr0=0.0001",
            "增加训练轮数: epochs=500+",
            "轻度数据增强",
            "增加预热轮数: warmup_epochs=10",
            "使用较小的模型: yolo11n.pt"
        ]
    }
    
    for problem, solutions in recommendations.items():
        print(f"\n{problem}:")
        for solution in solutions:
            print(f"  • {solution}")

if __name__ == "__main__":
    # 显示建议
    show_hyperparameter_recommendations()
    
    # 选择运行哪个配置
    print("\n" + "="*60)
    print("请根据您的需求选择配置:")
    print("1. 基础配置 - 适合初学者和快速测试")
    print("2. 高级配置 - 适合生产环境")  
    print("3. 小数据集配置 - 适合数据量较少的情况")
    print("="*60)
    
    # 这里可以添加交互式选择，暂时注释掉自动执行
    # main() 