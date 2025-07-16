#!/usr/bin/env python3
"""
YOLOv8/YOLOv11 训练超参数完整指南
"""

from ultralytics import YOLO

def show_all_hyperparameters():
    """
    展示所有可用的训练超参数
    """
    print("=" * 80)
    print("YOLO训练超参数完整列表")
    print("=" * 80)
    
    hyperparameters = {
        # 基本训练参数
        "基本训练参数": {
            "data": "数据集配置文件路径 (如 'coco8.yaml', 'path/to/dataset.yaml')",
            "epochs": "训练轮数 (默认: 100)",
            "patience": "早停耐心值，多少个epoch无改善后停止 (默认: 50)",
            "batch": "批次大小，-1为自动批次大小 (默认: 16)",
            "imgsz": "输入图像尺寸 (默认: 640)",
            "save": "是否保存训练检查点 (默认: True)",
            "save_period": "每隔多少epoch保存一次模型 (默认: -1, 即仅保存最后和最佳)",
            "cache": "数据缓存，True/False/'ram'/'disk' (默认: False)",
            "device": "训练设备 'cpu', 0, [0,1,2,3] (默认: None, 自动选择)",
            "workers": "数据加载器工作进程数 (默认: 8)",
            "project": "项目名称 (默认: 'runs/detect')",
            "name": "实验名称 (默认: 'train')",
        },
        
        # 模型架构参数
        "模型架构参数": {
            "pretrained": "是否使用预训练权重 (默认: True)",
            "optimizer": "优化器类型 'SGD', 'Adam', 'AdamW', 'RMSProp' (默认: 'auto')",
            "verbose": "是否详细输出 (默认: True)",
            "seed": "随机种子 (默认: 0)",
            "deterministic": "是否使用确定性训练 (默认: True)",
            "single_cls": "是否作为单类数据集训练 (默认: False)",
            "rect": "是否使用矩形训练 (默认: False)",
            "cos_lr": "是否使用余弦学习率调度 (默认: False)",
        },
        
        # 学习率和优化器参数
        "学习率和优化器": {
            "lr0": "初始学习率 (默认: 0.01)",
            "lrf": "最终学习率比例 = lr0 * lrf (默认: 0.01)",
            "momentum": "SGD动量/Adam beta1 (默认: 0.937)",
            "weight_decay": "权重衰减 (默认: 0.0005)",
            "warmup_epochs": "预热轮数 (默认: 3.0)",
            "warmup_momentum": "预热动量 (默认: 0.8)",
            "warmup_bias_lr": "预热偏置学习率 (默认: 0.1)",
        },
        
        # 数据增强参数
        "数据增强": {
            "hsv_h": "色调增强范围 (默认: 0.015)",
            "hsv_s": "饱和度增强范围 (默认: 0.7)",
            "hsv_v": "明度增强范围 (默认: 0.4)",
            "degrees": "旋转角度范围 (+/-deg) (默认: 0.0)",
            "translate": "平移范围 (+/-fraction) (默认: 0.1)",
            "scale": "缩放范围 (+/-gain) (默认: 0.5)",
            "shear": "剪切角度范围 (+/-deg) (默认: 0.0)",
            "perspective": "透视变换范围 (+/-fraction) (默认: 0.0)",
            "flipud": "垂直翻转概率 (默认: 0.0)",
            "fliplr": "水平翻转概率 (默认: 0.5)",
            "mosaic": "马赛克增强概率 (默认: 1.0)",
            "mixup": "混合增强概率 (默认: 0.0)",
            "copy_paste": "复制粘贴增强概率 (默认: 0.0)",
        },
        
        # 损失函数参数
        "损失函数": {
            "box": "边界框损失权重 (默认: 7.5)",
            "cls": "分类损失权重 (默认: 0.5)",
            "dfl": "DFL损失权重 (默认: 1.5)",
            "pose": "姿态损失权重，仅用于姿态模型 (默认: 12.0)",
            "kobj": "关键点obj损失权重，仅用于姿态模型 (默认: 2.0)",
            "label_smoothing": "标签平滑 (默认: 0.0)",
        },
        
        # 验证和测试参数
        "验证和测试": {
            "val": "是否在训练期间验证 (默认: True)",
            "fraction": "训练数据集的使用比例 (默认: 1.0)",
            "freeze": "冻结的层数，None表示不冻结 (默认: None)",
        },
        
        # 高级参数
        "高级参数": {
            "amp": "是否使用自动混合精度训练 (默认: True)",
            "multi_scale": "是否使用多尺度训练 (默认: False)",
            "overlap_mask": "训练期间掩码是否应重叠，仅分割 (默认: True)",
            "mask_ratio": "掩码下采样比例，仅分割 (默认: 4)",
            "dropout": "分类dropout概率，仅分类 (默认: 0.0)",
            "plots": "是否保存训练/验证图表 (默认: False)",
            "resume": "从最后一个检查点恢复训练 (默认: False)",
        }
    }
    
    for category, params in hyperparameters.items():
        print(f"\n{'='*10} {category} {'='*10}")
        for param, description in params.items():
            print(f"{param:20} : {description}")
    
    print("\n" + "="*80)

def example_configurations():
    """
    展示不同场景的配置示例
    """
    print("\n" + "="*80)
    print("常用配置示例")
    print("="*80)
    
    examples = {
        "基础训练": """
model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)""",
        
        "高性能训练": """
model.train(
    data="custom.yaml",
    epochs=300,
    imgsz=640,
    batch=32,
    device=[0,1,2,3],  # 多GPU
    cache='ram',       # 数据缓存到内存
    amp=True,          # 混合精度
    workers=16,        # 更多工作进程
    patience=50
)""",
        
        "小数据集训练": """
model.train(
    data="small_dataset.yaml",
    epochs=200,
    imgsz=416,        # 较小图像尺寸
    batch=8,          # 较小批次
    lr0=0.001,        # 较小学习率
    warmup_epochs=5,
    patience=30,
    augment=True
)""",
        
        "快速原型": """
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=320,        # 快速训练
    batch=64,
    cache=True,
    plots=True,       # 生成图表
    verbose=True
)""",
        
        "生产环境": """
model.train(
    data="production.yaml",
    epochs=500,
    imgsz=640,
    batch=16,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    device=0,
    multi_scale=False,
    rect=False,
    cos_lr=False,
    patience=100,
    save_period=10,
    project='production_runs',
    name='final_model'
)""",
        
        "调试模式": """
model.train(
    data="debug.yaml",
    epochs=5,
    imgsz=320,
    batch=2,
    cache=False,
    plots=True,
    verbose=True,
    save_period=1,    # 每轮都保存
    project='debug',
    name='test_run'
)"""
    }
    
    for title, code in examples.items():
        print(f"\n--- {title} ---")
        print(code)

def performance_tips():
    """
    性能优化建议
    """
    print("\n" + "="*80)
    print("性能优化建议")
    print("="*80)
    
    tips = [
        "1. 批次大小 (batch): 根据GPU内存调整，越大越好（通常16-32）",
        "2. 图像尺寸 (imgsz): 必须是32的倍数，常用320/416/640/832",
        "3. 缓存 (cache): 'ram'最快，'disk'节省内存，False最慢",
        "4. 工作进程 (workers): 通常设为CPU核心数",
        "5. 混合精度 (amp): 在支持的GPU上启用可提速并节省内存",
        "6. 多GPU训练: device=[0,1,2,3] 可显著提升训练速度",
        "7. 预训练模型: pretrained=True 通常效果更好",
        "8. 学习率: 从较小值开始（0.001-0.01）",
        "9. 数据增强: 根据数据集特点调整增强强度",
        "10. 早停: patience设置合理值避免过拟合"
    ]
    
    for tip in tips:
        print(tip)

def main():
    """
    主函数
    """
    show_all_hyperparameters()
    example_configurations()
    performance_tips()
    
    print("\n" + "="*80)
    print("使用建议:")
    print("="*80)
    print("1. 从基础配置开始，逐步调整参数")
    print("2. 监控训练过程中的loss变化")
    print("3. 使用验证集评估模型性能")
    print("4. 根据具体任务调整数据增强策略")
    print("5. 保存最佳模型用于推理")

if __name__ == "__main__":
    main() 