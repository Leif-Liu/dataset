#!/usr/bin/env python3
"""
COCO8标签格式分析和可视化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# COCO8 类别名称（简化版）
COCO8_CLASSES = {
    0: 'person',
    1: 'bicycle', 
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',  # 这是示例中的类别25
    26: 'handbag'
}

def parse_yolo_label(label_line):
    """
    解析YOLO格式的标签行
    """
    parts = label_line.strip().split()
    if len(parts) != 5:
        return None
    
    class_id = int(parts[0])
    center_x = float(parts[1])
    center_y = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    
    return {
        'class_id': class_id,
        'class_name': COCO8_CLASSES.get(class_id, f'class_{class_id}'),
        'center_x': center_x,
        'center_y': center_y,
        'width': width,
        'height': height
    }

def yolo_to_bbox(center_x, center_y, width, height, img_width, img_height):
    """
    将YOLO格式转换为边界框坐标
    """
    # 转换为像素坐标
    center_x_px = center_x * img_width
    center_y_px = center_y * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # 计算边界框四个角点
    left = int(center_x_px - width_px / 2)
    top = int(center_y_px - height_px / 2)
    right = int(center_x_px + width_px / 2)
    bottom = int(center_y_px + height_px / 2)
    
    return left, top, right, bottom

def analyze_sample_labels():
    """
    分析示例标签数据
    """
    sample_labels = [
        "25 0.475759 0.414523 0.951518 0.672422",
        "0 0.671279 0.617945 0.645759 0.726859"
    ]
    
    # 假设图像尺寸为640x640（YOLO常用尺寸）
    img_width, img_height = 640, 640
    
    print("=" * 60)
    print("COCO8 YOLO标签格式分析")
    print("=" * 60)
    print(f"图像尺寸: {img_width} × {img_height}")
    print(f"坐标系统: 原点在左上角 (0,0)")
    print(f"坐标格式: 归一化坐标 (0.0 ~ 1.0)")
    print()
    
    for i, label_line in enumerate(sample_labels, 1):
        print(f"--- 对象 {i} ---")
        print(f"原始标签: {label_line}")
        
        label_info = parse_yolo_label(label_line)
        if label_info:
            print(f"类别ID: {label_info['class_id']}")
            print(f"类别名称: {label_info['class_name']}")
            print(f"归一化坐标:")
            print(f"  中心点: ({label_info['center_x']:.6f}, {label_info['center_y']:.6f})")
            print(f"  尺寸: {label_info['width']:.6f} × {label_info['height']:.6f}")
            
            # 转换为像素坐标
            left, top, right, bottom = yolo_to_bbox(
                label_info['center_x'], label_info['center_y'],
                label_info['width'], label_info['height'],
                img_width, img_height
            )
            
            center_x_px = label_info['center_x'] * img_width
            center_y_px = label_info['center_y'] * img_height
            width_px = label_info['width'] * img_width
            height_px = label_info['height'] * img_height
            
            print(f"像素坐标:")
            print(f"  中心点: ({center_x_px:.1f}, {center_y_px:.1f})")
            print(f"  尺寸: {width_px:.1f} × {height_px:.1f}")
            print(f"  边界框: 左上({left}, {top}) -> 右下({right}, {bottom})")
            print()

def create_visualization():
    """
    创建坐标系统可视化图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：坐标系统说明
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.invert_yaxis()  # Y轴向下
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('YOLO Coordinate System (Normalized Coordinates)')
    ax1.set_xlabel('X axis (horizontal)')
    ax1.set_ylabel('Y axis (vertical)')
    
    # 标注原点
    ax1.plot(0, 0, 'ro', markersize=10)
    ax1.annotate('Origin (0,0)', xy=(0, 0), xytext=(0.1, 0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 绘制示例边界框
    sample_boxes = [
        {'center_x': 0.475759, 'center_y': 0.414523, 'width': 0.951518, 'height': 0.672422, 'class': 'umbrella'},
        {'center_x': 0.671279, 'center_y': 0.617945, 'width': 0.645759, 'height': 0.726859, 'class': 'person'}
    ]
    
    colors = ['blue', 'green']
    for i, box in enumerate(sample_boxes):
        left = box['center_x'] - box['width'] / 2
        top = box['center_y'] - box['height'] / 2
        
        # 绘制边界框
        rect = plt.Rectangle((left, top), box['width'], box['height'], 
                           fill=False, edgecolor=colors[i], linewidth=2)
        ax1.add_patch(rect)
        
        # 标注中心点
        ax1.plot(box['center_x'], box['center_y'], 'o', color=colors[i], markersize=8)
        ax1.annotate(f"{box['class']}\n({box['center_x']:.3f}, {box['center_y']:.3f})", 
                    xy=(box['center_x'], box['center_y']), 
                    xytext=(box['center_x'] + 0.1, box['center_y'] - 0.1),
                    fontsize=9, color=colors[i])
    
    # 右图：像素坐标示例
    img_size = 640
    ax2.set_xlim(0, img_size)
    ax2.set_ylim(0, img_size)
    ax2.invert_yaxis()  # Y轴向下
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Pixel Coordinate Example ({img_size}×{img_size})')
    ax2.set_xlabel('X axis (pixels)')
    ax2.set_ylabel('Y axis (pixels)')
    
    # 转换为像素坐标并绘制
    for i, box in enumerate(sample_boxes):
        center_x_px = box['center_x'] * img_size
        center_y_px = box['center_y'] * img_size
        width_px = box['width'] * img_size
        height_px = box['height'] * img_size
        
        left = center_x_px - width_px / 2
        top = center_y_px - height_px / 2
        
        rect = plt.Rectangle((left, top), width_px, height_px, 
                           fill=False, edgecolor=colors[i], linewidth=2)
        ax2.add_patch(rect)
        
        ax2.plot(center_x_px, center_y_px, 'o', color=colors[i], markersize=8)
        ax2.annotate(f"{box['class']}\n({center_x_px:.0f}, {center_y_px:.0f})", 
                    xy=(center_x_px, center_y_px), 
                    xytext=(center_x_px + 50, center_y_px - 50),
                    fontsize=9, color=colors[i])
    
    plt.tight_layout()
    plt.savefig('yolo_coordinate_system.png', dpi=300, bbox_inches='tight')
    print("坐标系统可视化图已保存为: yolo_coordinate_system.png")
    plt.show()

def main():
    """
    主函数
    """
    analyze_sample_labels()
    
    print("=" * 60)
    print("坐标系统特点总结:")
    print("=" * 60)
    print("1. 原点位置: 左上角 (0,0)")
    print("2. X轴方向: 从左到右 (水平)")
    print("3. Y轴方向: 从上到下 (垂直)")
    print("4. 坐标范围: 0.0 ~ 1.0 (归一化)")
    print("5. 中心点表示: (center_x, center_y)")
    print("6. 尺寸表示: (width, height)")
    print("7. 格式: class_id center_x center_y width height")
    print()
    
    try:
        create_visualization()
    except ImportError:
        print("注意: 需要安装matplotlib来生成可视化图表")
        print("安装命令: pip install matplotlib")

if __name__ == "__main__":
    main() 