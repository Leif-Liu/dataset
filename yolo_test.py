#!/usr/bin/env python3
"""
YOLO网络图像预测脚本
解决网络连接和图像读取问题
"""

import requests
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

def download_image_safely(url, output_path, timeout=30):
    """
    安全下载网络图像
    """
    try:
        # 添加用户代理头，避免被服务器拒绝
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"正在下载图像: {url}")
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # 将图像数据写入文件
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"图像已下载到: {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False

def verify_image(image_path):
    """
    验证图像文件是否有效
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"图像读取失败: {image_path}")
            return False
        print(f"图像验证成功: {image_path}, 尺寸: {img.shape}")
        return True
    except Exception as e:
        print(f"图像验证失败: {e}")
        return False

def run_yolo_prediction(model_path, source_path):
    """
    运行YOLO预测
    """
    try:
        # 加载模型
        model = YOLO(model_path)
        
        # 运行预测
        results = model(source_path)
        
        # 显示结果
        for result in results:
            result.show()  # 显示结果
            result.save()  # 保存结果
            
        print("预测完成!")
        return True
        
    except Exception as e:
        print(f"YOLO预测失败: {e}")
        return False

def main():
    """
    主函数
    """
    # 测试URL（使用多个备选URL）
    test_urls = [
        'https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg',
        'https://ultralytics.com/images/bus.jpg',
        'https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg'
    ]
    
    # 模型路径
    model_path = 'yolo11n.pt'
    
    # 输出目录
    output_dir = Path('downloaded_images')
    output_dir.mkdir(exist_ok=True)
    
    # 尝试下载图像
    downloaded = False
    for i, url in enumerate(test_urls):
        output_path = output_dir / f'test_bus_{i}.jpg'
        
        if download_image_safely(url, output_path):
            if verify_image(output_path):
                print(f"使用图像: {output_path}")
                
                # 运行YOLO预测
                if run_yolo_prediction(model_path, str(output_path)):
                    downloaded = True
                    break
    
    # 如果都下载失败，使用本地生成的测试图像
    if not downloaded:
        print("网络图像下载失败，使用本地测试图像...")
        local_test_image = 'test_image.jpg'
        
        if os.path.exists(local_test_image):
            if verify_image(local_test_image):
                run_yolo_prediction(model_path, local_test_image)
            else:
                print("本地测试图像也无效!")
        else:
            print("请先创建测试图像!")

if __name__ == "__main__":
    main() 