#!/usr/bin/env python3
import os
import torch
from torch import nn, optim
from transformers import CLIPModel, CLIPProcessor

# 设备配置：如果可用则使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 CLIP 模型和处理器：优先使用本地微调模型，否则使用预训练模型
local_model_path = "/home/liufeng/sdk-ragflow/fine-tuned-clip"
if os.path.exists(local_model_path):
    print(f"Found local fine-tuned model at {local_model_path}, loading...")
    model = CLIPModel.from_pretrained(local_model_path)
    processor = CLIPProcessor.from_pretrained(local_model_path)
    print("✓ Successfully loaded fine-tuned model")
else:
    print("Local fine-tuned model not found, loading pre-trained model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("✓ Successfully loaded pre-trained model")
model = model.to(device)

## 2. 构建自定义图文配对数据集
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, img_paths, captions):
        """
        img_paths: 包含所有图像文件路径的列表
        captions: 对应的文本描述列表，与 img_paths 对齐
        """
        assert len(img_paths) == len(captions), "img_paths 与 captions 长度必须一致"
        self.img_paths = img_paths
        self.captions = captions

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        text = self.captions[idx]
        return {"image": image, "text": text}


def collate_fn(batch):
    """将一批样本通过 CLIPProcessor 统一打包和填充到等长张量。"""
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return inputs


def ensure_example_data():
    """确保存在最小示例数据，若不存在则自动生成两张占位图片。"""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    img1 = os.path.join(data_dir, "img1.jpg")
    img2 = os.path.join(data_dir, "img2.jpg")
    if not os.path.exists(img1):
        Image.new("RGB", (256, 256), color=(200, 200, 255)).save(img1)
    if not os.path.exists(img2):
        Image.new("RGB", (256, 256), color=(255, 200, 200)).save(img2)
    captions = ["A blue-ish square.", "A red-ish square."]
    return [img1, img2], captions

# 组装数据
img_paths, captions = ensure_example_data()
batch_size = 8
dataset = ImageTextDataset(img_paths, captions)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

## 3. 定义训练循环与损失函数

# 优化器（可以根据需求使用 AdamW 等）
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# 训练配置
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # 将输入移动到 device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 前向计算：使用模型内置的投影与温度标度，直接得到对比学习 logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_loss=False)
        logits_per_image = outputs.logits_per_image  # [B, B]
        logits_per_text = outputs.logits_per_text    # [B, B]

        # 真实标签：对角线上匹配正样本，计算Loss前会进行一次softmax操作，所以batch 中图文对角线位置Label对应1, 其他位置Label为0；当前代码的labels 示例，torch.arange(batch_size)  # [0, 1, 2, 3, ...]
        # 真实标签：对角线上匹配，使用类别索引格式 [0,1,2,...]  # CrossEntropyLoss内部会进行softmax，期望对角线位置概率最高
        labels = torch.arange(pixel_values.size(0)).to(device)

        # 对称交叉熵损失（InfoNCE）
        loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

## 4. 支持 GPU 训练
## 5. 保存训练好的模型参数
# 保存模型和处理器
print(f"Saving model to {local_model_path}...")
model.save_pretrained(local_model_path)
processor.save_pretrained(local_model_path)
print("✓ Model saved successfully!")

## 6. 示例数据和输出说明
# 已在 ensure_example_data 中创建并说明