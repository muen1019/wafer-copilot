"""
分類模型效能評估腳本
計算 Confusion Matrix, Precision, Recall, F1-Score
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# 與 train.py 一致
CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']

class WaferDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), class_idx))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate(model_path="models/resnet_wm811k.pth", data_dir="data/raw", batch_size=32, seed=42):
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 重建模型
    model = models.resnet18(pretrained=False) # 評估時不需要載入 pretrained weights，因為會 load_state_dict
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return

    model = model.to(device)
    model.eval()

    # 重現驗證集分割 (使用相同的 Seed)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 這裡的邏輯必須嚴格與 train.py 一致才能確保評估的是 Validation Set
    full_dataset = WaferDataset(data_dir, transform=None)
    if len(full_dataset) == 0:
        print("❌ Dataset empty.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size], 
                                                  generator=torch.Generator().manual_seed(seed))
    
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    val_data = TransformedSubset(val_subset, transform_val)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Evaluating on {len(val_data)} validation samples...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 計算 Metrics
    print("\n" + "="*50)
    print("RESNET-18 CLASSIFICATION REPORT")
    print("="*50)
    
    # 使用 sklearn 生成報告
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)
    
    # 輸出 Markdown 格式表格 (方便貼到報告)
    print("\n[Markdown Table for Report]")
    print("| 瑕疵類別 (Class) | Precision (%) | Recall (%) | F1-Score (%) |")
    print("| :--- | :---: | :---: | :---: |")
    
    avg_precision = 0
    avg_recall = 0
    avg_f1 = 0
    
    for cls in CLASSES:
        if cls in report:
            prec = report[cls]['precision'] * 100
            rec = report[cls]['recall'] * 100
            f1 = report[cls]['f1-score'] * 100
            print(f"| {cls} | {prec:.1f} | {rec:.1f} | {f1:.1f} |")
    
    acc = accuracy_score(all_labels, all_preds) * 100
    print(f"| **Average** | **-** | **-** | **{report['weighted avg']['f1-score']*100:.1f}** |")
    print(f"\nOverall Accuracy: {acc:.1f}%")

if __name__ == "__main__":
    evaluate()
