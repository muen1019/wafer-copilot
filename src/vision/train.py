"""
訓練 WM-811K 晶圓瑕疵分類模型的腳本
使用 ResNet18 進行遷移學習
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import argparse
import random
import numpy as np
import os

# WM-811K 的標準 8 大類別
CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']


class WaferDataset(Dataset):
    """
    自定義資料集類別
    需要將 WM-811K 資料集整理成以下結構：
    data/raw/
        ├── Center/
        │   ├── img1.png
        │   └── ...
        ├── Donut/
        └── ...
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            class_idx
                        ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_model(data_dir="data/raw", epochs=20, batch_size=32, lr=0.001, 
                use_pretrained=True, use_augmentation=True, seed=42):
    """
    訓練模型的主函數
    """
    # 設定亂數種子以確保可重現性
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    # 資料轉換 (Augmentation Ablation Study)
    if use_augmentation:
        print("✅ 啟用資料增強 (Data Augmentation)")
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        print("❌ 停用資料增強 (No Augmentation)")
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 驗證集不需要 Augmentation
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 建立資料集與 DataLoader (這裡簡化，直接用 dataset 物件兩次但帶入不同 transform 會比較複雜，
    # 為了簡單起見，我們在 split 後覆寫 dataset.transform，或是更好的做法是使用 Subset 並包裝 transform)
    # 這裡採用簡單策略：讀取兩次 Dataset，一次給 Train 一次給 Val (效率稍差但程式碼最少變動)
    
    full_dataset = WaferDataset(data_dir, transform=None)
    
    if len(full_dataset) == 0:
        print("❌ 找不到訓練資料！請確認資料目錄結構。")
        return
    
    # 分割訓練集與驗證集 (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size], 
                                                             generator=torch.Generator().manual_seed(seed))
    
    # 自定義 Dataset Wrapper 來套用不同的 transform
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

    train_data = TransformedSubset(train_subset, transform_train)
    val_data = TransformedSubset(val_subset, transform_val)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"訓練集: {train_size} 張, 驗證集: {val_size} 張")
    
    # 建立模型 (Pretrained Ablation Study)
    print(f"型號設定: Pretrained={use_pretrained}")
    model = models.resnet18(pretrained=use_pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model = model.to(device)
    
    # 損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # 訓練階段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        print(f"訓練 Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        
        # 驗證階段
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        print(f"驗證 Acc: {val_acc:.2f}%")
        
        # 儲存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/resnet_wm811k.pth")
            print(f"✅ 已儲存最佳模型 (Acc: {val_acc:.2f}%)")
        
        scheduler.step()
    
    print(f"\n訓練完成！最佳驗證準確率: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wafer Defect Classification Training")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable transfer learning (use random init)")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        use_pretrained=not args.no_pretrained,
        use_augmentation=not args.no_aug
    )
