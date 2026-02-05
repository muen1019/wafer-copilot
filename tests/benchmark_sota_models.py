
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import os
from PIL import Image

# Configuration
DATA_DIR = "data/raw"
CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
BATCH_SIZE = 32
EPOCHS = 8 # Balanced for benchmarking speed and convergence trend
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WaferDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for class_name in CLASSES:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), CLASS_TO_IDX[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), label

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = WaferDataset(DATA_DIR, transform=transform)
    
    if len(dataset) == 0:
        return None, None
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

def train_eval(model, model_name):
    print(f"\nTraining {model_name}...")
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower LR for finetuning
    criterion = nn.CrossEntropyLoss()
    
    train_loader, val_loader = get_dataloader()
    if not train_loader:
        print("No data found.")
        return 0.0

    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Val Acc {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            
    print(f"Best {model_name} Accuracy: {best_acc:.2f}%")
    return best_acc

def run_benchmarks():
    print(f"Starting Benchmarks on Local Dataset (8-class) for {EPOCHS} Epochs...")
    
    results = {}

    # 1. ResNet-18 (Ours)
    print("\n[1/3] Benchmarking ResNet-18 (Ours)...")
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, len(CLASSES))
    results['ResNet-18'] = train_eval(resnet18, "ResNet-18")

    # 2. ResNet-50 (Transfer Learning Representative)
    print("\n[2/3] Benchmarking ResNet-50...")
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, len(CLASSES))
    results['ResNet-50'] = train_eval(resnet50, "ResNet-50")
    
    # 3. Vision Transformer (ViT-B-16) (SOTA Representative)
    print("\n[3/3] Benchmarking ViT-B/16...")
    try:
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # ViT head is usually 'heads'
        vit.heads.head = nn.Linear(vit.heads.head.in_features, len(CLASSES))
        results['ViT-B/16'] = train_eval(vit, "ViT-B/16")
    except Exception as e:
        print(f"Skipping ViT: {e}")
        results['ViT-B/16'] = "N/A"

    print("\n=== Final Benchmark Results (Fair Comparison) ===")
    print(f"{'Model':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for model, acc in results.items():
        val = f"{acc:.2f}%" if isinstance(acc, float) else acc
        print(f"{model:<15} | {val:<10}")

if __name__ == "__main__":
    run_benchmarks()
