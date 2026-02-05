"""
視覺辨識模型測試腳本

功能：
1. 測試分類模型對各類瑕疵的辨識準確度
2. 展示 Top-K 預測結果
3. 生成分類報告與混淆矩陣

輸出：
- 分類準確度統計
- 各類別的信心度分佈
- 混淆矩陣 (若有標籤資料)

使用方式：
    cd wafer_copilot
    python -m tests.test_classifier
"""

import os
import sys
import json
from datetime import datetime
from collections import defaultdict

# 確保可以找到 src 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.vision.classifier import WaferClassifier, CLASSES


def load_test_images(test_dir="data/test"):
    """
    載入測試圖片（按類別資料夾組織）
    
    結構：
        data/test/
            ├── Center/
            ├── Donut/
            └── ...
    """
    images = []
    
    for class_name in CLASSES:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    images.append({
                        "path": os.path.join(class_dir, img_name),
                        "true_label": class_name,
                        "filename": img_name
                    })
    
    return images


def test_single_prediction(classifier, image_path):
    """測試單張圖片預測"""
    result = classifier.predict(image_path, generate_cam=False)
    return result


def test_topk_prediction(classifier, image_path, k=3):
    """
    取得 Top-K 預測結果
    """
    if classifier.mock_mode:
        return [
            {"label": "Edge-Ring", "probability": 0.85},
            {"label": "Donut", "probability": 0.10},
            {"label": "Center", "probability": 0.05}
        ]
    
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(classifier.device)
        
        with torch.no_grad():
            outputs = classifier.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_indices = probs.topk(k, dim=1)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                "label": CLASSES[idx.item()],
                "probability": round(prob.item(), 4)
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e)}]


def run_classification_test():
    """執行完整分類測試"""
    print("=" * 70)
    print("🔬 WM-811K 視覺辨識模型測試")
    print("=" * 70)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 初始化分類器
    print("📦 載入模型...")
    classifier = WaferClassifier()
    print(f"   模式: {'真實模型' if not classifier.mock_mode else '模擬模式'}")
    print(f"   裝置: {classifier.device}")
    print()
    
    # 載入測試圖片
    test_images = load_test_images()
    print(f"📁 找到 {len(test_images)} 張測試圖片")
    
    # 統計每類數量
    class_counts = defaultdict(int)
    for img in test_images:
        class_counts[img["true_label"]] += 1
    
    print("   各類別數量:")
    for cls in CLASSES:
        print(f"      - {cls}: {class_counts[cls]} 張")
    print()
    
    if not test_images:
        print("⚠️ 沒有測試圖片，使用 sample_images 進行展示")
        sample_dir = "data/sample_images"
        if os.path.exists(sample_dir):
            for img_name in os.listdir(sample_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    test_images.append({
                        "path": os.path.join(sample_dir, img_name),
                        "true_label": "Unknown",
                        "filename": img_name
                    })
    
    # 執行預測
    print("🔍 執行分類預測...")
    print("-" * 70)
    
    results = []
    correct = 0
    total = 0
    
    for i, img_info in enumerate(test_images[:20], 1):  # 限制前 20 張展示
        result = test_single_prediction(classifier, img_info["path"])
        topk = test_topk_prediction(classifier, img_info["path"], k=3)
        
        pred_label = result.get("label", "Error")
        confidence = result.get("confidence", 0)
        true_label = img_info["true_label"]
        
        is_correct = pred_label == true_label
        if true_label != "Unknown":
            total += 1
            if is_correct:
                correct += 1
        
        status = "✅" if is_correct else "❌" if true_label != "Unknown" else "❓"
        
        print(f"\n[{i:02d}] {img_info['filename']}")
        print(f"     真實標籤: {true_label}")
        print(f"     預測結果: {pred_label} ({confidence:.1%}) {status}")
        print(f"     Top-3: {', '.join([f'{t['label']}({t['probability']:.1%})' for t in topk[:3]])}")
        
        results.append({
            "filename": img_info["filename"],
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "top_k": topk,
            "correct": is_correct if true_label != "Unknown" else None
        })
    
    # 計算準確度
    print("\n" + "=" * 70)
    print("📊 分類結果統計")
    print("=" * 70)
    
    if total > 0:
        accuracy = correct / total
        print(f"   準確度: {accuracy:.1%} ({correct}/{total})")
    else:
        print("   (無標籤資料，無法計算準確度)")
    
    # 各類別統計
    print("\n   各類別預測分佈:")
    pred_counts = defaultdict(int)
    for r in results:
        pred_counts[r["predicted_label"]] += 1
    
    for cls in CLASSES:
        print(f"      - {cls}: {pred_counts[cls]} 次")
    
    # 儲存結果
    output_dir = "tests/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "test_time": datetime.now().isoformat(),
        "model_mode": "real" if not classifier.mock_mode else "mock",
        "total_images": len(test_images),
        "accuracy": correct / total if total > 0 else None,
        "results": results
    }
    
    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 報告已儲存至: {report_path}")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    run_classification_test()
