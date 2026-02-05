"""
Grad-CAM 熱力圖展示腳本

功能：
1. 對各類別瑕疵生成 Grad-CAM 熱力圖
2. 產生原圖/熱力圖/疊加圖的對照組
3. 生成適合放入報告的展示圖片

輸出：
- tests/outputs/gradcam/ 目錄下的熱力圖
- 各類別瑕疵的視覺化解釋

使用方式：
    cd wafer_copilot
    python -m tests.test_gradcam
"""

import os
import sys
from datetime import datetime

# 確保可以找到 src 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.vision.classifier import WaferClassifier, CLASSES
from src.vision.gradcam import GradCAM, apply_colormap, generate_heatmap_only, create_comparison_image


def create_labeled_comparison(original, heatmap, overlay, label, confidence):
    """
    創建帶標籤的比較圖（適合報告使用）
    
    Layout:
    +------------------+------------------+------------------+
    |    原始圖片      |    熱力圖        |    疊加圖        |
    |   (Original)     |   (Heatmap)      |   (Overlay)      |
    +------------------+------------------+------------------+
    |        預測結果: {label} (信心度: {confidence})         |
    +-------------------------------------------------------+
    """
    size = (224, 224)
    padding = 10
    header_height = 40
    footer_height = 50
    
    # 調整圖片大小
    original = original.resize(size)
    heatmap = heatmap.resize(size)
    overlay = overlay.resize(size)
    
    # 計算總尺寸
    total_width = size[0] * 3 + padding * 4
    total_height = size[1] + header_height + footer_height + padding * 2
    
    # 創建畫布
    canvas = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 嘗試載入字體
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # 繪製標題
    titles = ["Original", "Grad-CAM Heatmap", "Overlay"]
    for i, title in enumerate(titles):
        x = padding + i * (size[0] + padding) + size[0] // 2
        draw.text((x, 15), title, fill=(50, 50, 50), font=title_font, anchor="mm")
    
    # 貼上圖片
    y_offset = header_height
    canvas.paste(original, (padding, y_offset))
    canvas.paste(heatmap, (padding * 2 + size[0], y_offset))
    canvas.paste(overlay, (padding * 3 + size[0] * 2, y_offset))
    
    # 繪製預測結果
    footer_y = y_offset + size[1] + padding + 20
    result_text = f"Prediction: {label}  |  Confidence: {confidence:.1%}"
    draw.text((total_width // 2, footer_y), result_text, fill=(0, 100, 0), font=title_font, anchor="mm")
    
    return canvas


def run_gradcam_demo():
    """執行 Grad-CAM 展示"""
    print("=" * 70)
    print("🔥 Grad-CAM 熱力圖展示")
    print("=" * 70)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 初始化分類器
    print("📦 載入模型...")
    classifier = WaferClassifier()
    print(f"   模式: {'真實模型' if not classifier.mock_mode else '模擬模式'}")
    
    if classifier.mock_mode:
        print("⚠️ 模型為模擬模式，無法生成真實 Grad-CAM")
        print("   請先訓練模型或下載預訓練權重")
        return
    
    # 準備輸出目錄
    output_dir = "tests/outputs/gradcam"
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集測試圖片
    test_images = []
    
    # 優先從 sample_images 取
    sample_dir = "data/sample_images"
    if os.path.exists(sample_dir):
        for img_name in os.listdir(sample_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                # 從檔名推斷類別
                label = img_name.split('_')[0] if '_' in img_name else "Unknown"
                test_images.append({
                    "path": os.path.join(sample_dir, img_name),
                    "expected_label": label,
                    "filename": img_name
                })
    
    # 從 test 資料夾補充
    test_dir = "data/test"
    for class_name in CLASSES:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                # 每類取一張
                test_images.append({
                    "path": os.path.join(class_dir, files[0]),
                    "expected_label": class_name,
                    "filename": files[0]
                })
    
    print(f"\n📁 找到 {len(test_images)} 張測試圖片")
    print()
    
    # 生成 Grad-CAM
    print("🎨 生成 Grad-CAM 熱力圖...")
    print("-" * 70)
    
    generated = []
    
    for i, img_info in enumerate(test_images, 1):
        print(f"\n[{i:02d}] 處理: {img_info['filename']}")
        
        try:
            # 執行預測（含 Grad-CAM）
            result = classifier.predict(img_info["path"], generate_cam=True)
            
            if "error" in result:
                print(f"     ❌ 錯誤: {result['error']}")
                continue
            
            print(f"     預測: {result['label']} ({result['confidence']:.1%})")
            
            # 載入生成的圖片
            original = Image.open(img_info["path"]).convert('RGB')
            
            if result.get("cam_path") and os.path.exists(result["cam_path"]):
                heatmap = Image.open(result["cam_path"])
                overlay = Image.open(result["cam_overlay_path"])
                
                # 創建帶標籤的比較圖
                labeled = create_labeled_comparison(
                    original, heatmap, overlay,
                    result["label"], result["confidence"]
                )
                
                # 儲存
                output_name = f"gradcam_{result['label']}_{i:02d}.png"
                output_path = os.path.join(output_dir, output_name)
                labeled.save(output_path)
                
                print(f"     ✅ 已儲存: {output_path}")
                
                generated.append({
                    "source": img_info["filename"],
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "output": output_path
                })
            else:
                print(f"     ⚠️ 未生成熱力圖")
        
        except Exception as e:
            print(f"     ❌ 例外: {e}")
    
    # 生成總覽圖
    print("\n" + "=" * 70)
    print("📊 生成結果總覽")
    print("=" * 70)
    
    if generated:
        print(f"\n成功生成 {len(generated)} 張 Grad-CAM 圖片:")
        for item in generated:
            print(f"   - {item['label']}: {item['source']} -> {os.path.basename(item['output'])}")
        
        # 創建多圖總覽
        create_overview_grid(generated, output_dir)
    
    print(f"\n💾 所有輸出已儲存至: {output_dir}")
    print("=" * 70)


def create_overview_grid(generated, output_dir):
    """
    創建多圖總覽網格
    """
    if len(generated) < 2:
        return
    
    # 載入所有生成的圖片
    images = []
    for item in generated[:8]:  # 最多 8 張
        if os.path.exists(item["output"]):
            images.append(Image.open(item["output"]))
    
    if not images:
        return
    
    # 計算網格佈局
    n = len(images)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    
    # 取得單張圖片尺寸
    img_w, img_h = images[0].size
    
    # 創建總覽畫布
    grid = Image.new('RGB', (img_w * cols + 20, img_h * rows + 20), (240, 240, 240))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * img_w + 10
        y = row * img_h + 10
        grid.paste(img, (x, y))
    
    overview_path = os.path.join(output_dir, "gradcam_overview.png")
    grid.save(overview_path)
    print(f"\n📸 總覽圖已儲存: {overview_path}")


if __name__ == "__main__":
    run_gradcam_demo()
