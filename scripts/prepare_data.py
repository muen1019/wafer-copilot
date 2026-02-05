"""
WM-811K 資料集下載與預處理腳本
此腳本會從 Kaggle 下載資料集並轉換為訓練用的圖片格式
"""

import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# WM-811K 的標準 9 大類別對應
LABEL_MAP = {
    0: 'None',
    1: 'Center',
    2: 'Donut', 
    3: 'Edge-Loc',
    4: 'Edge-Ring',
    5: 'Loc',
    6: 'Scratch',
    7: 'Random',
    8: 'Near-full'
}

# 字串標籤對應（WM-811K 原始格式使用字串）
STRING_LABEL_MAP = {
    'none': 'None',
    'center': 'Center',
    'donut': 'Donut',
    'edge-loc': 'Edge-Loc',
    'edge-ring': 'Edge-Ring',
    'loc': 'Loc',
    'scratch': 'Scratch',
    'random': 'Random',
    'near-full': 'Near-full',
    # 大小寫變體
    'None': 'None',
    'Center': 'Center',
    'Donut': 'Donut',
    'Edge-Loc': 'Edge-Loc',
    'Edge-Ring': 'Edge-Ring',
    'Loc': 'Loc',
    'Scratch': 'Scratch',
    'Random': 'Random',
    'Near-full': 'Near-full',
}

# 瑕疵類別（不含 None）
DEFECT_CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Scratch', 'Random', 'Near-full']

def download_dataset():
    """
    下載 WM-811K 資料集
    需要先安裝 kaggle: pip install kaggle
    並設定 Kaggle API credentials
    """
    print("=" * 60)
    print("📥 WM-811K 資料集下載指南")
    print("=" * 60)
    print("""
方法一：使用 Kaggle CLI（推薦）
--------------------------------
1. 安裝 kaggle CLI:
   pip install kaggle

2. 設定 Kaggle API credentials:
   - 前往 https://www.kaggle.com/settings
   - 點擊 "Create New Token" 下載 kaggle.json
   - 將 kaggle.json 放到 ~/.kaggle/ 目錄
   - chmod 600 ~/.kaggle/kaggle.json

3. 下載資料集:
   kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw/
   
4. 解壓縮:
   cd data/raw && unzip wm811k-wafer-map.zip

方法二：手動下載
--------------------------------
1. 前往: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
2. 點擊 "Download" 按鈕
3. 將下載的 LSWMD.pkl 放到 data/raw/ 目錄

方法三：使用 Mirror 資料集
--------------------------------
如果無法使用 Kaggle，可以使用以下備用來源：
- https://github.com/Junliangwangdhu/WaferMap
- https://zenodo.org/record/4394344
    """)
    print("=" * 60)


def load_wm811k(pkl_path="data/raw/LSWMD.pkl"):
    """
    載入 WM-811K pickle 檔案
    處理舊版 pandas 相容性問題
    """
    import pandas as pd
    import sys
    
    if not os.path.exists(pkl_path):
        print(f"❌ 找不到 {pkl_path}")
        print("請先下載資料集，執行: python prepare_data.py --download")
        return None
    
    print(f"📂 載入資料集: {pkl_path}")
    
    # 修復舊版 pandas 相容性問題
    # WM-811K 是用舊版 pandas 建立的，需要做 module 映射
    class PandasUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # 處理舊版 pandas 的 module 路徑變更
            renamed_modules = {
                'pandas.indexes': 'pandas',
                'pandas.indexes.base': 'pandas',
                'pandas.core.indexes': 'pandas',
                'pandas.core.indexes.base': 'pandas',
            }
            if module in renamed_modules:
                module = renamed_modules[module]
            if module == 'pandas' and name == 'Index':
                return pd.Index
            return super().find_class(module, name)
    
    try:
        with open(pkl_path, 'rb') as f:
            data = PandasUnpickler(f).load()
    except Exception as e:
        print(f"⚠️ 標準載入失敗，嘗試替代方法...")
        # 備用方法：使用 pandas 直接讀取
        try:
            data = pd.read_pickle(pkl_path)
        except Exception as e2:
            print(f"❌ 載入失敗: {e2}")
            print("\n💡 建議解決方案:")
            print("   pip install pandas==1.5.3")
            print("   然後重新執行此腳本")
            return None
    
    print(f"✅ 資料集載入完成！")
    print(f"   - 總樣本數: {len(data)}")
    print(f"   - 欄位: {data.columns.tolist()}")
    
    return data


def wafer_to_image(wafer_map, size=(224, 224)):
    """
    將晶圓圖譜矩陣轉換為 RGB 圖片
    0: 背景 (黑色)
    1: 正常區域 (灰色)
    2: 瑕疵區域 (紅色)
    """
    # 建立 RGB 圖片
    h, w = wafer_map.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 設定顏色
    img[wafer_map == 0] = [0, 0, 0]       # 背景: 黑色
    img[wafer_map == 1] = [100, 100, 100] # 正常: 灰色
    img[wafer_map == 2] = [255, 0, 0]     # 瑕疵: 紅色
    
    # 轉換為 PIL Image 並調整大小
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size, Image.Resampling.LANCZOS)
    
    return pil_img


def prepare_dataset(pkl_path="data/raw/LSWMD.pkl", output_dir="data/raw", max_per_class=2000):
    """
    將 WM-811K 資料集轉換為圖片格式
    
    Args:
        pkl_path: pickle 檔案路徑
        output_dir: 輸出目錄
        max_per_class: 每個類別最多取多少張（平衡資料集）
    """
    import pandas as pd
    
    # 載入資料
    data = load_wm811k(pkl_path)
    if data is None:
        return
    
    # 過濾有標籤的資料
    # failureType 欄位格式: [[label_id]]
    labeled_data = data[data['failureType'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False)]
    print(f"📊 有標籤的資料: {len(labeled_data)} 筆")
    
    # 建立類別目錄
    for class_name in LABEL_MAP.values():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # 統計每個類別的數量
    class_counts = {name: 0 for name in LABEL_MAP.values()}
    
    print("\n🔄 開始轉換圖片...")
    
    for idx, row in labeled_data.iterrows():
        try:
            # 取得標籤（WM-811K 使用字串格式，如 [['none']], [['Center']]）
            failure_type = row['failureType']
            if isinstance(failure_type, (list, np.ndarray)) and len(failure_type) > 0:
                if isinstance(failure_type[0], (list, np.ndarray)) and len(failure_type[0]) > 0:
                    label_raw = failure_type[0][0]
                else:
                    label_raw = failure_type[0]
            else:
                continue
            
            # 將標籤轉換為標準類別名稱
            label_str = str(label_raw)
            if label_str not in STRING_LABEL_MAP:
                continue
            class_name = STRING_LABEL_MAP[label_str]
            
            # 檢查是否超過上限
            if class_counts[class_name] >= max_per_class:
                continue
            
            # 取得晶圓圖譜
            wafer_map = row['waferMap']
            if wafer_map is None or not isinstance(wafer_map, np.ndarray):
                continue
            
            # 轉換為圖片
            img = wafer_to_image(wafer_map)
            
            # 儲存圖片
            img_name = f"{class_name}_{class_counts[class_name]:05d}.png"
            img_path = os.path.join(output_dir, class_name, img_name)
            img.save(img_path)
            
            class_counts[class_name] += 1
            
            # 進度顯示
            total = sum(class_counts.values())
            if total % 500 == 0:
                print(f"   已處理 {total} 張圖片...")
                
        except Exception as e:
            continue
    
    # 顯示統計
    print("\n" + "=" * 40)
    print("📊 資料集統計")
    print("=" * 40)
    total_images = 0
    for class_name, count in class_counts.items():
        print(f"   {class_name:12s}: {count:5d} 張")
        total_images += count
    print("-" * 40)
    print(f"   {'總計':12s}: {total_images:5d} 張")
    print("=" * 40)
    print(f"\n✅ 資料集準備完成！圖片已儲存至: {output_dir}")


def prepare_dataset_with_split(pkl_path="data/raw/LSWMD.pkl", train_dir="data/train", test_dir="data/test", 
                               test_ratio=0.1, max_per_class=100000, exclude_none=True):
    """
    將 WM-811K 資料集轉換為圖片格式，並自動分割為訓練集和測試集
    
    Args:
        pkl_path: pickle 檔案路徑
        train_dir: 訓練資料輸出目錄
        test_dir: 測試資料輸出目錄
        test_ratio: 測試集比例 (預設 0.1 = 10%)
        max_per_class: 每個類別最多取多少張
        exclude_none: 是否排除 None 類別 (預設 True)
    """
    import pandas as pd
    import random
    
    # 載入資料
    data = load_wm811k(pkl_path)
    if data is None:
        return
    
    # 過濾有標籤的資料
    labeled_data = data[data['failureType'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False)]
    print(f"📊 有標籤的資料: {len(labeled_data)} 筆")
    
    # 決定要使用的類別
    if exclude_none:
        classes_to_use = DEFECT_CLASSES
        print("⚠️  已排除 None 類別，只處理瑕疵類型")
    else:
        classes_to_use = list(LABEL_MAP.values())
    
    # 建立類別目錄
    for class_name in classes_to_use:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    # 先收集所有資料
    class_data = {name: [] for name in classes_to_use}
    
    print("\n🔄 收集資料中...")
    
    for idx, row in labeled_data.iterrows():
        try:
            # 取得標籤
            failure_type = row['failureType']
            if isinstance(failure_type, (list, np.ndarray)) and len(failure_type) > 0:
                if isinstance(failure_type[0], (list, np.ndarray)) and len(failure_type[0]) > 0:
                    label_raw = failure_type[0][0]
                else:
                    label_raw = failure_type[0]
            else:
                continue
            
            # 將標籤轉換為標準類別名稱
            label_str = str(label_raw)
            if label_str not in STRING_LABEL_MAP:
                continue
            class_name = STRING_LABEL_MAP[label_str]
            
            # 跳過不需要的類別
            if class_name not in classes_to_use:
                continue
            
            # 檢查是否超過上限
            if len(class_data[class_name]) >= max_per_class:
                continue
            
            # 取得晶圓圖譜
            wafer_map = row['waferMap']
            if wafer_map is None or not isinstance(wafer_map, np.ndarray):
                continue
            
            class_data[class_name].append(wafer_map)
            
        except Exception as e:
            continue
    
    # 分割並儲存
    print("\n🔄 分割並儲存圖片...")
    
    train_counts = {name: 0 for name in classes_to_use}
    test_counts = {name: 0 for name in classes_to_use}
    
    for class_name, wafer_maps in class_data.items():
        # 隨機打亂
        random.shuffle(wafer_maps)
        
        # 計算分割點
        n_test = int(len(wafer_maps) * test_ratio)
        n_train = len(wafer_maps) - n_test
        
        # 分割資料
        train_data = wafer_maps[:n_train]
        test_data = wafer_maps[n_train:]
        
        # 儲存訓練資料
        for i, wafer_map in enumerate(train_data):
            img = wafer_to_image(wafer_map)
            img_name = f"{class_name}_{i:05d}.png"
            img_path = os.path.join(train_dir, class_name, img_name)
            img.save(img_path)
            train_counts[class_name] += 1
        
        # 儲存測試資料
        for i, wafer_map in enumerate(test_data):
            img = wafer_to_image(wafer_map)
            img_name = f"{class_name}_test_{i:05d}.png"
            img_path = os.path.join(test_dir, class_name, img_name)
            img.save(img_path)
            test_counts[class_name] += 1
        
        print(f"   {class_name:12s}: 訓練 {train_counts[class_name]:5d} 張, 測試 {test_counts[class_name]:4d} 張")
    
    # 顯示統計
    print("\n" + "=" * 50)
    print("📊 資料集統計")
    print("=" * 50)
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    print(f"   訓練集: {total_train:5d} 張 ({100-test_ratio*100:.0f}%)")
    print(f"   測試集: {total_test:5d} 張 ({test_ratio*100:.0f}%)")
    print(f"   總計:   {total_train + total_test:5d} 張")
    print("=" * 50)
    print(f"\n✅ 資料集準備完成！")
    print(f"   訓練集: {train_dir}")
    print(f"   測試集: {test_dir}")


def prepare_test_dataset(pkl_path="data/raw/LSWMD.pkl", train_dir="data/raw", test_dir="data/test", max_per_class=1000):
    """
    提取測試資料集（排除已用於訓練的資料）
    
    Args:
        pkl_path: pickle 檔案路徑
        train_dir: 訓練資料目錄（用於計算已使用的數量）
        test_dir: 測試資料輸出目錄
        max_per_class: 每個類別最多取多少張測試資料
    """
    import pandas as pd
    
    # 載入資料
    data = load_wm811k(pkl_path)
    if data is None:
        return
    
    # 過濾有標籤的資料
    labeled_data = data[data['failureType'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False)]
    print(f"📊 有標籤的資料: {len(labeled_data)} 筆")
    
    # 計算每個類別已用於訓練的數量
    train_counts = {}
    for class_name in LABEL_MAP.values():
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            train_counts[class_name] = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
        else:
            train_counts[class_name] = 0
    
    print("\n📊 訓練集統計:")
    for class_name, count in train_counts.items():
        print(f"   {class_name:12s}: {count:5d} 張")
    
    # 建立測試資料目錄
    os.makedirs(test_dir, exist_ok=True)
    for class_name in LABEL_MAP.values():
        class_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # 統計：跳過訓練資料，提取測試資料
    skip_counts = {name: 0 for name in LABEL_MAP.values()}  # 需要跳過的數量
    test_counts = {name: 0 for name in LABEL_MAP.values()}  # 已提取的測試資料數量
    
    print("\n🔄 開始提取測試資料...")
    
    for idx, row in labeled_data.iterrows():
        try:
            # 取得標籤
            failure_type = row['failureType']
            if isinstance(failure_type, (list, np.ndarray)) and len(failure_type) > 0:
                if isinstance(failure_type[0], (list, np.ndarray)) and len(failure_type[0]) > 0:
                    label_raw = failure_type[0][0]
                else:
                    label_raw = failure_type[0]
            else:
                continue
            
            # 將標籤轉換為標準類別名稱
            label_str = str(label_raw)
            if label_str not in STRING_LABEL_MAP:
                continue
            class_name = STRING_LABEL_MAP[label_str]
            
            # 跳過已用於訓練的資料
            if skip_counts[class_name] < train_counts[class_name]:
                skip_counts[class_name] += 1
                continue
            
            # 檢查是否已達測試資料上限
            if test_counts[class_name] >= max_per_class:
                continue
            
            # 取得晶圓圖譜
            wafer_map = row['waferMap']
            if wafer_map is None or not isinstance(wafer_map, np.ndarray):
                continue
            
            # 轉換為圖片
            img = wafer_to_image(wafer_map)
            
            # 儲存圖片
            img_name = f"{class_name}_test_{test_counts[class_name]:05d}.png"
            img_path = os.path.join(test_dir, class_name, img_name)
            img.save(img_path)
            
            test_counts[class_name] += 1
            
            # 進度顯示
            total = sum(test_counts.values())
            if total % 500 == 0:
                print(f"   已處理 {total} 張測試圖片...")
                
        except Exception as e:
            continue
    
    # 顯示統計
    print("\n" + "=" * 40)
    print("📊 測試資料集統計")
    print("=" * 40)
    total_images = 0
    for class_name, count in test_counts.items():
        print(f"   {class_name:12s}: {count:5d} 張")
        total_images += count
    print("-" * 40)
    print(f"   {'總計':12s}: {total_images:5d} 張")
    print("=" * 40)
    print(f"\n✅ 測試資料集準備完成！圖片已儲存至: {test_dir}")


def visualize_samples(data_dir="data/raw", samples_per_class=3):
    """
    視覺化各類別的樣本
    """
    fig, axes = plt.subplots(len(LABEL_MAP), samples_per_class, figsize=(12, 24))
    
    for row, (label_id, class_name) in enumerate(LABEL_MAP.items()):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) if f.endswith('.png')][:samples_per_class]
        
        for col, img_name in enumerate(images):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(class_name, fontsize=12)
    
    plt.tight_layout()
    plt.savefig("data/sample_visualization.png", dpi=150)
    print("📊 樣本視覺化已儲存至: data/sample_visualization.png")
    plt.show()


def split_existing_images(source_dir="data/raw", train_dir="data/train", test_dir="data/test", test_ratio=0.1):
    """
    將現有的圖片分割為訓練集和測試集
    """
    import shutil
    import random
    
    print(f"🔄 開始從 {source_dir} 分割資料...")
    print(f"   測試集比例: {test_ratio} (訓練: {1-test_ratio:.1f}, 測試: {test_ratio:.1f})")
    
    if not os.path.exists(source_dir):
        print(f"❌ 找不到來源目錄: {source_dir}")
        return

    # 確保輸出目錄存在
    if os.path.exists(train_dir):
        print(f"⚠️  警告: 訓練目錄已存在 {train_dir}")
    if os.path.exists(test_dir):
        print(f"⚠️  警告: 測試目錄已存在 {test_dir}")
        
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 統計
    train_counts = {}
    test_counts = {}
    
    # 遍歷類別目錄
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        # 檢查是否為目錄且是我們關心的類別
        if not os.path.isdir(class_path) or class_name not in LABEL_MAP.values():
            continue
            
        # 建立類別目錄
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # 取得所有圖片
        images = [f for f in os.listdir(class_path) if f.endswith('.png')]
        random.shuffle(images)
        
        # 計算分割點
        n_test = int(len(images) * test_ratio)
        n_train = len(images) - n_test
        
        train_imgs = images[:n_train]
        test_imgs = images[n_train:]
        
        # 複製圖片
        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)
            
        for img in test_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)
            
        train_counts[class_name] = len(train_imgs)
        test_counts[class_name] = len(test_imgs)
        
        print(f"   {class_name:12s}: 訓練 {len(train_imgs):5d} 張, 測試 {len(test_imgs):4d} 張")
        
    print("\n" + "=" * 50)
    print(" 📊 資料分割完成")
    print("=" * 50)
    print(f"   訓練集總數: {sum(train_counts.values())}")
    print(f"   測試集總數: {sum(test_counts.values())}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WM-811K 資料集準備工具")
    parser.add_argument("--download", action="store_true", help="顯示下載指南")
    parser.add_argument("--prepare", action="store_true", help="準備訓練資料集（將 pkl 轉為圖片）")
    parser.add_argument("--prepare-split", action="store_true", help="準備資料集並自動分割 (推薦)")
    parser.add_argument("--split-existing", action="store_true", help="分割現有的 data/raw 圖片")
    parser.add_argument("--prepare-test", action="store_true", help="準備測試資料集（排除訓練資料）")
    parser.add_argument("--visualize", action="store_true", help="視覺化樣本")
    parser.add_argument("--pkl", type=str, default="data/raw/LSWMD.pkl", help="pkl 檔案路徑")
    parser.add_argument("--output", type=str, default="data/raw", help="訓練資料輸出目錄")
    parser.add_argument("--train-output", type=str, default="data/train", help="訓練資料輸出目錄 (用於 --prepare-split)")
    parser.add_argument("--test-output", type=str, default="data/test", help="測試資料輸出目錄")
    parser.add_argument("--max-per-class", type=int, default=100000, help="每類別最大數量")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="測試集比例 (預設 0.1 = 10%)")
    parser.add_argument("--include-none", action="store_true", help="包含 None 類別 (預設排除)")
    
    args = parser.parse_args()
    
    if args.download:
        download_dataset()
    elif args.split_existing:
        split_existing_images(
            args.output, 
            args.train_output, 
            args.test_output, 
            test_ratio=args.test_ratio
        )
    elif args.prepare:
        prepare_dataset(args.pkl, args.output, args.max_per_class)
    elif args.prepare_split:
        prepare_dataset_with_split(
            args.pkl, 
            args.train_output, 
            args.test_output, 
            test_ratio=args.test_ratio,
            max_per_class=args.max_per_class,
            exclude_none=not args.include_none
        )
    elif args.prepare_test:
        prepare_test_dataset(args.pkl, args.output, args.test_output, args.max_per_class)
    elif args.visualize:
        visualize_samples(args.output)
    else:
        # 預設顯示說明
        print("""
WM-811K 資料集準備工具
=====================

使用方式:
  python prepare_data.py --download       # 顯示下載指南
  python prepare_data.py --prepare-split  # 自動分割為訓練/測試集 (推薦)
  python prepare_data.py --prepare        # 將 pkl 轉換為訓練圖片
  python prepare_data.py --prepare-test   # 提取測試資料集
  python prepare_data.py --visualize      # 視覺化樣本

推薦流程 (自動分割 9:1，排除 None):
  1. python prepare_data.py --download    # 查看如何下載
  2. (手動下載 LSWMD.pkl 到 data/raw/)
  3. python prepare_data.py --prepare-split  # 自動分割為 train/test
  4. python -m src.vision.train           # 開始訓練

參數說明:
  --test-ratio 0.1      # 測試集比例 (預設 10%)
  --include-none        # 包含 None 類別
  --max-per-class 5000  # 每類別最多取多少張
        """)
