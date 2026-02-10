# Wafer Copilot - 半導體 AI 智慧助理

![Wafer Copilot](https://img.shields.io/badge/Status-Beta-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

Wafer Copilot 是一個專為半導體製造設計的 AI 智慧助理，結合深度學習 (CNN)、大型語言模型 (LLM) 與數位孿生 (Digital Twin) 技術，協助工程師快速診斷晶圓瑕疵、查詢製程知識並模擬參數調整結果。

---

## 🌟 核心功能

1.  **🔍 智慧視覺與生產履歷整合診斷**
    *   **雙重驗證**：不僅基於 CNN 模型識別晶圓圖譜，更自動關聯模擬的生產履歷 (Lot History) 與機台日誌 (FDC Log)。
    *   **Grad-CAM 可解釋性**：視覺化模型關注區域，確保 AI 看到正確的瑕疵特徵。
    *   **根因分析 (RCA)**：自動比對視覺瑕疵特徵與機台異常訊號 (如 Etcher 流量警示)，提供更精準的根本原因分析。

2.  **💬 多輪對話與 RAG 知識檢索**
    *   整合 LangChain 與 LLM (Gemini/Groq)，提供自然語言對話介面。
    *   **RAG (Retrieval-Augmented Generation)**：檢索內建的半導體製程知識庫與 SOP，提供有憑據的專業建議。

3.  **🧪 Digital Twin 數位孿生模擬**
    *   **模組化製程模擬**：針對 Center (CMP)、Donut (Etch)、Random (Environment) 等參數型瑕疵提供物理模擬。
    *   **參數尋優**：在虛擬環境中測試溫度、壓力、流量等參數調整對良率的影響，降低實機測試風險。
    *   **智慧分流**：自動區分「可模擬瑕疵」與「硬體/機械故障」(如 Scratch)，給出正確的處置建議。

---

## 🛠️ 安裝與設定

### 1. 環境需求
*   Linux / macOS / Windows
*   Python 3.10+
*   CUDA (選用，用於加速模型訓練)

### 2. 專案安裝
```bash
# 複製專案
git clone <repo_url>
cd wafer_copilot

# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

### 3. 設定 API Key
請設定 `GOOGLE_API_KEY` (使用 Gemini) 或 `GROQ_API_KEY`：

```bash
export GOOGLE_API_KEY="your_api_key_here"
# 或者
export GROQ_API_KEY="your_api_key_here"
```

---

## 🚀 系統使用方式

### 啟動 Web UI
```bash
streamlit run app.py
```
啟動後，瀏覽器將自動開啟 `http://localhost:8501`。

### 使用流程
1.  **上傳圖片**：從側邊欄上傳晶圓圖譜 (支援 PNG/JPG)。
2.  **AI 整合診斷**：
    *   點擊「開始 AI 診斷」。
    *   系統識別瑕疵類型 (Vision) 並自動拉取該批號的生產履歷數據 (Legacy Data)。
    *   LLM 綜合分析影像特徵與機台 Log，生成包含「可能異常機台」與「良率影響」的完整報告。
3.  **專家對話**：針對診斷結果提問（例：「如何改善 Center 瑕疵？」）。
4.  **數位模擬**：若瑕疵屬於可模擬類型，點擊「模擬參數測試」或直接要求 AI 模擬建議的參數。

---

## 🧪 Digital Twin 模組說明

本系統採用 **Factory Pattern** 構建模組化模擬器，根據瑕疵特性自動選擇：

### 支援模擬的瑕疵 (Simulatable)
| 瑕疵類型 | 對應製程 (Simulator) | 關鍵參數範例 |
|:---:|:---|:---|
| **Center** | CMP (化學機械研磨) | 研磨壓力 (psi), 研磨液流量 (ml/min) |
| **Donut** | Etch (電漿蝕刻) | ESC 溫度 (°C), 氦氣冷卻壓力 (Torr) |
| **Random** | Environment (環境控制) | 潔淨室微粒數, 壓差 (inch H₂O) |

### 不可模擬的瑕疵 (Non-Simulatable)
| 瑕疵類型 | 原因 | 建議處置 |
|:---:|:---|:---|
| **Scratch** | 機械手臂/傳送刮傷 | 停機清潔 Robot End Effector, 更換 FOUP |
| **Edge-Ring**| 硬體老化 | 更換 Edge Ring 或 Heater |
| **Loc/Edge-Loc** | 局部污染/對準異常 | 檢查晶圓對準器 (Notch Finder) |

---

## 🧠 模型訓練 (WM-811K)

若需重新訓練視覺模型，請遵循以下步驟：

### 1. 準備資料
WM-811K 原始檔為 pickle 格式。請先下載並轉換為圖片：
```bash
# 下載資料集 (需設定 Kaggle API)
kaggle datasets download -d qingyi/wm811k-wafer-map -p data/raw/
unzip data/raw/wm811k-wafer-map.zip -d data/raw/

# 轉換資料格式
python prepare_data.py --prepare
```

### 2. 執行訓練
```bash
python -m src.vision.train
```
*   預設參數：Epochs=20, Batch=32, Model=ResNet18
*   模型輸出路徑：`models/resnet_wm811k.pth`

---

## ✅ 測試與驗證

本專案包含完整的測試套件：

```bash
# 執行所有測試
bash tests/run_all_tests.sh

# 僅測試數位孿生模組
python tests/test_digital_twin.py
```

重點測試檔案：
*   `tests/test_digital_twin.py`: 驗證工廠模式與物理公式計算準確性。
*   `tests/test_classifier.py`: 驗證 CNN 模型推論。
*   `tests/test_agent_integration.py`: 驗證 LangChain 工具調用流程。

---

## ❓ 常見問題 (FAQ)

**Q: 為什麼 Scratch 瑕疵顯示「無法模擬」？**
A: Scratch (刮痕) 通常是由機械手臂摩擦或異物造成的物理損傷，這屬於「硬體故障」而非「製程參數偏差」。解決方案是更換零件或清潔，無法透過調整溫度/壓力來修復，因此不適用參數模擬。

**Q: 支援哪些 LLM？**
A: 目前原生支援 **Google Gemini** (推薦，多模態能力佳) 與 **Groq (Llama 3)** (速度快)。可透過 UI 側邊欄切換。

**Q: 如何新增一種製程模擬？**
A: 請繼承 `src.digital_twin.simulator.BaseSimulator`，實作 `run_simulation` 方法，並在 `DigitalTwinFactory` 中註冊。

---
**Wafer Copilot Team** | [GitHub Repo](#)
