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

## 📁 程式碼架構

Wafer Copilot 採用**模組化設計**，各功能分離獨立，便於擴展與維護。

### 整體架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI (app.py)                 │
│          ↓ 圖片上傳、診斷觸發、多輪對話交互                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────────┐
        │   LLM Agent (src/agent/bot.py, tools.py) │
        │  支持: Groq / Gemini / Ollama (本地)     │
        └──────────────────────────────────────────┘
         ↙                 ↓                ↘
    Vision          Knowledge Base      Digital Twin
    Module          (RAG 檢索)           模擬器
       ↓                 ↓                  ↓
```

### 核心模塊說明

#### 1️⃣ **視覺識別模塊** (`src/vision/`)
負責晶圓瑕疵的圖像分類與可解釋性分析。

| 文件 | 功能 |
|:---|:---|
| `classifier.py` | ResNet18 分類器，支持 8 大瑕疵類別 (WM-811K)，可選 Mock Mode |
| `gradcam.py` | Grad-CAM 熱力圖生成，視覺化模型關注區域 |
| `train.py` | 模型訓練腳本，支持 CUDA 加速 |

**主要流程**：
```python
from src.vision.classifier import WaferClassifier

classifier = WaferClassifier("models/resnet_wm811k.pth")
result = classifier.predict(image_path, generate_cam=True)
# 返回: {label, confidence, cam_path}
```

---

#### 2️⃣ **知識檢索模塊** (`src/knowledge/`)
整合簡易查詢與向量檢索（RAG），提供結構化的維修建議。

| 文件 | 功能 |
|:---|:---|
| `retriever.py` | 基礎知識庫，支持簡易查詢與結構化輸出 |
| `vector_retriever.py` | 向量知識庫，使用 LangChain 與向量數據庫進行語義搜尋 |
| `solutions.json` | 瑕疵類別 → 維修建議的對照表 |
| `wafer_maintenance_manual.json` | 詳細的製程 SOP 與維修手冊 |

**主要流程**：
```python
from src.knowledge.retriever import KnowledgeBase

kb = KnowledgeBase()
# 簡易查詢
advice = kb.get_solution("Center")
# 詳細查詢（含向量檢索）
detailed = kb.get_detailed_solution("Donut")
```

---

#### 3️⃣ **數位孿生模塊** (`src/digital_twin/`)
使用 **Factory Pattern** 構建模組化模擬器，針對不同瑕疵類型提供物理模擬。

| 文件 | 功能 |
|:---|:---|
| `simulator.py` | 基礎模擬器類與具體實現 (CMPSimulator, EtchSimulator, EnvironmentSimulator) |
| `tools.py` | 與 LLM 集成的工具函數，提供模擬 API |

**支持的模擬器**：

| 瑕疵類型 | 模擬器 | 模擬參數 | 應用場景 |
|:---:|:---|:---|:---|
| **Center** | CMPSimulator | 研磨壓力、研磨液流量、墊片壽命 | CMP 工藝參數調整 |
| **Donut** | EtchSimulator | ESC 溫度、氦氣冷卻、偏壓 | Etch 工藝參數優化 |
| **Random** | EnvironmentSimulator | 潔淨室微粒數、壓差、溫度 | 環境控制改善 |

**主要流程**：
```python
from src.digital_twin.simulator import DigitalTwinFactory

factory = DigitalTwinFactory()
simulator = factory.create_simulator("Center")
result = simulator.run_simulation({
    "polishing_pressure_center": 3.2,
    "slurry_flow_center": 180.0
})
# 返回: {status, predicted_yield, estimated_defects, ...}
```

---

#### 4️⃣ **AI 助理模塊** (`src/agent/`)
核心邏輯層，協調視覺識別、知識檢索與模擬器，提供多輪對話能力。

| 文件 | 功能 |
|:---|:---|
| `bot.py` | LLM 初始化、ReAct Agent 構建、診斷與追問邏輯 |
| `tools.py` | LangChain 工具定義，綁定視覺/知識/模擬功能 |

**支持的 LLM 提供者**：
- **Groq**: `llama-3.3-70b-versatile`（雲端最快）
- **Google Gemini**: `gemini-3-flash-preview`（中文穩定）
- **Ollama (本地)**: `gemma4:e2b`、`qwen3.5:2b`、`llama3.2:3b`

**主要函數**：
```python
from src.agent.bot import analyze_and_report, invoke_followup

# 初始診斷
diagnosis = analyze_and_report(image_path, defect_type)

# 多輪對話
response = invoke_followup(user_query, defect_type, mode="hybrid")
```

---

#### 5️⃣ **模擬數據生成模塊** (`src/simulation/`)
模擬生產履歷與機台日誌，用於測試與演示。

| 文件 | 功能 |
|:---|:---|
| `data_generator.py` | 生成 Mock 批號、Lot History、FDC Log 數據 |

**示例**：
```python
from src.simulation.data_generator import get_mock_context

context = get_mock_context(batch_id="L2026042801")
# 返回模擬的生產履歷與機台異常信息
```

---

### 系統流程圖

```
用戶上傳圖片
    ↓
Streamlit UI (app.py)
    ↓
Vision Classifier 識別瑕疵類別
    ↓
LLM Agent (bot.py)
    ├─→ 查詢生產履歷 (Simulation Module)
    ├─→ 檢索知識庫 (Knowledge Retriever)
    ├─→ 生成根因分析 (RCA)
    ↓
用戶追問
    ├─→ 「如何改善？」→ Digital Twin 模擬
    ├─→ 「具體參數？」→ Knowledge Base 檢索
    ├─→ 「測試結果？」→ Simulator 運算
    ↓
返回結構化建議與模擬結果
```

---

### 模塊依賴關係

```
app.py (UI層)
    ↓
src/agent/bot.py (協調層)
    ├─ src/vision/classifier.py (分類)
    ├─ src/knowledge/retriever.py (知識檢索)
    ├─ src/digital_twin/simulator.py (模擬)
    ├─ src/digital_twin/tools.py
    ├─ src/agent/tools.py
    └─ src/simulation/data_generator.py (數據生成)
```

---

### 檔案結構詳細說明

```
wafer_copilot/
├── app.py                          # Streamlit Web UI 主程式
├── requirements.txt                # Python 依賴
├── run.sh                          # 啟動腳本
├── README.md                       # 本文件
│
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── bot.py                  # LLM 邏輯 + ReAct Agent
│   │   └── tools.py                # LangChain 工具綁定
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── classifier.py           # ResNet18 分類器
│   │   ├── gradcam.py              # Grad-CAM 熱力圖
│   │   └── train.py                # 訓練腳本
│   │
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── retriever.py            # 簡易知識庫
│   │   ├── vector_retriever.py     # RAG 向量檢索
│   │   ├── solutions.json          # 對照表
│   │   ├── wafer_maintenance_manual.json  # SOP 手冊
│   │   └── .vector_cache/          # 向量緩存
│   │
│   ├── digital_twin/
│   │   ├── simulator.py            # Factory + 模擬器實現
│   │   └── tools.py                # 模擬工具函數
│   │
│   └── simulation/
│       └── data_generator.py       # Mock 數據生成
│
├── models/
│   └── resnet_wm811k.pth           # 預訓練模型權重
│
├── scripts/
│   ├── prepare_data.py             # WM-811K 數據轉換
│   ├── evaluate_model.py           # 模型評估
│   ├── evaluate_retrieval.py       # 知識檢索評估
│   └── ...
│
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py          # 視覺模塊單元測試
│   ├── test_digital_twin.py        # 模擬器單元測試
│   ├── test_knowledge_retrieval.py # 知識檢索測試
│   ├── test_agent_integration.py   # 集成測試
│   ├── test_gradcam.py             # Grad-CAM 測試
│   ├── test_multiturn_chat.py      # 多輪對話測試
│   ├── run_all_tests.sh            # 全量測試腳本
│   └── outputs/                    # 測試報告
│
└── data/
    ├── raw/                        # 原始數據集
    ├── train/                      # 訓練集
    ├── test/                       # 測試集
    ├── sample_images/              # 示例圖片
    └── gradcam_outputs/            # Grad-CAM 輸出
```

---

### 擴展指南

#### 新增瑕疵類型的模擬器
1. 在 `src/digital_twin/simulator.py` 中繼承 `BaseSimulator`：
```python
class NewSimulator(BaseSimulator):
    def __init__(self):
        super().__init__("新製程名稱")
        self.optimal_params = {...}
    
    def run_simulation(self, params):
        # 實現模擬邏輯
        return {...}
```

2. 在 `DigitalTwinFactory.create_simulator()` 中註冊：
```python
elif defect_type == "NewType":
    return NewSimulator()
```

#### 新增知識庫資訊
更新 `src/knowledge/solutions.json` 或 `wafer_maintenance_manual.json`，系統會自動進行向量索引。

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

### Ollama 啟動
若使用本地模型，請先啟動 Ollama：
```bash
ollama serve
```

---

## 📊 實測結果

測試條件：
- 任務流程：`analyze_and_report` + `invoke_followup`（固定 `hybrid`，章節 + 語義）
- 測試圖片：`data/raw/Donut/Donut_00167.png`
- 量測時間：2026-04-26 (Ollama GPU: `OLLAMA_NUM_GPU=999`, Context: 4096)

### A. 目前 App 本地模型實測

| Provider | 模型 | 初始診斷 (s) | 追問 hybrid (s) | 總時間 (s) | 備註 |
|:--|:--|--:|--:|--:|:--|
| Ollama | gemma4:e2b | 10.37 | 12.58 | 22.96 | 預設，報告較完整 |
| Ollama | qwen3.5:2b | 18.58 | 7.48 | 26.06 | 中文輸出穩定 |
| Ollama | llama3.2:3b | 7.38 | 1.28 | 8.67 | 本地最快 |


### B. 雲端模型參考（不受本地 GPU 綁定影響）

| Provider | 模型 | 初始診斷 (s) | 追問 hybrid (s) | 總時間 (s) | 備註 |
|:--|:--|--:|--:|--:|:--|
| Groq | llama-3.3-70b-versatile | 2.15 | 1.40 | 3.55 | 雲端最快 |
| Gemini | gemini-3-flash-preview | 10.42 | 9.97 | 20.38 | 雲端穩定 |

### C. 本地模型建議

- 目前預設本地模型為 `gemma4:e2b`。
- `qwen3.5:2b` 與 `llama3.2:3b` 已在 UI 中提供選項；使用前需確認本機 Ollama 已下載對應模型並啟動服務。

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
A: 目前支援 **Ollama (本地: gemma4:e2b / qwen3.5:2b / llama3.2:3b)**、**Google Gemini** 與 **Groq**。可透過 UI 側邊欄切換。

**Q: 如何新增一種製程模擬？**
A: 請繼承 `src.digital_twin.simulator.BaseSimulator`，實作 `run_simulation` 方法，並在 `DigitalTwinFactory` 中註冊。

---
**Wafer Copilot Team** | [GitHub Repo](#)
