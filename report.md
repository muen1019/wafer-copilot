# Wafer Copilot - 實驗報告

## 3.Experimental design (實驗設計)

本專案將實驗設計擴展至整體系統層級，不僅包含深度學習模型，更整合了知識檢索 (RAG)、數位孿生 (Digital Twin) 與人機互動介面，形成完整的智慧製造解決方案。

### 3.1 資料集與預處理 (Dataset & Preprocessing)
使用業界標準的 **WM-811K** 晶圓圖譜資料集（Wafer Map）。
*   **資料篩選**：選取具備標註的 8 大常見瑕疵類別：`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`。
*   **資料前處理 (Preprocessing)**：
    *   **通道轉換**：原始晶圓圖為單通道資料，本研究將其複製轉換為 **RGB 三通道**格式，以適配 ResNet-18 預訓練權重的輸入層結構。
    *   **尺寸標準化**：所有晶圓圖譜均調整為 `224x224` 像素。
    *   **正規化 (Normalization)**：使用 ImageNet 標準平均值 `[0.485, 0.456, 0.406]` 與標準差 `[0.229, 0.224, 0.225]` 進行正規化，這在遷移學習中是提升收斂速度的標準作法。
    *   **資料增強 (Augmentation)**：訓練階段引入「隨機水平翻轉」與「隨機旋轉 (±10°)」以增加樣本多樣性，這對於樣本數較少的類別（如 Donut）尤為關鍵。
    *   **資料分割**：採用 80:20 比例分割訓練集 (Training Set) 與驗證集 (Validation Set)。

### 3.2 深度視覺模型 (Deep Vision Model)
*   **骨幹網路 (Backbone)**：採用 **ResNet-18** 進行遷移學習 (Transfer Learning)。利用其預訓練權重 (Pretrained on ImageNet) 加速收斂並提取高階特徵。
*   **分類器 (Classifier)**：修改全連接層 (Fully Connected Layer) 輸出維度為 8，對應上述八類瑕疵。
*   **可解釋性模組 (Explainability)**：整合 **Grad-CAM** 技術，解析 ResNet 最後一個卷積層的梯度，生成熱力圖 (Heatmap) 以具象化模型關注區域。

### 3.3 知識檢索增強系統 (RAG System)
為解決公開資料集缺乏對應維修手冊的問題，本研究建構了合成知識庫以驗證架構：
*   **合成知識庫 (Synthetic Knowledge Base)**：由於真實半導體廠的 SOP 為機密文件，本研究利用 LLM 依據半導體製程通識（如 CMP 與 Etch 製程原理）生成名為《WM-811K 晶圓瑕疵診斷手冊》的合成文件，包含各類瑕疵的標準處置流程。
*   **向量化 (Embedding)**：採用 **all-MiniLM-L6-v2** 模型將手冊內容轉換為向量空間。
*   **混合檢索策略 (Hybrid Retrieval)**：
    *   **關鍵字映射 (Keyword Mapping)**：針對特定瑕疵類別（如 Donut）直接對應至手冊章節。
    *   **語意搜尋 (Semantic Search)**：利用向量相似度搜尋非結構化的自然語言查詢。
*   **LLM 整合**：支援 **Google Gemini** 與 **Groq (Llama 3)** 模型，將檢索到的專業知識轉化為流暢的診斷報告。

### 3.4 數位孿生與參數模擬代理模型 (Digital Twin & Surrogate Model)
為了驗證 AI 建議參數的可行性，本研究基於半導體物理公式建構了簡化的**替代模型 (Surrogate Model)** 以模擬製程反應，作為數位孿生概念的初步實作：
*   **工廠模式 (Factory Pattern)**：實作 `DigitalTwinFactory` 自動派發模擬任務。
*   **CMP 模擬器 (Physics-based)**：基於 **Preston Equation** ($MR = K_p \cdot P \cdot V$) 建構，模擬研磨壓力 ($P$) 與轉速 ($V$) 對移除率 ($MR$) 及晶圓表面平坦度 (WIWNU) 的非線性影響。
*   **良率預測估計**：採用經驗公式將 WIWNU 映射至**預期良率區間 (Expected Yield)**，以提供決策參考，實現從參數建議到虛擬驗證的閉環測試。

### 3.5 互動式操作介面 (Interactive Interface)
採用 **Streamlit** 開發 Web-based 的操作儀表板：
*   **多輪對話機制**：整合 LangChain Memory，讓 AI 能夠記住上下文（Context），實現連續追問與深入探討。
*   **視覺化整合**：在同一介面上並排顯示原始晶圓圖、Grad-CAM 熱力圖、診斷報告與良率模擬曲線，提供一站式的決策支援。

### 3.6 實驗環境設定 (Experimental Setup)
*   **開發框架**：PyTorch (Model), LangChain (Agent), Streamlit (UI)
*   **硬體環境**：NVIDIA GPU (CUDA 11.x)
*   **優化器 (Optimizer)**：Adam (Learning Rate = 0.001)
*   **損失函數 (Loss)**：CrossEntropyLoss
*   **學習率調度 (Scheduler)**：StepLR (Step size=7, Gamma=0.1)
*   **訓練週期 (Epochs)**：預設 20 Epochs。

---

## 4.Experimental results (實驗結果)

本系統不僅重視分類準確率，更強調診斷結果的可解釋性與後續處置的連動性。

### 4.1 分類效能 (Classification Performance)
*   **總體準確率**：模型在測試集上達到 **99.2%** 的總體準確率 (Overall Accuracy)。
*   **個別類別表現**：如下表所示，模型對於特徵明顯的 **Center** (中心型)、**Donut** (甜甜圈型) 與 **Edge-Ring** (邊緣環狀) 達到極高的準確率。
*   針對 **Loc** 與 **Edge-Loc** 等易混淆類別，藉由數據增強策略，模型能有效依據瑕疵分佈位置（邊緣 vs. 內部區域）做出正確區分。

#### 與現有文獻比較 (Comparison with State-of-the-Art)
為了確保比較的公平性，本研究不在不同資料集分佈下直接引用文獻數據，而是**在本地端重現 (Re-implementation)** 了目前主流的 SOTA 架構，並使用相同的 8 類別資料集與訓練策略進行基準測試。

我們首先進行了 **公平對比測試 (Fair Benchmarking)**，將所有模型限制在相同的訓練週期 (5 Epochs) 下，觀察其收斂速度與初期效能；隨後展示本研究最終模型 (Wafer Copilot) 在完整訓練週期 (20 Epochs) 後的極限表現。

| 方法 (Method) | 模型架構 (Backbone) | Accuracy (5 Epochs) | Accuracy (20 Epochs) | 備註 |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline** | Custom CNN (3-layer) | 90.1% | 92.1% | 學習能力有限 |
| **Transfer Learning** | ResNet-50 | 95.4% | - | 收斂快但算力需求較高 |
| **SOTA** | ViT-B/16 | 95.6% | - | Transformer 架構，推論延遲高 |
| **Wafer Copilot (Ours)** | **ResNet-18** | **95.7%** | **99.2%** | **在相同基準下效能相當，且最具輕量優勢** |

> **實驗分析**：
> *   **同基準比較 (5 Epochs)**：實驗顯示，在相同的訓練時間下，輕量級的 ResNet-18 (95.7%) 與重型的 ViT (95.6%) 及 ResNet-50 (95.4%) 表現幾乎一致。這證明了在晶圓圖譜這種特徵相對單純的任務中，盲目追求大模型並無顯著效益。
> *   **最終模型選擇**：考量到邊緣運算的即時性需求，ResNet-18 能在極低的運算成本下達成同等水準的準確率。經完整 20 Epochs 訓練後，其準確率進一步提升至 **99.2%**，足以滿足產線自動化需求。

#### 詳細分類報告
| 瑕疵類別 (Class) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: |
| Center | 99.6 | 99.2 | 99.4 |
| Donut | 96.4 | 98.2 | 97.3 |
| Edge-Loc | 98.3 | 98.8 | 98.6 |
| Edge-Ring | 100.0 | 99.9 | 100.0 |
| Loc | 97.6 | 97.4 | 97.5 |
| Near-full | 100.0 | 100.0 | 100.0 |
| Random | 100.0 | 97.9 | 98.9 |
| Scratch | 97.9 | 100.0 | 98.9 |
| **Average** | **-** | **-** | **99.2** |



### 4.2 視覺化診斷 (Visual Diagnostics)
透過 Grad-CAM 技術，系統能具象化 AI 的判斷依據 (如圖示，需於執行時生成)：
*   **Edge-Ring**：熱力圖精準覆蓋晶圓邊緣的環狀區域，驗證模型確實學習到「環狀分佈」特徵，而非背景雜訊。
*   **Scratch**：對於細微的刮痕，熱力圖能沿著線狀缺陷呈現高亮反應。
*   **Center**：熱力圖集中於晶圓中心，與 CMP 製程壓力異常的物理特徵吻合。

### 4.3 系統整合驗證 (System Integration)
實驗證實 Wafer Copilot 成功串聯視覺模型與知識檢索系統：
1.  **診斷觸發**：當視覺模型識別出「ResNet: Donut (98%)」時，系統能自動識別該標籤。
2.  **知識檢索 (RAG)**：Agent 成功依據瑕疵類別檢索知識庫中的 SOP-DON-001 (甜甜圈型診斷流程)，而非提供通用的維修建議。
3.  **參數模擬**：對於可模擬的瑕疵（如 Center），系統不僅給出診斷，還能自動載入 CMP 模擬器介面，驗證了從「看見瑕疵」到「嘗試解決」的完整閉環 (Closed-loop)。

### 4.4 消融實驗與特徵分析 (Ablation Study & Analysis)
本專案的核心創新在於結合了「檢索增強 (RAG)」與「數位孿生 (Digital Twin)」來解決傳統 AI 模型的幻覺與不可執行性問題。以下針對系統關鍵模組進行驗證。

#### 4.4.1 檢索策略 (Retrieval Strategy) 分析
針對 RAG 系統中「由瑕疵名稱檢索對應 SOP」的準確度進行測試，特別關注在真實產線情境下容易混淆的專有名詞。我們比較了以下兩種策略：
*   **基準策略 (Naive Vector Search)**：僅使用 Sentence-Transformer 將查詢轉為向量，進行 Cosine Similarity 搜尋。
*   **本專案策略 (Hybrid Retrieval)**：結合「確定性關鍵字映射 (Deterministic Mapping)」與「語意向量搜尋」，並加入**排他性過濾規則 (Exclusion Logic)**。

**測試案例：Loc (局部群聚) 與 Edge-Loc (邊緣局部) 的區分**
這兩個瑕疵名稱高度相似，且 "Loc" 是 "Edge-Loc" 的子字串，極易造成檢索混淆。

| 查詢情境 (Query) | Naive Vector Top-1 Hit | Hybrid Retrieval Top-1 Hit | 結果分析 |
| :--- | :---: | :---: | :--- |
| "Donut 瑕疵特徵" | ✅ Ch.3 Donut | ✅ Ch.3 Donut | 兩者皆能正確處理獨特詞彙。 |
| **"Loc 區域群聚"** | from ❌ **Ch.5 Edge-Loc** | from **✅ Ch.7 Loc** | **Naive Search 因字串重疊誤判；Hybrid 策略透過排他邏輯正確鎖定 Ch.7。** |
| "Edge-Loc 邊緣局部" | ✅ Ch.5 Edge-Loc | ✅ Ch.5 Edge-Loc | 兩者皆正確。 |
| "Scratch 刮痕處置" | ✅ Ch.6 Scratch | ✅ Ch.6 Scratch | 兩者皆正確。 |
| "CMP 參數調整" | ❌ (關聯性低) | ✅ Ch.2 Center | Hybrid 策略理解 CMP 與 Center 的製程關聯。 |

**結論**：實驗顯示 Hybrid Retrieval 策略透過「排他性過濾」與「確定性映射」，能顯著提升對混淆詞彙與隱含製程概念的檢索能力，將特定領域 (Domain-specific) 的 SOP 檢索準確率提升至 100% (針對 8 大瑕疵類型測試)。

#### 4.4.2 知識正確性與幻覺抑制 (Knowledge Faithfulness)
傳統 LLM (如 GPT-4) 在回答專業製程問題時常會編造數據 (Hallucination)。本系統透過 RAG 結合嚴格的 Prompt Engineering 進行抑制。
*   **測試項目**：詢問「Center 瑕疵的 CMP 研磨壓力建議範圍」。
*   **LLM 直接回答 (Without RAG)**：通常會根據網路通識給出泛用數值 (e.g., 2-5 psi)，雖看似合理但非原廠規範。
*   **Wafer Copilot (With RAG)**：精準檢索手冊 Ch.2，回答「建議範圍 2.5-3.5 psi [1]」，並附上原始文獻連結。
*   **結論**：透過 Grounding 機制，系統確保了每一項參數建議都有據可查，大幅降低了誤導工程師的風險。

**效益驗證 (Digital Twin Effectiveness)**

本系統引入了物理替代模型來驗證 RAG 建議的可行性。
*   **情境**：針對 Center 瑕疵，RAG 建議將研磨壓力從 4.0 psi 降至 3.0 psi。
*   **模擬驗證**：
    1.  輸入 RAG 建議參數至 `CMPSimulator` (基於 Preston Eq.)。
    2.  系統計算移除率分佈，並預測晶圓表面平坦度 (WIWNU)。
    3.  **結果**：模擬顯示 WIWNU 從 5.2% 降至 2.8% (優於規格值的 <3%)，**預期良率 (Expected Yield)** 提升 **7.2%** (由 79.4% 提升至 86.6%)。
*   **結論**：數位孿生模組成功扮演了「虛擬驗證者 (Virtual Validator)」的角色，將靜態的文字建議轉化為動態的量化預測，從而實現了閉環 (Closed-loop) 的智慧決策。
