"""
Wafer Copilot - LLM Bot
支援 Groq (Llama 3.3) 和 Google Gemini 雙模式
"""

import os
from dotenv import load_dotenv
from src.agent.tools import analyze_wafer_defect, search_maintenance_knowledge

load_dotenv()

# ============================================================
# LLM Provider 設定
# ============================================================
# 可選值: "groq", "gemini"
# 可透過環境變數 LLM_PROVIDER 設定，或在程式中動態切換
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

_current_provider = DEFAULT_LLM_PROVIDER


def set_llm_provider(provider: str):
    """
    設定 LLM 提供者
    
    Args:
        provider: "groq" 或 "gemini"
    """
    global _current_provider
    if provider not in ["groq", "gemini"]:
        raise ValueError(f"不支援的 LLM provider: {provider}，請選擇 'groq' 或 'gemini'")
    _current_provider = provider
    print(f"✅ LLM provider 已切換為: {provider}")


def get_llm_provider() -> str:
    """取得目前的 LLM 提供者"""
    return _current_provider


def _get_llm():
    """根據設定建立對應的 LLM 實例"""
    if _current_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:  # groq
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )


def get_llm_info() -> dict:
    """取得目前 LLM 的資訊"""
    if _current_provider == "gemini":
        return {
            "provider": "Google",
            "model": "Gemini 3 Flash Preview",
            "icon": "🔷"
        }
    else:
        return {
            "provider": "Groq",
            "model": "Llama 3.3 70B",
            "icon": "🟠"
        }


def _extract_text_content(content) -> str:
    """
    從 LLM 回應中提取純文字內容
    處理 Gemini 的 multimodal content blocks 格式
    
    Args:
        content: response.content，可能是 str 或 list
    
    Returns:
        純文字字串
    """
    # 如果已經是字串，直接返回
    if isinstance(content, str):
        return content
    
    # 如果是 list（Gemini multimodal 格式）
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # 格式: {'type': 'text', 'text': '...'}
                if item.get('type') == 'text' and 'text' in item:
                    text_parts.append(item['text'])
            elif isinstance(item, str):
                text_parts.append(item)
        return '\n'.join(text_parts)
    
    # 其他情況，嘗試轉字串
    return str(content)


def analyze_and_report(image_path: str) -> dict:
    """
    不使用 function calling 的穩定版本：
    1. Python 直接調用工具
    2. 把結果給 LLM 生成格式化報告
    
    Returns:
        dict: {"report": str, "defect_type": str, "vision_result": dict}
    """
    llm = _get_llm()
    
    # Step 1: 直接調用視覺辨識工具
    vision_result = analyze_wafer_defect.invoke({"image_path": image_path})
    
    if isinstance(vision_result, str) and "錯誤" in vision_result:
        return {"report": f"視覺辨識失敗: {vision_result}", "defect_type": "Unknown", "vision_result": {}}
    
    defect_type = vision_result.get("defect_type", "Unknown")
    raw_confidence = vision_result.get("confidence", 0)
    priority = vision_result.get("priority_level", "NORMAL")
    
    # 修正信心度格式：將 0-1 的數值轉換為百分比
    if raw_confidence <= 1:
        confidence_percent = round(raw_confidence * 100, 1)
    else:
        confidence_percent = round(raw_confidence, 1)
    
    # Step 2: 獲取知識庫資訊
    # 優先使用 vision_result 中已有的建議（來自確定性章節檢索），若無則進行搜尋
    if "maintenance_advice" in vision_result and vision_result["maintenance_advice"]:
        print(f"✅ 使用工具預先檢索的知識庫建議 ({defect_type})")
        knowledge_result = vision_result["maintenance_advice"]
    else:
        print(f"⚠️ 工具未返回建議，執行備用搜尋 ({defect_type})")
        knowledge_query = f"{defect_type} 瑕疵參數調整建議"
        knowledge_result = search_maintenance_knowledge.invoke({"query": knowledge_query})
    
    # Step 3: 讓 LLM 根據工具結果生成報告（不使用 function calling）
    # 重新設計 prompt，強調引用必須對應實際來源
    system_prompt = """你是擁有 20 年經驗的資深半導體製程與設備維護專家。你的任務是根據自動化檢測系統的數據與知識庫檢索結果，撰寫一份專業、準確且具備可執行性的瑕疵診斷報告。

### ⚠️ 核心原則 (違反將導致嚴重扣分)
1. **事實至上**：所有的參數建議 (如溫度、壓力、流量) 必須 100% 來自【知識庫檢索結果】，嚴禁依據通用知識編造數值。
2. **精確引用**：每一項建議或成因分析後，必須標註來源編號 (如 [1], [2])，且該編號必須對應到提供的知識庫內容。
3. **誠實不知**：若知識庫中沒有相關資訊，請明確寫出「知識庫中無此資訊」，不要強行推論。

### 🎯 輸出報告結構

---
### 📊 診斷摘要
| 項目 | 結果 |
|------|------|
| 瑕疵類別 | (填入提供的類別) |
| 信心度 | (填入提供的百分比) |
| 優先級別 | (填入提供的級別) |

### 💡 瑕疵特徵說明
(請描述此類瑕疵在晶圓上的典型分佈特徵與可能影響的晶粒範圍。若知識庫有提到此瑕疵的判定標準，請一併列出。) [來源編號]

### 🔍 根本原因分析
請依照邏輯推演進行分析：
1. **現象觀察**：(描述此瑕疵通常伴隨的製程異狀)
2. **潛在環節**：(指出可能出問題的具體製程步驟，如蝕刻、鍍膜等) [來源編號]
3. **關鍵因子**：(具體指出哪個硬體部件或參數異常導致此結果) [來源編號]

### 🔧 建議改善對策
請提供具體、可執行的排查步驟。若知識庫包含具體參數，務必使用表格呈現：

| 檢查項目 / 模組 | 建議設定值 / 標準規範 | 資料來源 |
|----------------|----------------------|----------|
| (例如：ESC 溫度) | (例如：60±2°C) | [來源編號] |

(若無具體數值，請列出詳細的排查清單。**注意：嚴禁編造「預計時間」或「步驟時長」，若知識庫未提及時間，請直接省略該資訊。**)

### 🧪 數位孿生驗證
請根據瑕疵類型判斷是否適合進行模擬：
- **可模擬類型 (Center, Donut, Random)**：若您的建議對策包含具體的製程參數調整，告知使用者「您可以要求我在數位孿生系統中模擬這些參數組合，以評估其可行性與風險。」
- **不可模擬類型 (Scratch, Edge-Ring, Edge-Loc, Loc, Near-full)**：這些瑕疵的處置以硬體更換、清潔程序或緊急停機為主，無法進行參數模擬。若使用者詢問模擬，請說明原因。

### 📚 參考文獻
[1] (對應知識庫結果 1 的標題)
[2] (對應知識庫結果 2 的標題)
---

用繁體中文，專業且語氣肯定地回答。"""
    
    user_prompt = f"""請根據以下資訊生成診斷報告：

【視覺辨識結果】
- 瑕疵類別: {defect_type}
- 信心度: {confidence_percent}%
- 優先級別: {priority}
- Grad-CAM 路徑: {vision_result.get('gradcam_overlay', 'N/A')}

【知識庫檢索結果】
{knowledge_result}

⚠️ 請注意：
1. 信心度直接使用 {confidence_percent}%，不要修改
2. 引用標註必須對應上方「結果 1」=「[1]」、「結果 2」=「[2]」、「結果 3」=「[3]」
3. 建議對策的數值必須從知識庫原文直接複製，例如「60±2°C」「10±1 Torr」等

請生成完整的診斷報告。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = llm.invoke(messages)
    report_text = _extract_text_content(response.content)
    
    return {
        "report": report_text,
        "defect_type": defect_type,
        "vision_result": vision_result
    }


def get_agent():
    """
    建立具備完整工具能力的 Agent (包含 Digital Twin 模擬)
    用於需要呼叫外部工具的複雜任務 (如視覺辨識、知識檢索、模擬驗證)
    """
    llm = _get_llm()
    from langgraph.prebuilt import create_react_agent
    from src.agent.tools import analyze_wafer_defect, search_maintenance_knowledge, simulate_defect_solution
    
    tools = [analyze_wafer_defect, search_maintenance_knowledge, simulate_defect_solution]
    
    system_prompt = """你是半導體製程專家，具備以下能力：
1. 視覺辨識瑕疵 (analyze_wafer_defect)
2. 查詢維修知識庫 (search_maintenance_knowledge)
3. 模擬數位孿生系統 (simulate_defect_solution)

### 數位孿生使用規則
當使用者詢問參數調整建議時：
- 若涉及具體數值調整且瑕疵類型為 **Center, Donut, Random** 之一，請主動提供「可透過數位孿生系統進行模擬驗證」的選項。
- 若使用者要求驗證或測試參數，請使用 simulate_defect_solution 工具。
- 若瑕疵類型為 **Scratch, Edge-Ring, Edge-Loc, Loc, Near-full**，這些屬於硬體更換、清潔程序或緊急處置，無法進行參數模擬，請誠實告知使用者並說明原因。

用繁體中文回答。"""
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent


def get_followup_llm():
    """建立用於追問的 LLM（不使用工具，直接回答）"""
    return _get_llm()


def invoke_agent_with_history(agent, user_message: str, chat_history: list) -> str:
    """
    帶有對話歷史的 Agent 調用
    
    Args:
        agent: LangGraph Agent
        user_message: 使用者的新訊息
        chat_history: 對話歷史 [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        Agent 的回應文字
    """
    # 構建完整的訊息列表
    messages = []
    
    for msg in chat_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # 加入新的使用者訊息
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # 調用 Agent
    response = agent.invoke({"messages": messages})
    
    # 取得最後一則回應
    return response["messages"][-1].content


def invoke_followup(user_message: str, chat_history: list, diagnosis_context: str = "", retrieval_strategy: str = "hybrid") -> str:
    """
    處理追問（不使用工具，但自動搜尋知識庫作為上下文）
    
    Args:
        user_message: 使用者的追問
        chat_history: 對話歷史
        diagnosis_context: 首次診斷的上下文（如瑕疵類型）
        retrieval_strategy: 檢索策略 ("hybrid" | "vector_only" | "no_rag")
    
    Returns:
        LLM 的回應
    """
    from src.knowledge.vector_retriever import VectorKnowledgeBase
    from langgraph.prebuilt import create_react_agent
    from src.agent.tools import simulate_defect_solution
    
    llm = get_followup_llm()
    
    # [Strategy: No RAG] 若策略為關閉 RAG，直接回傳
    if retrieval_strategy == "no_rag":
        system_msg = "你是半導體製程專家。請依據你的專業知識回答問題。若不確定請誠實告知。"
        messages = [{"role": "system", "content": system_msg}]
        messages.extend([{"role": m["role"], "content": m["content"]} for m in chat_history[-4:]])
        messages.append({"role": "user", "content": user_message})
        response = llm.invoke(messages)
        return _extract_text_content(response.content)

    # 1. 偵測對話中涉及的所有瑕疵類型
    # 建立映射表以支援中英文與多種稱呼
    defect_mapping = {
        "center": "Center", "中心": "Center",
        "donut": "Donut", "甜甜圈": "Donut",
        "edge-ring": "Edge-Ring", "邊緣環狀": "Edge-Ring",
        "edge-loc": "Edge-Loc", "邊緣局部": "Edge-Loc",
        "loc": "Loc", "局部": "Loc",
        "near-full": "Near-full", "近全區": "Near-full",
        "random": "Random", "隨機": "Random",
        "scratch": "Scratch", "刮痕": "Scratch"
    }

    # [新增] 通用查詢關鍵字 (偵測是否詢問總覽、分類、列表)
    general_keywords = [
        "分類", "幾種", "種類", "類型", "types", "categories", 
        "classification", "總覽", "列表", "list", "概覽"
    ]
    
    found_defects = set()
    
    # Check for general keywords
    is_general_query = any(k in user_message.lower() for k in general_keywords)
    if is_general_query:
        # 添加 "總覽" 以觸發第一章的確定性檢索
        found_defects.add("總覽")
    
    # [新增] 若有明確的診斷上下文 (Active Defect)，優先加入
    # 但若發現是「通用查詢」(比如問有幾種瑕疵)，則此上下文權重應降低，
    # 雖然這裡還是加入，但在後續 Prompt 提示模型優先回答使用者問題。
    if diagnosis_context and diagnosis_context in defect_mapping.values():
        found_defects.add(diagnosis_context)
    
    # 優先檢查當前使用者訊息
        found_defects.add(diagnosis_context)
    
    # 優先檢查當前使用者訊息
    for key, val in defect_mapping.items():
        if key.lower() in user_message.lower():
            found_defects.add(val)
            
    # 若當前訊息無瑕疵關鍵字 且 無診斷上下文，才回溯歷史 (取最近幾則)
    if not found_defects:
        history_text = " ".join([m.get("content", "") for m in chat_history[-3:]])
        for key, val in defect_mapping.items():
            if key.lower() in history_text.lower():
                found_defects.add(val)
    
    # 2. 檢索知識庫 (混合策略：確定性章節 + 語意搜尋)
    kb = VectorKnowledgeBase()
    knowledge_context = ""
    source_docs = [] # 用於去重和排序
    seen_identifiers = set() # (title, content_snippet)
    
    # (A) 針對每個偵測到的瑕疵，撈取完整章節 (確保有參數與 SOP)
    if retrieval_strategy == "hybrid":
        for defect_type in list(found_defects)[:3]: # 限制最多處理3個，避免上下文過長
            try:
                # 使用確定性匹配撈取
                sol = kb.get_solution_by_defect(defect_type, top_k=6)
                if sol.get("found"):
                    for sec in sol.get("sections", []):
                        # 嘗試從標題還原章節資訊，或直接使用標題
                        # 格式: 章節 > 小節
                        chapter = sec.get("chapter", "")
                        title = sec.get("title", "")
                        if chapter:
                            identifier = f"{chapter} > {title}"
                        else:
                            identifier = title
                            
                        if identifier not in seen_identifiers:
                            source_docs.append({
                                "source": identifier,
                                "content": sec["content"]
                            })
                            seen_identifiers.add(identifier)
            except Exception as e:
                print(f"瑕疵章節檢索失敗 ({defect_type}): {e}")

    # (B) 針對使用者具體問題進行語意搜尋 (補充跨章節或細節資訊)
    try:
        # 增加 top_k 確保能抓到比較細微的差異
        search_query = user_message
        if found_defects and retrieval_strategy == "hybrid":
            search_query = f"{' '.join(found_defects)} {user_message}"
            
        semantic_results = kb.search(search_query, top_k=8 if retrieval_strategy == "hybrid" else 3)
        if semantic_results:
            for r in semantic_results:
                chapter = r.get('chapter', r.get('metadata', {}).get('chapter', ''))
                section = r.get('title', r.get('metadata', {}).get('title', 'Unknown')) # title 欄位通常存小節標題
                full_title = f"{chapter} > {section}" if chapter else section
                
                identifier = full_title # 統一使用完整標題作為去重依據
                if identifier not in seen_identifiers:
                    source_docs.append({
                        "source": identifier,
                        "content": r.get('content', r.get('text', ''))
                    })
                    seen_identifiers.add(identifier)
    except Exception as e:
        print(f"語意搜尋失敗: {e}")
    
    # [更新] 不再進行主動過濾，而是將所有檢索到的資訊提供給模型進行判斷
    # 這是為了確保如 Appendix 等不包含特定瑕疵關鍵字的通用資訊也能被保留
    final_docs = source_docs
            
    # 3. 構建上下文與 Prompt
    if final_docs:
        knowledge_context = "\n\n📚 可用知識庫資訊 (請自行判斷相關性)：\n"
        for i, doc in enumerate(final_docs, 1):
            knowledge_context += f"[{i}] 【{doc['source']}】\n{doc['content']}\n\n"
    else:
        knowledge_context = "\n（知識庫中未找到直接相關資訊，請基於半導體製程通識謹慎回答，並告知使用者資料不足。）"
    
    # 根據診斷上下文生成針對性的系統提示
    current_defect_info = ""
    if diagnosis_context:
        current_defect_info = f"\n### 🔍 目前診斷上下文\n**當前討論的瑕疵類型**: {diagnosis_context}\n請優先聚焦於與此瑕疵相關的資訊。\n"

    system_msg = f"""你是半導體製程專家。

### 任務
回答工程師的追問。
{current_defect_info}
### 知識庫資訊
以下提供檢索到的相關文件，其中可能包含多個不同瑕疵或通用的資訊。
**請發揮專家判斷力，僅篩選與使用者問題「真正相關」的資訊進行回答。**
若資訊與問題無關（例如問 A 瑕疵但提供了 B 瑕疵的資料），請直接忽略該部分。

{knowledge_context}

### 🧪 數位孿生 (Digital Twin) 模擬規則 (重要)
使用者可能會要求你「模擬參數」或「預測良率」。請嚴格遵守以下規則，**絕不可編造模擬數據**：

1. **可模擬類型** (Center, Donut, Random)：
   - 請使用 `simulate_defect_solution` 工具進行真實模擬。
   - **報告結果時的強制格式**：
     必須先列出**「測試參數組合」**（如：ESC 溫度=60°C, 氦氣壓力=10 Torr），再列出工具回傳的「預測良率」與「評估結果」。**切勿只回報良率而忽略參數。**

2. **不可模擬類型** (Scratch, Edge-Ring, Edge-Loc, Loc, Near-full)：
   - **處置原則**：這些瑕疵通常源於硬體故障 (如 Robot 刮傷)、耗材老化 (如 Edge Ring 磨損) 或清潔問題，**不涉及參數調整**，因此無法進行物理模擬。
   - **回應方式**：若使用者要求模擬這些瑕疵，請**明確拒絕**，並說明原因。
     - 錯誤範例："模擬結果顯示良率 95%..." (❌ 絕對禁止 hallucination)
     - 正確範例："Edge-Loc 屬於硬體與機械相關瑕疵，無法透過數位孿生進行參數模擬。建議檢查 Notch Finder 或 Robot 傳送路徑。" (✅ 正確)

### ⚠️ 極重要規則
1. **引用來源**：
   - 所有的回答內容若涉及具體知識、步驟或參數，**必須**在該句結尾標注來源編號。**即使是檢查清單的每一細項，若有依據也須標註。**
   - 格式：...建議調整溫度 [1]。
   - 若某段回答完全基於你的通識而非知識庫，請勿標注編號。
2. **禁止幻覺 (時間、數值與成因)**：
   - **時間/數值**：若知識庫中沒有「預計時間」或「具體數值」，**絕對不要自己預估**。請誠實說「知識庫中未提及具體時間」。
   - **其他成因**：若使用者詢問「還有沒有其他原因？」，**僅能**從知識庫中尋找尚未提及的資訊。若知識庫已無更多資訊，請回答「根據目前知識庫，已列出所有已知成因」，**嚴禁**使用外部知識編造未經證實的真因。
   - **瑕疵類型錯置**：請小心區分不同章節。例如：不要將第五章 Edge-Loc 的成因（如 Robot 碰撞）套用到第七章 Loc 瑕疵上，除非知識庫明確指出兩者相關。
3. **參考文獻區塊規則 (絕對執行)**：
   - 只有當你在回答的內文中確實寫出了 `[1]` 或 `[2]` 等引用標記時，才允許在文末建立「📚 參考文獻」區塊。
   - **若你的回答中完全沒有引用任何編號（例如僅是拒絕模擬或一般性閒聊），嚴禁顯示參考文獻區塊。**
   - 請自我檢查：內文無 `[x]` → 文末無參考文獻。

用繁體中文回答。"""
    
    # 建立具備「模擬工具」的 Agent
    # 這樣當使用者要求模擬時，LLM 才能真正執行 simulate_defect_solution
    tools = [simulate_defect_solution]
    agent = create_react_agent(llm, tools, prompt=system_msg)
    
    # 構建輸入訊息 (排除 System Message，因為已包含在 agent prompt 中)
    input_messages = []
    
    # 只取最近 4 輪對話
    recent_history = chat_history[-8:] if len(chat_history) > 8 else chat_history
    for msg in recent_history:
        input_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    input_messages.append({
        "role": "user",
        "content": user_message
    })
    
    # 執行 Agent
    result = agent.invoke({"messages": input_messages})
    content = _extract_text_content(result["messages"][-1].content)

    # 後處理：若內文沒有引用任何文獻 [x]，則移除參考文獻區塊
    # 這是為了防止 LLM 在拒絕模擬或閒聊時仍習慣性加上參考文獻
    import re
    if "📚 參考文獻" in content:
        split_token = "📚 參考文獻"
    elif "參考文獻" in content:
        split_token = "參考文獻"
    else:
        split_token = None
        
    if split_token:
        # 分割內文與參考文獻
        parts = content.split(split_token)
        body = parts[0]
        
        # 檢查內文是否有 [數值] 引用
        # 排除可能是標題的 [1], [2] (通常標題不會這樣寫，但保險起見)
        has_citation = bool(re.search(r'\[\d+\]', body))
        
        if not has_citation:
            # 若內文無引用，直接捨棄參考文獻部分
            content = body.strip()

    return content
