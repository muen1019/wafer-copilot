"""
Wafer Copilot - 半導體 AI 智慧助理
支援多輪對話的 Streamlit Web UI
支援 Groq / Gemini 雙 LLM 模式
"""

import streamlit as st
import os
from PIL import Image
from src.agent.bot import (
    analyze_and_report, 
    invoke_followup, 
    set_llm_provider, 
    get_llm_provider,
    get_llm_info
)

# ============================================================
# 頁面設定
# ============================================================
st.set_page_config(
    page_title="半導體 AI 智慧助理",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# Session State 初始化
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "current_image_path" not in st.session_state:
    st.session_state.current_image_path = None

if "diagnosis_done" not in st.session_state:
    st.session_state.diagnosis_done = False

if "latest_gradcam" not in st.session_state:
    st.session_state.latest_gradcam = None

if "current_defect" not in st.session_state:
    st.session_state.current_defect = None

if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = get_llm_provider()


# ============================================================
# 側邊欄
# ============================================================
with st.sidebar:
    st.title("🔧 控制面板")
    
    # LLM 選擇器
    st.markdown("### 🧠 LLM 模型選擇")
    llm_options = {
        "groq": "🟠 Groq (Llama 3.3 70B) - 免費快速",
        "gemini": "🔷 Gemini 3 Flash Preview - 穩定準確"
    }
    selected_llm = st.selectbox(
        "選擇語言模型",
        options=list(llm_options.keys()),
        format_func=lambda x: llm_options[x],
        index=0 if st.session_state.llm_provider == "groq" else 1,
        key="llm_selector"
    )
    
    # 如果選擇改變，更新 provider
    if selected_llm != st.session_state.llm_provider:
        st.session_state.llm_provider = selected_llm
        set_llm_provider(selected_llm)
        st.success(f"✅ 已切換至 {llm_options[selected_llm].split(' - ')[0]}")
    else:
        # 確保 bot 模組同步
        set_llm_provider(st.session_state.llm_provider)
    
    st.markdown("---")
    
    # 圖片上傳
    st.markdown("### 📤 上傳晶圓圖譜")
    uploaded_file = st.file_uploader(
        "選擇 PNG/JPG 圖片",
        type=["png", "jpg", "jpeg"],
        key="wafer_upload"
    )
    
    # 處理上傳的圖片
    if uploaded_file is not None:
        # 檢查是否為新圖片
        if st.session_state.current_image_path != os.path.join("data/sample_images", uploaded_file.name):
            # 儲存圖片
            temp_dir = "data/sample_images"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            image = Image.open(uploaded_file)
            image.save(file_path)
            
            # 更新狀態
            st.session_state.current_image = image
            st.session_state.current_image_path = file_path
            st.session_state.diagnosis_done = False
            st.session_state.latest_gradcam = None
            st.session_state.current_defect = None
            
            # 清除對話歷史（新圖片）
            st.session_state.messages = []
            
            st.success(f"✅ 已上傳: {uploaded_file.name}")
        
        # 顯示當前圖片
        st.markdown("### 📷 當前分析圖片")
        st.image(st.session_state.current_image, width="stretch")
    
    st.markdown("---")
    
    # 快速操作按鈕
    st.markdown("### ⚡ 快速操作")
    
    if st.button("🗑️ 清除對話", use_container_width=True):
        st.session_state.messages = []
        st.session_state.diagnosis_done = False
        st.session_state.pending_message = None
        st.session_state.current_defect = None
        st.rerun()
    
    if st.button("🔄 重新分析", use_container_width=True, disabled=not st.session_state.current_image_path):
        st.session_state.messages = []
        st.session_state.diagnosis_done = False
        st.rerun()
    
    st.markdown("---")
    
    # 說明
    st.markdown("### 💡 使用說明")
    st.markdown("""
    1. 上傳晶圓圖譜
    2. 點擊「開始診斷」
    3. 查看診斷報告
    4. 在對話框中追問細節
    
    **追問範例：**
    - 請詳細說明 CMP 壓力調整方法
    - ESC 溫度應該設定多少？
    - 這種瑕疵的緊急處理步驟是什麼？
    """)
    
    st.markdown("---")
    st.markdown("### 🔬 Grad-CAM 說明")
    st.markdown("""
    - 🔴 紅色：高度關注區域
    - 🟡 黃色：中度關注區域
    - 🔵 藍色：低關注區域
    """)


# ============================================================
# 主畫面
# ============================================================
st.title("🤖 GenAI 半導體智慧製造助理")

# 顯示目前 LLM 資訊
llm_info = get_llm_info()
st.caption(f"基於 WM-811K + RAG + LLM 的多輪對話診斷系統 | 目前模型: {llm_info['icon']} {llm_info['model']}")

# 分為左右兩欄：左邊是 Grad-CAM，右邊是對話
if st.session_state.current_image_path:
    col_visual, col_chat = st.columns([1, 2])
    
    # ============================================================
    # 左側：視覺化區域
    # ============================================================
    with col_visual:
        st.markdown("### 🔥 Grad-CAM 分析")
        
        if st.session_state.latest_gradcam and os.path.exists(st.session_state.latest_gradcam):
            st.image(st.session_state.latest_gradcam, width="stretch")
            st.caption("模型關注區域視覺化")
        elif st.session_state.diagnosis_done:
            # 嘗試載入最新的 Grad-CAM
            gradcam_dir = "data/gradcam_outputs"
            if os.path.exists(gradcam_dir):
                files = sorted(
                    [f for f in os.listdir(gradcam_dir) if f.startswith("overlay_")],
                    key=lambda x: os.path.getmtime(os.path.join(gradcam_dir, x)),
                    reverse=True
                )
                if files:
                    st.session_state.latest_gradcam = os.path.join(gradcam_dir, files[0])
                    st.image(st.session_state.latest_gradcam, width="stretch")
                    st.caption("模型關注區域視覺化")
        else:
            st.info("點擊「開始診斷」後將顯示熱力圖")
        
        # 顯示原始圖片（縮小版）
        st.markdown("### 📷 原始圖片")
        st.image(st.session_state.current_image, width="stretch")
    
    # ============================================================
    # 右側：對話區域
    # ============================================================
    with col_chat:
        st.markdown("### 💬 AI 診斷對話")
        
        # 對話容器
        chat_container = st.container(height=500)
        
        with chat_container:
            # 顯示歷史訊息
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # ============================================================
        # 處理待處理的追問訊息
        # ============================================================
        if st.session_state.pending_message and st.session_state.diagnosis_done:
            pending = st.session_state.pending_message
            st.session_state.pending_message = None  # 清除待處理訊息
            
            # 加入使用者訊息
            st.session_state.messages.append({
                "role": "user",
                "content": pending
            })
            
            # 追問使用不帶工具的 LLM（更穩定）
            with st.spinner("🤔 思考中..."):
                try:
                    # 使用 invoke_followup 而非 agent
                    response = invoke_followup(
                        user_message=pending,
                        chat_history=st.session_state.messages[:-1],
                        diagnosis_context=st.session_state.current_defect
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"回應過程發生錯誤: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # 初始診斷按鈕（只在尚未診斷時顯示）
        if not st.session_state.diagnosis_done:
            if st.button("🔍 開始 AI 診斷", use_container_width=True, type="primary"):
                # 加入使用者訊息
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"📷 上傳圖片進行分析: {os.path.basename(st.session_state.current_image_path)}"
                })
                
                # 使用穩定版本（不使用 function calling）
                with st.spinner("🔬 正在進行視覺辨識與知識檢索..."):
                    try:
                        result_data = analyze_and_report(st.session_state.current_image_path)
                        response = result_data["report"]
                        st.session_state.current_defect = result_data["defect_type"]
                        
                        # 加入助理回應
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        st.session_state.diagnosis_done = True
                        
                        # 載入 Grad-CAM
                        gradcam_dir = "data/gradcam_outputs"
                        if os.path.exists(gradcam_dir):
                            files = sorted(
                                [f for f in os.listdir(gradcam_dir) if f.startswith("overlay_")],
                                key=lambda x: os.path.getmtime(os.path.join(gradcam_dir, x)),
                                reverse=True
                            )
                            if files:
                                st.session_state.latest_gradcam = os.path.join(gradcam_dir, files[0])
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"診斷過程發生錯誤: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # 追問輸入框
        st.markdown("---")
        
        # 使用 chat_input 進行追問
        if prompt := st.chat_input("繼續詢問（例如：請詳細說明參數調整方法）", disabled=not st.session_state.diagnosis_done):
            st.session_state.pending_message = prompt
            st.rerun()
        
        # 追問建議（始終顯示）
        if st.session_state.diagnosis_done:
            st.markdown("#### 💡 快速追問：")
            
            suggest_col1, suggest_col2 = st.columns(2)
            
            with suggest_col1:
                if st.button("🔧 詳細參數調整步驟", use_container_width=True):
                    st.session_state.pending_message = "請更詳細說明具體的參數調整步驟和數值範圍"
                    st.rerun()
                
                if st.button("⚠️ 緊急處理程序", use_container_width=True):
                    st.session_state.pending_message = "如果這是緊急情況，應該採取什麼緊急處理程序?"
                    st.rerun()
                
                if st.button("🧪 模擬參數測試", use_container_width=True):
                    st.session_state.pending_message = "請在數位孿生系統中測試你建議的參數組合，看看預測良率如何"
                    st.rerun()
            
            with suggest_col2:
                if st.button("🔍 更多成因分析", use_container_width=True):
                    st.session_state.pending_message = "請從其他角度分析可能的成因，有沒有其他潛在因素?"
                    st.rerun()
                
                if st.button("📋 排查檢查清單", use_container_width=True):
                    st.session_state.pending_message = "請提供完整的排查檢查清單，包括每個步驟的預計時間"
                    st.rerun()

else:
    # ============================================================
    # 歡迎畫面
    # ============================================================
    st.info("👈 請從左側上傳一張晶圓圖譜以開始分析。")
    
    st.markdown("---")
    st.markdown("### 🎯 系統特色")
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        #### 🧠 智慧診斷
        結合 CNN 視覺模型與 LLM 推理能力，自動識別 8 種晶圓瑕疵模式
        """)
    
    with feat_col2:
        st.markdown("""
        #### 🔥 Grad-CAM 可解釋性
        視覺化 AI 的決策依據，讓工程師「看見」模型關注的區域
        """)
    
    with feat_col3:
        st.markdown("""
        #### 🧪 Digital Twin 模擬
        在虛擬孿生系統中測試參數調整建議，降低實機風險
        """)
    
    st.markdown("---")
    st.markdown("### 📊 支援的瑕疵類型")
    
    defect_cols = st.columns(4)
    defects = [
        ("Center", "中心型"),
        ("Donut", "甜甜圈型"),
        ("Edge-Loc", "邊緣局部型"),
        ("Edge-Ring", "邊緣環狀型"),
        ("Loc", "局部群聚型"),
        ("Near-full", "近全區型"),
        ("Random", "隨機分佈型"),
        ("Scratch", "刮痕型")
    ]
    
    for i, (eng, chi) in enumerate(defects):
        with defect_cols[i % 4]:
            st.markdown(f"- **{eng}** ({chi})")
    
    st.markdown("---")
    st.markdown("### 🚀 快速開始")
    st.markdown("""
    1. 在左側面板上傳晶圓圖譜（PNG/JPG）
    2. 點擊「開始 AI 診斷」按鈕
    3. 查看診斷報告與 Grad-CAM 熱力圖
    4. 在對話框中追問任何問題
    """)
