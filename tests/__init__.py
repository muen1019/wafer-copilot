"""
Wafer Copilot 測試套件

此資料夾包含所有測試腳本，用於驗證各元件功能並生成展示用資料。

測試腳本列表：
- test_classifier.py: 視覺辨識模型測試（分類準確度、Top-K 預測）
- test_gradcam.py: Grad-CAM 熱力圖生成與展示
- test_knowledge_retrieval.py: RAG 知識庫檢索功能測試
- test_agent_integration.py: 完整 Agent 流程整合測試
- demo_full_pipeline.py: 完整流程展示腳本（適合文稿撰寫）

執行方式：
    cd wafer_copilot
    python -m tests.test_classifier
    python -m tests.test_gradcam
    python -m tests.demo_full_pipeline
"""
