"""
檢索系統效能評估腳本
測試 "Naive Vector Search" vs "Hybrid Retrieval"
"""

import sys
import os
# Add project root to path
sys.path.append(os.getcwd())

from src.knowledge.vector_retriever import VectorKnowledgeBase
import pandas as pd

def evaluate_retrieval():
    kb = VectorKnowledgeBase()
    # 強制載入 (如果尚未初始化)
    if not kb._documents:
        kb._build_index()

    test_cases = [
        {"query": "Donut 瑕疵特徵", "expected_chapter_id": "CH03", "defect_type": "Donut", "desc": "Standard Donut"},
        {"query": "Loc 區域群聚", "expected_chapter_id": "CH07", "defect_type": "Loc", "desc": "Confusing Term (Loc vs Edge-Loc)"},
        {"query": "Edge-Loc 邊緣局部", "expected_chapter_id": "CH05", "defect_type": "Edge-Loc", "desc": "Standard Edge-Loc"},
        {"query": "Scratch 刮痕處置", "expected_chapter_id": "CH06", "defect_type": "Scratch", "desc": "Standard Scratch"},
        {"query": "CMP 參數調整", "expected_chapter_id": "CH02", "defect_type": "Center", "desc": "Concept Query (CMP -> Center)"}
    ]
    
    results = []
    
    print(f"\nEvaluating Retrieval Strategies on {len(test_cases)} test cases...\n")
    
    for case in test_cases:
        defect = case["defect_type"]
        
        # 1. 模擬 Naive Vector Search (關閉 Keyword Mapping 邏輯)
        # 這裡我們直接呼叫 search() 而不使用 get_solution_by_defect() 的特別邏輯
        # 並且刻意不傳入 defect_type 讓 search() 純粹依賴查詢字串
        naive_hits = kb.search(case["query"], top_k=1)
        naive_result = "Fail"
        naive_doc = "None"
        if naive_hits:
            # 檢查 ID 是否包含預期的 Chapter ID (例如 CH03)
            doc_id = naive_hits[0].get("id", "") # e.g., CH03-S01
            naive_doc = naive_hits[0].get("chapter", "") + " - " + naive_hits[0].get("title", "")
            if case["expected_chapter_id"] in doc_id:
                naive_result = "Pass"
        
        # 2. 模擬 Hybrid Retrieval (使用 get_solution_by_defect)
        hybrid_solution = kb.get_solution_by_defect(defect, top_k=1)
        hybrid_result = "Fail"
        hybrid_doc = "None"
        if hybrid_solution["found"]:
            # 取第一個 section 檢查
            sections = hybrid_solution.get("sections", [])
            if sections:
                # 這裡 get_solution_by_defect 回傳的結構稍微不同，通常是整章
                # 我們假設 get_solution_by_defect 的 "確定性匹配" 會抓到正確章節
                # 由於它是依賴 defect_type 參數，只要 defect_type 對了，通常就會抓對
                # 在此我們檢查回傳內容是否包含正確章節標題
                
                # 不過，get_solution_by_defect 回傳的是 sections list
                # 我們檢查第一個 section
                first_section = sections[0]
                # 這裡沒有 ID 欄位，只能檢查 content 或 title
                # 但我們的 vector_retriever.py 裡面的 get_solution_by_defect 
                # 是比對 defect_type 是否在 chapter title 裡
                
                # 重新檢查 vector_retriever logic...
                # 它比對 if defect_type.lower() in doc['chapter'].lower()
                # 這樣對於 Loc 來說，如果它排除 Edge-Loc，就會正確抓到 CH07
                # 如果沒排除，可能會抓到 CH05 (如果 Edge-Loc 出現在更前面)
                
                match_chapter = first_section.get("chapter", "")
                hybrid_doc = match_chapter
                
                # 簡單驗證：檢查 expected ID 對應的中文是否在 chapter title
                # CH03 -> Donut, CH07 -> Loc
                if defect.lower() in match_chapter.lower():
                     # Double check for Loc vs Edge-Loc
                    if defect.lower() == "loc" and "edge-loc" in match_chapter.lower():
                        hybrid_result = "Fail (Confused)"
                    else:
                        hybrid_result = "Pass"
                else:
                    hybrid_result = "Fail"
        
        results.append({
            "Query Type": case["desc"],
            "Query": case["query"],
            "Naive Result": naive_result,
            "Hybrid Result": hybrid_result,
            "Naive Top-1 Doc": naive_doc[:30] + "...",
            "Hybrid Doc": hybrid_doc[:30] + "..."
        })

    # 輸出表格
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    print("\n\n[Markdown Table for Report]")
    print("| Query Defect Type | Naive Vector Top-1 Hit | Hybrid Retrieval Top-1 Hit | 備註 |")
    print("| :--- | :---: | :---: | :--- |")
    for _, row in df.iterrows():
        defect = row["Query"].split()[0] # 簡單抓取
        naive_icon = "✅ Correct" if row["Naive Result"] == "Pass" else f"❌ {row['Naive Result']}"
        hybrid_icon = "✅ Correct" if "Pass" in row["Hybrid Result"] else f"❌ {row['Hybrid Result']}"
        
        # 特別處理 Loc 的顯示 (為了報告效果)
        if "Loc" == defect and "Edge-Loc" in row.get("Naive Top-1 Doc", ""):
            naive_icon = "❌ Edge-Loc (Ch.5)"
        
        print(f"| {defect} | {naive_icon} | {hybrid_icon} | {row['Query Type']} |")

if __name__ == "__main__":
    evaluate_retrieval()
