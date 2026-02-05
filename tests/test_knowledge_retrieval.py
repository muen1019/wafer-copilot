"""
RAG 知識庫檢索測試腳本

功能：
1. 測試向量知識庫的語意檢索能力
2. 展示各類瑕疵的詳細診斷建議
3. 驗證參數建議與來源引用功能

輸出：
- 檢索準確度評估
- 各類別查詢結果範例
- 知識庫覆蓋度統計

使用方式：
    cd wafer_copilot
    python -m tests.test_knowledge_retrieval
"""

import os
import sys
import json
from datetime import datetime

# 確保可以找到 src 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge.retriever import KnowledgeBase
from src.vision.classifier import CLASSES


def test_defect_queries():
    """測試各類瑕疵的查詢"""
    print("\n" + "=" * 70)
    print("📋 各類瑕疵查詢測試")
    print("=" * 70)
    
    kb = KnowledgeBase()
    results = {}
    
    for defect in CLASSES:
        print(f"\n{'─' * 70}")
        print(f"🔍 瑕疵類型: {defect}")
        print("─" * 70)
        
        # 詳細查詢
        detailed = kb.get_detailed_solution(defect)
        
        print(f"   找到相關資料: {'✅ 是' if detailed.get('found') else '❌ 否'}")
        print(f"   優先級別: {detailed.get('priority_level', 'N/A')}")
        print(f"   相關章節數: {len(detailed.get('sections', []))}")
        
        if detailed.get('sections'):
            print("\n   📚 相關章節:")
            for i, section in enumerate(detailed['sections'][:3], 1):
                score = section.get('relevance_score', 0)
                print(f"      {i}. {section['title']}")
                print(f"         相關度: {score:.1%}")
                print(f"         內容預覽: {section['content'][:80]}...")
        
        if detailed.get('all_parameters'):
            print("\n   🔧 建議參數:")
            for param, value in list(detailed['all_parameters'].items())[:3]:
                print(f"      - {param}: {value}")
        
        if detailed.get('source_references'):
            print("\n   📖 資料來源:")
            for ref in detailed['source_references'][:2]:
                print(f"      - {ref}")
        
        results[defect] = {
            "found": detailed.get('found', False),
            "priority": detailed.get('priority_level'),
            "sections_count": len(detailed.get('sections', [])),
            "parameters_count": len(detailed.get('all_parameters', {}))
        }
    
    return results


def test_semantic_search():
    """測試語意搜尋功能"""
    print("\n" + "=" * 70)
    print("🔎 語意搜尋測試")
    print("=" * 70)
    
    kb = KnowledgeBase()
    
    # 測試查詢（模擬工程師可能會問的問題）
    test_queries = [
        ("CMP 研磨頭壓力要怎麼調整？", ["CMP", "壓力", "Center"]),
        ("ESC 溫度異常導致的問題", ["ESC", "溫度", "Donut"]),
        ("刮痕瑕疵的緊急處理步驟", ["刮痕", "Scratch", "緊急"]),
        ("邊緣瑕疵可能是什麼原因造成的？", ["Edge", "邊緣"]),
        ("如何減少隨機微粒汙染？", ["Random", "微粒", "汙染"]),
        ("Focus Ring 磨損會造成什麼影響？", ["Focus Ring", "Edge-Ring"]),
        ("近全區瑕疵需要立即處理嗎？", ["Near-full", "緊急"]),
        ("光阻塗佈氣泡問題", ["Loc", "氣泡", "光阻"])
    ]
    
    search_results = []
    
    for query, expected_keywords in test_queries:
        print(f"\n{'─' * 70}")
        print(f"❓ 查詢: {query}")
        print("─" * 70)
        
        results = kb.search_knowledge(query, top_k=3)
        
        if results:
            hit_keywords = []
            
            for i, doc in enumerate(results, 1):
                print(f"\n   結果 {i}: {doc['title']}")
                print(f"   相關度: {doc['relevance_score']:.1%}")
                print(f"   關鍵字: {', '.join(doc.get('keywords', []))}")
                
                # 檢查是否命中預期關鍵字
                for kw in expected_keywords:
                    if kw.lower() in doc['title'].lower() or kw.lower() in doc['content'].lower():
                        hit_keywords.append(kw)
            
            hit_keywords = list(set(hit_keywords))
            hit_rate = len(hit_keywords) / len(expected_keywords) if expected_keywords else 0
            
            print(f"\n   ✅ 命中關鍵字: {', '.join(hit_keywords)} ({hit_rate:.0%})")
            
            search_results.append({
                "query": query,
                "results_count": len(results),
                "hit_keywords": hit_keywords,
                "hit_rate": hit_rate
            })
        else:
            print("   ⚠️ 無搜尋結果（向量功能可能未啟用）")
            search_results.append({
                "query": query,
                "results_count": 0,
                "hit_keywords": [],
                "hit_rate": 0
            })
    
    # 統計搜尋效能
    if search_results:
        avg_hit_rate = sum(r["hit_rate"] for r in search_results) / len(search_results)
        print(f"\n📊 搜尋效能統計: 平均關鍵字命中率 {avg_hit_rate:.1%}")
    
    return search_results


def test_formatted_output():
    """測試 LLM 格式化輸出"""
    print("\n" + "=" * 70)
    print("📝 LLM 格式化輸出測試")
    print("=" * 70)
    
    kb = KnowledgeBase()
    
    # 選擇幾個代表性類別
    test_defects = ["Scratch", "Donut", "Random"]
    
    for defect in test_defects:
        print(f"\n{'─' * 70}")
        print(f"瑕疵類型: {defect}")
        print("─" * 70)
        
        formatted = kb.format_advice_for_llm(defect)
        
        # 顯示前 500 字
        print(formatted[:500])
        if len(formatted) > 500:
            print(f"\n... (共 {len(formatted)} 字)")


def run_knowledge_test():
    """執行完整知識庫測試"""
    print("=" * 70)
    print("📚 RAG 知識庫檢索功能測試")
    print("=" * 70)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化知識庫
    print("\n📦 初始化知識庫...")
    kb = KnowledgeBase()
    print(f"   向量檢索: {'✅ 啟用' if kb.use_vector else '❌ 未啟用'}")
    
    # 執行各項測試
    defect_results = test_defect_queries()
    search_results = test_semantic_search()
    test_formatted_output()
    
    # 儲存報告
    output_dir = "tests/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "test_time": datetime.now().isoformat(),
        "vector_enabled": kb.use_vector,
        "defect_queries": defect_results,
        "semantic_search": search_results
    }
    
    report_path = os.path.join(output_dir, "knowledge_retrieval_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 70)
    print("📊 測試總結")
    print("=" * 70)
    print(f"   瑕疵類別覆蓋: {sum(1 for d in defect_results.values() if d['found'])}/{len(CLASSES)}")
    
    if search_results:
        success_searches = sum(1 for r in search_results if r['results_count'] > 0)
        print(f"   語意搜尋成功: {success_searches}/{len(search_results)}")
    
    print(f"\n💾 報告已儲存至: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_knowledge_test()
