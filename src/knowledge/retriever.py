"""
知識庫檢索模組
提供簡易查詢與向量檢索兩種模式
"""

import json
import os
from typing import Dict, Any, List

# 嘗試匯入向量知識庫
try:
    from src.knowledge.vector_retriever import VectorKnowledgeBase
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False


class KnowledgeBase:
    """
    整合型知識庫 - 結合簡單查詢與向量檢索
    
    Features:
        1. 簡易查詢：直接透過瑕疵類別查詢對應建議
        2. 向量檢索：使用語意相似度搜尋更細緻的知識
        3. 結構化輸出：提供參數建議與來源引用
    """
    
    def __init__(self, json_path: str = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 載入簡易對照表
        simple_path = os.path.join(self.base_dir, "solutions.json")
        try:
            with open(simple_path, 'r', encoding='utf-8') as f:
                self.simple_solutions = json.load(f)
        except FileNotFoundError:
            self.simple_solutions = {}
            print("⚠️ 找不到簡易知識庫 JSON 檔")
        
        # 初始化向量知識庫
        self.vector_kb = None
        self.use_vector = False
        
        if VECTOR_AVAILABLE:
            try:
                self.vector_kb = VectorKnowledgeBase()
                self.use_vector = True
                print("✅ 向量知識庫已啟用")
            except Exception as e:
                print(f"⚠️ 向量知識庫初始化失敗: {e}，使用簡易模式")
        else:
            print("ℹ️ 向量檢索依賴未安裝，使用簡易模式")
    
    def get_solution(self, defect_label: str) -> str:
        """
        取得瑕疵的維修建議（簡易版本，向後相容）
        
        Args:
            defect_label: 瑕疵類別標籤
            
        Returns:
            維修建議文字
        """
        return self.simple_solutions.get(
            defect_label, 
            "資料庫中無此瑕疵類別的特定維修建議，建議人工排查。"
        )
    
    def get_detailed_solution(self, defect_label: str) -> Dict[str, Any]:
        """
        取得詳細的診斷建議（使用向量檢索）
        
        Args:
            defect_label: 瑕疵類別標籤
            
        Returns:
            包含以下資訊的字典：
            - defect_type: 瑕疵類型
            - sections: 相關章節列表
            - all_parameters: 建議參數
            - source_references: 來源引用
        """
        if self.use_vector and self.vector_kb:
            return self.vector_kb.get_solution_by_defect(defect_label)
        else:
            # 降級到簡易模式
            simple = self.get_solution(defect_label)
            return {
                "defect_type": defect_label,
                "found": True,
                "sections": [{
                    "title": "基本建議",
                    "content": simple,
                    "keywords": [],
                    "relevance_score": 1.0
                }],
                "all_parameters": {},
                "priority_level": "NORMAL",
                "source_references": ["solutions.json (簡易模式)"]
            }
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        語意搜尋知識庫
        
        Args:
            query: 搜尋查詢
            top_k: 返回結果數量
            
        Returns:
            相關文件列表
        """
        if self.use_vector and self.vector_kb:
            return self.vector_kb.search(query, top_k=top_k)
        else:
            return []
    
    def format_advice_for_llm(self, defect_label: str) -> str:
        """
        格式化建議以供 LLM 使用
        
        Args:
            defect_label: 瑕疵類別
            
        Returns:
            格式化的建議文字（包含來源引用）
        """
        result = self.get_detailed_solution(defect_label)
        
        if not result.get("found", False):
            return "知識庫中無此瑕疵類別的相關資訊，建議人工排查。"
        
        output_parts = []
        
        # 優先級警告
        if result.get("priority_level") in ["CRITICAL", "EMERGENCY"]:
            output_parts.append(f"⚠️ **警告等級: {result['priority_level']}** - 需立即處理！\n")
        
        # 各章節內容
        for i, section in enumerate(result.get("sections", []), 1):
            output_parts.append(f"### 參考資料 {i}: {section['title']}")
            output_parts.append(section['content'])
            if section.get('keywords'):
                output_parts.append(f"*關鍵字: {', '.join(section['keywords'])}*")
            output_parts.append("")
        
        # 參數建議
        params = result.get("all_parameters", {})
        if params:
            output_parts.append("### 建議參數範圍")
            for param_name, param_value in params.items():
                if isinstance(param_value, dict):
                    param_str = ", ".join(f"{k}: {v}" for k, v in param_value.items())
                    output_parts.append(f"- **{param_name}**: {param_str}")
                else:
                    output_parts.append(f"- **{param_name}**: {param_value}")
            output_parts.append("")
        
        # 來源引用
        refs = result.get("source_references", [])
        if refs:
            output_parts.append("### 📚 資料來源")
            for ref in refs:
                output_parts.append(f"- {ref}")
        
        return "\n".join(output_parts)
