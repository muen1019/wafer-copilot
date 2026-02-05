from langchain.tools import tool
from src.vision.classifier import WaferClassifier
from src.knowledge.retriever import KnowledgeBase
from src.digital_twin.simulator import DigitalTwinFactory
from typing import Optional

# 初始化單例模式
classifier = WaferClassifier()
kb = KnowledgeBase()
digital_twin_factory = DigitalTwinFactory()

@tool
def analyze_wafer_defect(image_path: str) -> dict:
    """
    分析晶圓圖片，生成瑕疵診斷報告與 Grad-CAM 熱力圖。
    
    Args:
        image_path: 圖片的檔案路徑
        
    Returns:
        包含瑕疵類別、信心度、熱力圖路徑及維修建議的字典
    """
    # 1. 視覺辨識 (含 Grad-CAM)
    result = classifier.predict(image_path, generate_cam=True)
    if "error" in result:
        return f"視覺模型分析錯誤: {result['error']}"
    
    label = result['label']
    
    # 2. 知識檢索 (RAG) - 使用詳細版本
    detailed_advice = kb.get_detailed_solution(label)
    formatted_advice = kb.format_advice_for_llm(label)
    
    return {
        "defect_type": label,
        "confidence": result['confidence'],
        "gradcam_heatmap": result.get('cam_path'),
        "gradcam_overlay": result.get('cam_overlay_path'),
        "comparison_image": result.get('comparison_path'),
        "maintenance_advice": formatted_advice,
        "priority_level": detailed_advice.get("priority_level", "NORMAL"),
        "recommended_parameters": detailed_advice.get("all_parameters", {}),
        "source_references": detailed_advice.get("source_references", [])
    }


@tool
def search_maintenance_knowledge(query: str) -> str:
    """
    在維修知識庫中搜尋相關資訊。
    
    Args:
        query: 搜尋查詢，例如 "CMP 壓力調整" 或 "ESC 溫度設定"
        
    Returns:
        相關的維修知識與建議
    """
    results = kb.search_knowledge(query, top_k=6)
    
    if not results:
        return "知識庫中找不到相關資訊，請嘗試其他關鍵字或聯繫資深工程師。"
    
    output_parts = [f"找到 {len(results)} 筆相關資料：\n"]
    
    for i, doc in enumerate(results, 1):
        output_parts.append(f"### 結果 {i}: {doc['title']}")
        output_parts.append(f"*章節: {doc['chapter']}*")
        output_parts.append(doc['content'])
        if doc.get('keywords'):
            output_parts.append(f"*關鍵字: {', '.join(doc['keywords'])}*")
        output_parts.append(f"*相關度: {doc['relevance_score']:.2%}*")
        output_parts.append("")
    
    return "\n".join(output_parts)


@tool
def simulate_defect_solution(defect_type: str, parameters: dict) -> dict:
    """
    在數位孿生 (Digital Twin) 系統中模擬測試瑕疵解決方案的參數調整建議。
    系統會根據瑕疵類型自動選擇對應的製程模擬器 (CMP, Etch, Environmental 等)。
    
    Args:
        defect_type: 瑕疵類型 (Center, Donut, Random, Scratch, Edge-Ring, Edge-Loc, Loc, Near-full)
        parameters: 製程參數字典，根據瑕疵類型而異
            - Center 瑕疵: {"polishing_pressure_center": float, "slurry_flow_center": float, "pad_lifetime_hours": int}
            - Donut 瑕疵: {"esc_temp_inner": float, "esc_temp_outer": float, "he_pressure_inner": float, "he_pressure_outer": float}
            - Random 瑕疵: {"cleanroom_particle_count": float, "gas_filter_age_months": float, "differential_pressure": float}
        
    Returns:
        dict: 包含模擬結果的完整報告
            - status: 模擬狀態 (success/not_available)
            - process: 製程名稱
            - defect_type: 瑕疵類型
            - inputs: 輸入的參數值
            - results: 預測良率、瑕疵數、製程指標
            - feedback: 專家級評估與風險警告
            - unavailable_reason: (若無法模擬) 說明原因
    
    Example:
        >>> simulate_defect_solution("Donut", {"esc_temp_inner": 62, "esc_temp_outer": 67, "he_pressure_inner": 10.5, "he_pressure_outer": 14.5})
        {
            "status": "success",
            "process": "Etch (Plasma Etching)",
            "defect_type": "Donut",
            "results": {"predicted_yield": "96.2%", ...}
        }
    """
    # 檢查該瑕疵類型是否可模擬
    if not digital_twin_factory.is_simulatable(defect_type):
        return {
            "status": "not_available",
            "defect_type": defect_type,
            "message": f"❌ {defect_type} 瑕疵目前無法進行數位孿生模擬",
            "unavailable_reason": digital_twin_factory.get_unavailable_reason(defect_type),
            "supported_defects": digital_twin_factory.get_supported_defects()
        }
    
    # 獲取對應的模擬器
    simulator = digital_twin_factory.get_simulator(defect_type)
    
    # 執行模擬
    try:
        result = simulator.run_simulation(parameters)
        return result
    except Exception as e:
        return {
            "status": "error",
            "defect_type": defect_type,
            "message": f"模擬過程發生錯誤: {str(e)}",
            "inputs": parameters
        }


