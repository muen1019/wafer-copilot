"""
Agent 整合測試腳本

功能：
1. 測試完整的視覺辨識 + RAG + LLM 流程
2. 模擬使用者上傳圖片並獲得診斷報告
3. 評估 Agent 回應品質

輸出：
- 完整診斷報告範例
- 處理時間統計
- 各元件運作狀態

使用方式：
    cd wafer_copilot
    python -m tests.test_agent_integration

注意：需要設定 GROQ_API_KEY 環境變數
"""

import os
import sys
import json
import time
from datetime import datetime

# 確保可以找到 src 模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def check_prerequisites():
    """檢查前置條件"""
    print("=" * 70)
    print("🔍 前置條件檢查")
    print("=" * 70)
    
    checks = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "模型檔案": os.path.exists("models/resnet_wm811k.pth"),
        "知識庫": os.path.exists("src/knowledge/wafer_maintenance_manual.json"),
        "測試圖片": os.path.exists("data/sample_images") or os.path.exists("data/test")
    }
    
    all_passed = True
    for name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {name}")
        if not status:
            all_passed = False
    
    return all_passed


def test_tool_directly():
    """直接測試工具函數（不經過 LLM）"""
    print("\n" + "=" * 70)
    print("🔧 直接工具測試（不經過 LLM）")
    print("=" * 70)
    
    from src.agent.tools import analyze_wafer_defect
    
    # 找測試圖片
    test_image = None
    
    # 嘗試 sample_images
    sample_dir = "data/sample_images"
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                test_image = os.path.join(sample_dir, f)
                break
    
    # 嘗試 test 資料夾
    if not test_image:
        test_dir = "data/test"
        for cls in os.listdir(test_dir) if os.path.exists(test_dir) else []:
            cls_dir = os.path.join(test_dir, cls)
            if os.path.isdir(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.endswith(('.png', '.jpg', '.jpeg')):
                        test_image = os.path.join(cls_dir, f)
                        break
            if test_image:
                break
    
    if not test_image:
        print("❌ 找不到測試圖片")
        return None
    
    print(f"\n測試圖片: {test_image}")
    
    # 執行工具
    start_time = time.time()
    result = analyze_wafer_defect.invoke(test_image)
    elapsed = time.time() - start_time
    
    print(f"\n執行時間: {elapsed:.2f} 秒")
    print("\n工具回傳結果:")
    print("-" * 50)
    
    if isinstance(result, dict):
        for key, value in result.items():
            if key == "maintenance_advice":
                print(f"\n{key}:")
                print(str(value)[:500])
                if len(str(value)) > 500:
                    print("...")
            else:
                print(f"{key}: {value}")
    else:
        print(result)
    
    return {
        "image": test_image,
        "result": result if isinstance(result, dict) else {"message": str(result)},
        "elapsed": elapsed
    }


def test_full_agent():
    """測試完整 Agent 流程 (使用 analyze_and_report)"""
    print("\n" + "=" * 70)
    print("🤖 完整 Agent 測試（含 LLM 與 模擬資料整合）")
    print("=" * 70)
    
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("❌ 未設定 API KEY，跳過此測試")
        return None
    
    from src.agent.bot import analyze_and_report
    
    # 找測試圖片
    test_image = None
    sample_dir = "data/sample_images"
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                test_image = os.path.join(sample_dir, f)
                break
    
    if not test_image:
        print("❌ 找不到測試圖片")
        return None
    
    print(f"\n測試圖片: {test_image}")
    print("\n正在呼叫 analyze_and_report...")
    
    try:
        start_time = time.time()
        
        # 使用 analyze_and_report，這會包含 get_mock_context 的模擬數據
        result = analyze_and_report(test_image)
        
        elapsed = time.time() - start_time
        
        output = result.get("report", "No report generated")
        
        print(f"\n執行時間: {elapsed:.2f} 秒")
        print("\n" + "=" * 70)
        print("📋 Agent 診斷報告")
        print("=" * 70)
        print(output)
        
        # 簡單驗證是否包含新生產履歷資訊
        has_log_info = "機台日誌" in output or "異常機台" in output
        llm_failed = "回應失敗" in output or "Connection error" in output
        print(f"\n🔍 檢查模擬數據整合: {'✅ 成功' if has_log_info else '⚠️ 未發現機台日誌資訊'}")
        if llm_failed:
            print("⚠️ LLM 回應失敗，完整 Agent 報告未完成驗證")
        
        return {
            "image": test_image,
            "response": output,
            "elapsed": elapsed,
            "success": not llm_failed,
            "has_log_info": has_log_info
        }
    
    except Exception as e:
        print(f"\n❌ Agent 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "success": False
        }


def run_agent_test():
    """執行完整 Agent 測試"""
    print("=" * 70)
    print("🚀 Agent 整合測試")
    print("=" * 70)
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 前置檢查
    prereq_ok = check_prerequisites()
    
    if not prereq_ok:
        print("\n⚠️ 部分前置條件未滿足，測試可能不完整")
    
    # 工具測試
    tool_result = test_tool_directly()
    
    # Agent 測試
    agent_result = test_full_agent()
    
    # 儲存報告
    output_dir = "tests/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "test_time": datetime.now().isoformat(),
        "prerequisites": {
            "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
            "model_file": os.path.exists("models/resnet_wm811k.pth"),
        },
        "tool_test": tool_result,
        "agent_test": {
            "success": agent_result.get("success") if agent_result else False,
            "elapsed": agent_result.get("elapsed") if agent_result else None
        } if agent_result else None
    }
    
    report_path = os.path.join(output_dir, "agent_integration_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    # 總結
    print("\n" + "=" * 70)
    print("📊 測試總結")
    print("=" * 70)
    print(f"   工具測試: {'✅ 成功' if tool_result else '❌ 失敗'}")
    print(f"   Agent 測試: {'✅ 成功' if agent_result and agent_result.get('success') else '❌ 失敗/跳過'}")
    print(f"\n💾 報告已儲存至: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_agent_test()
