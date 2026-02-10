"""
多輪對話功能測試

此腳本測試 Agent 的多輪對話能力
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.bot import get_agent, invoke_agent_with_history, analyze_and_report

def test_multi_turn_conversation():
    """測試多輪對話"""
    print("=" * 60)
    print("🧪 測試多輪對話功能")
    print("=" * 60)
    
    # 初始化 Agent
    print("\n⏳ 初始化 Agent...")
    agent = get_agent()
    print("✅ Agent 已初始化")
    
    # 模擬對話歷史
    chat_history = []
    
    # 第一輪：分析圖片 (使用 analyze_and_report 確保包含模擬數據)
    print("\n" + "-" * 40)
    print("📝 第一輪：分析晶圓圖片 (模擬 App 流程)")
    print("-" * 40)
    
    image_path = "data/sample_images/00a018e04.png"
    if not os.path.exists(image_path):
        # 嘗試其他路徑
        for alt_path in ["data/test/Edge-Ring_1.png", "data/test/Center_1.png"]:
            if os.path.exists(alt_path):
                image_path = alt_path
                break
    
    first_query = f"請分析這張晶圓圖片: {image_path}"
    print(f"👤 使用者: {first_query}")
    
    # 這裡模擬 App 行為：先呼叫 analyze_and_report
    # 這會觸發 視覺辨識 + 生產履歷模擬 + 知識庫檢索 + 報告生成
    analysis_result = analyze_and_report(image_path)
    report_text = analysis_result["report"]
    
    print(f"\n🤖 助理 (Diagnosis Report):\n{report_text[:500]}...")  # 只顯示前 500 字
    
    # 更新對話歷史
    chat_history.append({"role": "user", "content": first_query})
    chat_history.append({"role": "assistant", "content": report_text}) # 將完整報告放入歷史
    
    # 第二輪：追問參數細節
    print("\n" + "-" * 40)
    print("📝 第二輪：追問參數細節")
    print("-" * 40)
    
    # 若上一輪的報告已包含 log 資訊，接下來的追問應該能參考到
    second_query = "根據剛剛的機台日誌，請問我們應該優先檢查哪個模組？"
    print(f"👤 使用者: {second_query}")
    
    response2 = invoke_agent_with_history(agent, second_query, chat_history)
    print(f"\n🤖 助理:\n{response2[:500]}...")
    
    # 更新對話歷史
    chat_history.append({"role": "user", "content": second_query})
    chat_history.append({"role": "assistant", "content": response2})
    
    # 第三輪：詢問緊急處理
    print("\n" + "-" * 40)
    print("📝 第三輪：詢問緊急處理程序")
    print("-" * 40)
    
    third_query = "如果這是緊急情況，應該優先做什麼？"
    print(f"👤 使用者: {third_query}")
    
    response3 = invoke_agent_with_history(agent, third_query, chat_history)
    print(f"\n🤖 助理:\n{response3[:500]}...")
    
    # 結果摘要
    print("\n" + "=" * 60)
    print("📊 測試結果摘要")
    print("=" * 60)
    print(f"✅ 對話輪數: {len(chat_history) // 2 + 1} 輪")
    print(f"✅ 總對話歷史長度: {len(chat_history)} 則訊息")
    print(f"✅ 第一輪回應長度: {len(response1)} 字")
    print(f"✅ 第二輪回應長度: {len(response2)} 字")
    print(f"✅ 第三輪回應長度: {len(response3)} 字")
    print("=" * 60)


def test_knowledge_followup():
    """測試知識庫追問"""
    print("\n" + "=" * 60)
    print("🧪 測試知識庫追問功能")
    print("=" * 60)
    
    agent = get_agent()
    chat_history = []
    
    # 直接詢問維修知識
    query1 = "請問 Edge-Ring 瑕疵的主要成因有哪些？"
    print(f"\n👤 使用者: {query1}")
    
    response1 = invoke_agent_with_history(agent, query1, chat_history)
    print(f"\n🤖 助理:\n{response1[:400]}...")
    
    chat_history.append({"role": "user", "content": query1})
    chat_history.append({"role": "assistant", "content": response1})
    
    # 追問
    query2 = "ESC 溫度應該設定多少？"
    print(f"\n👤 使用者: {query2}")
    
    response2 = invoke_agent_with_history(agent, query2, chat_history)
    print(f"\n🤖 助理:\n{response2}")
    
    print("\n✅ 知識庫追問測試完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多輪對話測試")
    parser.add_argument("--full", action="store_true", help="執行完整對話測試")
    parser.add_argument("--knowledge", action="store_true", help="測試知識庫追問")
    args = parser.parse_args()
    
    if args.full:
        test_multi_turn_conversation()
    elif args.knowledge:
        test_knowledge_followup()
    else:
        print("🧪 快速測試：invoke_agent_with_history 函式")
        print("-" * 40)
        
        agent = get_agent()
        print("✅ Agent 初始化成功")
        
        # 簡單測試
        response = invoke_agent_with_history(
            agent,
            "請問 Donut 瑕疵通常是什麼原因造成的？",
            []
        )
        print(f"✅ Agent 回應: {response[:200]}...")
        print("\n💡 執行 --full 進行完整多輪對話測試")
        print("💡 執行 --knowledge 測試知識庫追問功能")
