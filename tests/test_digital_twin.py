"""
測試 Digital Twin 模組化架構
驗證不同瑕疵類型的模擬能力
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.digital_twin.simulator import DigitalTwinFactory
import json

def test_all_defects():
    """測試所有瑕疵類型的模擬能力"""
    factory = DigitalTwinFactory()
    
    all_defects = ["Center", "Donut", "Random", "Scratch", "Edge-Ring", "Edge-Loc", "Loc", "Near-full"]
    
    print("=" * 80)
    print("數位孿生系統 - 瑕疵類型模擬能力測試")
    print("=" * 80)
    
    for defect in all_defects:
        print(f"\n📌 瑕疵類型: {defect}")
        print("-" * 80)
        
        if factory.is_simulatable(defect):
            print(f"✅ 可模擬 - 使用 {factory.get_simulator(defect).process_name} 模擬器")
            
            # 根據類型執行示範模擬
            simulator = factory.get_simulator(defect)
            
            if defect == "Center":
                params = {
                    "polishing_pressure_center": 3.0,
                    "slurry_flow_center": 175,
                    "pad_lifetime_hours": 250
                }
            elif defect == "Donut":
                params = {
                    "esc_temp_inner": 60,
                    "esc_temp_outer": 65,
                    "he_pressure_inner": 10,
                    "he_pressure_outer": 15,
                    "rf_inner_outer_ratio": 1.0
                }
            elif defect == "Random":
                params = {
                    "cleanroom_particle_count": 50,
                    "gas_filter_age_months": 3.0,
                    "differential_pressure": 0.04
                }
            
            result = simulator.run_simulation(params)
            print(f"   最佳參數模擬結果:")
            print(f"   - 預測良率: {result['results']['predicted_yield']}")
            print(f"   - 評估: {result['feedback']}")
            
        else:
            print(f"❌ 不可模擬")
            print(f"   原因: {factory.get_unavailable_reason(defect)}")
    
    print("\n" + "=" * 80)
    print(f"✅ 支援模擬的瑕疵類型: {factory.get_supported_defects()}")
    print(f"📊 總計支援 {len(factory.get_supported_defects())} / {len(all_defects)} 種瑕疵模擬")
    print("=" * 80)


def test_tool_integration():
    """測試 Tool 整合層"""
    from src.agent.tools import simulate_defect_solution
    
    print("\n" + "=" * 80)
    print("工具整合層測試 - simulate_defect_solution")
    print("=" * 80)
    
    # 測試 1: 可模擬的瑕疵 (Donut)
    print("\n📝 測試 1: 可模擬瑕疵 (Donut)")
    result = simulate_defect_solution.invoke({
        "defect_type": "Donut",
        "parameters": {
            "esc_temp_inner": 62,
            "esc_temp_outer": 67,
            "he_pressure_inner": 10.5,
            "he_pressure_outer": 14.5
        }
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 測試 2: 不可模擬的瑕疵 (Scratch)
    print("\n📝 測試 2: 不可模擬瑕疵 (Scratch)")
    result = simulate_defect_solution.invoke({
        "defect_type": "Scratch",
        "parameters": {}
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_all_defects()
    test_tool_integration()
