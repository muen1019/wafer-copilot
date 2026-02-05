import random
import json
from typing import Dict, Any, Optional

class BaseSimulator:
    """基礎模擬器類別，所有製程模擬器都繼承此類"""
    
    def __init__(self, process_name: str):
        self.process_name = process_name
        self.optimal_params = {}
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """執行模擬，由子類實作"""
        raise NotImplementedError("Subclass must implement run_simulation")
    
    def _calculate_deviation(self, params: Dict[str, float]) -> float:
        """計算參數偏差度（通用方法）"""
        total_error = 0.0
        count = 0
        
        for key, optimal_val in self.optimal_params.items():
            if key in params and isinstance(optimal_val, (int, float)):
                actual_val = params[key]
                deviation = abs(actual_val - optimal_val) / optimal_val
                total_error += deviation
                count += 1
        
        return total_error / count if count > 0 else 0.0
    
    def _generate_feedback(self, error: float) -> str:
        """生成風險評估回饋"""
        if error < 0.1:
            return "✅ Simulation Successful: Parameters are within optimal processing window."
        elif error < 0.3:
            return "⚠️ Simulation Warning: Parameters are deviating from optimal settings. Yield may be impacted."
        else:
            return "❌ Simulation Critical: High deviation detected. Process stability is at risk."


class CMPSimulator(BaseSimulator):
    """CMP (Chemical Mechanical Polishing) 製程模擬器 - 用於 Center 瑕疵"""
    
    def __init__(self):
        super().__init__("CMP")
        self.optimal_params = {
            "polishing_pressure_center": 3.0,  # psi (範圍: 2.5-3.5)
            "slurry_flow_center": 175.0,       # ml/min (範圍: 150-200)
            "pad_lifetime_hours": 250           # 研磨墊使用時數 (最大: 500片 ≈ 250hr)
        }
        
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """模擬 CMP 製程參數對晶圓中心區域的影響"""
        
        # 解析輸入參數
        pressure = params.get("polishing_pressure_center", self.optimal_params["polishing_pressure_center"])
        flow = params.get("slurry_flow_center", self.optimal_params["slurry_flow_center"])
        pad_hours = params.get("pad_lifetime_hours", self.optimal_params["pad_lifetime_hours"])
        
        # 計算參數偏差
        total_error = self._calculate_deviation({
            "polishing_pressure_center": pressure,
            "slurry_flow_center": flow,
            "pad_lifetime_hours": pad_hours
        })
        
        # 模擬良率與瑕疵
        base_yield = 98.0
        simulated_yield = max(0.0, base_yield - (total_error * 40.0))
        
        # Center 瑕疵數量與壓力偏差高度相關
        pressure_error = abs(pressure - self.optimal_params["polishing_pressure_center"]) / self.optimal_params["polishing_pressure_center"]
        center_defect_count = int(3 + (pressure_error * 80))
        
        # 中心區域均勻性
        uniformity = 1.2 + (total_error * 8.0)
        
        # 研磨墊磨損警告
        pad_warning = ""
        if pad_hours > 400:
            pad_warning = "⚠️ 研磨墊使用時數接近上限，建議盡快更換"
            simulated_yield -= 2.0
        
        noise = random.uniform(-0.5, 0.5)
        simulated_yield = min(100.0, max(0.0, simulated_yield + noise))
        
        return {
            "status": "success",
            "process": "CMP (Chemical Mechanical Polishing)",
            "defect_type": "Center",
            "inputs": {
                "polishing_pressure_center": pressure,
                "slurry_flow_center": flow,
                "pad_lifetime_hours": pad_hours
            },
            "results": {
                "predicted_yield": f"{simulated_yield:.2f}%",
                "estimated_center_defects": center_defect_count,
                "center_uniformity": f"{uniformity:.2f}%",
                "pad_warning": pad_warning
            },
            "feedback": self._generate_feedback(total_error)
        }


class EtchSimulator(BaseSimulator):
    """Etch (蝕刻) 製程模擬器 - 用於 Donut 瑕疵"""
    
    def __init__(self):
        super().__init__("Etch")
        self.optimal_params = {
            "esc_temp_inner": 60.0,      # °C (目標: 60±2)
            "esc_temp_outer": 65.0,      # °C (目標: 65±2)
            "he_pressure_inner": 10.0,   # Torr (目標: 10±1)
            "he_pressure_outer": 15.0,   # Torr (目標: 15±1)
            "rf_inner_outer_ratio": 1.0  # 內外圈功率比 (預設 1:1)
        }
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """模擬蝕刻製程參數對環狀瑕疵的影響"""
        
        # 解析輸入參數
        temp_in = params.get("esc_temp_inner", self.optimal_params["esc_temp_inner"])
        temp_out = params.get("esc_temp_outer", self.optimal_params["esc_temp_outer"])
        he_in = params.get("he_pressure_inner", self.optimal_params["he_pressure_inner"])
        he_out = params.get("he_pressure_outer", self.optimal_params["he_pressure_outer"])
        rf_ratio = params.get("rf_inner_outer_ratio", self.optimal_params["rf_inner_outer_ratio"])
        
        # 計算溫度差異（Donut 瑕疵與內外圈溫差高度相關）
        temp_diff = abs(temp_out - temp_in)
        optimal_temp_diff = 5.0  # 目標溫差
        temp_diff_error = abs(temp_diff - optimal_temp_diff) / optimal_temp_diff
        
        # 計算整體偏差
        total_error = self._calculate_deviation({
            "esc_temp_inner": temp_in,
            "esc_temp_outer": temp_out,
            "he_pressure_inner": he_in,
            "he_pressure_outer": he_out,
            "rf_inner_outer_ratio": rf_ratio
        })
        
        # 溫差偏差會加重瑕疵
        total_error = (total_error + temp_diff_error * 1.5) / 2
        
        # 模擬良率
        base_yield = 97.5
        simulated_yield = max(0.0, base_yield - (total_error * 45.0))
        
        # Donut 環狀瑕疵數量
        donut_defect_count = int(4 + (temp_diff_error * 60) + (total_error * 40))
        
        # 蝕刻均勻性
        etch_uniformity = 2.0 + (total_error * 12.0)
        
        noise = random.uniform(-0.5, 0.5)
        simulated_yield = min(100.0, max(0.0, simulated_yield + noise))
        
        return {
            "status": "success",
            "process": "Etch (Plasma Etching)",
            "defect_type": "Donut",
            "inputs": {
                "esc_temp_inner": temp_in,
                "esc_temp_outer": temp_out,
                "he_pressure_inner": he_in,
                "he_pressure_outer": he_out,
                "rf_inner_outer_ratio": rf_ratio,
                "actual_temp_diff": f"{temp_diff:.1f}°C"
            },
            "results": {
                "predicted_yield": f"{simulated_yield:.2f}%",
                "estimated_donut_defects": donut_defect_count,
                "etch_uniformity": f"{etch_uniformity:.2f}%",
                "temp_diff_status": "✅ Optimal" if abs(temp_diff - 5.0) < 1.0 else "⚠️ Suboptimal"
            },
            "feedback": self._generate_feedback(total_error)
        }


class EnvironmentalSimulator(BaseSimulator):
    """環境控制模擬器 - 用於 Random 瑕疵"""
    
    def __init__(self):
        super().__init__("Environmental")
        self.optimal_params = {
            "cleanroom_particle_count": 50.0,  # particles/ft³ (Class 100 標準: <100)
            "gas_filter_age_months": 3.0,      # 過濾器使用月數 (最大: 6個月)
            "differential_pressure": 0.04       # inch H2O (範圍: 0.03-0.05)
        }
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """模擬環境參數對隨機瑕疵的影響"""
        
        particle_count = params.get("cleanroom_particle_count", self.optimal_params["cleanroom_particle_count"])
        filter_age = params.get("gas_filter_age_months", self.optimal_params["gas_filter_age_months"])
        diff_pressure = params.get("differential_pressure", self.optimal_params["differential_pressure"])
        
        # 計算偏差
        total_error = self._calculate_deviation({
            "cleanroom_particle_count": particle_count,
            "gas_filter_age_months": filter_age,
            "differential_pressure": diff_pressure
        })
        
        # 微粒數直接影響 Random 瑕疵
        particle_impact = (particle_count / 100.0) * 0.5  # 超過 100 會嚴重影響
        
        # 模擬良率
        base_yield = 99.0
        simulated_yield = max(0.0, base_yield - (total_error * 30.0) - (particle_impact * 20.0))
        
        # Random 瑕疵數量
        random_defect_count = int(2 + (particle_count / 10) + (filter_age * 3))
        
        noise = random.uniform(-0.3, 0.3)
        simulated_yield = min(100.0, max(0.0, simulated_yield + noise))
        
        # 警告訊息
        warnings = []
        if particle_count > 100:
            warnings.append("❌ 潔淨度超標！需立即檢查 FFU 與過濾系統")
        if filter_age > 5:
            warnings.append("⚠️ 氣體過濾器即將到期，建議更換")
        if diff_pressure < 0.03 or diff_pressure > 0.05:
            warnings.append("⚠️ 壓差異常，檢查風速與靜壓箱")
        
        return {
            "status": "success",
            "process": "Environmental Control",
            "defect_type": "Random",
            "inputs": {
                "cleanroom_particle_count": particle_count,
                "gas_filter_age_months": filter_age,
                "differential_pressure": diff_pressure
            },
            "results": {
                "predicted_yield": f"{simulated_yield:.2f}%",
                "estimated_random_defects": random_defect_count,
                "cleanroom_status": "✅ Pass" if particle_count < 100 else "❌ Fail",
                "warnings": warnings if warnings else ["✅ All environmental parameters within spec"]
            },
            "feedback": self._generate_feedback(total_error + particle_impact)
        }


class DigitalTwinFactory:
    """數位孿生工廠 - 根據瑕疵類型選擇對應的模擬器"""
    
    DEFECT_TO_SIMULATOR = {
        "Center": CMPSimulator,
        "Donut": EtchSimulator,
        "Random": EnvironmentalSimulator,
        # 其他瑕疵類型若無量化參數可模擬，則不提供模擬器
        # "Scratch": 無量化參數，屬於緊急清潔程序
        # "Edge-Ring": 硬體更換為主（Focus Ring, Edge Heater）
        # "Edge-Loc": 機械校正為主（Robot, Notch Finder）
        # "Loc": 光罩檢查與清潔程序
        # "Near-full": 緊急停機與完整 PM
    }
    
    @staticmethod
    def get_simulator(defect_type: str) -> Optional[BaseSimulator]:
        """根據瑕疵類型獲取對應的模擬器"""
        simulator_class = DigitalTwinFactory.DEFECT_TO_SIMULATOR.get(defect_type)
        if simulator_class:
            return simulator_class()
        return None
    
    @staticmethod
    def is_simulatable(defect_type: str) -> bool:
        """檢查該瑕疵類型是否可模擬"""
        return defect_type in DigitalTwinFactory.DEFECT_TO_SIMULATOR
    
    @staticmethod
    def get_supported_defects() -> list:
        """獲取所有支援模擬的瑕疵類型"""
        return list(DigitalTwinFactory.DEFECT_TO_SIMULATOR.keys())
    
    @staticmethod
    def get_unavailable_reason(defect_type: str) -> str:
        """獲取無法模擬的原因說明"""
        reasons = {
            "Scratch": "Scratch 瑕疵為機械損傷，主要處置為立即停機清潔與更換受損零件，無量化參數可模擬",
            "Edge-Ring": "Edge-Ring 瑕疵處置以硬體更換為主（Focus Ring, Edge Heater），非參數調整類型",
            "Edge-Loc": "Edge-Loc 瑕疵需進行機械校正（Robot, Notch Finder），屬於位置偏差問題而非參數問題",
            "Loc": "Loc 瑕疵通常與光罩缺陷或局部汙染有關，需執行檢查與清潔程序，不適合參數模擬",
            "Near-full": "Near-full 為嚴重系統性問題，需立即停機並執行完整 PM，不適合模擬測試"
        }
        return reasons.get(defect_type, "此瑕疵類型尚未建立模擬模型")


# 向後兼容的別名（保留舊的 API）
class DigitalTwinSimulator(CMPSimulator):
    """向後兼容的 CMP 模擬器別名"""
    pass


if __name__ == "__main__":
    # 測試範例
    print("=" * 60)
    print("測試 1: CMP 模擬器 (Center 瑕疵)")
    print("=" * 60)
    cmp_sim = CMPSimulator()
    result = cmp_sim.run_simulation({
        "polishing_pressure_center": 3.2,
        "slurry_flow_center": 180,
        "pad_lifetime_hours": 300
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("測試 2: Etch 模擬器 (Donut 瑕疵)")
    print("=" * 60)
    etch_sim = EtchSimulator()
    result = etch_sim.run_simulation({
        "esc_temp_inner": 62,
        "esc_temp_outer": 67,
        "he_pressure_inner": 10.5,
        "he_pressure_outer": 14.5,
        "rf_inner_outer_ratio": 1.1
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("測試 3: Environmental 模擬器 (Random 瑕疵)")
    print("=" * 60)
    env_sim = EnvironmentalSimulator()
    result = env_sim.run_simulation({
        "cleanroom_particle_count": 85,
        "gas_filter_age_months": 4.5,
        "differential_pressure": 0.042
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("測試 4: 使用 Factory 模式")
    print("=" * 60)
    factory = DigitalTwinFactory()
    print(f"支援的瑕疵類型: {factory.get_supported_defects()}")
    print(f"Center 可模擬: {factory.is_simulatable('Center')}")
    print(f"Scratch 可模擬: {factory.is_simulatable('Scratch')}")
    if not factory.is_simulatable('Scratch'):
        print(f"Scratch 無法模擬原因: {factory.get_unavailable_reason('Scratch')}")
