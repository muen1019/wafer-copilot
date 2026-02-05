"""
Digital Twin Agent Tool - 將建議參數送入數位孿生系統進行模擬
"""

from langchain.tools import tool
from src.digital_twin.simulator import DigitalTwinSimulator

@tool
def simulate_process_parameters(params: dict):
    """
    在數位孿生系統中測試製程參數建議。
    
    Args:
        params (dict): 包含製程參數的字典，例如:
            {
                "rotation_speed": 100.0,  # rpm
                "pad_pressure": 5.0,      # psi
                "slurry_flow": 200.0      # ml/min
            }
    
    Returns:
        dict: 模擬結果，包含預測良率、瑕疵數、均勻度等指標
    """
    simulator = DigitalTwinSimulator()
    result = simulator.run_simulation(params)
    return result
