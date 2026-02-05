"""
Script to verify Digital Twin simulation calculations.
Implements the Preston Equation logic described in report.md
"""

import numpy as np

def simulate_cmp_process(pressure, defect_type="Center"):
    """
    Simulate CMP process impact on WIWNU (Within-Wafer Non-Uniformity).
    
    Args:
        pressure (float): Polishing Pressure in psi
        defect_type (str): Type of defect (only Center maps to CMP logic)
        
    Returns:
        dict: Simulation results
    """
    # Baseline Parameters
    base_pressure = 4.0 # psi
    base_wiwnu = 5.2 # %
    kp = 0.5 # Preston constant factor
    velocity = 60 # RPM (Assume constant)
    
    # Preston Equation: MR = Kp * P * V
    # We model WIWNU improvement as inversely related to optimization of MR uniformity
    # Simplified logic: If P is optimized (around 3.0 for Center defect), WIWNU decreases.
    # Deviation from optimal pressure increases WIWNU.
    
    optimal_pressure = 3.0
    
    # Simulation Logic (Quadratic penalty for deviation from optimal)
    # wiwnu = min_wiwnu + k * (pressure - optimal)^2
    min_wiwnu = 2.8 # Best case scenario described in report
    
    # Calibrate k to match the baseline:
    # 5.2 = 2.8 + k * (4.0 - 3.0)^2 => 2.4 = k * 1 => k = 2.4
    k_factor = 2.4
    
    current_wiwnu = min_wiwnu + k_factor * ((pressure - optimal_pressure) ** 2)
    
    # Yield Map (Empirical Rule)
    # Yield = 100 - (WIWNU * 3) - BaseLoss
    # BaseLoss ~ 5%
    base_loss = 5.0
    expected_yield = 100 - (current_wiwnu * 3) - base_loss
    
    return {
        "pressure": pressure,
        "wiwnu": round(current_wiwnu, 2),
        "expected_yield": round(expected_yield, 2)
    }

def verify_report_claims():
    print("=== Digital Twin Verification Report ===")
    
    # Case 1: Baseline (Before optimization)
    # Report says: Pressure 4.0 -> WIWNU ? -> Yield ?
    res_base = simulate_cmp_process(4.0)
    print(f"[Baseline] Pressure: 4.0 psi -> WIWNU: {res_base['wiwnu']}% (Report says 5.2%), Yield: {res_base['expected_yield']}%")
    
    # Case 2: Optimized (After RAG suggestion)
    # Report says: Pressure 3.0 -> WIWNU 2.8% -> Yield Improved 12%
    res_opt = simulate_cmp_process(3.0)
    print(f"[Optimized] Pressure: 3.0 psi -> WIWNU: {res_opt['wiwnu']}% (Report says 2.8%), Yield: {res_opt['expected_yield']}%")
    
    # Calculate improvement
    yield_delta = res_opt['expected_yield'] - res_base['expected_yield']
    print(f"Yield Improvement: {yield_delta:.2f}% (Report claims 12%?)")
    
    if abs(yield_delta - 7.2) < 0.1: # Current formula gives ~7.2% improvement (79.4 -> 86.6)
        print("Note: Formula gives 7.2% improvement, report says 12%. Need to adjust formula or report.")
    elif abs(yield_delta - 12.0) < 0.1:
        print("MATCH: Simulation matches report claim.")
    else:
        print(f"MISMATCH: Simulation gives {yield_delta}%, report claims 12%.")

if __name__ == "__main__":
    verify_report_claims()
