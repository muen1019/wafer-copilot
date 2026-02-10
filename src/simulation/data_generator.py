import random
import uuid
from datetime import datetime

def get_mock_context(defect_type: str) -> dict:
    """
    Generate simulated production context data based on defect type.
    
    Args:
        defect_type (str): The classification result from the vision model.
        
    Returns:
        dict: Context containing lot_id, step_history, batch_yield, and machine_log.
    """
    lot_id = f"LOT-{uuid.uuid4().hex[:8].upper()}"
    
    # Base flow
    step_history = ["Cleaner-01", "Coater-02", "Scanner-05", "Developer-02", "Etcher-01", "Stripper-01"]
    batch_yield = round(random.uniform(0.95, 0.99), 4)
    machine_log = "All systems nominal. Process completed within standard cycle time."
    
    if defect_type == 'Donut':
        # Specific rule for Donut defect
        # Replace Etcher-01 with Etcher-03 to match requirement
        if "Etcher-01" in step_history:
            step_history[step_history.index("Etcher-01")] = "Etcher-03"
        else:
            step_history.append("Etcher-03")
            
        machine_log = "[WARNING] Etcher-03: Helium flow warning. Backside cooling pressure fluctuation detected."
        batch_yield = round(random.uniform(0.80, 0.88), 4)
        
    elif defect_type == 'Scratch':
        # Specific rule for Scratch defect
        # Scratches often happen during handling
        step_history.append("Robot-Arm-B")
        machine_log = "[ALERT] Robot-Arm-B: Unexpected torque spike recorded during cluster transfer."
        batch_yield = round(random.uniform(0.85, 0.92), 4)
    
    # Generic handling for other defects to make it realistic
    elif defect_type == 'Center':
        step_history[1] = "Coater-03" # Maybe spinner issue
        machine_log = "[INFO] Coater-03: Spin speed deviation observed (+1.2%)."
        batch_yield = round(random.uniform(0.88, 0.94), 4)
        
    elif defect_type == 'Edge-Ring':
        if "Etcher-01" in step_history:
            step_history[step_history.index("Etcher-01")] = "Etcher-02"
        machine_log = "[INFO] Etcher-02: Edge clamp alignment check required. Plasma uniformity draft detected at edge."
        batch_yield = round(random.uniform(0.87, 0.93), 4)

    elif defect_type == 'Loc':
        # Localized defect, often lithography focus or particle
        machine_log = "[WARNING] Scanner-05: Localized focus offset deviation (-0.15um) detected in quadrant 3."
        batch_yield = round(random.uniform(0.90, 0.96), 4)

    elif defect_type == 'Edge-Loc':
        # Edge Local, often handling or detailed edge bead removal issue
        step_history[-1] = "Stripper-02"
        machine_log = "[INFO] Stripper-02: Chemical dispense nozzle pressure oscillating during edge rinse."
        batch_yield = round(random.uniform(0.88, 0.94), 4)

    elif defect_type == 'Random':
        # Random distribution usually means dirty environment or cleaning issue
        if "Cleaner-01" in step_history:
             step_history[step_history.index("Cleaner-01")] = "Cleaner-01-Backup"
        machine_log = "[ALERT] Cleaner-01-Backup: Particle count monitor (P-Mon) spike > 50 counts/hour. Filter replacement overdue."
        batch_yield = round(random.uniform(0.75, 0.85), 4)

    elif defect_type == 'Near-full':
        # Near full wafer failure, often major process step failure (like developer)
        machine_log = "[CRITICAL] Developer-02: Developer solution concentration below threshold (-5%). Efficacy significantly dropped."
        batch_yield = round(random.uniform(0.50, 0.70), 4)

    return {
        "lot_id": lot_id,
        "step_history": step_history,
        "batch_yield": f"{batch_yield:.2%}",
        "machine_log": machine_log,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
