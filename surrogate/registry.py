import csv
import os
import datetime

REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_registry.csv")

def log_run(run_type, module, equation, operator, ic, path, metadata=None):
    """
    Logs an executed run (from main or ml) into the master registry.
    
    run_type: 'single' or 'sweep'
    module: 'main' or 'ml'
    """
    file_exists = os.path.isfile(REGISTRY_FILE)
    
    with open(REGISTRY_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Module", "Run Type", "Equation", "Operator", "Initial Condition", "Path", "Metadata"])
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_str = str(metadata) if metadata else ""
        writer.writerow([timestamp, module, run_type, equation, operator, ic, path, metadata_str])
