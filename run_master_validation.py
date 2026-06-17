import subprocess
import os
import glob
import time
import csv

configs = [
    "configs/advection.py",
    "configs/advec_conv.py",
    "configs/diffusion.py",
    "configs/diff_conv.py",
    "configs/wave.py",
    "configs/wave_conv.py",
    "configs/shallow_validation.py",
    "configs/shallow_convergence.py",
    "configs/dam_validation.py",
    "configs/dam_convergence.py"
]

start_time = time.time()

for cfg in configs:
    print(f"\n========================================")
    print(f"Executing {cfg}")
    print(f"========================================")
    result = subprocess.run(["python", "main.py", cfg], env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if result.returncode != 0:
        print(f"CRITICAL ERROR running {cfg}. Stopping.")
        exit(1)

print("\n\nAll runs completed successfully!")
print("Extracting Convergence Slopes from Sweeps...\n")

csv_files = glob.glob("pipeline_results/sweeps/**/convergence_*.csv", recursive=True)
recent_csvs = [f for f in csv_files if os.path.getmtime(f) > start_time]

for file in recent_csvs:
    with open(file, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        slope = float(row['observed_order'])
        metric = row['error_metric']
        name = os.path.basename(file).replace('convergence_', '').replace('.csv', '')
        print(f"-> {name}")
        print(f"   Slope ({metric}): {slope:.3f}")

print("\nValidation Suite complete.")
