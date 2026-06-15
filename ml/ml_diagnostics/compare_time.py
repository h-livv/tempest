import os
import sys
import time
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.solver import solver
from src.equations import advection
from src.integrators import rk4
from src.operators import upwind
from src.init_conditions import advec_gauss
from src.boundaries import periodic

class ConvSurrogate(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(1, 1, 25))
        self.temperature = 1000.0

    def forward(self, x):
        w = F.softmax(self.raw_weights * self.temperature, dim=-1)
        x_padded = F.pad(x.unsqueeze(1), (12, 12), mode='circular')
        return F.conv1d(x_padded, w).squeeze(1)

def get_latest_run(base_dirs):
    all_runs = []
    for d in base_dirs:
        if os.path.exists(d):
            for r in os.listdir(d):
                full_path = os.path.join(d, r)
                if os.path.isdir(full_path):
                    all_runs.append(full_path)
    if not all_runs:
        return None
    all_runs.sort(key=lambda x: os.path.getmtime(x))
    return all_runs[-1]

def main():
    print("=== Inference Speedup Benchmark ===")
    
    run_dir = get_latest_run(["ml/runs/sweeps", "ml/runs/single"])
    if not run_dir:
        print("Error: Could not find ML run directory.")
        sys.exit(1)
        
    model_path = os.path.join(run_dir, "advec_model.pt")
    
    # 1. Benchmark Numerical Solver
    print("Running True Numerical Physics (RK4 Upwind, 10,000 steps)...")
    start_num = time.time()
    
    # Suppress print statements from solver
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    sim_output = solver(
        N=5000,
        init_state=advec_gauss,
        boundary=periodic,
        operator=upwind,
        equation=advection,
        integrator=rk4,
        coefficient=1.0,
        dt=0.005,
        dx=0.05,
        FINAL_TIME=50.0,
        STEPS_PER_FRAME=20,
        RECORD_INTERVAL=20
    )
    
    sys.stdout.close()
    sys.stdout = original_stdout
    
    end_num = time.time()
    time_num = end_num - start_num
    print(f"✅ Numerical Time: {time_num:.4f} seconds")
    
    # 2. Benchmark ML Inference
    print("\nRunning ML Surrogate Inference (10,000 auto-regressive steps)...")
    
    device = torch.device("cpu")
    model = ConvSurrogate(5000).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Generate initial state mathematically to match solver
    x_grid = np.linspace(0, 5000 * 0.05, 5000, endpoint=False)
    initial_array = advec_gauss(5000, x_grid)
    current_state = torch.tensor(initial_array, dtype=torch.float32).to(device)
    
    start_ml = time.time()
    with torch.no_grad():
        for _ in range(10000):
            current_state = model(current_state)
    end_ml = time.time()
    
    time_ml = end_ml - start_ml
    print(f"✅ ML Inference Time: {time_ml:.4f} seconds")
    
    speedup = time_num / time_ml
    
    print("\n=== Results ===")
    print(f"The CNN Surrogate is {speedup:.2f}x faster than the Numerical Solver!")
    
    summary_text = (
        "=== Computation Time Comparison (Micro-Benchmark) ===\n"
        f"Numerical Solver (RK4 Upwind): {time_num:.4f} seconds\n"
        f"ML Inference (CNN Surrogate): {time_ml:.4f} seconds\n"
        f"Speedup: {speedup:.2f}x faster\n"
    )
    
    results_path = os.path.join(run_dir, "long_term_diagnostics", "compare_time_results.txt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        f.write(summary_text)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
