import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import glob
import sys

# Define the exact model architecture to load the weights
import torch.nn.functional as F

class ConvSurrogate(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(1, 1, 25))

    def forward(self, x):
        w = F.softmax(self.raw_weights * 1000.0, dim=-1)
        x_padded = F.pad(x.unsqueeze(1), (12, 12), mode='circular')
        return F.conv1d(x_padded, w).squeeze(1)

def get_latest_dir(base_dir):
    all_runs = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            full_path = os.path.join(base_dir, d)
            if os.path.isdir(full_path):
                all_runs.append(full_path)
    if not all_runs:
        return None
    all_runs.sort(key=lambda x: os.path.getmtime(x))
    return all_runs[-1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Find latest ML run to load model
    # Note: script is run from project root typically
    ml_runs_dir = os.path.join("ml", "runs", "sweeps")
    latest_ml_run = get_latest_dir(ml_runs_dir)
    if not latest_ml_run:
        print("No ML runs found.")
        sys.exit(1)
        
    model_path = os.path.join(latest_ml_run, "advec_model.pt")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. You need to train advec_cnn.py first.")
        sys.exit(1)
        
    model = ConvSurrogate(grid_size=5000).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded frozen model from {model_path}")
    
    # 2. Find latest numerical dataset
    data_runs_dir = os.path.join("pipeline_results", "sweeps")
    latest_data_run = get_latest_dir(data_runs_dir)
    if not latest_data_run:
        print("No dataset runs found.")
        sys.exit(1)
        
    dataset_files = []
    if os.path.isfile(os.path.join(latest_data_run, 'spatial_data.npz')):
        dataset_files.append(os.path.join(latest_data_run, 'spatial_data.npz'))
    else:
        dataset_files = glob.glob(os.path.join(latest_data_run, '*', 'spatial_data.npz'))
        
    if not dataset_files:
        print("No spatial_data.npz files found.")
        sys.exit(1)
        
    print(f"Found {len(dataset_files)} datasets in {latest_data_run}")
    
    # 3. Create diagnostics output directory
    diag_dir = os.path.join(latest_ml_run, "long_term_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    
    max_steps_ml = 10000
    
    import json
    for df in dataset_files:
        ic_name = "unknown_ic"
        config_path = os.path.join(os.path.dirname(df), "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                    ic_name = cfg.get("initial_condition", ic_name).replace(" ", "_").lower()
            except Exception:
                pass
                
        print(f"\nProcessing {ic_name}...")
        
        numerical_data = np.load(df)['ml_tensor_data']
        # numerical_data shape is [T_num, N]
        T_num = numerical_data.shape[0]
        N = numerical_data.shape[1]
        
        initial_condition = numerical_data[0]
        
        # Unroll ML model up to max_steps_ml
        current_state = torch.tensor(initial_condition, dtype=torch.float32).unsqueeze(0).to(device)
        ml_rollout = [initial_condition]
        
        # Soft energy projection parameters
        initial_energy = torch.mean(current_state**2)
        
        with torch.no_grad():
            for _ in range(max_steps_ml):
                current_state = model(current_state)
                # Energy projection
                current_energy = torch.mean(current_state**2)
                current_state = current_state * (0.8 + 0.2 * torch.sqrt(initial_energy / (current_energy + 1e-8)))
                ml_rollout.append(current_state.squeeze().cpu().numpy())
                
        ml_rollout = np.array(ml_rollout)
        
        times_ml = np.arange(max_steps_ml + 1)
        times_num = np.arange(T_num)
        
        l2_num = np.zeros(T_num)
        phase_num = np.zeros(T_num)
        mass_num = np.zeros(T_num)
        energy_num = np.zeros(T_num)
        
        l2_ml = np.zeros(max_steps_ml + 1)
        phase_ml = np.zeros(max_steps_ml + 1)
        mass_ml = np.zeros(max_steps_ml + 1)
        energy_ml = np.zeros(max_steps_ml + 1)
        
        print("Calculating errors...")
        for t in range(max_steps_ml + 1):
            ana_t = np.roll(initial_condition, shift=2 * t)
            
            # ML Errors
            ml_t = ml_rollout[t]
            l2_ml[t] = np.sqrt(np.mean((ml_t - ana_t)**2))
            phase_ml[t] = np.abs(np.argmax(ml_t) - np.argmax(ana_t))
            mass_ml[t] = np.sum(np.abs(ml_t)) - np.sum(np.abs(ana_t))
            energy_ml[t] = np.sum(ml_t**2) - np.sum(ana_t**2)
            
            # Num Errors
            if t < T_num:
                num_t = numerical_data[t]
                l2_num[t] = np.sqrt(np.mean((num_t - ana_t)**2))
                phase_num[t] = np.abs(np.argmax(num_t) - np.argmax(ana_t))
                mass_num[t] = np.sum(np.abs(num_t)) - np.sum(np.abs(ana_t))
                energy_num[t] = np.sum(num_t**2) - np.sum(ana_t**2)
                
        phase_ml = np.minimum(phase_ml, N - phase_ml)
        phase_num = np.minimum(phase_num, N - phase_num)
        
        print(f"Plotting {ic_name}...")
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Diagnostic Errors vs Time ({ic_name})\nNumerical vs ML Surrogate against Analytical Solution", fontsize=16)
        
        # L2 Error
        axs[0, 0].plot(times_num, l2_num, label="Numerical (FD) L2 Error", color="blue")
        axs[0, 0].plot(times_ml, l2_ml, label="ML Surrogate L2 Error", color="red", linestyle="--")
        axs[0, 0].set_title("L2 Error")
        axs[0, 0].set_xlabel("Timestep")
        axs[0, 0].set_ylabel("Error")
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # Phase Error
        axs[0, 1].plot(times_num, phase_num, label="Numerical Phase Error", color="blue")
        axs[0, 1].plot(times_ml, phase_ml, label="ML Phase Error", color="red", linestyle="--")
        axs[0, 1].set_title("Phase Error (Index Difference)")
        axs[0, 1].set_xlabel("Timestep")
        axs[0, 1].set_ylabel("Grid Cells")
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        
        # Mass Error
        axs[1, 0].plot(times_num, mass_num, label="Numerical Mass Error", color="blue")
        axs[1, 0].plot(times_ml, mass_ml, label="ML Mass Error", color="red", linestyle="--")
        axs[1, 0].set_title("Mass Error")
        axs[1, 0].set_xlabel("Timestep")
        axs[1, 0].set_ylabel("Error")
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # Energy Error
        axs[1, 1].plot(times_num, energy_num, label="Numerical Energy Error", color="blue")
        axs[1, 1].plot(times_ml, energy_ml, label="ML Energy Error", color="red", linestyle="--")
        axs[1, 1].set_title("Energy Error")
        axs[1, 1].set_xlabel("Timestep")
        axs[1, 1].set_ylabel("Error")
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, f"diagnostics_{ic_name}.png"), dpi=150)
        plt.close()
        
        print(f"Saved {ic_name} diagnostics to {diag_dir}/diagnostics_{ic_name}.png")

if __name__ == "__main__":
    main()
