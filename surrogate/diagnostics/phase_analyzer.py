import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

def unwrap_phase(signal_matrix):
    """
    Tracks the peak of a wave and unwraps the position across periodic boundaries.
    signal_matrix: shape (T, N)
    """
    T, N = signal_matrix.shape
    raw_peaks = np.argmax(signal_matrix, axis=1)
    
    unwrapped_peaks = np.zeros(T, dtype=float)
    unwrapped_peaks[0] = raw_peaks[0]
    
    cumulative_offset = 0
    for t in range(1, T):
        diff = raw_peaks[t] - raw_peaks[t-1]
        
        # Wrapped right (e.g. 4999 -> 0)
        if diff < -N/2.0:
            cumulative_offset += N
        # Wrapped left (e.g. 0 -> 4999)
        elif diff > N/2.0:
            cumulative_offset -= N
            
        unwrapped_peaks[t] = raw_peaks[t] + cumulative_offset
        
    return unwrapped_peaks

def compute_c_eff_discrete(unwrapped_physical_pos, dt_frame):
    """
    Computes effective velocity in physical units using central differences.
    (Kept for the CSV output, but not used for global metrics).
    """
    T = len(unwrapped_physical_pos)
    c_eff = np.zeros(T, dtype=float)
    if T < 2:
        return c_eff
        
    c_eff[0] = (unwrapped_physical_pos[1] - unwrapped_physical_pos[0]) / dt_frame
    c_eff[-1] = (unwrapped_physical_pos[-1] - unwrapped_physical_pos[-2]) / dt_frame
    for t in range(1, T-1):
        c_eff[t] = (unwrapped_physical_pos[t+1] - unwrapped_physical_pos[t-1]) / (2.0 * dt_frame)
        
    return c_eff

def main():
    parser = argparse.ArgumentParser(description="Analyze numerical dispersion and phase lag.")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to the ML run directory")
    args = parser.parse_args()
    
    if args.run_dir:
        run_dir = args.run_dir
    else:
        print("No run directory provided. Searching for the latest run...")
        runs_base_dir = os.path.join("ml", "CNN", "runs")
        all_runs = []
        for category in ["single", "sweeps"]:
            cat_dir = os.path.join(runs_base_dir, category)
            if os.path.exists(cat_dir):
                for d in os.listdir(cat_dir):
                    full_path = os.path.join(cat_dir, d)
                    if os.path.isdir(full_path):
                        all_runs.append(full_path)
        
        if not all_runs:
            print("Error: No ML runs found.")
            sys.exit(1)
            
        all_runs.sort(key=lambda x: os.path.getmtime(x))
        run_dir = all_runs[-1]
        print(f"✅ Automatically selected latest run: {os.path.basename(run_dir)}")

    # Extract Physical Parameters
    hyper_path = os.path.join(run_dir, "hyperparameters.json")
    if not os.path.exists(hyper_path):
        print(f"Error: Could not find hyperparameters.json in {run_dir}")
        sys.exit(1)
        
    with open(hyper_path, "r") as f:
        hyper = json.load(f)
        
    training_data_path = hyper.get("training_data_path")
    if not training_data_path or not os.path.exists(training_data_path):
        print(f"Error: Invalid training_data_path in hyperparameters: {training_data_path}")
        sys.exit(1)
        
    # Grab the first config.json we can find in the training data path
    config_files = glob.glob(os.path.join(training_data_path, "**", "config.json"), recursive=True)
    if not config_files:
        print("Error: Could not find any config.json in the numerical training data.")
        sys.exit(1)
        
    with open(config_files[0], "r") as f:
        config = json.load(f)
        
    dx = config.get("dx", 1.0)
    dt_math = config.get("dt", 1.0)
    record_interval = config.get("record_interval", 1)
    
    # Analytical wave speed 'c'
    if "coefficient" in config:
        c_ana = float(config["coefficient"])
    else:
        # Fallback if coefficient not explicitly saved
        cfl = config.get("CFL", 0.1)
        c_ana = cfl * dx / dt_math
        
    dt_frame = dt_math * record_interval
    
    print(f"\n--- Physical Parameters Extracted ---")
    print(f"Spatial Grid dx : {dx}")
    print(f"Numerical dt    : {dt_math}")
    print(f"Record Interval : {record_interval}")
    print(f"Frame dt        : {dt_frame}")
    print(f"Analytical c    : {c_ana}")
    print("-------------------------------------")

    rollout_files = glob.glob(os.path.join(run_dir, "rollout_*.npz"))
    if not rollout_files:
        print(f"Error: No rollout_*.npz files found in {run_dir}")
        sys.exit(1)

    plots_dir = os.path.join(run_dir, "long_term_diagnostics")
    os.makedirs(plots_dir, exist_ok=True)
    
    summary_path = os.path.join(plots_dir, "velocity_summary.txt")
    with open(summary_path, 'w') as f_sum:
        f_sum.write("=== Global Effective Velocity Summary ===\n")
        f_sum.write(f"Analytical speed (c): {c_ana}\n\n")

    for file_path in rollout_files:
        ic_name = os.path.basename(file_path).replace("rollout_", "").replace(".npz", "")
        
        data = np.load(file_path)
        rollout_ml = data['rollout']
        numerical_true = data['numerical']
        
        min_T = min(rollout_ml.shape[0], numerical_true.shape[0])
        rollout_ml = rollout_ml[:min_T]
        numerical_true = numerical_true[:min_T]
        
        # 1. Unwrap indices
        pos_true_idx = unwrap_phase(numerical_true)
        pos_ml_idx = unwrap_phase(rollout_ml)
        
        # 2. Convert to Physical units
        x_true = pos_true_idx * dx
        x_ml = pos_ml_idx * dx
        t_array = np.arange(min_T) * dt_frame
        
        # 3. Calculate Global Effective Velocity (Linear Regression)
        c_eff_true_global = np.polyfit(t_array, x_true, 1)[0]
        c_eff_ml_global = np.polyfit(t_array, x_ml, 1)[0]
        
        err_true_pct = abs(c_eff_true_global - c_ana) / c_ana * 100.0
        err_ml_pct = abs(c_eff_ml_global - c_ana) / c_ana * 100.0
        
        # Console output
        print(f"\n[{ic_name}] Velocity Diagnostics:")
        print(f"{'Method':<15} | {'C_eff':<10} | {'Error (%)':<10}")
        print("-" * 40)
        print(f"{'Analytical':<15} | {c_ana:<10.5f} | {'0.00%':<10}")
        print(f"{'Numerical':<15} | {c_eff_true_global:<10.5f} | {err_true_pct:<9.5f}%")
        print(f"{'ML Surrogate':<15} | {c_eff_ml_global:<10.5f} | {err_ml_pct:<9.5f}%")
        
        # File output summary
        with open(summary_path, 'a') as f_sum:
            f_sum.write(f"[{ic_name}]\n")
            f_sum.write(f"Numerical    : {c_eff_true_global:.6f}  (Err: {err_true_pct:.4f}%)\n")
            f_sum.write(f"ML Surrogate : {c_eff_ml_global:.6f}  (Err: {err_ml_pct:.4f}%)\n\n")
        
        # Calculate discrete velocities for CSV logging
        c_eff_true_discrete = compute_c_eff_discrete(x_true, dt_frame)
        c_eff_ml_discrete = compute_c_eff_discrete(x_ml, dt_frame)
        positional_error = x_true - x_ml
        
        # 4. Save CSV
        csv_path = os.path.join(run_dir, f"phase_diagnostics_{ic_name}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_phys', 'true_pos_phys', 'ml_pos_phys', 'positional_error_phys', 'true_c_eff_discrete', 'ml_c_eff_discrete'])
            for t in range(min_T):
                writer.writerow([t_array[t], x_true[t], x_ml[t], positional_error[t], c_eff_true_discrete[t], c_eff_ml_discrete[t]])
        
        # 5. Generate and Save Plot (Unchanged visually, but using physical arrays)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.plot(t_array, x_true, label='True Physics (Numerical)', color='blue', linewidth=2)
        ax1.plot(t_array, x_ml, label='ML Surrogate', color='red', linestyle='--', linewidth=2)
        ax1.set_title(f"Cumulative Phase Tracking: {ic_name}")
        ax1.set_ylabel("Unwrapped Position (Physical x)")
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(t_array, positional_error, label=r'$\Delta x$ (True - ML)', color='purple', linewidth=2)
        ax2.set_xlabel("Physical Time (t)")
        ax2.set_ylabel("Positional Error (Physical dx)")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"phase_tracking_{ic_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    main()
