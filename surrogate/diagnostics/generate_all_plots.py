import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def generate_plots_for_run(run_dir):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    rollout_files = glob.glob(os.path.join(run_dir, "rollout_*.npz"))
    if not rollout_files:
        print(f"Skipping {run_dir}: No rollout files found.")
        return
        
    print(f"Processing run: {os.path.basename(run_dir)}")
    
    for file_path in rollout_files:
        ic_name = os.path.basename(file_path).replace("rollout_", "").replace(".npz", "")
        print(f"  -> Generating plots for {ic_name}")
        
        data = np.load(file_path)
        ml_rollout = data['rollout']
        numerical_data = data['numerical']
        
        T_num = min(ml_rollout.shape[0], numerical_data.shape[0])
        N = ml_rollout.shape[1]
        
        initial_condition = numerical_data[0]
        
        # 1. Shape Comparison Snapshots
        for history_idx in [2000, 5000, 10000]:
            if history_idx < T_num:
                plt.figure(figsize=(10, 5))
                analytical = np.roll(initial_condition, shift=2 * history_idx)
                plt.plot(analytical, label=f"Analytical (T={history_idx})", color="green", linestyle=":", linewidth=2, alpha=0.8)
                plt.plot(numerical_data[history_idx], label=f"Numerical (T={history_idx})", color="blue", alpha=0.6)
                plt.plot(ml_rollout[history_idx], label=f"ML Rollout (T={history_idx})", linestyle="--", color="red")
                
                plt.title(f"Wave Propagation Comparison at T={history_idx} ({ic_name})")
                plt.xlabel("Grid Cell")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.grid(True)
                
                save_path = os.path.join(plots_dir, f"snapshot_{ic_name}_T{history_idx}.png")
                plt.savefig(save_path, dpi=150)
                plt.close()

        # 2. Continuous Error Diagnostics
        times = np.arange(T_num)
        l2_num = np.zeros(T_num)
        phase_num = np.zeros(T_num)
        l2_ml = np.zeros(T_num)
        phase_ml = np.zeros(T_num)
        
        for t in range(T_num):
            ana_t = np.roll(initial_condition, shift=2 * t)
            
            ml_t = ml_rollout[t]
            l2_ml[t] = np.sqrt(np.mean((ml_t - ana_t)**2))
            phase_ml[t] = np.abs(np.argmax(ml_t) - np.argmax(ana_t))
            
            num_t = numerical_data[t]
            l2_num[t] = np.sqrt(np.mean((num_t - ana_t)**2))
            phase_num[t] = np.abs(np.argmax(num_t) - np.argmax(ana_t))
            
        phase_ml = np.minimum(phase_ml, N - phase_ml)
        phase_num = np.minimum(phase_num, N - phase_num)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f"Diagnostic Errors vs Time ({ic_name})\nNumerical vs ML Surrogate against Analytical Solution", fontsize=16)
        
        # L2 Error
        axs[0].plot(times, l2_num, label="Numerical (FD) L2 Error", color="blue")
        axs[0].plot(times, l2_ml, label="ML Surrogate L2 Error", color="red", linestyle="--")
        axs[0].set_title("L2 Error")
        axs[0].set_xlabel("Timestep")
        axs[0].set_ylabel("Error")
        axs[0].grid(True)
        axs[0].legend()
        
        # Phase Error
        axs[1].plot(times, phase_num, label="Numerical Phase Error", color="blue")
        axs[1].plot(times, phase_ml, label="ML Phase Error", color="red", linestyle="--")
        axs[1].set_title("Phase Error (Index Difference)")
        axs[1].set_xlabel("Timestep")
        axs[1].set_ylabel("Grid Cells")
        axs[1].grid(True)
        axs[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"error_tracking_{ic_name}.png"), dpi=150)
        plt.close()

def main():
    sweeps_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CNN", "runs", "sweeps")
    if not os.path.exists(sweeps_dir):
        print(f"Sweeps directory not found at {sweeps_dir}")
        return
        
    run_dirs = [os.path.join(sweeps_dir, d) for d in os.listdir(sweeps_dir) if os.path.isdir(os.path.join(sweeps_dir, d))]
    
    print(f"Found {len(run_dirs)} sweep runs. Generating plots...")
    for run_dir in run_dirs:
        generate_plots_for_run(run_dir)
        
    print("\n✅ All historical plots generated successfully!")

if __name__ == "__main__":
    main()
