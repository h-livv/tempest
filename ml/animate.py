import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import glob

# Define where your runs are stored
runs_base_dir = os.path.join("ml", "runs")

# 1. Determine which run directory to use
if len(sys.argv) < 2:
    print("No run directory provided. Searching for the latest run...")
    try:
        all_runs = []
        for category in ["single", "sweeps"]:
            cat_dir = os.path.join(runs_base_dir, category)
            if os.path.exists(cat_dir):
                for d in os.listdir(cat_dir):
                    full_path = os.path.join(cat_dir, d)
                    if os.path.isdir(full_path):
                        all_runs.append(full_path)
        
        if not all_runs:
            raise FileNotFoundError
        
        # Sort strictly chronologically by modification time
        all_runs.sort(key=lambda x: os.path.getmtime(x))
        
        # The newest run is at the end of the list
        run_dir = all_runs[-1]
        latest_run_name = os.path.basename(run_dir)
        print(f"✅ Automatically selected latest run: {latest_run_name}")
        
    except (FileNotFoundError, OSError):
        print(f"Error: Could not automatically find any run directories in {runs_base_dir}/")
        print("Usage: python ml/animate.py [path/to/ml/runs/specific_run]")
        sys.exit(1)
else:
    run_dir = sys.argv[1]

# 2. Find all rollout comparison files in the run directory
rollout_files = glob.glob(os.path.join(run_dir, "rollout_*.npz"))

if not rollout_files:
    print(f"Error: No rollout_*.npz files found in {run_dir}")
    sys.exit(1)

print(f"Found {len(rollout_files)} rollout dataset(s) for video generation.")

# Create the videos folder if it doesn't exist
videos_dir = os.path.join(run_dir, "videos")
os.makedirs(videos_dir, exist_ok=True)

# Loop over each rollout file and generate the comparison video
for data_path in rollout_files:
    ic_name = os.path.basename(data_path).replace("rollout_", "").replace(".npz", "")
    print(f"\nProcessing comparison video for initial condition: {ic_name}")
    
    # Load the data
    data = np.load(data_path)
    rollout = data['rollout']      # ML predictions
    numerical = data['numerical']  # Ground truth numerical
    
    # 3. Setup the Matplotlib Canvas
    fig, ax = plt.subplots(figsize=(10, 5))
    x_axis = np.arange(rollout.shape[1])
    
    ax.set_xlim(0, rollout.shape[1])
    
    # Determine stable y-limits with margins
    y_min = float(np.min(numerical)) - 0.1
    y_max = float(np.max(numerical)) + 0.1
    ax.set_ylim(y_min, y_max)
    ax.grid(True)
    ax.set_xlabel("Spatial Grid")
    ax.set_ylabel("Amplitude")
    
    # Initialize the lines (Analytical in thick green halo, Numerical in solid blue, ML in dashed red)
    line_ana, = ax.plot([], [], lw=6, color='green', alpha=0.35, linestyle='-', label='Analytical (Perfect Shift)')
    line_num, = ax.plot([], [], lw=2, color='blue', label='Numerical (True Physics)')
    line_ml, = ax.plot([], [], lw=2, color='red', linestyle='--', label='ML Predicted Rollout')
    ax.legend(loc="upper right")
    
    # 4. Animation Functions
    def init():
        line_ana.set_data([], [])
        line_num.set_data([], [])
        line_ml.set_data([], [])
        return line_ana, line_num, line_ml,
    
    def animate(i):
        # The true analytical operator shifts by exactly 2 cells per recorded step
        y_ana = np.roll(numerical[0], shift=2 * i)
        y_num = numerical[i]
        y_ml = rollout[i]
        
        line_ana.set_data(x_axis, y_ana)
        line_num.set_data(x_axis, y_num)
        line_ml.set_data(x_axis, y_ml)
        
        # Compute real-time L2 error between prediction and numerical solution
        l2_err = np.sqrt(np.mean((y_num - y_ml) ** 2))
        ax.set_title(f"Comparison ({ic_name}) | Timestep: {i} | L2 Error: {l2_err:.5f}")
        return line_ana, line_num, line_ml,
    
    # 5. Render and Save
    print(f"Rendering comparison video for {ic_name}... This might take a minute.")
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                   frames=rollout.shape[0], interval=20, blit=True)
    
    save_path = os.path.join(videos_dir, f"comparison_{ic_name}.mp4")
    
    # Save the animation using ffmpeg
    anim.save(save_path, writer='ffmpeg', fps=30)
    print(f"✅ Comparison video successfully saved to {save_path}")
    plt.close(fig)

print("\n🎉 All comparison videos have been successfully rendered!")