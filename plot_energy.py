import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

# Find the latest wave_rk4_laplacian_wave_gauss_sweep dir
sweeps = glob.glob("pipeline_results/sweeps/wave_*_sweep_*")
sweeps.sort(key=os.path.getmtime, reverse=True)
sweep_dir = sweeps[0] if sweeps else ""

# Determine path for each integrator using wildcards for the timestamp
runs = {
    "Explicit Euler": glob.glob(os.path.join(sweep_dir, "wave_euler_*/time_history.csv")),
    "RK4": glob.glob(os.path.join(sweep_dir, "wave_rk4_*/time_history.csv")),
    "Symplectic Leapfrog": glob.glob(os.path.join(sweep_dir, "wave_leapfrog_*/time_history.csv"))
}


style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 10), dpi=150)

colors = {
    "Explicit Euler": "#cc3311", 
    "RK4": "#33aa33", 
    "Symplectic Leapfrog": "#0077bb"
}

# Pre-calculate shared y-limits for RK4 and Leapfrog to prevent auto-scaling noise
max_rel_err = 1e-6
data_store = {}

for name, path_list in runs.items():
    if path_list and os.path.exists(path_list[0]):
        full_path = path_list[0]
        df = pd.read_csv(full_path)
        
        e0 = df["energy"].iloc[0]
        e0 = e0 if e0 != 0 else 1.0
        rel_energy = (df["energy"] - e0) / e0
        data_store[name] = (df["time"], rel_energy)
        
        if name in ["RK4", "Symplectic Leapfrog"]:
            max_rel_err = max(max_rel_err, rel_energy.abs().max())

shared_ylim = (-max_rel_err * 1.2, max_rel_err * 1.2)

for name, (time_arr, rel_energy) in data_store.items():
    if name == "Explicit Euler":
        ax0.plot(time_arr, rel_energy, label=name, color=colors[name], lw=2)
        ax0.set_title("Explicit Euler (Unconditionally Unstable Explosion)", fontsize=12, fontweight="bold")
        ax0.set_ylabel(r"Relative Error $\Delta E / E_0$", fontsize=10)
        ax0.set_xlim(0, 50)
        ax0.grid(True, linestyle="--", alpha=0.5)
        ax0.legend(loc="upper left")
        
    elif name == "RK4":
        ax1.plot(time_arr, rel_energy, label=name, color=colors[name], lw=2)
        ax1.set_title("RK4 Energy Drift (4th Order Error)", fontsize=12, fontweight="bold")
        ax1.set_ylabel(r"Relative Error $\Delta E / E_0$", fontsize=10)
        ax1.set_xlim(0, 5000)
        ax1.set_ylim(shared_ylim)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")
        
    elif name == "Symplectic Leapfrog":
        ax2.plot(time_arr, rel_energy, label=name, color=colors[name], lw=2)
        ax2.set_title("Symplectic Leapfrog (Exact Hamiltonian Conservation)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Physical Time", fontsize=12)
        ax2.set_ylabel(r"Relative Error $\Delta E / E_0$", fontsize=10)
        ax2.set_xlim(0, 5000)
        ax2.set_ylim(shared_ylim)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right")

fig.suptitle("Total System Energy Drift (Wave Equation)", fontsize=16, fontweight="bold")
fig.tight_layout()

out_path = "pipeline_results/wave_energy_comparison.png"
fig.savefig(out_path, bbox_inches="tight")
print(f"Energy plot saved to {out_path}")

