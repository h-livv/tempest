#Imports
import os
import argparse
import importlib.util
import sys

# Dynamically set Matplotlib backend to Agg (headless) if VISUAL_MODE is not enabled in the config
visual_mode = False
if len(sys.argv) > 1:
    try:
        config_path = os.path.abspath(sys.argv[1])
        if os.path.exists(config_path) and config_path.endswith('.py'):
            spec = importlib.util.spec_from_file_location("temp_config", config_path)
            temp_cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_cfg)
            visual_mode = getattr(temp_cfg, "VISUAL_MODE", False)
    except Exception:
        pass

import matplotlib
if not visual_mode:
    matplotlib.use("Agg")
import itertools
import numpy as np
import csv
import math
import json
import datetime
import concurrent.futures
from pathlib import Path
from collections import defaultdict

# Module imports
from src import solver                          # kept intact; not yet removed
from src.core.config import SimulationConfig
from src.core.simulation import Simulation
from src.init_conditions import make_ic
from src.diagnostics.plots import TempestPlotter
from ml.registry import log_run

# Address NumPy Thread Contention. Limits one process to one thread.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# IST timesteps for tracking
ist_timezone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
timestamp = datetime.datetime.now(ist_timezone).strftime("%Y%m%d_%H%M%S")

# Load the configuration profile
def load_config(config_path):
    """Dynamically loads an external Python file as a module."""
    try:
        spec = importlib.util.spec_from_file_location("current_config", config_path)
        cfg = importlib.util.module_from_spec(spec)
        sys.modules["current_config"] = cfg
        spec.loader.exec_module(cfg)
        return cfg
    except Exception as e:
        print(f"Error loading configuration file '{config_path}':\n{e}")
        sys.exit(1)

# For convergence runs, creates a dictionary to store dx values and error metrics
def _init_convergence_entry():
    return {
        "dx_values": [],
        "metrics": {name: [] for name in TempestPlotter.CONVERGENCE_METRICS},
    }

def safe_log(val):
    return math.log(max(val, 1e-16))

# 
def run_single_simulation(params):

    # Unpack parameters
    (combo, output_dir, parent_sweep_dir, timestamp, is_sweep, final_time, steps_per_frame, record_interval, verbose, visual_mode) = params
    grid, ic, bc, op, eq, int_func, coeff = combo
    N = grid["N"]
    dx = grid["dx"]
    dt = grid["dt"]

    # Extract function names for logging
    eq_name = eq.__name__
    int_name = int_func.__name__
    op_name = op.__name__
    ic_name = ic.__name__
    bc_name = bc.__name__

    if verbose:
        print(f'''Pipeline launching job: Running {eq_name} | Integrator={int_name} | Operator={op_name} | Initial state={ic_name} | Boundary={bc_name} | N={N} | dt={dt} | dx={dx}''')

    # ------------------------------------------------------------------
    # Build SimulationConfig and run via Simulation
    # ------------------------------------------------------------------
    # N and dx may be scalars (1-D) or tuples (2-D); normalise to tuples.
    shape   = N  if isinstance(N,  tuple) else (N,)
    spacing = dx if isinstance(dx, tuple) else (dx,)

    sim_config = SimulationConfig(
        shape=shape,
        spacing=spacing,
        dt=dt,
        final_time=final_time,
        steps_per_frame=steps_per_frame,
        record_interval=record_interval,
        equation=eq,
        operator=op,
        boundary=bc,
        integrator=int_func,
        coefficient=coeff,
        initial_condition=make_ic(ic),
    )

    results = Simulation(sim_config).run()

    # ------------------------------------------------------------------
    # Map SimulationResults attributes to the names used below
    # (mirrors the old dict keys so the rest of the function is unchanged)
    # ------------------------------------------------------------------
    # TODO: remove _SimOutputCompat once the pipeline fully adopts
    #       SimulationResults and all downstream code reads attributes
    #       directly (results.history, results.final_numerical, etc.).
    class _SimOutputCompat:
        """Temporary compatibility adapter: maps SimulationResults attributes
        to the legacy dict-style keys used by the rest of this function."""
        def __init__(self, r):
            self.history_dataframe = r.history
            self.grid              = r.grid
            self.x                 = (
                r.grid.coordinates[0]
                if r.grid.ndim == 1
                else r.grid.coordinates
            )
            self.final_numerical   = r.final_numerical
            self.final_analytic    = r.final_analytical
            self.raw_tensor_data   = r.raw_tensor_data
            self.energy_history    = r.energy_history

        def __getitem__(self, key):       # legacy dict-style access
            return getattr(self, key)

        def get(self, key, default=None):
            return getattr(self, key, default)

    legacy_output = _SimOutputCompat(results)

    # Extract and calculate errors
    run_history_df = legacy_output["history_dataframe"]

    stats = run_history_df[['l2_error', 'l1_error']].agg(['mean', 'median'])
    l2, l1 = run_history_df['l2_error'].iloc[-1], run_history_df['l1_error'].iloc[-1]
    avg_l2, median_l2 = stats.loc['mean', 'l2_error'], stats.loc['median', 'l2_error']
    avg_l1, median_l1 = stats.loc['mean', 'l1_error'], stats.loc['median', 'l1_error']
    peak_max_error = run_history_df['max_error'].max()

    metric_values = {
        "avg_l2": stats.loc['mean', 'l2_error'],
        "final_l2": l2,
        "avg_l1": stats.loc['mean', 'l1_error'],
        "final_l1": l1,
    }

    grid_obj = legacy_output["grid"]
    char_spacing = grid_obj.characteristic_spacing()
    mesh_size = grid_obj.mesh_size()

    # Physics stability check
    spatial_order = getattr(eq, "spatial_order", 1)
    stability_ratio = dt / (char_spacing ** spatial_order)
    group_key = (eq_name, int_name, op_name, ic_name, bc_name, round(stability_ratio, 6))

    # Data packaging
    row_data = {
        'Equation': eq_name, 'N': N, 'Initial Condition': ic_name, 'Boundary Function': bc_name,
        'DX': dx, 'DT': dt, 'L2 Error': l2, 'Avg L2 Error': avg_l2, 'Median L2 Error': median_l2,
        'L1 Error': l1, 'Avg L1 Error': avg_l1, 'Median L1 Error': median_l1, 'Peak Max Error': peak_max_error,
        'log(dx)': safe_log(mesh_size), 'log(L2)': safe_log(l2), 'log(mean L2)': safe_log(avg_l2), 'log(median L2)': safe_log(median_l2),
        'log(L1)': safe_log(l1), 'log(mean L1)': safe_log(avg_l1), 'log(median L1)': safe_log(median_l1)
    }

    if not visual_mode:
        # Directory creation and file saving
        clean_eq_name = str(eq_name).replace(" ", "_").lower()
        run_dir_name = f"{clean_eq_name}_{int_name}_{op_name}_{ic_name}_N{N}_{timestamp}"
        run_dir_path = Path(output_dir) / "runs" / run_dir_name
        run_dir_path.mkdir(parents=True, exist_ok=True)
        (run_dir_path / "plots").mkdir(exist_ok=True)
        (run_dir_path / "data").mkdir(exist_ok=True)

        log_run("sweep" if is_sweep else "single", "main", eq_name, op_name, ic_name, run_dir_path, metadata={"N": N, "dt": dt, "dx": dx})

        #Plotting
        plotter = TempestPlotter(output_dir=run_dir_path / "plots")
        display_solver_name = int_name if getattr(int_func, "is_direct_solver", False) else f"{int_name} + {op_name}"

        plotter.plot_validation(
            time_history_df=run_history_df,
            eq_name=eq_name,
            solver_name=display_solver_name,
            run_id="transient_errors",
            N=N,
            dx=dx,
            dt=dt,
            x=legacy_output["x"],
            u_numerical=legacy_output["final_numerical"],
            u_analytical=legacy_output["final_analytic"],
            raw_tensor_data=legacy_output.get("raw_tensor_data"),
            energy_history=legacy_output.get("energy_history"),
        )

        # Disk writing
        with open(run_dir_path / 'data' / 'metrics.csv', 'w', newline='') as local_file:
            csv_writer = csv.DictWriter(local_file, fieldnames=list(row_data.keys()))
            csv_writer.writeheader()
            csv_writer.writerow(row_data)

        run_history_df.to_csv(run_dir_path / "data" / "time_history.csv", index=False)

        # Config data
        config_data = {
            "equation": eq_name,
            "integrator": int_name,
            "operator": op_name,
            "initial_condition": ic_name,
            "boundary_function": bc_name,
            "N": N,
            "dt": dt,
            "dx": dx,
            "coefficient": coeff,
            "final_time": final_time,
            "steps_per_frame": steps_per_frame,
            "record_interval": record_interval
        }

        with open(run_dir_path / "data" / "config.json", "w") as f:
            json.dump(config_data, f, indent=4)

        np.savez_compressed(
            run_dir_path / "data" / "spatial_data.npz",
            x=legacy_output["x"], 
            u_numerical=legacy_output["final_numerical"],
            u_analytical=legacy_output["final_analytic"], 
            ml_tensor_data=legacy_output["raw_tensor_data"]
        )
    
    # Return data
    return {
        "row_data": row_data,
        "group_key": group_key,
        "mesh_size": mesh_size,
        "metric_values": metric_values,
        "operator_order": getattr(op, "convergence_order", 2),
        "ic_order": getattr(ic, "convergence_order", None),
        "is_direct": getattr(int_func, "is_direct_solver", False)
    }


# The Multiprocessing Guard
if __name__ == '__main__':

    # Run a custom config file
    parser = argparse.ArgumentParser(description="Project Tempest: Pipeline Execution")
    parser.add_argument("config", help="Path to the config profile script")
    args = parser.parse_args()

    cfg = load_config(args.config)

    visual_mode = getattr(cfg, "VISUAL_MODE", False)
    verbose = not visual_mode
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    convergence_metrics = getattr(cfg, "CONVERGENCE_METRICS", ["avg_l2", "avg_l1"])

    # Cartesian product of all configs
    param_combinations = list(itertools.product(
        cfg.grid_configs,
        cfg.initial_conditions,
        cfg.boundary_functions,
        cfg.operators_list,
        cfg.equations_list,
        cfg.integrators_list,
        cfg.coefficients
    ))

    output_dir = Path('pipeline_results')
    output_dir.mkdir(exist_ok=True)

    # Write to master csv
    csv_columns = [
        'Equation', 'N', 'Initial Condition', 'Boundary Function', 'DX', 'DT',
        'L2 Error', 'Avg L2 Error', 'Median L2 Error',
        'L1 Error', 'Avg L1 Error', 'Median L1 Error', 'Peak Max Error',
        'log(dx)', 'log(L2)', 'log(mean L2)', 'log(median L2)',
        'log(L1)', 'log(mean L1)', 'log(median L1)'
    ]

    master_csv_path = output_dir / 'master_metrics.csv'

    if not master_csv_path.exists():
        with open(master_csv_path, 'w', newline='') as master_file:
            csv.DictWriter(master_file, fieldnames=csv_columns).writeheader()
        vprint(f"Initialized new global metrics database at '{master_csv_path}'")
    else:
        vprint(f"Existing global database found. Appending run data to '{master_csv_path}'")

    convergence_tracker = defaultdict(_init_convergence_entry)

    is_sweep = len(param_combinations) > 1
    parent_sweep_dir = None

    final_time = getattr(cfg, "FINAL_TIME", 2000)
    steps_per_frame = getattr(cfg, "STEPS_PER_FRAME", 50)
    record_interval = getattr(cfg, "RECORD_INTERVAL", 1)

    # Package all necessary parameters for ProcessPoolExecutor mapping
    execution_params = [
        (combo, str(output_dir), str(parent_sweep_dir) if parent_sweep_dir else None, 
         timestamp, is_sweep, final_time, steps_per_frame, record_interval, verbose, visual_mode)
        for combo in param_combinations
    ]

    if visual_mode:
        # Run visual simulation directly in the main thread and skip CSV updates/convergence steps
        run_single_simulation(execution_params[0])
        import sys; sys.exit(0)

    vprint(f"\nInitiating {'parallel sweep' if is_sweep else 'single run'}...\n")

    # Open the CSV once in append mode for the duration of the execution
    with open(master_csv_path, 'a', newline='') as master_file:
        csv_writer = csv.DictWriter(master_file, fieldnames=csv_columns)
        
        if is_sweep:
            max_workers = max(1, os.cpu_count() - 2)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks to the pool
                futures = {executor.submit(run_single_simulation, p): p for p in execution_params}
                
                # Yield results as they finish
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    
                    # Write to disk immediately
                    csv_writer.writerow(res["row_data"])
                    master_file.flush() # Force write to prevent data loss on crash
                    
                    # Track convergence data in memory
                    group_key = res["group_key"]
                    entry = convergence_tracker[group_key]
                    if "operator_order" not in entry:
                        entry["operator_order"] = res["operator_order"]
                        entry["ic_order"] = res["ic_order"]
                        entry["is_direct"] = res["is_direct"]
                    entry["dx_values"].append(res["mesh_size"])
                    for m_name, m_val in res["metric_values"].items():
                        entry["metrics"][m_name].append(m_val)
        else:
            # Fallback for single run
            res = run_single_simulation(execution_params[0])
            csv_writer.writerow(res["row_data"])
            
            group_key = res["group_key"]
            entry = convergence_tracker[group_key]
            if "operator_order" not in entry:
                entry["operator_order"] = res["operator_order"]
                entry["ic_order"] = res["ic_order"]
                entry["is_direct"] = res["is_direct"]
            entry["dx_values"].append(res["mesh_size"])
            for m_name, m_val in res["metric_values"].items():
                entry["metrics"][m_name].append(m_val)

    print("\nEvaluating data for automated grid convergence studies...")
    for group_key, data in convergence_tracker.items():
        unique_dx = len(set(data["dx_values"]))
        if unique_dx < 2:
            print(
                f"Skipping convergence for {group_key[0]}: "
                f"only {unique_dx} unique dx value(s) in this session."
            )
            continue
            
        eq_n, int_n, op_n, ic_n, bc_n, cfl_n = group_key
        study_name = f"{eq_n}_{int_n}_{op_n}_{ic_n}_{bc_n}"
        group_plot_dir = Path(output_dir) / "runs" / f"{eq_n}_{int_n}_{op_n}_{ic_n}_convergence_{timestamp}".lower()
        (group_plot_dir / "plots").mkdir(parents=True, exist_ok=True)
        (group_plot_dir / "data").mkdir(parents=True, exist_ok=True)
        plotter = TempestPlotter(output_dir=group_plot_dir / "plots")

        print(
            f"Executing log-log regression for: {study_name} "
            f"(CFL dt/dx={cfl_n})"
        )

        target_order = data["ic_order"]
        if target_order is None:
            target_order = data["operator_order"]

        display_solver_name = int_n if data["is_direct"] else f"{int_n} + {op_n}"

        plotter.plot_convergence_suite(
            dx_values=data["dx_values"],
            metrics_map=data["metrics"],
            eq_name=study_name,
            expected_order=target_order,
            metrics=convergence_metrics,
            title_display_name=f"{eq_n} ({display_solver_name})",
        )

    print(f"\nPipeline Complete. Datasets successfully written to '{output_dir}/'")
