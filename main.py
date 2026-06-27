import os
# 1. Address NumPy Thread Contention (CRITICAL)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import importlib.util
import sys
import itertools
import numpy as np
import csv
import math
import json
import datetime
import concurrent.futures

# Module imports
from src import solver
from diagnostics.plots import TempestPlotter
from ml.registry import log_run

ist_timezone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
timestamp = datetime.datetime.now(ist_timezone).strftime("%Y%m%d_%H%M%S")

# Loads the configuration profile
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

def _init_convergence_entry():
    return {
        "dx_values": [],
        "metrics": {name: [] for name in TempestPlotter.CONVERGENCE_METRICS},
    }

def safe_log(val):
    return math.log(max(val, 1e-16))

# 2. Encapsulate the Simulation Logic
def run_single_simulation(params):
    combo, output_dir, parent_sweep_dir, timestamp, is_sweep, final_time, steps_per_frame, record_interval = params
    grid, ic, bc, op, eq, int_func, coeff = combo
    N = grid["N"]
    dx = grid["dx"]
    dt = grid["dt"]

    eq_name = eq.__name__
    int_name = int_func.__name__
    op_name = op.__name__
    ic_name = ic.__name__
    bc_name = bc.__name__

    print(f'''Pipeline launching job: Running {eq_name} | Integrator={int_name} | Operator={op_name} | Initial state={ic_name} | Boundary={bc_name} | N={N} | dt={dt} | dx={dx}''')

    sim_output = solver.solver(
        N=N,
        init_state=ic,
        boundary=bc,
        operator=op,
        equation=eq,
        integrator=int_func,
        coefficient=coeff,
        dt=dt,
        dx=dx,
        FINAL_TIME=final_time,
        STEPS_PER_FRAME=steps_per_frame,
        RECORD_INTERVAL=record_interval,
    )

    run_history_df = sim_output["history_dataframe"]

    # Calculate errors
    l2 = run_history_df['l2_error'].iloc[-1]
    avg_l2 = run_history_df['l2_error'].mean()
    median_l2 = run_history_df['l2_error'].median()

    l1 = run_history_df['l1_error'].iloc[-1]
    avg_l1 = run_history_df['l1_error'].mean()
    median_l1 = run_history_df['l1_error'].median()

    peak_max_error = run_history_df['max_error'].max()

    metric_values = {
        "avg_l2": avg_l2,
        "final_l2": l2,
        "avg_l1": avg_l1,
        "final_l1": l1,
    }

    is_diffusion = "diff" in eq_name.lower()
    if is_diffusion:
        stability_ratio = dt / (dx ** 2)
    else:
        stability_ratio = dt / dx
    group_key = (eq_name, int_name, op_name, ic_name, bc_name, round(stability_ratio, 6))

    clean_eq_name = str(eq_name).replace(" ", "_").lower()
    run_dir_name = f"{clean_eq_name}_{int_name}_{op_name}_{ic_name}_N{N}_{timestamp}"
    if is_sweep:
        run_dir_path = os.path.join(parent_sweep_dir, run_dir_name)
    else:
        run_dir_path = os.path.join(output_dir, "runs", run_dir_name)
    os.makedirs(run_dir_path, exist_ok=True)

    log_run("sweep" if is_sweep else "single", "main", eq_name, op_name, ic_name, run_dir_path, metadata={"N": N, "dt": dt, "dx": dx})

    plotter = TempestPlotter(output_dir=run_dir_path)
    if int_name.startswith("lax") or int_name in ["maccormack"]:
        display_solver_name = int_name
    else:
        display_solver_name = f"{int_name} + {op_name}"

    plotter.plot_validation(
        time_history_df=run_history_df,
        eq_name=eq_name,
        solver_name=display_solver_name,
        run_id="transient_errors",
        N=N,
        dx=dx,
        dt=dt,
        x=sim_output["x"],
        u_numerical=sim_output["final_numerical"],
        u_analytical=sim_output["final_analytic"],
    )

    row_data = {
        'Equation': eq_name, 'N': N, 'Initial Condition': ic_name, 'Boundary Function': bc_name,
        'DX': dx, 'DT': dt, 'L2 Error': l2, 'Avg L2 Error': avg_l2, 'Median L2 Error': median_l2,
        'L1 Error': l1, 'Avg L1 Error': avg_l1, 'Median L1 Error': median_l1, 'Peak Max Error': peak_max_error,
        'log(dx)': safe_log(dx), 'log(L2)': safe_log(l2), 'log(mean L2)': safe_log(avg_l2), 'log(median L2)': safe_log(median_l2),
        'log(L1)': safe_log(l1), 'log(mean L1)': safe_log(avg_l1), 'log(median L1)': safe_log(median_l1)
    }

    local_csv_columns = list(row_data.keys())
    local_csv_path = os.path.join(run_dir_path, 'metrics.csv')
    with open(local_csv_path, 'w', newline='') as local_file:
        csv_writer = csv.DictWriter(local_file, fieldnames=local_csv_columns)
        csv_writer.writeheader()
        csv_writer.writerow(row_data)

    run_history_df.to_csv(os.path.join(run_dir_path, "time_history.csv"), index=False)

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
    with open(os.path.join(run_dir_path, "config.json"), "w") as f:
        json.dump(config_data, f, indent=4)

    np.savez_compressed(
        os.path.join(run_dir_path, "spatial_data.npz"),
        x=sim_output["x"],
        u_numerical=sim_output["final_numerical"],
        u_analytical=sim_output["final_analytic"],
        ml_tensor_data=sim_output["raw_tensor_data"]
    )
    
    # Return data to be aggregated sequentially by main thread
    return {
        "row_data": row_data,
        "group_key": group_key,
        "dx": dx,
        "metric_values": metric_values
    }


# 4. The Multiprocessing Guard (CRITICAL)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Project Tempest: Pipeline Execution")
    parser.add_argument("config", help="Path to the config profile script (e.g., Configs/wave_sweep.py)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    convergence_metrics = getattr(cfg, "CONVERGENCE_METRICS", ["avg_l2", "avg_l1"])

    param_combinations = list(itertools.product(
        cfg.grid_configs,
        cfg.initial_conditions,
        cfg.boundary_functions,
        cfg.operators_list,
        cfg.equations_list,
        cfg.integrators_list,
        cfg.coefficients
    ))

    output_dir = 'pipeline_results'
    os.makedirs(output_dir, exist_ok=True)

    csv_columns = [
        'Equation', 'N', 'Initial Condition', 'Boundary Function', 'DX', 'DT',
        'L2 Error', 'Avg L2 Error', 'Median L2 Error',
        'L1 Error', 'Avg L1 Error', 'Median L1 Error', 'Peak Max Error',
        'log(dx)', 'log(L2)', 'log(mean L2)', 'log(median L2)',
        'log(L1)', 'log(mean L1)', 'log(median L1)'
    ]

    master_csv_path = os.path.join(output_dir, 'master_metrics.csv')
    file_exists = os.path.exists(master_csv_path)

    if not file_exists:
        with open(master_csv_path, 'w', newline='') as master_file:
            csv_writer = csv.DictWriter(master_file, fieldnames=csv_columns)
            csv_writer.writeheader()
        print(f"Initialized new global metrics database at '{master_csv_path}'")
    else:
        print(f"Existing global database found. Appending run data to '{master_csv_path}'")

    convergence_tracker = {}

    is_sweep = len(param_combinations) > 1
    
    if is_sweep:
        base_eq_name = param_combinations[0][4].__name__
        base_op_name = param_combinations[0][3].__name__
        base_int_name = param_combinations[0][5].__name__
        ic_names = "_".join(sorted(list(set([combo[1].__name__ for combo in param_combinations]))))
        parent_sweep_dir = os.path.join(output_dir, "sweeps", f"{base_eq_name}_{base_int_name}_{base_op_name}_{ic_names}_sweep_{timestamp}".lower())
        os.makedirs(parent_sweep_dir, exist_ok=True)
    else:
        parent_sweep_dir = None

    final_time = getattr(cfg, "FINAL_TIME", 2000)
    steps_per_frame = getattr(cfg, "STEPS_PER_FRAME", 50)
    record_interval = getattr(cfg, "RECORD_INTERVAL", 1)

    # Package all necessary parameters for ProcessPoolExecutor mapping
    execution_params = []
    for combo in param_combinations:
        execution_params.append((
            combo, output_dir, parent_sweep_dir, timestamp, is_sweep, 
            final_time, steps_per_frame, record_interval
        ))

    results = []

    # 3. Implement the Parallel Execution Pool
    if is_sweep:
        max_workers = max(1, os.cpu_count() - 2)
        print(f"\nInitiating parallel sweep across {max_workers} CPU cores...\n")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map returns results in the order the iterables were given
            for res in executor.map(run_single_simulation, execution_params):
                results.append(res)
    else:
        print(f"\nInitiating single run sequentially...\n")
        results.append(run_single_simulation(execution_params[0]))

    # Aggregate results synchronously to avoid race conditions on the master CSV and convergence tracker
    with open(master_csv_path, 'a', newline='') as master_file:
        csv_writer = csv.DictWriter(master_file, fieldnames=csv_columns)
        for res in results:
            csv_writer.writerow(res["row_data"])
            
            group_key = res["group_key"]
            if group_key not in convergence_tracker:
                convergence_tracker[group_key] = _init_convergence_entry()
            
            convergence_tracker[group_key]["dx_values"].append(res["dx"])
            for metric_name, metric_value in res["metric_values"].items():
                convergence_tracker[group_key]["metrics"][metric_name].append(metric_value)

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
        group_plot_dir = os.path.join(parent_sweep_dir if parent_sweep_dir else output_dir, f"{eq_n}_{int_n}_{op_n}_{ic_n}_convergence".lower())
        plotter = TempestPlotter(output_dir=group_plot_dir)

        print(
            f"Executing log-log regression for: {study_name} "
            f"(CFL dt/dx={cfl_n})"
        )

        if "upwind" in op_n.lower():
            target_order = 1
        elif "shallow_dam" in ic_n.lower():
            target_order = {"avg_l1": 1.0, "avg_l2": 0.5, "final_l1": 1.0, "final_l2": 0.5}
        else:
            target_order = 2

        if int_n.startswith("lax") or int_n in ["maccormack"]:
            display_solver_name = int_n
        else:
            display_solver_name = f"{int_n} + {op_n}"

        plotter.plot_convergence_suite(
            dx_values=data["dx_values"],
            metrics_map=data["metrics"],
            eq_name=study_name,
            expected_order=target_order,
            metrics=convergence_metrics,
            title_display_name=f"{eq_n} ({display_solver_name})",
        )

    print(f"\nPipeline Complete. Datasets successfully written to '{output_dir}/'")
