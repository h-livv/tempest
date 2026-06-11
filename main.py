import argparse
import importlib.util
import sys
import itertools
import numpy as np
import os
import csv
import math

# Module imports
from src import solver
from diagnostics.plots import TempestPlotter

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


def main():
    # 1. Coordinate command-line execution args
    parser = argparse.ArgumentParser(description="Project Tempest: Pipeline Execution")
    parser.add_argument("config", help="Path to the config profile script (e.g., Configs/wave_sweep.py)")
    args = parser.parse_args()

    # 2. Dynamically load the configuration profile
    cfg = load_config(args.config)

    # Metrics to include in automated convergence plots (matches manual study docs).
    convergence_metrics = getattr(cfg, "CONVERGENCE_METRICS", ["avg_l2", "avg_l1"])

    # 3. Pull parameter lists directly from the injected module
    param_combinations = list(itertools.product(
        cfg.grid_configs,
        cfg.initial_conditions,
        cfg.boundary_functions,
        cfg.operators_list,
        cfg.equations_list,
        cfg.integrators_list,
        cfg.coefficients
    ))

    # Output directory
    output_dir = 'pipeline_results'
    os.makedirs(output_dir, exist_ok=True)

    # CSV file structure
    csv_columns = [
        'Equation', 'N', 'Initial Condition', 'Boundary Function', 'DX', 'DT',
        'L2 Error', 'Avg L2 Error', 'Median L2 Error',
        'L1 Error', 'Avg L1 Error', 'Median L1 Error', 'Peak Max Error',
        'log(dx)', 'log(L2)', 'log(mean L2)', 'log(median L2)',
        'log(L1)', 'log(mean L1)', 'log(median L1)'
    ]

    master_csv_path = os.path.join(output_dir, 'master_metrics.csv')

    # Check if the master file already exists to prevent wiping past runs
    file_exists = os.path.exists(master_csv_path)

    # ONLY write the header if this is the first time the file is being created
    if not file_exists:
        with open(master_csv_path, 'w', newline='') as master_file:
            csv_writer = csv.DictWriter(master_file, fieldnames=csv_columns)
            csv_writer.writeheader()
        print(f"Initialized new global metrics database at '{master_csv_path}'")
    else:
        print(f"Existing global database found. Appending run data to '{master_csv_path}'")

    def safe_log(val):
        return math.log(max(val, 1e-16))

    convergence_tracker = {}

    # Loops through each combination and passes it to the solver
    for combo in param_combinations:
        grid, ic, bc, op, eq, int_func, coeff = combo
        N = grid["N"]
        dx = grid["dx"]
        dt = grid["dt"]

        # Extract global settings dynamically with built-in fallbacks
        final_time = getattr(cfg, "FINAL_TIME", 2000)
        steps_per_frame = getattr(cfg, "STEPS_PER_FRAME", 50)
        record_interval = getattr(cfg, "RECORD_INTERVAL", 1)

        eq_name = eq.__name__
        int_name = int_func.__name__
        op_name = op.__name__
        ic_name = ic.__name__
        bc_name = bc.__name__

        print(f'''\nPipeline launching job: Running {eq_name} | Integrator={int_name} | Operator={op_name} | Initial state={ic_name} |
              Boundary={bc_name} | N={N} | dt={dt} | dx={dx}''')

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

        # Group by scheme + IC + BC + CFL ratio to avoid mixing incompatible sweeps.
        is_diffusion = "diff" in eq_name.lower()
        
        if is_diffusion:
            # Quadratic scaling for diffusion: dt / dx^2
            stability_ratio = dt / (dx ** 2)
        else:
            # Linear scaling for hyperbolic/advective equations: dt / dx
            stability_ratio = dt / dx
        group_key = (eq_name, int_name, op_name, ic_name, bc_name, round(stability_ratio, 6))
        
        if group_key not in convergence_tracker:
            convergence_tracker[group_key] = _init_convergence_entry()

        convergence_tracker[group_key]["dx_values"].append(dx)
        for metric_name, metric_value in metric_values.items():
            convergence_tracker[group_key]["metrics"][metric_name].append(metric_value)

        # Per-run output directory (created before plotting so plots land here).
        run_folder_name = (
            f"{eq_name}_{int_name}_{op_name}_{ic_name}_{bc_name}_N{N}_dt{dt}_dx{dx}"
        )
        run_dir_path = os.path.join(output_dir, run_folder_name)
        os.makedirs(run_dir_path, exist_ok=True)

        plotter = TempestPlotter(output_dir=run_dir_path)
        plotter.plot_validation(
            time_history_df=run_history_df,
            eq_name=eq_name,
            solver_name=int_name,
            run_id="transient_errors",
            N=N,
            dx=dx,
            dt=dt,
        )

        # Row data
        row_data = {
            'Equation': eq_name, 'N': N, 'Initial Condition': ic_name, 'Boundary Function': bc_name,
            'DX': dx, 'DT': dt, 'L2 Error': l2, 'Avg L2 Error': avg_l2, 'Median L2 Error': median_l2,
            'L1 Error': l1, 'Avg L1 Error': avg_l1, 'Median L1 Error': median_l1, 'Peak Max Error': peak_max_error,
            'log(dx)': safe_log(dx), 'log(L2)': safe_log(l2), 'log(mean L2)': safe_log(avg_l2), 'log(median L2)': safe_log(median_l2),
            'log(L1)': safe_log(l1), 'log(mean L1)': safe_log(avg_l1), 'log(median L1)': safe_log(median_l1)
        }
        # Master CSV with metrics of all runs
        with open(master_csv_path, 'a', newline='') as master_file:
            csv_writer = csv.DictWriter(master_file, fieldnames=csv_columns)
            csv_writer.writerow(row_data)

        # Localized metrics for each run
        local_csv_path = os.path.join(run_dir_path, 'metrics.csv')
        with open(local_csv_path, 'w', newline='') as local_file:
            csv_writer = csv.DictWriter(local_file, fieldnames=csv_columns)
            csv_writer.writeheader()
            csv_writer.writerow(row_data)

        # Save frame histories
        run_history_df.to_csv(os.path.join(run_dir_path, "time_history.csv"), index=False)

        # Save spatial grid histories
        np.savez_compressed(
            os.path.join(run_dir_path, "spatial_data.npz"),
            x=sim_output["x"],
            u_numerical=sim_output["final_numerical"],
            u_analytical=sim_output["final_analytic"]
        )

    print("\nEvaluating data for automated grid convergence studies...")
    for group_key, data in convergence_tracker.items():
        eq_n, int_n, op_n, ic_n, bc_n, cfl_n = group_key
        study_name = f"{eq_n}_{int_n}_{op_n}_{ic_n}_{bc_n}"
        group_plot_dir = os.path.join(output_dir, f"{eq_n}_{int_n}_{op_n}")
        plotter = TempestPlotter(output_dir=group_plot_dir)

        unique_dx = len(set(data["dx_values"]))
        if unique_dx < 2:
            print(
                f"Skipping convergence for {study_name}: "
                f"only {unique_dx} unique dx value(s) in this session."
            )
            continue

        print(
            f"Executing log-log regression for: {study_name} "
            f"(CFL dt/dx={cfl_n})"
        )

        target_order = 1 if "upwind" in op_n.lower() else 2

        plotter.plot_convergence_suite(
            dx_values=data["dx_values"],
            metrics_map=data["metrics"],
            eq_name=study_name,
            expected_order=target_order,
            metrics=convergence_metrics,
        )

    print(f"\nPipeline Complete. Datasets successfully written to '{output_dir}/'")


if __name__ == "__main__":
    main()
