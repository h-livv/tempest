import argparse
import importlib.util
import sys
import itertools
import numpy as np
import os
import csv
import math

# Module imports
from Core import solver

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

def main():
    # 1. Coordinate command-line execution args
    parser = argparse.ArgumentParser(description="Project Tempest: Pipeline Execution")
    parser.add_argument("config", help="Path to the config profile script (e.g., Configs/wave_sweep.py)")
    args = parser.parse_args()

    # 2. Dynamically load the configuration profile
    cfg = load_config(args.config)
    
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

    # Loops through each combination and passes it to the solver
    for combo in param_combinations:
        grid, ic, bc, op, eq, int_func, coeff = combo
        N = grid["N"]
        dx = grid["dx"]
        dt = grid["dt"]
        
        # Extract global settings dynamically with built-in fallbacks
        final_time = getattr(cfg, "FINAL_TIME", 2000)
        steps_per_frame = getattr(cfg, "STEPS_PER_FRAME", 50)
        
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
            STEPS_PER_FRAME=steps_per_frame
        )
        
        # Extract data
        x_grid = sim_output["x"]
        u_final_num = sim_output["final_numerical"]
        u_final_ana = sim_output["final_analytic"]
        run_history_df = sim_output["history_dataframe"]
        
        # Calculate errors
        l2 = run_history_df['l2_error'].iloc[-1]
        avg_l2 = run_history_df['l2_error'].mean()
        median_l2 = run_history_df['l2_error'].median()
        
        l1 = run_history_df['l1_error'].iloc[-1]
        avg_l1 = run_history_df['l1_error'].mean()
        median_l1 = run_history_df['l1_error'].median()
        
        peak_max_error = run_history_df['max_error'].max()

        # Create easily accessible directory structure
        run_folder_name = f"{eq_name}_{int_name}_{op_name}_{ic_name}_{bc_name}_N{N}_dt{dt}_dx{dx}"
        run_dir_path = os.path.join(output_dir, run_folder_name)
        os.makedirs(run_dir_path, exist_ok=True)

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
        
    print(f"\nPipeline Complete. Datasets successfully written to '{output_dir}/'")

if __name__ == "__main__":
    main()