#Tool imports
import itertools
import numpy as np
import pandas as pd
import os
import csv
import math

#Module imports
from Core import boundaries, solver, operators, equations, integrators
from Core import init_conditions

#Define stable grid configurations (protects CFL stability)
grid_configs = [
    {"N": 5000, "dx": 0.5, "dt": 0.05},
    '''{"N": 5000, "dx": 0.05, "dt": 0.005},
    {"N": 10000, "dx": 0.025, "dt": 0.0025},
    {"N": 20000, "dx": 0.0125, "dt": 0.00125}'''
]

#Define custom conditions for automated pipeline, as many parameters as required
initial_conditions = [init_conditions.wave_gauss]
boundary_functions = [boundaries.periodic]
operators_list = [operators.laplacian]
equations_list = [equations.wave]
integrators_list = [integrators.rk4]
coefficients = [1.0]

#Generates a cartesian product using the given conditions
param_combinations = list(itertools.product(
    grid_configs,
    initial_conditions,
    boundary_functions,
    operators_list,
    equations_list,
    integrators_list,
    coefficients
))

#Output directory
output_dir = 'pipeline_results'
os.makedirs(output_dir, exist_ok=True)

#CSV file structure
csv_columns = [
    'Equation', 'N', 'Initial Condition', 'Boundary Function', 'DX', 'DT',
    'L2 Error', 'Avg L2 Error', 'Median L2 Error',
    'L1 Error', 'Avg L1 Error', 'Median L1 Error', 'Peak Max Error',
    'log(dx)', 'log(L2)', 'log(mean L2)', 'log(median L2)', 
    'log(L1)', 'log(mean L1)', 'log(median L1)'
]

master_csv_path = os.path.join(output_dir, 'master_metrics.csv')
with open(master_csv_path, 'w', newline='') as master_file:
    csv_writer = csv.DictWriter(master_file, fieldnames=csv_columns)
    csv_writer.writeheader()

# Safe log function to prevent runtime crashes if error hits 0
def safe_log(val):
    return math.log(max(val, 1e-16))

#Loops through each combination and passes it to the solver
for combo in param_combinations:
    grid, ic, bc, op, eq, int_func, coeff = combo
    N = grid["N"]
    dx = grid["dx"]
    dt = grid["dt"]
    
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
        FINAL_TIME=500,
        STEPS_PER_FRAME=200
    )
    
    #Definition of required metrics
    
    #Extract data
    x_grid = sim_output["x"]
    u_final_num = sim_output["final_numerical"]
    u_final_ana = sim_output["final_analytic"]
    run_history_df = sim_output["history_dataframe"]
    
    #Calculate errors
    l2 = run_history_df['l2_error'].iloc[-1]
    avg_l2 = run_history_df['l2_error'].mean()
    median_l2 = run_history_df['l2_error'].median()
    
    l1 = run_history_df['l1_error'].iloc[-1]
    avg_l1 = run_history_df['l1_error'].mean()
    median_l1 = run_history_df['l1_error'].median()
    
    peak_max_error = run_history_df['max_error'].max()

    #Create easily accessible directory structure
    run_folder_name = f"{eq_name}_{int_name}_{op_name}_{ic_name}_{bc_name}_N{N}_dt{dt}_dx{dx}"
    run_dir_path = os.path.join(output_dir, run_folder_name)
    os.makedirs(run_dir_path, exist_ok=True)

    #Row data
    row_data = {
        'Equation': eq_name,
        'N': N,
        'Initial Condition': ic_name,
        'Boundary Function': bc_name,
        'DX': dx,
        'DT': dt,
        'L2 Error': l2,
        'Avg L2 Error': avg_l2,
        'Median L2 Error': median_l2,
        'L1 Error': l1,
        'Avg L1 Error': avg_l1,
        'Median L1 Error': median_l1,
        'Peak Max Error': peak_max_error,
        'log(dx)': safe_log(dx),
        'log(L2)': safe_log(l2),
        'log(mean L2)': safe_log(avg_l2),
        'log(median L2)': safe_log(median_l2),
        'log(L1)': safe_log(l1),
        'log(mean L1)': safe_log(avg_l1),
        'log(median L1)': safe_log(median_l1)
    }
    
    #Master CSV with metrics of all runs
    with open(master_csv_path, 'a', newline='') as master_file:
        csv_writer = csv.DictWriter(master_file, fieldnames=csv_columns)
        csv_writer.writerow(row_data)
    
    #Localized metrics for each run
    local_csv_path = os.path.join(run_dir_path, 'metrics.csv')
    with open(local_csv_path, 'w', newline='') as local_file:
        csv_writer = csv.DictWriter(local_file, fieldnames=csv_columns)
        csv_writer.writeheader()
        csv_writer.writerow(row_data)
    
    #Save frame histories
    run_history_df.to_csv(os.path.join(run_dir_path, "time_history.csv"), index=False)
    
    #Save spatial grid histories
    np.savez_compressed(
        os.path.join(run_dir_path, "spatial_data.npz"),
        x=sim_output["x"], 
        u_numerical=sim_output["final_numerical"], 
        u_analytical=sim_output["final_analytic"]
    )
    
    
print(f"\nPipeline Complete. Datasets successfully written to '{output_dir}/'")