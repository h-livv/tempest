import itertools
import numpy as np
import pandas as pd
import os
import csv
import math

from Validation import validation
from Core import boundaries
from Core import solver
from Core import boundaries
from Core import operators
from Core import equations
from Core import integrators
from Experiments import init_conditions

N_value = [2500]
initial_condition = [init_conditions.wave_gauss]
boundary_function = [boundaries.periodic]
operator = [operators.laplacian]
equation = [equations.wave]
integrator = [integrators.rk4]
coeff = [1.0]
dt = [0.01]
dx = [0.1]

#For convergence study
'''grid_configs = [
    {"N": 2500, "dx": 0.1, "dt": 0.01},
    {"N": 5000, "dx": 0.05, "dt": 0.005},
    {"N": 10000, "dx": 0.025, "dt": 0.0025},
    {"N": 20000, "dx": 0.0125, "dt": 0.00125}
]

param_combinations = list(itertools.product(
    grid_configs,
    initial_condition,
    boundary_function,
    operator,
    equation,
    integrator,
    coeff
))'''

param_combinations = list(itertools.product(N_value, initial_condition, boundary_function, operator, equation, integrator, coeff, dt, dx))

output_dir = 'pipeline_results'
spatial_dir = os.path.join(output_dir, 'spatial_history')
history_dir = os.path.join(output_dir, 'time_histories')
os.makedirs(spatial_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

csv_path = os.path.join(output_dir, 'metrics.csv')
csv_columns = [
    'Equation',
    'N',
    'Initial Condition',
    'Boundary Function',
    'DX',
    'L2 Error',
    'Avg L2 Error',
    'Median L2 Error',
    'L1 Error',
    'Avg L1 Error',
    'Median L1 Error',
    'Peak Max Error',
    'log(dx)',
    'log(mean L2)',
    'log(median L2)',
    'log(mean L1)'
]

with open(csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

for combo in param_combinations:
    N, ic, bc, op, eq, int_func, coeff, dt, dx = combo
    '''grid, ic, bc, op, eq, int_func, coeff = combo

    N = grid["N"]
    dx = grid["dx"]
    dt = grid["dt"]'''
    
    eq_name = eq.__name__
    ic_name = ic.__name__
    bc_name = bc.__name__
    
    print(f"\nPipeline launching job: Run {eq_name} | N={N} | BC={bc_name} | DX={dx}")
    
    sim_output = solver.solver(
        N=N, 
        init_state=ic, 
        boundary=bc, 
        operator=op, 
        equation=eq, 
        integrator=int_func, 
        coefficient=coeff, 
        dt=dt, 
        dx=dx
    )
    
    x_grid = sim_output["x"]
    u_final_num = sim_output["final_numerical"]
    u_final_ana = sim_output["final_analytic"]
    run_history_df = sim_output["history_dataframe"]
    l2 = run_history_df['l2_error'].iloc[-1]
    avg_l2 = run_history_df['l2_error'].mean()
    median_l2 = run_history_df['l2_error'].median()
    l1 = run_history_df['l1_error'].iloc[-1]
    avg_l1 = run_history_df['l1_error'].mean()
    median_l1 = run_history_df['l1_error'].median()
    
    peak_max_error = run_history_df['max_error'].max()

    
    with open(csv_path, 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        csv_writer.writerow({
            'Equation': eq_name,
            'N': N,
            'Initial Condition': ic_name,
            'Boundary Function': bc_name,
            'DX': dx,
            'L2 Error': l2,
            'Avg L2 Error': avg_l2,
            'Median L2 Error': median_l2,
            'L1 Error': l1,
            'Avg L1 Error': avg_l1,
            'Median L1 Error': median_l1,
            'Peak Max Error': peak_max_error,
            'log(mean L2)': math.log(avg_l2),
            'log(median L2)': math.log(median_l2),
            'log(mean L1)': math.log(avg_l1),
            'log(median L2)': math.log(median_l2),
            'log(dx)': math.log(dx),
        })
        
    history_filename = f"history_{eq_name}_N{N}_bc_{bc_name}_dx{dx}.csv"
    run_history_df.to_csv(os.path.join(history_dir, history_filename), index=False)
    
    spatial_filename = f"spatial_{eq_name}_N{N}_bc_{bc_name}_dx{dx}.npz"
    np.savez_compressed(
        os.path.join(spatial_dir, spatial_filename),
        x=x_grid, 
        u_numerical=u_final_num, 
        u_analytical=u_final_ana
    )

print(f"\n✅ Pipeline Complete! Datasets successfully written to '{output_dir}/'")