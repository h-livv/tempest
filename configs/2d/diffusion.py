from src import boundaries, operators, equations, integrators, init_conditions

# Define grid configuration for 2D (Optimized grid size & stable dt for fast smooth rendering)
grid_configs = [
    {"N": (60, 60), "dx": (0.3, 0.3), "dt": 0.01}  # Diffusion dt is constrained by dx^2 / 4D = 0.09 / 4 = 0.0225. dt=0.01 is stable.
]

initial_conditions = [init_conditions.diff_gauss_2d]
boundary_functions = [boundaries.edge]
operators_list = [operators.laplacian]
equations_list = [equations.diffusion]
integrators_list = [integrators.rk4]
coefficients = [1.0]

FINAL_TIME = 5.0         # Fast diffusion to flat state
STEPS_PER_FRAME = 3      # Balanced for smooth fluid evolution and speed
RECORD_INTERVAL = 10

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
