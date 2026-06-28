from src import boundaries, operators, equations, integrators, init_conditions

# Define grid configuration for 2D (Optimized grid size & stable dt for fast smooth rendering)
grid_configs = [
    {"N": (60, 60), "dx": (0.3, 0.3), "dt": 0.01}
]

initial_conditions = [init_conditions.GaussianIC(sigma=2.0)]
boundary_functions = [boundaries.edge]
operators_list = [operators.laplacian]
equations_list = [equations.DiffusionEquation(diffusivity=1.0)]
integrators_list = [integrators.rk4]

FINAL_TIME = 10.0         # Fast diffusion to flat state
STEPS_PER_FRAME = 3      # Balanced for smooth fluid evolution and speed
RECORD_INTERVAL = 10

# Toggle to True to bypass data export and run the visual dashboard in the main thread
VISUAL_MODE = True
