import numpy as np
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions

# ── Grid ──────────────────────────────────────────────────────────────────────
# 100×100 points, dx=dy=0.05  →  domain [0, 5] × [0, 5]
# sigma=0.3 gives sigma/dx=6 (well-resolved) and sigma/L=0.06 (genuinely local)
grid_configs = [
    {
        "N":  (100, 100),
        "dx": (0.05, 0.05),
        "dt": 0.05,
    }
]

# ── Initial condition ─────────────────────────────────────────────────────────
initial_conditions = [init_conditions.BarotropicVoricityDoubleGaussianIC(offset=0.2, sigma=0.3, amplitude=1.0, num_fields=1, active_field=0, speed=0.0)]

# ── Numerics ──────────────────────────────────────────────────────────────────
boundary_functions = [boundaries.Dirichlet(left_val=0.0, right_val=0.0)]
operators_list     = [operators.gradient]
integrators_list   = [integrators.rk4]

equations_list = [equations.BarotropicVorticity(beta=0.0, nu=0.0)]

# ── Run parameters ────────────────────────────────────────────────────────────
# 20 forcing periods (T=2 each)
FINAL_TIME      = 100
STEPS_PER_FRAME = 10     # one frame every 0.2 time units
RECORD_INTERVAL = 2
VISUAL_MODE     = True
