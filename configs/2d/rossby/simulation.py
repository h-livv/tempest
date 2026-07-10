import numpy as np
from src.mesh import boundaries
from src.numerics import operators, integrators
from src.physics import equations, init_conditions
from src.physics.sources import ZeroMeanGaussianSource

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
# Zero vorticity: let the source build the wave field entirely from scratch.
initial_conditions = [init_conditions.ConstantIC(val=0.0, num_fields=1)]

# ── Numerics ──────────────────────────────────────────────────────────────────
boundary_functions = [boundaries.Dirichlet(left_val=0.0, right_val=0.0)]
operators_list     = [operators.upwind]
integrators_list   = [integrators.rk4]

# ── Forcing ───────────────────────────────────────────────────────────────────
# ZeroMeanGaussianSource: profile = G(x,y) − mean(G)
#   → ∫∫ F dx dy = 0 for all t (no net PV injected)
#
# Parameters chosen for classical β-plane Rossby wave radiation:
#   center_ratio=0.6  source east of center, leaving ~3 units for westward waves
#   sigma=0.3         6 grid cells per e-folding; ~6 % of domain width
#   beta=1.0          non-dimensional planetary vorticity gradient
#   omega=pi          forcing period T = 2 time units
#   amplitude=1.0     moderate forcing; q stays O(1) over the run
_source = ZeroMeanGaussianSource(
    center_ratio=0.5,
    sigma=0.1,
    amplitude=0.25,
    omega=(np.pi),
    num_fields=1,
    active_field=0,
)

equations_list = [equations.RossbyWave(beta=1.0, source=_source)]

# ── Run parameters ────────────────────────────────────────────────────────────
# 20 forcing periods (T=2 each) → ample time for Rossby wave trains to develop
FINAL_TIME      = 100
STEPS_PER_FRAME = 10     # one frame every 0.2 time units
RECORD_INTERVAL = 2
VISUAL_MODE     = True
