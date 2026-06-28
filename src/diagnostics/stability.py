import numpy as np
from src.numerics import operators

def tracking(state, grid, boundary, equation):
    """
    Computes generic physical stability metrics (potential, kinetic, and total energies).
    
    This diagnostic module now delegates physics-specific calculations to the Equation class, 
    adhering strictly to dimension-independent Grid and Field abstractions.
    """
    if hasattr(equation, 'compute_energies'):
        return equation.compute_energies(state, boundary)
    
    # Generic fallback (L2 norm) if equation does not provide compute_energies
    dV = np.prod(grid.spacing)
    data = state.data if hasattr(state, 'data') else np.asarray(state)
    total_e = np.sum(data**2) * dV
    return 0.0, 0.0, total_e