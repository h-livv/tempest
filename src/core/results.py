"""
SimulationResults – a plain dataclass returned by Simulation.run().

Provides attribute-style access to simulation outputs instead of a raw
dictionary.  All fields are optional except *grid* and *final_state* so
that partial results (e.g. from a failed run) can still be returned.

Fields
------
grid : Grid
    The Grid object used during the simulation.  Carries shape, spacing,
    coordinates, and domain information for every downstream consumer.

final_state : Field
    The numerical state at the end of the simulation, wrapped in the
    appropriate Field subclass (ScalarField or VectorField).

history : pd.DataFrame | None
    Time-series diagnostic data recorded at every *record_interval* step.
    Columns are determined by DataTracker (e.g. time, l2_error, l1_error,
    max_error, total_energy).

final_numerical : np.ndarray | None
    Raw NumPy snapshot of the numerical solution at the final time step.
    Retained for downstream consumers (plotting, npz export) that work
    directly with arrays rather than Field objects.

final_analytical : np.ndarray | None
    Analytical solution evaluated at the final time, if available for the
    active equation.  None when no closed-form solution exists.

raw_tensor_data : np.ndarray | None
    Full time-series tensor of recorded numerical snapshots with shape
    ``(n_snapshots, *grid.shape)``.  Useful for ML pipelines and
    space-time visualisations.

energy_history : dict | None
    Dictionary of energy-related time-series arrays recorded by the
    visualiser (keys: "time", "pe", "ke", "total", "loss").
    None when energy tracking is disabled or unavailable.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SimulationResults:
    # ------------------------------------------------------------------
    # Required outputs
    # ------------------------------------------------------------------
    grid: Any                               # Grid object
    final_state: Any                        # Field (ScalarField or VectorField)

    # ------------------------------------------------------------------
    # Diagnostic history
    # ------------------------------------------------------------------
    history: Any = None                     # DataFrame from DataTracker
    final_numerical: Any = None             # np.ndarray at final time
    final_analytical: Any = None            # analytical solution at final time
    raw_tensor_data: Any = None             # shape (n_snapshots, *grid.shape)
    energy_history: dict | None = None      # keys: time, pe, ke, total, loss
