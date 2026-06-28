"""
SimulationResults – an immutable snapshot of a completed Tempest simulation.

This is a pure data container.  It must not contain simulation logic,
helper methods, or any mutable state.  All fields are populated by
``Simulation.run()`` and should be treated as read-only by consumers.

Two representations are kept deliberately:

* **High-level objects** (``grid``, ``final_state``) – for user-facing code
  that works with the Grid/Field API.
* **Raw numerical arrays** (``final_numerical``, ``raw_tensor_data``) – for
  plotting, export (CSV/npz), and ML pipelines that operate on plain NumPy
  arrays directly.

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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

# Use TYPE_CHECKING guard to avoid runtime circular imports.
# Grid and Field live in src.grid / src.fields today and will move to
# src.core.grid / src.core.field in a later step; string annotations keep
# results.py decoupled from that path change.
if TYPE_CHECKING:
    import pandas as pd
    from src.mesh.grid import Grid
    from src.mesh.fields import Field


@dataclass
class SimulationResults:
    # ------------------------------------------------------------------
    # High-level objects  (user-facing API)
    # ------------------------------------------------------------------
    grid: "Grid"                            # dimension-agnostic Grid
    final_state: "Field"                    # ScalarField or VectorField

    # ------------------------------------------------------------------
    # Diagnostic history
    # ------------------------------------------------------------------
    history: "pd.DataFrame | None" = None  # time-series from DataTracker

    # ------------------------------------------------------------------
    # Raw numerical arrays  (plotting / export / ML)
    # ------------------------------------------------------------------
    final_numerical: np.ndarray | None = None    # numerical solution at T_final
    final_analytical: np.ndarray | None = None   # analytical solution at T_final
    raw_tensor_data: np.ndarray | None = None    # shape: (n_snapshots, *grid.shape)
    energy_history: dict | None = None           # keys: time, pe, ke, total, loss
