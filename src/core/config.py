"""
SimulationConfig – a plain dataclass that collects every parameter needed to
run a Tempest simulation.

All physics objects (equation, operator, boundary, integrator) are passed as
plain callables or lightweight objects.  No validation framework is used.

Fields
------
shape : tuple[int, ...]
    Number of grid points per axis, e.g. ``(200,)`` for 1-D or ``(100, 100)``
    for 2-D.

spacing : tuple[float, ...]
    Grid spacing per axis (must have the same length as *shape*), e.g.
    ``(0.05,)`` or ``(0.05, 0.05)``.

dt : float
    Time-step size.

final_time : float
    Physical end-time of the simulation.

steps_per_frame : int
    How many time steps are taken between two consecutive visualiser renders
    (animation cadence only; does not affect accuracy).

record_interval : int
    Record a diagnostic snapshot every *record_interval* time steps.
    Default is 1 (record every step).

equation : callable
    A callable (function or object with a ``__call__`` method) that returns
    the time derivative of the state.  Must accept the arguments expected by
    the active integrator.

operator : callable
    A spatial-discretisation operator, e.g. an upwind or central-difference
    stencil.  Must accept the arguments expected by *equation*.

boundary : callable
    A boundary-condition function that pads the state array before spatial
    operators are applied.

integrator : callable
    A time-integration scheme (e.g. ``rk4``, ``euler``).  Receives the
    current state together with the other physics objects.

initial_condition : callable
    A function with the signature ``initial_condition(grid) -> Field | np.ndarray``.
    It must return either a ``Field`` instance or a raw NumPy array of the
    correct spatial shape.  The ``Simulation`` constructor will wrap raw
    arrays into the appropriate ``ScalarField`` or ``VectorField``.

    Because the initial condition is independent of the equation, the same
    callable (e.g. a Gaussian) can be reused across different equations.
"""

from dataclasses import dataclass
from typing import Callable, Union
from src.physics.init_conditions import InitialCondition
from src.physics.equations import Equation
from src.physics.sources import Source


@dataclass
class SimulationConfig:
    # ------------------------------------------------------------------
    # Grid geometry
    # ------------------------------------------------------------------
    shape: tuple[int, ...]      # e.g. (200,) for 1-D or (100, 100) for 2-D
    spacing: tuple[float, ...]  # grid spacing per axis, same length as shape

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    dt: float
    final_time: float
    steps_per_frame: int
    record_interval: int = 1

    # ------------------------------------------------------------------
    # Physics and numerics
    # ------------------------------------------------------------------
    equation: Equation | None = None    # physical equation object
    operator: Callable | None = None    # spatial discretisation stencil
    boundary: Callable | None = None    # boundary padding function
    integrator: Callable | None = None  # time-integration scheme

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------
    # Signature: initial_condition(grid) -> Field | np.ndarray
    # Independent of the equation; reusable across any PDE.
    initial_condition: Union[Callable, InitialCondition, None] = None

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------
    source: Union[Callable, Source, None] = None