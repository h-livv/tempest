from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def benchmark_numerical_solver(data_path: Path | str) -> float:
    """Re-run the Tempest numerical solver for a validation case and return wall time."""
    from src.core.config import SimulationConfig
    from src.core.simulation import Simulation
    from src.mesh import boundaries
    from src.numerics import operators, integrators
    from src.physics import equations, init_conditions

    data_path = Path(data_path)
    with open(data_path.parent / "config.json") as f:
        cfg = json.load(f)

    ic_map = {
        "gaussian": init_conditions.GaussianIC(),
        "square": init_conditions.SquareIC(),
        "shifted_gaussian": init_conditions.ShiftedGaussianIC(),
        "sine_wave": init_conditions.SineWaveIC(),
        "double_gaussian": init_conditions.DoubleGaussianIC(),
        "spike": init_conditions.SpikeIC(),
    }
    integrator_map = {
        "rk4": integrators.rk4,
        "euler": integrators.euler,
        "leapfrog": integrators.leapfrog,
    }
    operator_map = {
        "upwind": operators.upwind,
        "gradient": operators.gradient,
        "laplacian": operators.laplacian,
    }
    equation_map = {
        "advection": lambda cfg: equations.AdvectionEquation(velocity=float(cfg.get("velocity", 1.0))),
        "wave": lambda cfg: equations.WaveEquation(wave_speed=float(cfg.get("wave_speed", 1.0))),
    }
    boundary_map = {
        "periodic": boundaries.periodic,
        "dirichlet": boundaries.Dirichlet(0.0, 0.0),
    }

    n_val = cfg["N"]
    dx_val = cfg["dx"]
    shape = (n_val,) if isinstance(n_val, int) else tuple(n_val)
    spacing = (dx_val,) if isinstance(dx_val, (int, float)) else tuple(dx_val)

    equation_name = cfg.get("equation", "advection")
    if equation_name not in equation_map:
        raise ValueError(f"Unsupported equation '{equation_name}' in benchmark_numerical_solver")

    wave_fields = 2 if equation_name == "wave" else 1

    def make_ic(name: str):
        if name == "gaussian":
            return init_conditions.GaussianIC(num_fields=wave_fields)
        if name == "square":
            return init_conditions.SquareIC(num_fields=wave_fields)
        if name == "shifted_gaussian":
            return init_conditions.ShiftedGaussianIC(num_fields=wave_fields)
        if name == "sine_wave":
            return init_conditions.SineWaveIC(num_fields=wave_fields)
        if name == "double_gaussian":
            return init_conditions.DoubleGaussianIC(num_fields=wave_fields, speed=0.0)
        if name == "spike":
            return init_conditions.SpikeIC(num_fields=wave_fields)
        raise KeyError(name)

    ic_name = cfg["initial_condition"]
    if ic_name not in ic_map and ic_name not in {
        "gaussian", "square", "shifted_gaussian", "sine_wave", "double_gaussian", "spike"
    }:
        raise KeyError(f"Unknown initial condition '{ic_name}'")

    sim_config = SimulationConfig(
        shape=shape,
        spacing=spacing,
        dt=float(cfg["dt"]),
        final_time=float(cfg["final_time"]),
        steps_per_frame=int(cfg["steps_per_frame"]),
        record_interval=int(cfg["record_interval"]),
        equation=equation_map[equation_name](cfg),
        operator=operator_map[cfg["operator"]],
        boundary=boundary_map[cfg["boundary_function"]],
        integrator=integrator_map[cfg["integrator"]],
        initial_condition=make_ic(ic_name),
    )

    start = time.perf_counter()
    Simulation(sim_config).run()
    return time.perf_counter() - start


def print_timing_comparison(timing: dict[str, Any], train_ics: list[str]) -> None:
    """Print numerical solver vs surrogate wall-clock comparison."""
    print("\n" + "=" * 26)
    print("Timing Comparison")
    print("=" * 26)
    print(f"\nSurrogate training ({', '.join(train_ics)}): {timing['surrogate_training']:.3f} s")
    print(f"Surrogate inference (all test ICs):          {timing['surrogate_inference']:.3f} s")
    print(f"Surrogate total:                             {timing['surrogate_total']:.3f} s")
    print("\nNumerical solver (per test IC):")
    for ic_name, elapsed in timing["numerical_solver_by_ic"].items():
        print(f"  {ic_name:<18} {elapsed:.3f} s")
    print(f"Numerical solver total:                      {timing['numerical_solver_total']:.3f} s")
    print(f"Speedup (numerical total / surrogate total): {timing['speedup']:.2f}x")


def save_timing(
    timing: dict[str, Any],
    output_dir: Path,
    train_ics: list[str],
    test_ics: list[str],
    equation: str = "wave",
) -> None:
    """Write timing summary to JSON in the output directory."""
    from ml.core.data import find_validation_data

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **timing,
        "equation": equation,
        "train_ics": train_ics,
        "test_ics": test_ics,
        "train_data_paths": {
            ic: str(find_validation_data(ic, equation=equation)) for ic in train_ics
        },
        "test_data_paths": {
            ic: str(find_validation_data(ic, equation=equation)) for ic in test_ics
        },
    }
    with open(output_dir / "timings.json", "w") as f:
        json.dump(payload, f, indent=2)
