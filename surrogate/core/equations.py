"""Equation metadata and CLI helpers for ML experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EquationSpec:
    name: str
    validation_subpath: str
    default_train_ics: tuple[str, ...]
    default_test_ics: tuple[str, ...]
    known_ics: tuple[str, ...]
    description: str
    # Periodic linear advection / wave conserve ∫u; Dirichlet Burgers does not.
    conserves_mass: bool = True


EQUATION_SPECS: dict[str, EquationSpec] = {
    "wave": EquationSpec(
        name="wave",
        validation_subpath="wave/1D/validation",
        default_train_ics=("gaussian", "square", "sine_wave"),
        default_test_ics=(
            "gaussian",
            "square",
            "double_gaussian",
            "shifted_gaussian",
            "sine_wave",
        ),
        known_ics=(
            "gaussian",
            "square",
            "double_gaussian",
            "shifted_gaussian",
            "sine_wave",
        ),
        description="1D wave equation (leapfrog validation data)",
        conserves_mass=True,
    ),
    "advection": EquationSpec(
        name="advection",
        validation_subpath="advection/1D/validation",
        default_train_ics=("gaussian", "square", "sine_wave"),
        default_test_ics=(
            "gaussian",
            "square",
            "double_gaussian",
            "shifted_gaussian",
            "sine_wave",
            "spike",
        ),
        known_ics=(
            "gaussian",
            "square",
            "double_gaussian",
            "shifted_gaussian",
            "sine_wave",
            "spike",
        ),
        description="1D linear advection (Lax-Friedrichs validation data)",
        conserves_mass=True,
    ),
    "advection_2d": EquationSpec(
        name="advection_2d",
        validation_subpath="advection/2D/validation",
        default_train_ics=("gaussian",),
        default_test_ics=("gaussian",),
        known_ics=("gaussian",),
        description="2D linear advection (sliced to 1D for ML)",
        conserves_mass=True,
    ),
    "burgers": EquationSpec(
        name="burgers",
        validation_subpath="burgers/1D/validation",
        default_train_ics=("burgers_traveling_shock",),
        default_test_ics=("burgers_traveling_shock",),
        known_ics=(
            "burgers_traveling_shock",
            "burgers_stationary_shock",
            "burgers_traveling_smooth",
        ),
        description="1D viscous Burgers equation (validation data)",
        conserves_mass=False,
    ),
    "burgers_2d": EquationSpec(
        name="burgers_2d",
        validation_subpath="burgers/2D/validation",
        default_train_ics=("burgers_traveling_shock",),
        default_test_ics=("burgers_traveling_shock", "gaussian"),
        known_ics=("burgers_traveling_shock", "gaussian"),
        description="2D Burgers equation (sliced to 1D for ML)",
        conserves_mass=False,
    ),
    "rossby_wave": EquationSpec(
        name="rossby_wave",
        validation_subpath="rossby_wave/2D/validation",
        default_train_ics=("rossby_gaussian_vortex",),
        default_test_ics=("rossby_gaussian_vortex", "constant"),
        known_ics=("rossby_gaussian_vortex", "constant"),
        description="2D Rossby wave equation (sliced to 1D for ML)",
        conserves_mass=False,
    ),
}

EQUATION_ALIASES: dict[str, str] = {
    "burgers_1d": "burgers",
    "1d_burgers": "burgers",
}


def normalize_equation(name: str) -> str:
    key = name.strip().lower()
    key = EQUATION_ALIASES.get(key, key)
    if key not in EQUATION_SPECS:
        available = ", ".join(sorted(EQUATION_SPECS))
        raise ValueError(f"Unknown equation '{name}'. Available: {available}")
    return key


def get_equation_spec(equation: str) -> EquationSpec:
    return EQUATION_SPECS[normalize_equation(equation)]


def get_validation_dir(equation: str) -> Path:
    spec = get_equation_spec(equation)
    return PROJECT_ROOT / "outputs" / spec.validation_subpath


def parse_ic_list(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    items = [item.strip() for item in value.split() if item.strip()]
    if not items:
        raise ValueError("Initial condition list cannot be empty.")
    return items


def format_equation_help() -> str:
    lines = ["Available equations:"]
    for key, spec in sorted(EQUATION_SPECS.items()):
        ics = ", ".join(spec.known_ics)
        lines.append(f"  {key}: {spec.description}")
        lines.append(f"    known ICs: {ics}")
    return "\n".join(lines)


def add_equation_cli_args(
    parser: argparse.ArgumentParser,
    *,
    default_equation: str = "wave",
) -> None:
    equation_help = (
        "PDE equation to use for validation data paths "
        f"(default: {default_equation}). {format_equation_help()}"
    )
    parser.add_argument(
        "--equation",
        "--pde",
        dest="equation",
        default=default_equation,
        help=equation_help,
    )
    parser.add_argument(
        "--train-ics",
        default=None,
        help=(
            "Space-separated training initial conditions "
            "(default: TRAIN_ICS list in the experiment script)"
        ),
    )
    parser.add_argument(
        "--test-ics",
        default=None,
        help=(
            "Space-separated test initial conditions "
            "(default: TEST_ICS list in the experiment script)"
        ),
    )


def resolve_experiment_ics(
    equation: str,
    train_ics: str | None = None,
    test_ics: str | None = None,
    *,
    default_train_ics: list[str] | tuple[str, ...] | None = None,
    default_test_ics: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, list[str], list[str]]:
    spec = get_equation_spec(equation)
    train_default = (
        tuple(default_train_ics)
        if default_train_ics is not None
        else spec.default_train_ics
    )
    test_default = (
        tuple(default_test_ics)
        if default_test_ics is not None
        else spec.default_test_ics
    )
    resolved_train = parse_ic_list(train_ics, train_default)
    resolved_test = parse_ic_list(test_ics, test_default)
    return spec.name, resolved_train, resolved_test
