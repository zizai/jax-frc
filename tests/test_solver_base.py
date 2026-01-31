"""Tests for Solver base class attributes."""

import pytest
from jax_frc.solvers.base import Solver


def test_solver_has_timestep_attributes():
    """Solver should have cfl_safety, dt_min, dt_max attributes."""
    # Can't instantiate ABC directly, but we can check class attributes
    assert hasattr(Solver, 'cfl_safety')
    assert hasattr(Solver, 'dt_min')
    assert hasattr(Solver, 'dt_max')
    assert hasattr(Solver, 'use_checked_step')
    assert hasattr(Solver, 'divergence_cleaning')


def test_solver_timestep_defaults():
    """Solver timestep attributes should have sensible defaults."""
    assert Solver.cfl_safety == 0.5
    assert Solver.dt_min == 1e-12
    assert Solver.dt_max == 1e-3


def test_solver_numerical_option_defaults():
    """Solver numerical options should have sensible defaults."""
    assert Solver.use_checked_step is True
    assert Solver.divergence_cleaning == "projection"
