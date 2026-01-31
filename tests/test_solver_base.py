# tests/test_solver_base.py
"""Tests for Solver base class timestep control attributes."""
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


def test_solver_default_values():
    """Solver should have sensible default values for timestep control."""
    assert Solver.cfl_safety == 0.5
    assert Solver.dt_min == 1e-12
    assert Solver.dt_max == 1e-3
    assert Solver.use_checked_step is True
    assert Solver.divergence_cleaning == "projection"
