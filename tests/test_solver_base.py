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


def test_solver_compute_dt():
    """Solver._compute_dt should compute timestep from model CFL."""
    from jax_frc.solvers.explicit import RK4Solver
    from jax_frc.models.extended_mhd import ExtendedMHD
    from jax_frc.core.state import State
    from jax_frc.core.geometry import Geometry
    
    solver = RK4Solver()
    model = ExtendedMHD(eta=1e-4)
    geometry = Geometry(nx=8, ny=8, nz=1)
    state = State.zeros(8, 8, 1)
    
    dt = solver._compute_dt(state, model, geometry)
    assert dt > 0
    assert dt <= solver.dt_max
    assert dt >= solver.dt_min


def test_solver_apply_constraints():
    """Solver._apply_constraints should enforce div(B)=0."""
    from jax_frc.solvers.explicit import RK4Solver
    from jax_frc.core.state import State
    from jax_frc.core.geometry import Geometry
    
    solver = RK4Solver()
    geometry = Geometry(nx=8, ny=8, nz=1)
    state = State.zeros(8, 8, 1)
    
    new_state = solver._apply_constraints(state, geometry)
    assert new_state is not None
    assert new_state.B.shape == state.B.shape


def test_solver_step_computes_dt_internally():
    """Solver.step() should compute dt internally, not require it as param."""
    from jax_frc.solvers.explicit import RK4Solver
    from jax_frc.models.extended_mhd import ExtendedMHD
    from jax_frc.core.state import State
    from jax_frc.core.geometry import Geometry
    
    solver = RK4Solver()
    model = ExtendedMHD(eta=1e-4)
    geometry = Geometry(nx=8, ny=8, nz=1)
    state = State.zeros(8, 8, 1)
    
    # New signature: step(state, model, geometry) - no dt parameter
    new_state = solver.step(state, model, geometry)
    assert new_state.time > state.time
