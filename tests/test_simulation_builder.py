# tests/test_simulation_builder.py
"""Tests for SimulationBuilder fluent API."""
import pytest
from jax_frc.simulation import State, Geometry
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.solvers.explicit import RK4Solver


def test_simulation_builder_basic():
    """SimulationBuilder should create Simulation with fluent API."""
    from jax_frc.simulation.simulation import Simulation
    
    geometry = Geometry(nx=8, ny=8, nz=1)
    model = ExtendedMHD(eta=1e-4)
    solver = RK4Solver()
    state = State.zeros(8, 8, 1)
    
    sim = Simulation.builder() \
        .geometry(geometry) \
        .model(model) \
        .solver(solver) \
        .initial_state(state) \
        .build()
    
    assert sim.geometry == geometry
    assert sim.model == model
    assert sim.solver == solver
    assert sim.state == state


def test_simulation_builder_missing_required():
    """SimulationBuilder should raise error if required fields missing."""
    from jax_frc.simulation.simulation import Simulation
    
    with pytest.raises(ValueError, match="geometry is required"):
        Simulation.builder().build()
