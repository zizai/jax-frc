# tests/test_presets.py
"""Tests for preset simulation factory functions."""
import pytest


def test_create_magnetic_diffusion():
    """create_magnetic_diffusion should return configured Simulation."""
    from jax_frc.simulation.presets import create_magnetic_diffusion
    
    sim = create_magnetic_diffusion(nx=16, ny=16)
    
    assert sim.geometry.nx == 16
    assert sim.geometry.ny == 16
    assert sim.state is not None
    assert sim.model is not None
    assert sim.solver is not None
