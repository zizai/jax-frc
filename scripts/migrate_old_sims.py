#!/usr/bin/env python
"""Migration script to run old-style simulations using the new JAX-FRC framework.

This script demonstrates how to migrate from the standalone simulation files
(resistive_mhd.py, extended_mhd.py, hybrid_kinetic.py) to the new OOP framework.

Usage:
    python scripts/migrate_old_sims.py --model resistive_mhd --steps 100
    python scripts/migrate_old_sims.py --model extended_mhd --steps 50
    python scripts/migrate_old_sims.py --model hybrid_kinetic --steps 100 --particles 1000
"""

import argparse
import jax.numpy as jnp
import jax.random as random

from jax_frc.core.simulation import Simulation
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models import ResistiveMHD, ExtendedMHD, HybridKinetic
from jax_frc.solvers import RK4Solver, SemiImplicitSolver, HybridSolver
from jax_frc.solvers.time_controller import TimeController
from jax_frc.diagnostics.probes import DiagnosticSet


def run_resistive_mhd(steps: int = 100, nx: int = 32, nz: int = 64):
    """Run Resistive MHD simulation matching old resistive_mhd.py behavior."""
    print("=" * 60)
    print("Resistive MHD Simulation (New Framework)")
    print("=" * 60)

    # Create geometry (matching old defaults)
    geometry = Geometry(
        coord_system='cylindrical',
        nr=nx, nz=nz,
        r_min=0.01, r_max=1.0,
        z_min=-1.0, z_max=1.0
    )

    # Create model with Chodura resistivity (matching old default)
    model = ResistiveMHD.from_config({
        'resistivity': {
            'type': 'chodura',
            'eta_0': 1e-6,
            'eta_anom': 1e-3,
            'threshold': 1e4
        }
    })

    # Create solver and time controller
    solver = RK4Solver()
    time_controller = TimeController(cfl_safety=0.25, dt_max=1e-4)

    # Create simulation
    sim = Simulation(
        geometry=geometry,
        model=model,
        solver=solver,
        time_controller=time_controller
    )

    # Initialize with FRC-like flux profile (matching old default)
    def psi_init(r, z):
        return (1 - r**2) * jnp.exp(-z**2)

    sim.initialize(psi_init=psi_init)

    # Setup diagnostics
    diagnostics = DiagnosticSet.default_set()
    diagnostics.measure_all(sim.state, geometry)

    # Run simulation
    print(f"Running {steps} steps...")
    for i in range(steps):
        sim.step()
        if (i + 1) % 10 == 0:
            results = diagnostics.measure_all(sim.state, geometry)
            print(f"  Step {i+1}: t={sim.state.time:.2e}, psi_max={results['psi_max']:.4f}")

    print(f"\nFinal state: t={sim.state.time:.2e}, psi_max={jnp.max(sim.state.psi):.4f}")
    return sim.state, diagnostics.get_history()


def run_extended_mhd(steps: int = 50, nx: int = 32, nz: int = 64):
    """Run Extended MHD simulation matching old extended_mhd.py behavior."""
    print("=" * 60)
    print("Extended MHD Simulation (New Framework)")
    print("=" * 60)

    # Create geometry
    geometry = Geometry(
        coord_system='cylindrical',
        nr=nx, nz=nz,
        r_min=0.01, r_max=1.0,
        z_min=-1.0, z_max=1.0
    )

    # Create model with Hall physics
    model = ExtendedMHD.from_config({
        'resistivity': {'type': 'spitzer', 'eta_0': 1e-4},
        'halo': {
            'halo_density': 1e16,
            'core_density': 1e19,
            'r_cutoff': 0.8
        }
    })

    # Use semi-implicit solver for Whistler stability
    solver = SemiImplicitSolver(damping_factor=1e6)
    time_controller = TimeController(cfl_safety=0.5, dt_max=1e-6)

    # Create simulation
    sim = Simulation(
        geometry=geometry,
        model=model,
        solver=solver,
        time_controller=time_controller
    )

    # Initialize state
    state = State.zeros(geometry.nr, geometry.nz)
    B_init = jnp.zeros((geometry.nr, geometry.nz, 3))
    B_init = B_init.at[:, :, 2].set(1.0 * jnp.exp(-geometry.r_grid**2 - geometry.z_grid**2))
    state = state.replace(
        B=B_init,
        n=jnp.ones((geometry.nr, geometry.nz)) * 1e19,
        p=jnp.ones((geometry.nr, geometry.nz)) * 1e3
    )
    sim.state = state

    # Run simulation
    print(f"Running {steps} steps...")
    for i in range(steps):
        sim.step()
        if (i + 1) % 10 == 0:
            B_max = jnp.max(jnp.abs(sim.state.B[:, :, 2]))
            print(f"  Step {i+1}: t={sim.state.time:.2e}, B_z_max={B_max:.4f}")

    print(f"\nFinal state: t={sim.state.time:.2e}, B_z_max={jnp.max(sim.state.B[:,:,2]):.4f}")
    return sim.state


def run_hybrid_kinetic(steps: int = 100, n_particles: int = 1000, nx: int = 32, nz: int = 64):
    """Run Hybrid Kinetic simulation matching old hybrid_kinetic.py behavior."""
    print("=" * 60)
    print("Hybrid Kinetic Simulation (New Framework)")
    print("=" * 60)

    # Create geometry
    geometry = Geometry(
        coord_system='cylindrical',
        nr=nx, nz=nz,
        r_min=0.01, r_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    # Create model
    model = HybridKinetic.from_config({
        'equilibrium': {
            'n0': 1e19,
            'T0': 100.0,  # eV
            'Omega': 1e5  # rad/s
        },
        'eta': 1e-4
    })

    # Use hybrid solver
    solver = HybridSolver()
    time_controller = TimeController(cfl_safety=0.1, dt_max=1e-8)

    # Create simulation
    sim = Simulation(
        geometry=geometry,
        model=model,
        solver=solver,
        time_controller=time_controller
    )

    # Initialize state with B field
    state = State.zeros(geometry.nr, geometry.nz)
    B_init = jnp.zeros((geometry.nr, geometry.nz, 3))
    B_init = B_init.at[:, :, 2].set(1.0 * jnp.exp(-geometry.r_grid**2 - geometry.z_grid**2))
    state = state.replace(
        B=B_init,
        n=jnp.ones((geometry.nr, geometry.nz)) * 1e19,
        p=jnp.ones((geometry.nr, geometry.nz)) * model.equilibrium.T0 * 1e19 * 1.602e-19
    )

    # Initialize particles
    key = random.PRNGKey(0)
    particles = HybridKinetic.initialize_particles(n_particles, geometry, model.equilibrium, key)
    state = state.replace(particles=particles)
    sim.state = state

    # Run simulation
    print(f"Running {steps} steps with {n_particles} particles...")
    for i in range(steps):
        sim.step()
        if (i + 1) % 20 == 0:
            w_max = jnp.max(jnp.abs(sim.state.particles.w))
            print(f"  Step {i+1}: t={sim.state.time:.2e}, w_max={w_max:.4f}")

    print(f"\nFinal state: t={sim.state.time:.2e}, n_particles={sim.state.particles.n_particles}")
    return sim.state


def main():
    parser = argparse.ArgumentParser(description="Run JAX-FRC simulations using new framework")
    parser.add_argument('--model', type=str, default='resistive_mhd',
                        choices=['resistive_mhd', 'extended_mhd', 'hybrid_kinetic'],
                        help='Physics model to run')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps')
    parser.add_argument('--nx', type=int, default=32, help='Grid size in r')
    parser.add_argument('--nz', type=int, default=64, help='Grid size in z')
    parser.add_argument('--particles', type=int, default=1000, help='Number of particles (hybrid only)')

    args = parser.parse_args()

    if args.model == 'resistive_mhd':
        run_resistive_mhd(steps=args.steps, nx=args.nx, nz=args.nz)
    elif args.model == 'extended_mhd':
        run_extended_mhd(steps=args.steps, nx=args.nx, nz=args.nz)
    elif args.model == 'hybrid_kinetic':
        run_hybrid_kinetic(steps=args.steps, n_particles=args.particles, nx=args.nx, nz=args.nz)


if __name__ == "__main__":
    main()
