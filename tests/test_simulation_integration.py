"""Integration tests for JAX-FRC simulation framework."""

import pytest
import jax.numpy as jnp
import jax.random as random

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.core.simulation import Simulation
from jax_frc.models import ResistiveMHD, ExtendedMHD, HybridKinetic
from jax_frc.solvers import RK4Solver, SemiImplicitSolver, HybridSolver
from tests.utils.cartesian import make_geometry


class TestSimulationIntegration:
    """Integration tests for the Simulation class."""

    def test_simulation_from_config(self):
        """Test creating simulation from config dictionary."""
        config = {
            'geometry': {
                'nx': 32,
                'ny': 4,
                'nz': 64,
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': -1.0, 'y_max': 1.0,
                'z_min': -1.0, 'z_max': 1.0,
                'bc_x': 'neumann',
                'bc_y': 'periodic',
                'bc_z': 'neumann'
            },
            'model': {'type': 'resistive_mhd', 'eta': 1e-4},
            'solver': {'type': 'rk4'},
            'time': {'cfl_safety': 0.25, 'dt_max': 1e-4}
        }

        sim = Simulation.from_config(config)
        sim.initialize(
            B_init=lambda x, y, z: jnp.stack(
                [jnp.zeros_like(x), jnp.zeros_like(x), (1 - x**2) * jnp.exp(-z**2)],
                axis=-1
            )
        )

        assert sim.state is not None
        assert sim.state.B.shape == (32, 4, 64, 3)

    def test_simulation_from_config_with_recipe(self):
        """Test Simulation supports numerics recipe configuration."""
        config = {
            'geometry': {
                'nx': 8,
                'ny': 2,
                'nz': 8,
                'x_min': -1.0,
                'x_max': 1.0,
                'y_min': -1.0,
                'y_max': 1.0,
                'z_min': -1.0,
                'z_max': 1.0,
                'bc_x': 'neumann',
                'bc_y': 'periodic',
                'bc_z': 'neumann'
            },
            'model': {'type': 'resistive_mhd', 'eta': 1e-4},
            'solver': {'type': 'rk4'},
            'time': {'cfl_safety': 0.25, 'dt_max': 1e-4},
            'numerics': {'divergence_strategy': 'none'}
        }

        sim = Simulation.from_config(config)
        sim.initialize()
        state = sim.step()

        assert state.step == 1

    def test_simulation_run_steps(self):
        """Test running simulation for fixed number of steps."""
        config = {
            'geometry': {
                'nx': 16,
                'ny': 4,
                'nz': 32,
                'x_min': -0.5, 'x_max': 0.5,
                'y_min': -0.5, 'y_max': 0.5,
                'z_min': -0.5, 'z_max': 0.5,
                'bc_x': 'neumann',
                'bc_y': 'periodic',
                'bc_z': 'neumann'
            },
            'model': {'type': 'resistive_mhd', 'eta': 1e-3},
            'solver': {'type': 'euler'},
            'time': {'cfl_safety': 0.1}
        }

        sim = Simulation.from_config(config)
        sim.initialize(
            B_init=lambda x, y, z: jnp.stack(
                [jnp.zeros_like(x), jnp.zeros_like(x), jnp.exp(-x**2 - z**2)],
                axis=-1
            )
        )

        initial_time = sim.state.time
        final = sim.run_steps(10)

        assert float(final.time) > float(initial_time)
        assert int(final.step) == 10


class TestResistiveMHD:
    """Tests for Resistive MHD model physics."""

    @pytest.fixture
    def setup(self):
        """Create geometry and model for tests."""
        geom = make_geometry(nx=32, ny=4, nz=64)
        model = ResistiveMHD(eta=1e-4)
        return geom, model

    def test_flux_diffusion(self, setup):
        """Test that flux diffuses according to resistive timescale."""
        geom, model = setup

        # Initialize with Gaussian flux profile
        state = State.zeros(geom.nx, geom.ny, geom.nz)
        B = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B = B.at[:, :, :, 2].set(jnp.exp(-geom.x_grid**2 - geom.z_grid**2))
        state = state.replace(B=B)

        # Run for several timesteps using new API (solver computes dt internally)
        solver = RK4Solver(cfl_safety=0.25)

        psi_max_initial = jnp.max(state.B[..., 2])

        for _ in range(100):
            state = solver.step(state, model, geom)

        psi_max_final = jnp.max(state.B[..., 2])

        # Flux should decrease due to resistive diffusion
        assert psi_max_final < psi_max_initial

    def test_boundary_conditions(self, setup):
        """Test that boundary conditions are enforced."""
        geom, model = setup

        state = State.zeros(geom.nx, geom.ny, geom.nz)
        B = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B = B.at[:, :, :, 2].set(1.0)
        state = state.replace(B=B)

        state = model.apply_constraints(state, geom)

        assert jnp.allclose(state.B, B)


class TestExtendedMHD:
    """Tests for Extended MHD model physics."""

    @pytest.fixture
    def setup(self):
        """Create geometry and model for tests."""
        geom = make_geometry(nx=32, ny=4, nz=64)
        model = ExtendedMHD(eta=1e-4, include_hall=True)
        return geom, model

    def test_hall_term_present(self, setup):
        """Test that Hall term contributes to E field."""
        geom, model = setup

        # Create state with B field and current
        state = State.zeros(geom.nx, geom.ny, geom.nz)
        B_init = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B_init = B_init.at[:, :, :, 2].set(0.1 * jnp.exp(-geom.x_grid**2))
        state = state.replace(
            B=B_init,
            n=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e19,
            p=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e3
        )

        rhs = model.compute_rhs(state, geom)

        # dB/dt should be non-zero due to Hall term
        assert jnp.max(jnp.abs(rhs.B)) > 0

    def test_whistler_cfl(self, setup):
        """Test that Whistler CFL is computed correctly."""
        geom, model = setup

        state = State.zeros(geom.nx, geom.ny, geom.nz)
        B_init = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B_init = B_init.at[:, :, :, 2].set(0.1)
        state = state.replace(
            B=B_init,
            n=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e19
        )

        dt = model.compute_stable_dt(state, geom)

        assert jnp.isfinite(dt)
        assert dt > 0


class TestHybridKinetic:
    """Tests for Hybrid Kinetic model physics."""

    @pytest.fixture
    def setup(self):
        """Create geometry and model for tests."""
        geom = make_geometry(nx=16, ny=4, nz=32)
        model = HybridKinetic.from_config({'equilibrium': {'n0': 1e19, 'T0': 1000.0, 'Omega': 1e5}})
        return geom, model

    def test_particle_initialization(self, setup):
        """Test particle initialization from equilibrium."""
        geom, model = setup

        key = random.PRNGKey(42)
        particles = HybridKinetic.initialize_particles(1000, geom, model.equilibrium, key)

        assert particles.x.shape[0] == 1000
        assert particles.x.shape == (1000, 3)
        assert particles.v.shape == (1000, 3)
        assert jnp.allclose(particles.w, 0)  # Delta-f weights start at zero

    def test_boris_pusher(self, setup):
        """Test that Boris pusher advances particles correctly."""
        geom, model = setup

        # Create state with uniform B field
        state = State.zeros(geom.nx, geom.ny, geom.nz)
        B_init = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B_init = B_init.at[:, :, :, 2].set(0.1)  # Uniform Bz
        state = state.replace(
            B=B_init,
            E=jnp.zeros((geom.nx, geom.ny, geom.nz, 3)),
            n=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e19,
            p=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e3
        )

        # Initialize particles
        key = random.PRNGKey(0)
        particles = HybridKinetic.initialize_particles(100, geom, model.equilibrium, key)
        state = state.replace(particles=particles)

        # Push particles
        dt = 1e-9
        state_new = model.push_particles(state, geom, dt)

        # Particles should have moved
        assert not jnp.allclose(state_new.particles.x, state.particles.x)

    def test_weight_conservation(self, setup):
        """Test that weights remain bounded."""
        geom, model = setup

        state = State.zeros(geom.nx, geom.ny, geom.nz)
        B_init = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B_init = B_init.at[:, :, :, 2].set(0.1)
        state = state.replace(
            B=B_init,
            E=jnp.zeros((geom.nx, geom.ny, geom.nz, 3)),
            n=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e19,
            p=jnp.ones((geom.nx, geom.ny, geom.nz)) * 1e3
        )

        key = random.PRNGKey(1)
        particles = HybridKinetic.initialize_particles(100, geom, model.equilibrium, key)
        state = state.replace(particles=particles)

        # Run several steps
        dt = 1e-9
        for _ in range(10):
            state = model.push_particles(state, geom, dt)

        # Weights should be bounded in [-1, 1]
        assert jnp.all(state.particles.w >= -1)
        assert jnp.all(state.particles.w <= 1)


class TestEnergyConservation:
    """Tests for energy conservation properties."""

    def test_magnetic_energy_decay(self):
        """Test that magnetic energy decays with resistivity."""
        geom = make_geometry(nx=32, ny=4, nz=64)
        model = ResistiveMHD(eta=1e-3)
        solver = RK4Solver(cfl_safety=0.25)

        # Initialize with B field
        state = State.zeros(geom.nx, geom.ny, geom.nz)
        B = jnp.zeros((geom.nx, geom.ny, geom.nz, 3))
        B = B.at[:, :, :, 2].set(jnp.exp(-geom.x_grid**2 - geom.z_grid**2))
        state = state.replace(B=B)

        # Compute initial magnetic energy proxy
        energy_init = jnp.sum(jnp.sum(state.B**2, axis=-1) * geom.cell_volumes)

        # Run simulation using new API (solver computes dt internally)
        for _ in range(50):
            state = solver.step(state, model, geom)

        energy_final = jnp.sum(jnp.sum(state.B**2, axis=-1) * geom.cell_volumes)

        # Energy should decrease due to resistive dissipation
        assert energy_final < energy_init


@pytest.mark.skip(reason="Rigid rotor equilibrium remains cylindrical; 3D migration pending.")
class TestEquilibriumSolvers:
    """Tests for equilibrium solvers."""

    def test_rigid_rotor_equilibrium(self):
        """Test rigid rotor equilibrium generation."""
        from jax_frc.equilibrium import RigidRotorEquilibrium, EquilibriumConstraints

        geom = Geometry('cylindrical', 32, 64, 0.01, 0.4, -1.0, 1.0)
        rr = RigidRotorEquilibrium(r_s=0.2, z_s=0.5, B_e=0.1, n0=1e19)
        constraints = EquilibriumConstraints()

        state = rr.solve(geom, constraints)

        # Should have valid psi profile
        assert jnp.max(state.psi) > 0

        # Should have density profile
        assert jnp.max(state.n) > 0

        # Should have B field
        assert jnp.max(jnp.abs(state.B)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
