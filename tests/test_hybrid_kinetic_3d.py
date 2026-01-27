"""Tests for 3D hybrid kinetic model with trilinear interpolation."""

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State, ParticleState
from jax_frc.models.hybrid_kinetic import HybridKinetic
from jax_frc.models.particle_pusher import interpolate_field_to_particles_3d


class TestInterpolation3D:
    """Tests for trilinear field interpolation to particle positions."""

    def test_interpolate_uniform_field(self):
        """Uniform field should interpolate to same value everywhere."""
        geom = Geometry(nx=8, ny=8, nz=8)
        field = jnp.ones((8, 8, 8, 3)) * 5.0
        key = random.PRNGKey(0)
        x = random.uniform(
            key,
            (100, 3),
            minval=jnp.array([geom.x_min, geom.y_min, geom.z_min]),
            maxval=jnp.array([geom.x_max, geom.y_max, geom.z_max]),
        )
        field_p = interpolate_field_to_particles_3d(field, x, geom)
        assert field_p.shape == (100, 3)
        assert jnp.allclose(field_p, 5.0, atol=1e-6)

    def test_interpolate_uniform_scalar_field(self):
        """Uniform scalar field should interpolate to same value everywhere."""
        geom = Geometry(nx=8, ny=8, nz=8)
        field = jnp.ones((8, 8, 8)) * 3.0
        key = random.PRNGKey(1)
        x = random.uniform(
            key,
            (50, 3),
            minval=jnp.array([geom.x_min, geom.y_min, geom.z_min]),
            maxval=jnp.array([geom.x_max, geom.y_max, geom.z_max]),
        )
        field_p = interpolate_field_to_particles_3d(field, x, geom)
        assert field_p.shape == (50,)
        assert jnp.allclose(field_p, 3.0, atol=1e-6)

    def test_interpolate_linear_field_x(self):
        """Linear field in x should interpolate correctly."""
        geom = Geometry(nx=16, ny=16, nz=16)
        # Create field that varies linearly with x
        x_vals = geom.x
        field = jnp.zeros((16, 16, 16))
        for i in range(16):
            field = field.at[i, :, :].set(x_vals[i])

        # Test at center point (should be 0.0 for domain [-1, 1])
        x = jnp.array([[0.0, 0.0, 0.0]])
        field_p = interpolate_field_to_particles_3d(field, x, geom)
        assert jnp.allclose(field_p[0], 0.0, atol=0.1)

        # Test at x=0.5
        x = jnp.array([[0.5, 0.0, 0.0]])
        field_p = interpolate_field_to_particles_3d(field, x, geom)
        assert jnp.allclose(field_p[0], 0.5, atol=0.1)

    def test_interpolate_output_shape_vector(self):
        """Test output shape for vector field."""
        geom = Geometry(nx=8, ny=8, nz=8)
        field = jnp.ones((8, 8, 8, 3))
        x = jnp.zeros((10, 3))  # 10 particles at origin
        field_p = interpolate_field_to_particles_3d(field, x, geom)
        assert field_p.shape == (10, 3)

    def test_interpolate_output_shape_scalar(self):
        """Test output shape for scalar field."""
        geom = Geometry(nx=8, ny=8, nz=8)
        field = jnp.ones((8, 8, 8))
        x = jnp.zeros((10, 3))  # 10 particles at origin
        field_p = interpolate_field_to_particles_3d(field, x, geom)
        assert field_p.shape == (10,)

    def test_interpolate_is_jittable(self):
        """Test that interpolation can be JIT compiled."""
        geom = Geometry(nx=8, ny=8, nz=8)
        field = jnp.ones((8, 8, 8, 3)) * 2.0
        x = jnp.zeros((5, 3))

        @jax.jit
        def interp(f, pos):
            return interpolate_field_to_particles_3d(f, pos, geom)

        field_p = interp(field, x)
        assert field_p.shape == (5, 3)
        assert jnp.allclose(field_p, 2.0, atol=1e-6)

    def test_interpolate_periodic_wrapping(self):
        """Test that interpolation handles periodic boundaries."""
        geom = Geometry(nx=8, ny=8, nz=8)
        # Set different values at opposite boundaries
        field = jnp.zeros((8, 8, 8, 3))
        field = field.at[0, :, :, :].set(1.0)
        field = field.at[-1, :, :, :].set(1.0)

        # Particle near boundary
        x = jnp.array([[geom.x_max - geom.dx / 4, 0.0, 0.0]])
        field_p = interpolate_field_to_particles_3d(field, x, geom)
        # Should interpolate between boundary values
        assert field_p.shape == (1, 3)


class TestHybridKinetic3D:
    """Tests for HybridKinetic model with 3D Cartesian geometry."""

    def test_model_creation(self):
        """Test creating hybrid model."""
        model = HybridKinetic.from_config({"eta": 1e-6})
        assert model.eta == 1e-6

    def test_model_creation_with_equilibrium(self):
        """Test creating hybrid model with equilibrium parameters."""
        config = {
            "eta": 1e-5,
            "collision_frequency": 1e4,
            "equilibrium": {
                "n0": 2e19,
                "T0": 500.0,
                "Omega": 5e4,
            },
        }
        model = HybridKinetic.from_config(config)
        assert model.eta == 1e-5
        assert model.collision_frequency == 1e4
        assert model.equilibrium.n0 == 2e19
        assert model.equilibrium.T0 == 500.0
        assert model.equilibrium.Omega == 5e4

    def test_compute_rhs_shape(self):
        """Test RHS has correct shape."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(0.1)
        state = state.replace(
            B=B, n=jnp.ones((8, 8, 8)) * 1e19, p=jnp.ones((8, 8, 8)) * 1e3
        )
        rhs = model.compute_rhs(state, geom)
        assert rhs.B.shape == (8, 8, 8, 3)

    def test_compute_rhs_asymmetric_grid(self):
        """Test RHS on asymmetric grid."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=6, ny=8, nz=10)
        state = State.zeros(nx=6, ny=8, nz=10)
        B = jnp.zeros((6, 8, 10, 3))
        B = B.at[..., 2].set(0.1)
        state = state.replace(
            B=B, n=jnp.ones((6, 8, 10)) * 1e19, p=jnp.ones((6, 8, 10)) * 1e3
        )
        rhs = model.compute_rhs(state, geom)
        assert rhs.B.shape == (6, 8, 10, 3)

    def test_uniform_field_no_change(self):
        """Uniform B field should have zero dB/dt (no current)."""
        model = HybridKinetic.from_config({"eta": 0.0})  # No resistivity
        geom = Geometry(nx=16, ny=16, nz=16)
        state = State.zeros(nx=16, ny=16, nz=16)
        B = jnp.zeros((16, 16, 16, 3))
        B = B.at[..., 2].set(1.0)  # Uniform Bz
        state = state.replace(
            B=B, n=jnp.ones((16, 16, 16)) * 1e19, p=jnp.zeros((16, 16, 16))
        )

        rhs = model.compute_rhs(state, geom)
        # Uniform field => J = curl(B)/mu0 = 0 => dB/dt = 0 (with eta=0, no pressure)
        assert jnp.max(jnp.abs(rhs.B)) < 1e-10

    def test_compute_rhs_jittable(self):
        """Test that compute_rhs can be JIT compiled."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(0.1)
        state = state.replace(
            B=B, n=jnp.ones((8, 8, 8)) * 1e19, p=jnp.ones((8, 8, 8)) * 1e3
        )

        def compute_rhs_wrapper(s):
            return model.compute_rhs(s, geom)

        compute_rhs_jit = jax.jit(compute_rhs_wrapper)
        rhs = compute_rhs_jit(state)
        assert rhs.B.shape == (8, 8, 8, 3)


class TestHybridStableDt3D:
    """Tests for stable timestep calculation."""

    def test_compute_stable_dt(self):
        """Test computing stable timestep."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(0.1)  # B_z = 0.1 T
        state = state.replace(B=B)

        dt = model.compute_stable_dt(state, geom)
        assert dt > 0
        assert dt < 1e-3  # Should be a small timestep for plasma timescales


class TestHybridConstraints3D:
    """Tests for boundary condition application."""

    def test_apply_constraints_shape(self):
        """Test constraint application preserves shape."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        B = jnp.ones((8, 8, 8, 3))
        state = state.replace(B=B)

        constrained = model.apply_constraints(state, geom)
        assert constrained.B.shape == (8, 8, 8, 3)


class TestHybridParticles3D:
    """Tests for particle operations in 3D."""

    def test_initialize_particles_3d(self):
        """Test particle initialization produces 3D Cartesian positions."""
        geom = Geometry(nx=8, ny=8, nz=8)
        model = HybridKinetic.from_config({})
        key = random.PRNGKey(42)

        particles = HybridKinetic.initialize_particles(
            n_particles=100, geometry=geom, equilibrium=model.equilibrium, key=key
        )

        assert particles.x.shape == (100, 3)
        assert particles.v.shape == (100, 3)
        assert particles.w.shape == (100,)

        # Positions should be within domain
        assert jnp.all(particles.x[:, 0] >= geom.x_min)
        assert jnp.all(particles.x[:, 0] <= geom.x_max)
        assert jnp.all(particles.x[:, 1] >= geom.y_min)
        assert jnp.all(particles.x[:, 1] <= geom.y_max)
        assert jnp.all(particles.x[:, 2] >= geom.z_min)
        assert jnp.all(particles.x[:, 2] <= geom.z_max)

    def test_deposit_current_shape_3d(self):
        """Test current deposition returns correct shape."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)
        key = random.PRNGKey(0)

        particles = HybridKinetic.initialize_particles(
            n_particles=50, geometry=geom, equilibrium=model.equilibrium, key=key
        )

        state = State.zeros(nx=8, ny=8, nz=8)
        state = state.replace(particles=particles)

        J = model.deposit_current(state, geom)
        assert J.shape == (8, 8, 8, 3)

    def test_deposit_current_no_particles(self):
        """Test current deposition with no particles returns zeros."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)

        J = model.deposit_current(state, geom)
        assert J.shape == (8, 8, 8, 3)
        assert jnp.allclose(J, 0.0)

    def test_push_particles_3d(self):
        """Test particle pushing in 3D."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)
        key = random.PRNGKey(0)

        particles = HybridKinetic.initialize_particles(
            n_particles=10, geometry=geom, equilibrium=model.equilibrium, key=key
        )

        state = State.zeros(nx=8, ny=8, nz=8)
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(0.1)  # Uniform Bz
        E = jnp.zeros((8, 8, 8, 3))
        state = state.replace(B=B, E=E, particles=particles)

        new_state = model.push_particles(state, geom, dt=1e-9)
        assert new_state.particles.x.shape == (10, 3)
        assert new_state.particles.v.shape == (10, 3)

    def test_particle_boundaries_periodic_3d(self):
        """Test periodic boundary conditions for particles."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)

        # Create particle outside domain
        x = jnp.array([[geom.x_max + 0.1, 0.0, 0.0]])
        v = jnp.array([[1.0, 0.0, 0.0]])
        w = jnp.array([0.0])
        particles = ParticleState(x=x, v=v, w=w, species="ion")

        wrapped = model._apply_particle_boundaries(particles, geom)

        # Should wrap to opposite side (for periodic)
        assert wrapped.x[0, 0] >= geom.x_min
        assert wrapped.x[0, 0] <= geom.x_max
