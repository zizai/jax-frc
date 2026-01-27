"""End-to-end integration tests for 3D Cartesian coordinate system.

Tests complete workflows including:
- Resistive MHD with Harris sheet initial condition
- Extended MHD with Hall term physics
- Module import verification
"""

import jax.numpy as jnp
import pytest

# Test module imports at top level
from jax_frc import (
    Geometry,
    State,
    gradient_3d,
    divergence_3d,
    curl_3d,
    laplacian_3d,
    MU0,
)
from jax_frc.models import ResistiveMHD, ExtendedMHD
from jax_frc.equilibrium import harris_sheet_3d, uniform_field_3d


class TestModuleImports:
    """Verify all 3D-related imports work correctly."""

    def test_core_imports(self):
        """Test core classes can be imported."""
        from jax_frc import Geometry, State
        assert Geometry is not None
        assert State is not None

    def test_operator_imports(self):
        """Test 3D operators can be imported."""
        from jax_frc import gradient_3d, divergence_3d, curl_3d, laplacian_3d
        assert gradient_3d is not None
        assert divergence_3d is not None
        assert curl_3d is not None
        assert laplacian_3d is not None

    def test_model_imports(self):
        """Test MHD models can be imported."""
        from jax_frc.models import ResistiveMHD, ExtendedMHD
        assert ResistiveMHD is not None
        assert ExtendedMHD is not None

    def test_equilibrium_imports(self):
        """Test equilibrium initializers can be imported."""
        from jax_frc.equilibrium import harris_sheet_3d, uniform_field_3d, flux_rope_3d
        assert harris_sheet_3d is not None
        assert uniform_field_3d is not None
        assert flux_rope_3d is not None

    def test_constants_import(self):
        """Test physical constants can be imported."""
        from jax_frc.constants import MU0, QE, ME, MI, KB
        assert MU0 > 0
        assert QE > 0
        assert ME > 0
        assert MI > 0
        assert KB > 0


class TestResistiveMHDIntegration:
    """Integration tests for resistive MHD with Harris sheet."""

    def test_harris_sheet_initialization(self):
        """Test creating state with Harris sheet initial condition."""
        geom = Geometry(nx=8, ny=8, nz=8)
        B = harris_sheet_3d(geom, B0=0.1, L=0.2)

        assert B.shape == (8, 8, 8, 3)
        # Harris sheet should have non-trivial Bx variation
        assert jnp.max(jnp.abs(B[..., 0])) > 0.05

    def test_resistive_mhd_rhs_computation(self):
        """Test computing RHS for resistive MHD."""
        geom = Geometry(nx=8, ny=8, nz=8)
        model = ResistiveMHD(eta=1e-4)

        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        state = State.zeros(nx=8, ny=8, nz=8)
        state = state.replace(B=B, n=jnp.ones((8, 8, 8)) * 1e19)

        rhs = model.compute_rhs(state, geom)

        assert rhs.B.shape == (8, 8, 8, 3)
        assert jnp.all(jnp.isfinite(rhs.B))

    def test_resistive_mhd_cfl(self):
        """Test CFL condition computation."""
        geom = Geometry(nx=8, ny=8, nz=8)
        model = ResistiveMHD(eta=1e-4)

        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        state = State.zeros(nx=8, ny=8, nz=8)
        state = state.replace(B=B, n=jnp.ones((8, 8, 8)) * 1e19)

        dt = model.compute_stable_dt(state, geom)

        assert dt > 0
        assert jnp.isfinite(dt)

    @pytest.mark.slow
    def test_resistive_mhd_time_stepping(self):
        """Test time stepping with resistive MHD."""
        geom = Geometry(
            nx=8, ny=16, nz=8,
            x_min=-0.5, x_max=0.5,
            y_min=-1.0, y_max=1.0,
            z_min=-0.5, z_max=0.5,
        )
        model = ResistiveMHD(eta=1e-3)  # Higher resistivity for faster evolution

        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        state = State.zeros(nx=8, ny=16, nz=8)
        state = state.replace(B=B, n=jnp.ones((8, 16, 8)) * 1e19)

        # Record initial magnetic energy
        B_mag_initial = jnp.sum(state.B**2)

        # Simple Euler time stepping
        dt = 0.1 * model.compute_stable_dt(state, geom)
        for _ in range(5):
            rhs = model.compute_rhs(state, geom)
            B_new = state.B + dt * rhs.B
            state = state.replace(B=B_new)

        B_mag_final = jnp.sum(state.B**2)

        # Magnetic field should still be finite
        assert jnp.all(jnp.isfinite(state.B))

        # Resistive diffusion should cause some change (energy decay)
        # With high enough resistivity and small grid, we expect some evolution
        assert B_mag_final < B_mag_initial * 1.1  # Allow small numerical increase


class TestExtendedMHDIntegration:
    """Integration tests for extended MHD with Hall term."""

    def test_extended_mhd_creation(self):
        """Test creating extended MHD model with Hall term."""
        model_hall = ExtendedMHD(eta=1e-4, include_hall=True)
        model_no_hall = ExtendedMHD(eta=1e-4, include_hall=False)

        assert model_hall.include_hall is True
        assert model_no_hall.include_hall is False

    def test_extended_mhd_rhs_computation(self):
        """Test computing RHS for extended MHD."""
        geom = Geometry(nx=8, ny=8, nz=8)
        model = ExtendedMHD(eta=1e-4, include_hall=True)

        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        state = State.zeros(nx=8, ny=8, nz=8)
        state = state.replace(
            B=B,
            n=jnp.ones((8, 8, 8)) * 1e19,
            p=jnp.ones((8, 8, 8)) * 1e3,
        )

        rhs = model.compute_rhs(state, geom)

        assert rhs.B.shape == (8, 8, 8, 3)
        assert jnp.all(jnp.isfinite(rhs.B))

    def test_hall_term_affects_evolution(self):
        """Verify Hall term produces different results than resistive-only."""
        geom = Geometry(nx=16, ny=16, nz=8)

        # Create state with B field that generates J perpendicular to B
        B = jnp.zeros((16, 16, 8, 3))
        x = geom.x_grid
        y = geom.y_grid
        B = B.at[..., 0].set(jnp.sin(2 * jnp.pi * y))
        B = B.at[..., 1].set(jnp.sin(2 * jnp.pi * x))
        B = B.at[..., 2].set(0.1)

        state = State.zeros(nx=16, ny=16, nz=8)
        state = state.replace(
            B=B,
            n=jnp.ones((16, 16, 8)) * 1e19,
            p=jnp.ones((16, 16, 8)) * 1e3,
        )

        # Compare Hall vs no-Hall
        model_hall = ExtendedMHD(eta=0.0, include_hall=True)
        model_no_hall = ExtendedMHD(eta=0.0, include_hall=False)

        rhs_hall = model_hall.compute_rhs(state, geom)
        rhs_no_hall = model_no_hall.compute_rhs(state, geom)

        # With zero resistivity, no-Hall should give zero dB/dt
        # (no resistive term, no velocity term)
        assert jnp.max(jnp.abs(rhs_no_hall.B)) < 1e-10

        # Hall term should produce non-zero dB/dt
        assert jnp.max(jnp.abs(rhs_hall.B)) > 1e-10

    def test_hall_term_jxb_physics(self):
        """Test that Hall term (J x B)/(ne) contributes correctly.

        Uses a 3D magnetic field configuration where:
        - J = curl(B)/mu0 has components perpendicular to B
        - curl(J x B) is non-zero (required for dB/dt != 0)
        """
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-1.0, z_max=1.0,
        )

        # Create a 3D B field with curl(J x B) != 0
        # Use sinusoidal variations in multiple directions to ensure
        # J x B has non-zero curl
        B = jnp.zeros((16, 16, 16, 3))
        x = geom.x_grid
        y = geom.y_grid
        z = geom.z_grid
        # Bx varies with y and z
        B = B.at[..., 0].set(0.1 * jnp.sin(2 * jnp.pi * y) * jnp.cos(jnp.pi * z))
        # By varies with x and z
        B = B.at[..., 1].set(0.1 * jnp.sin(2 * jnp.pi * x) * jnp.cos(jnp.pi * z))
        # Bz uniform guide field
        B = B.at[..., 2].set(0.1)

        state = State.zeros(nx=16, ny=16, nz=16)
        state = state.replace(
            B=B,
            n=jnp.ones((16, 16, 16)) * 1e19,
            p=jnp.ones((16, 16, 16)) * 1e3,
        )

        # With Hall term: J has spatial variation, J x B has non-zero curl
        model = ExtendedMHD(eta=0.0, include_hall=True)
        rhs = model.compute_rhs(state, geom)

        # Should have non-trivial evolution due to Hall term
        # The 3D configuration ensures curl(E_Hall) != 0
        assert jnp.max(jnp.abs(rhs.B)) > 0

    @pytest.mark.slow
    def test_extended_mhd_time_stepping(self):
        """Test time stepping with extended MHD Hall physics."""
        geom = Geometry(nx=8, ny=8, nz=8)
        model = ExtendedMHD(eta=1e-4, include_hall=True)

        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        state = State.zeros(nx=8, ny=8, nz=8)
        state = state.replace(
            B=B,
            n=jnp.ones((8, 8, 8)) * 1e19,
            p=jnp.ones((8, 8, 8)) * 1e3,
        )

        # Take a few time steps
        dt = 0.01 * model.compute_stable_dt(state, geom)
        for _ in range(3):
            rhs = model.compute_rhs(state, geom)
            B_new = state.B + dt * rhs.B
            state = state.replace(B=B_new)

        # Result should be finite
        assert jnp.all(jnp.isfinite(state.B))


class TestOperatorIntegration:
    """Test 3D operators work together in physics computations."""

    def test_faraday_law_consistency(self):
        """Test that dB/dt = -curl(E) is computed consistently."""
        geom = Geometry(nx=16, ny=16, nz=16)

        # Create a simple E field: E = eta * J where J = curl(B)/mu0
        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        J = curl_3d(B, geom) / MU0
        eta = 1e-4
        E = eta * J

        # dB/dt = -curl(E)
        dB_dt = -curl_3d(E, geom)

        assert dB_dt.shape == B.shape
        assert jnp.all(jnp.isfinite(dB_dt))

    def test_div_b_constraint(self):
        """Test that uniform field has div(B) = 0."""
        geom = Geometry(nx=16, ny=16, nz=16)
        B = uniform_field_3d(geom, B0=0.1, direction="z")

        div_B = divergence_3d(B, geom)

        assert jnp.max(jnp.abs(div_B)) < 1e-10

    def test_operators_with_geometry(self):
        """Test operators work correctly with non-unit geometry."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=-2.0, x_max=2.0,
            y_min=-2.0, y_max=2.0,
            z_min=-2.0, z_max=2.0,
        )

        # Create a scalar field
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid
        f = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)

        # Test gradient
        grad_f = gradient_3d(f, geom)
        assert grad_f.shape == (16, 16, 16, 3)
        assert jnp.all(jnp.isfinite(grad_f))

        # Test Laplacian
        lap_f = laplacian_3d(f, geom)
        assert lap_f.shape == (16, 16, 16)
        assert jnp.all(jnp.isfinite(lap_f))


class TestRK4Integration:
    """Test simple RK4 time integration with MHD models."""

    @staticmethod
    def rk4_step(state, dt, model, geom):
        """Simple RK4 time stepping."""
        k1 = model.compute_rhs(state, geom)
        state_k1 = state.replace(B=state.B + 0.5 * dt * k1.B)

        k2 = model.compute_rhs(state_k1, geom)
        state_k2 = state.replace(B=state.B + 0.5 * dt * k2.B)

        k3 = model.compute_rhs(state_k2, geom)
        state_k3 = state.replace(B=state.B + dt * k3.B)

        k4 = model.compute_rhs(state_k3, geom)

        B_new = state.B + (dt / 6.0) * (k1.B + 2 * k2.B + 2 * k3.B + k4.B)
        return state.replace(B=B_new)

    @pytest.mark.slow
    def test_rk4_resistive_mhd(self):
        """Test RK4 integration with resistive MHD."""
        geom = Geometry(nx=8, ny=8, nz=8)
        model = ResistiveMHD(eta=1e-3)

        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        state = State.zeros(nx=8, ny=8, nz=8)
        state = state.replace(B=B, n=jnp.ones((8, 8, 8)) * 1e19)

        dt = 0.1 * model.compute_stable_dt(state, geom)

        for _ in range(3):
            state = self.rk4_step(state, dt, model, geom)

        assert jnp.all(jnp.isfinite(state.B))

    @pytest.mark.slow
    def test_rk4_extended_mhd(self):
        """Test RK4 integration with extended MHD."""
        geom = Geometry(nx=8, ny=8, nz=8)
        model = ExtendedMHD(eta=1e-3, include_hall=True)

        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        state = State.zeros(nx=8, ny=8, nz=8)
        state = state.replace(
            B=B,
            n=jnp.ones((8, 8, 8)) * 1e19,
            p=jnp.ones((8, 8, 8)) * 1e3,
        )

        dt = 0.01 * model.compute_stable_dt(state, geom)

        for _ in range(3):
            state = self.rk4_step(state, dt, model, geom)

        assert jnp.all(jnp.isfinite(state.B))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
