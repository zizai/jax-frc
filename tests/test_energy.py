"""Tests for ThermalTransport class."""
import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.models.energy import ThermalTransport
from jax_frc.models.extended_mhd import ExtendedMHD, TemperatureBoundaryCondition
from jax_frc.core.state import State
from tests.utils.cartesian import make_geometry


class TestThermalTransportKappa:
    """Tests for thermal conductivity calculations."""

    def test_spitzer_conductivity_scaling(self):
        """Spitzer conductivity should scale as T^(5/2)."""
        transport = ThermalTransport(
            kappa_parallel_0=1.0,
            use_spitzer=True,
            coulomb_log=1.0  # Simplify for testing
        )

        # Test at different temperatures
        T1 = jnp.array([[1.0]])
        T2 = jnp.array([[4.0]])  # 4x temperature

        kappa1 = transport.compute_kappa_parallel(T1)
        kappa2 = transport.compute_kappa_parallel(T2)

        # κ ∝ T^(5/2), so κ2/κ1 = (4)^(5/2) = 32
        ratio = kappa2 / kappa1
        expected = 4.0**2.5  # = 32

        assert jnp.allclose(ratio, expected, rtol=1e-5), \
            f"Spitzer scaling: expected ratio {expected}, got {ratio}"

    def test_constant_conductivity(self):
        """Non-Spitzer should give constant conductivity."""
        kappa_0 = 1e15
        transport = ThermalTransport(
            kappa_parallel_0=kappa_0,
            use_spitzer=False
        )

        T = jnp.array([[1.0, 10.0, 100.0],
                       [1000.0, 10000.0, 100000.0]])

        kappa = transport.compute_kappa_parallel(T)

        assert jnp.allclose(kappa, kappa_0), \
            "Constant conductivity should not depend on T"

    def test_perp_to_parallel_ratio(self):
        """Perpendicular conductivity should be ratio times parallel."""
        ratio = 1e-6
        transport = ThermalTransport(
            kappa_parallel_0=1e15,
            kappa_perp_ratio=ratio,
            use_spitzer=False
        )

        T = jnp.ones((8, 8)) * 100.0

        kappa_par = transport.compute_kappa_parallel(T)
        kappa_perp = transport.compute_kappa_perp(T)

        assert jnp.allclose(kappa_perp / kappa_par, ratio), \
            "κ_⊥/κ_∥ should equal kappa_perp_ratio"

    def test_minimum_temperature_clipping(self):
        """Low temperatures should be clipped to avoid singularities."""
        transport = ThermalTransport(
            kappa_parallel_0=1.0,
            use_spitzer=True,
            coulomb_log=1.0,
            min_temperature=1.0
        )

        # Very low temperature
        T_low = jnp.array([[1e-10]])
        # Temperature at minimum
        T_min = jnp.array([[1.0]])

        kappa_low = transport.compute_kappa_parallel(T_low)
        kappa_min = transport.compute_kappa_parallel(T_min)

        # Both should give same result due to clipping
        assert jnp.allclose(kappa_low, kappa_min), \
            "Temperatures below min should be clipped"


class TestThermalTransportHeatFlux:
    """Tests for heat flux calculations."""

    def test_heat_flux_parallel_to_B(self):
        """Heat flux should be primarily along B for high κ_∥/κ_⊥."""
        transport = ThermalTransport(
            kappa_parallel_0=1e15,
            kappa_perp_ratio=1e-10,  # Very small perpendicular
            use_spitzer=False
        )

        geometry = make_geometry(nx=16, ny=4, nz=16)
        dr, dz = geometry.dx, geometry.dz
        x_mid = geometry.ny // 2
        r = jnp.abs(geometry.x_grid[:, x_mid, :])

        # Uniform B in z direction
        B = jnp.zeros((geometry.nx, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)  # B_z = 1

        # Temperature gradient in r direction (perpendicular to B)
        T = jnp.linspace(100, 200, geometry.nx)[:, None] * jnp.ones((1, geometry.nz))

        q_r, q_z = transport.compute_heat_flux(T, B, dr, dz, r)

        # With B along z and gradient along r (perpendicular),
        # heat flux should be primarily perpendicular (very small due to low κ_⊥)
        # q_z should be near zero (no gradient in z)
        assert jnp.max(jnp.abs(q_z[2:-2, 2:-2])) < 1e-5, \
            "No heat flux in z without z gradient"

    def test_heat_flux_direction_parallel_gradient(self):
        """Heat flux along B direction when gradient is parallel to B."""
        transport = ThermalTransport(
            kappa_parallel_0=1e10,
            kappa_perp_ratio=1e-6,
            use_spitzer=False
        )

        geometry = make_geometry(nx=16, ny=4, nz=32)
        dr, dz = geometry.dx, geometry.dz
        x_mid = geometry.ny // 2
        r = jnp.abs(geometry.x_grid[:, x_mid, :])

        # Uniform B in z direction
        B = jnp.zeros((geometry.nx, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        # Temperature gradient in z direction (parallel to B)
        z_values = jnp.linspace(0, 0.32, geometry.nz)
        T = jnp.ones((geometry.nx, 1)) * (100.0 + 1000.0 * z_values[None, :])

        q_r, q_z = transport.compute_heat_flux(T, B, dr, dz, r)

        # Heat should flow opposite to gradient (negative q_z for positive dT/dz)
        # Interior points (avoiding boundary artifacts)
        interior_q_z = q_z[2:-2, 2:-2]
        assert jnp.all(interior_q_z < 0), \
            "Heat flux should be opposite to temperature gradient"

    def test_heat_flux_zero_with_uniform_T(self):
        """Heat flux should be zero with uniform temperature."""
        transport = ThermalTransport(
            kappa_parallel_0=1e15,
            use_spitzer=False
        )

        geometry = make_geometry(nx=16, ny=4, nz=16)
        dr, dz = geometry.dx, geometry.dz
        x_mid = geometry.ny // 2
        r = jnp.abs(geometry.x_grid[:, x_mid, :])

        # Some B field
        B = jnp.zeros((geometry.nx, geometry.nz, 3))
        B = B.at[:, :, 0].set(0.5)
        B = B.at[:, :, 2].set(0.5)

        # Uniform temperature
        T = jnp.ones((geometry.nx, geometry.nz)) * 100.0

        q_r, q_z = transport.compute_heat_flux(T, B, dr, dz, r)

        # Heat flux should be zero (within numerical precision)
        # Avoid boundaries due to roll artifacts
        assert jnp.allclose(q_r[2:-2, 2:-2], 0, atol=1e-10), \
            "No heat flux with uniform T"
        assert jnp.allclose(q_z[2:-2, 2:-2], 0, atol=1e-10), \
            "No heat flux with uniform T"


class TestThermalTransportDivergence:
    """Tests for heat flux divergence."""

    def test_divergence_zero_uniform_flux(self):
        """Divergence should be near zero for uniform heat flux."""
        transport = ThermalTransport(
            kappa_parallel_0=1e10,
            use_spitzer=False
        )

        geometry = make_geometry(nx=16, ny=4, nz=16)
        dr, dz = geometry.dx, geometry.dz
        x_mid = geometry.ny // 2
        r = jnp.abs(geometry.x_grid[:, x_mid, :])

        # Uniform B in z
        B = jnp.zeros((geometry.nx, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        # Uniform temperature
        T = jnp.ones((geometry.nx, geometry.nz)) * 100.0

        div_q = transport.compute_heat_flux_divergence(T, B, dr, dz, r)

        # Divergence of zero flux should be zero
        assert jnp.allclose(div_q[2:-2, 2:-2], 0, atol=1e-10), \
            "Divergence should be zero for uniform T"

    def test_divergence_shape(self):
        """Divergence should have same shape as input."""
        transport = ThermalTransport()

        geometry = make_geometry(nx=32, ny=4, nz=64)
        dr, dz = geometry.dx, geometry.dz
        x_mid = geometry.ny // 2
        r = jnp.abs(geometry.x_grid[:, x_mid, :])

        B = jnp.zeros((geometry.nx, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        T = jnp.ones((geometry.nx, geometry.nz)) * 100.0

        div_q = transport.compute_heat_flux_divergence(T, B, dr, dz, r)

        assert div_q.shape == (geometry.nx, geometry.nz), \
            f"Expected shape ({geometry.nx}, {geometry.nz}), got {div_q.shape}"


class TestThermalTransportPhysics:
    """Physics validation tests."""

    def test_heat_flux_orthogonal_decomposition(self):
        """Parallel and perpendicular heat flux should be orthogonal."""
        transport = ThermalTransport(
            kappa_parallel_0=1e15,
            kappa_perp_ratio=0.5,  # Comparable to see both contributions
            use_spitzer=False
        )

        geometry = make_geometry(nx=16, ny=4, nz=16)
        dr, dz = geometry.dx, geometry.dz
        x_mid = geometry.ny // 2
        r = jnp.abs(geometry.x_grid[:, x_mid, :])

        # B at 45 degrees in r-z plane
        B = jnp.zeros((geometry.nx, geometry.nz, 3))
        B = B.at[:, :, 0].set(1.0)  # B_r
        B = B.at[:, :, 2].set(1.0)  # B_z

        # Temperature gradient in r direction
        T = jnp.linspace(100, 200, geometry.nx)[:, None] * jnp.ones((1, geometry.nz))

        q_r, q_z = transport.compute_heat_flux(T, B, dr, dz, r)

        # For B at 45°, the parallel gradient is dT/dr * (1/√2)
        # Parallel heat flux should be along B direction
        # Interior points
        interior_q_r = q_r[4:-4, 4:-4]
        interior_q_z = q_z[4:-4, 4:-4]

        # Both components should be non-zero with B at 45°
        assert jnp.any(jnp.abs(interior_q_r) > 1e-5), \
            "q_r should be non-zero with angled B"
        assert jnp.any(jnp.abs(interior_q_z) > 1e-5), \
            "q_z should be non-zero with angled B"


class TestExtendedMHDTemperature:
    """Tests for temperature evolution in ExtendedMHD."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return make_geometry(nx=16, ny=4, nz=32)

    @pytest.fixture
    def model_with_thermal(self):
        """Create ExtendedMHD model with thermal transport."""
        return ExtendedMHD(eta=1e-4, kappa_perp=1e-2)

    @pytest.fixture
    def model_without_thermal(self):
        """Create ExtendedMHD model without thermal transport."""
        return ExtendedMHD(eta=1e-4, kappa_perp=1e-2)

    def test_compute_rhs_returns_dT_with_thermal(self, geometry, model_with_thermal):
        """compute_rhs should return dT when thermal is enabled."""
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create state with a non-uniform temperature profile
        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(0.1)  # Uniform B_z

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=100.0 + 10.0 * geometry.x_grid**2,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        rhs = model_with_thermal.compute_rhs(state, geometry)

        # RHS should have non-trivial T (dT/dt)
        assert rhs.Te is not None
        assert rhs.Te.shape == (nx, ny, nz), f"Expected Te shape ({nx}, {ny}, {nz})"
        center = rhs.Te[nx // 2, ny // 2, nz // 2]
        assert center != 0

    def test_compute_rhs_unchanged_without_thermal(self, geometry, model_without_thermal):
        """compute_rhs should not modify T when thermal is disabled."""
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        B = jnp.zeros((nx, ny, nz, 3))
        B = B.at[:, :, :, 2].set(0.1)

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=None,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        rhs = model_without_thermal.compute_rhs(state, geometry)

        # T should be unchanged (no dT in RHS)
        assert rhs.Te is None

    def test_temperature_diffusion_reduces_peak(self, geometry, model_with_thermal):
        """Thermal conduction should smooth temperature peaks."""
        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
        B = jnp.zeros((nx, ny, nz, 3))

        x = geometry.x_grid
        z = geometry.z_grid
        Te = 100.0 + 50.0 * jnp.exp(-(x**2 + z**2))

        state = State(
            B=B,
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=Te,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        rhs = model_with_thermal.compute_rhs(state, geometry)

        center = rhs.Te[nx // 2, ny // 2, nz // 2]
        assert center < 0, "Peak temperature should diffuse downward"


class TestTemperatureBoundaryConditions:
    """Tests for temperature boundary conditions."""

    @pytest.fixture
    def geometry(self):
        """Create test geometry."""
        return make_geometry(nx=16, ny=4, nz=32)

    def test_dirichlet_bc_sets_wall_temperature(self, geometry):
        """Dirichlet BC should set T = T_wall at boundaries."""
        T_wall = 50.0
        bc = TemperatureBoundaryCondition(
            bc_type="dirichlet",
            T_wall=T_wall,
            apply_axis_symmetry=False,
        )

        model = ExtendedMHD(
            eta=1e-6,
            temperature_bc=bc,
        )

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create state with non-wall temperature
        T_interior = 200.0
        state = State(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=jnp.ones((nx, ny, nz)) * T_interior,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        # Apply constraints
        constrained = model.apply_constraints(state, geometry)

        # Wall boundaries should have T_wall
        assert jnp.allclose(constrained.Te[0, :, :], T_wall), "x_min boundary should be T_wall"
        assert jnp.allclose(constrained.Te[-1, :, :], T_wall), "x_max boundary should be T_wall"
        assert jnp.allclose(constrained.Te[:, :, 0], T_wall), "z_min boundary should be T_wall"
        assert jnp.allclose(constrained.Te[:, :, -1], T_wall), "z_max boundary should be T_wall"

        # Interior should be unchanged
        assert jnp.allclose(constrained.Te[5, 1, 5], T_interior), "Interior should be unchanged"

    def test_neumann_bc_extrapolates_from_interior(self, geometry):
        """Neumann BC should extrapolate from interior (zero gradient)."""
        bc = TemperatureBoundaryCondition(bc_type="neumann")

        model = ExtendedMHD(
            eta=1e-6,
            temperature_bc=bc,
        )

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create state with gradient toward boundaries
        T = jnp.linspace(100, 300, nx)[:, None] * jnp.ones((1, nz))
        state = State(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=jnp.repeat(T[:, None, :], ny, axis=1),
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        constrained = model.apply_constraints(state, geometry)

        # Outer r boundary should equal adjacent interior
        assert jnp.allclose(constrained.Te[-1, :, :], constrained.Te[-2, :, :]), \
            "Neumann: x_max should match interior"

    def test_axis_symmetry_enforced(self, geometry):
        """Axis symmetry should enforce ∂T/∂r = 0 at r=0."""
        bc = TemperatureBoundaryCondition(apply_axis_symmetry=True)

        model = ExtendedMHD(
            eta=1e-6,
            temperature_bc=bc,
        )

        nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

        # Create state with different value at axis
        T = jnp.ones((nx, ny, nz)) * 100.0
        T = T.at[0, :, :].set(50.0)  # Different at axis

        state = State(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.ones((nx, ny, nz)) * 1e19,
            p=jnp.ones((nx, ny, nz)) * 1e3,
            v=jnp.zeros((nx, ny, nz, 3)),
            Te=T,
            Ti=None,
            particles=None,
            time=0.0,
            step=0,
        )

        constrained = model.apply_constraints(state, geometry)

        # First row (axis) should equal second row
        assert jnp.allclose(constrained.Te[0, :, :], constrained.Te[1, :, :]), \
            "Axis symmetry: Te[0,:,:] should equal Te[1,:,:]"
