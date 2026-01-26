"""Tests for translation validation module."""

import pytest
import jax.numpy as jnp
from jax_frc.validation.translation import (
    AnalyticTrajectory,
    TranslationResult,
    TranslationBenchmark,
    ModelComparisonResult,
    compute_field_gradient_at_point,
    create_mirror_push_benchmark,
    create_uniform_gradient_benchmark,
    create_staged_acceleration_benchmark,
    traveling_wave_timing,
)
from jax_frc.fields import MirrorCoil, ThetaPinchArray


class TestAnalyticTrajectory:
    """Tests for analytic trajectory calculation."""

    def test_constant_velocity_when_no_gradient(self):
        """With zero gradient, position increases linearly."""
        traj = AnalyticTrajectory(
            magnetic_moment=1.0,
            frc_mass=1.0,
            field_gradient=0.0,
            initial_position=0.0,
            initial_velocity=1.0,
        )
        t = jnp.array([0.0, 1.0, 2.0])
        z = traj.position(t)
        assert jnp.allclose(z, jnp.array([0.0, 1.0, 2.0]))

    def test_constant_velocity_preserved_when_no_gradient(self):
        """With zero gradient, velocity remains constant."""
        traj = AnalyticTrajectory(
            magnetic_moment=1.0,
            frc_mass=1.0,
            field_gradient=0.0,
            initial_position=0.0,
            initial_velocity=5.0,
        )
        t = jnp.array([0.0, 1.0, 2.0, 10.0])
        v = traj.velocity(t)
        assert jnp.allclose(v, jnp.array([5.0, 5.0, 5.0, 5.0]))

    def test_acceleration_with_gradient(self):
        """With gradient, velocity changes linearly."""
        # F = -mu * dB/dz, a = F/m
        traj = AnalyticTrajectory(
            magnetic_moment=1.0,
            frc_mass=1.0,
            field_gradient=2.0,  # dB/dz = 2 T/m
            initial_position=0.0,
            initial_velocity=0.0,
        )
        # a = -1 * 2 / 1 = -2 m/s^2
        t = jnp.array([0.0, 1.0])
        v = traj.velocity(t)
        assert jnp.allclose(v, jnp.array([0.0, -2.0]))

    def test_acceleration_property(self):
        """acceleration property computes F/m correctly."""
        traj = AnalyticTrajectory(
            magnetic_moment=2.0,
            frc_mass=4.0,
            field_gradient=3.0,
            initial_position=0.0,
            initial_velocity=0.0,
        )
        # a = -mu * (dB/dz) / m = -2 * 3 / 4 = -1.5
        assert traj.acceleration == pytest.approx(-1.5)

    def test_position_with_acceleration(self):
        """Position follows kinematic equation with acceleration."""
        traj = AnalyticTrajectory(
            magnetic_moment=1.0,
            frc_mass=2.0,
            field_gradient=4.0,  # a = -1 * 4 / 2 = -2
            initial_position=10.0,
            initial_velocity=3.0,
        )
        # z = z0 + v0*t + 0.5*a*t^2 = 10 + 3*t - t^2
        t = jnp.array([0.0, 1.0, 2.0])
        z = traj.position(t)
        expected = jnp.array([10.0, 12.0, 12.0])
        assert jnp.allclose(z, expected)

    def test_negative_gradient_gives_positive_acceleration(self):
        """Negative gradient (decreasing B) gives positive force."""
        traj = AnalyticTrajectory(
            magnetic_moment=1.0,
            frc_mass=1.0,
            field_gradient=-1.0,
            initial_position=0.0,
            initial_velocity=0.0,
        )
        # a = -1 * (-1) / 1 = 1 m/s^2
        assert traj.acceleration == pytest.approx(1.0)


class TestTranslationResult:
    """Tests for TranslationResult data container."""

    def test_flux_loss_fraction_computation(self):
        """flux_loss_fraction correctly computes relative loss."""
        result = TranslationResult(
            times=jnp.array([0.0, 1.0, 2.0]),
            positions=jnp.array([0.0, 0.5, 1.0]),
            velocities=jnp.array([0.5, 0.5, 0.5]),
            flux_max=jnp.array([1.0, 0.9, 0.8]),
            E_thermal=jnp.array([100.0, 105.0, 110.0]),
        )
        # Loss = (1.0 - 0.8) / 1.0 = 0.2
        assert result.flux_loss_fraction() == pytest.approx(0.2)

    def test_flux_loss_fraction_with_zero_initial(self):
        """flux_loss_fraction returns 0 when initial flux is zero."""
        result = TranslationResult(
            times=jnp.array([0.0, 1.0]),
            positions=jnp.array([0.0, 1.0]),
            velocities=jnp.array([1.0, 1.0]),
            flux_max=jnp.array([0.0, 0.0]),
            E_thermal=jnp.array([100.0, 100.0]),
        )
        assert result.flux_loss_fraction() == 0.0

    def test_heating_fraction_computation(self):
        """heating_fraction correctly computes relative increase."""
        result = TranslationResult(
            times=jnp.array([0.0, 1.0, 2.0]),
            positions=jnp.array([0.0, 0.5, 1.0]),
            velocities=jnp.array([0.5, 0.5, 0.5]),
            flux_max=jnp.array([1.0, 1.0, 1.0]),
            E_thermal=jnp.array([100.0, 120.0, 150.0]),
        )
        # Heating = (150 - 100) / 100 = 0.5
        assert result.heating_fraction() == pytest.approx(0.5)

    def test_heating_fraction_with_zero_initial(self):
        """heating_fraction returns 0 when initial energy is zero."""
        result = TranslationResult(
            times=jnp.array([0.0, 1.0]),
            positions=jnp.array([0.0, 1.0]),
            velocities=jnp.array([1.0, 1.0]),
            flux_max=jnp.array([1.0, 1.0]),
            E_thermal=jnp.array([0.0, 0.0]),
        )
        assert result.heating_fraction() == 0.0


class TestTranslationBenchmark:
    """Tests for TranslationBenchmark."""

    def test_compute_error_without_analytic(self):
        """compute_error returns empty dict when no analytic trajectory."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=1000.0)
        benchmark = TranslationBenchmark(
            name="test",
            description="Test benchmark",
            coil_field=coil,
            analytic_trajectory=None,
        )
        result = TranslationResult(
            times=jnp.array([0.0, 1.0]),
            positions=jnp.array([0.0, 1.0]),
            velocities=jnp.array([1.0, 1.0]),
            flux_max=jnp.array([1.0, 1.0]),
            E_thermal=jnp.array([100.0, 100.0]),
        )
        errors = benchmark.compute_error(result)
        assert errors == {}

    def test_compute_error_with_analytic(self):
        """compute_error computes metrics vs analytic trajectory."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=1000.0)
        analytic = AnalyticTrajectory(
            magnetic_moment=1.0,
            frc_mass=1.0,
            field_gradient=0.0,
            initial_position=0.0,
            initial_velocity=1.0,
        )
        benchmark = TranslationBenchmark(
            name="test",
            description="Test benchmark",
            coil_field=coil,
            analytic_trajectory=analytic,
        )
        # Simulation result with small error
        times = jnp.array([0.0, 1.0, 2.0])
        result = TranslationResult(
            times=times,
            positions=jnp.array([0.0, 1.1, 2.1]),  # +0.1 error
            velocities=jnp.array([1.0, 1.0, 1.0]),  # exact
            flux_max=jnp.array([1.0, 1.0, 1.0]),
            E_thermal=jnp.array([100.0, 100.0, 100.0]),
        )
        errors = benchmark.compute_error(result)

        assert "max_position_error" in errors
        assert "max_velocity_error" in errors
        assert "rms_position_error" in errors
        assert "rms_velocity_error" in errors
        assert errors["max_position_error"] == pytest.approx(0.1, abs=1e-6)
        assert errors["max_velocity_error"] == pytest.approx(0.0, abs=1e-6)


class TestModelComparisonResult:
    """Tests for ModelComparisonResult."""

    def test_position_divergence(self):
        """position_divergence computes max pairwise differences."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=1000.0)
        benchmark = TranslationBenchmark(
            name="test", description="Test", coil_field=coil
        )

        times = jnp.array([0.0, 1.0, 2.0])
        result_a = TranslationResult(
            times=times,
            positions=jnp.array([0.0, 1.0, 2.0]),
            velocities=jnp.array([1.0, 1.0, 1.0]),
            flux_max=jnp.array([1.0, 1.0, 1.0]),
            E_thermal=jnp.array([100.0, 100.0, 100.0]),
        )
        result_b = TranslationResult(
            times=times,
            positions=jnp.array([0.0, 1.2, 2.5]),  # Different trajectory
            velocities=jnp.array([1.0, 1.2, 1.3]),
            flux_max=jnp.array([1.0, 0.95, 0.9]),
            E_thermal=jnp.array([100.0, 110.0, 120.0]),
        )

        comparison = ModelComparisonResult(
            benchmark=benchmark,
            model_names=["model_a", "model_b"],
            results={"model_a": result_a, "model_b": result_b},
        )

        divergence = comparison.position_divergence()
        assert "model_a_vs_model_b" in divergence
        assert divergence["model_a_vs_model_b"] == pytest.approx(0.5)

    def test_flux_loss_comparison(self):
        """flux_loss_comparison returns per-model flux loss."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=1000.0)
        benchmark = TranslationBenchmark(
            name="test", description="Test", coil_field=coil
        )

        times = jnp.array([0.0, 1.0])
        result_a = TranslationResult(
            times=times,
            positions=jnp.array([0.0, 1.0]),
            velocities=jnp.array([1.0, 1.0]),
            flux_max=jnp.array([1.0, 0.8]),  # 20% loss
            E_thermal=jnp.array([100.0, 100.0]),
        )
        result_b = TranslationResult(
            times=times,
            positions=jnp.array([0.0, 1.0]),
            velocities=jnp.array([1.0, 1.0]),
            flux_max=jnp.array([1.0, 0.9]),  # 10% loss
            E_thermal=jnp.array([100.0, 100.0]),
        )

        comparison = ModelComparisonResult(
            benchmark=benchmark,
            model_names=["model_a", "model_b"],
            results={"model_a": result_a, "model_b": result_b},
        )

        flux_loss = comparison.flux_loss_comparison()
        assert flux_loss["model_a"] == pytest.approx(0.2)
        assert flux_loss["model_b"] == pytest.approx(0.1)


class TestBenchmarkCreation:
    """Tests for benchmark creation functions."""

    def test_create_mirror_push_benchmark(self):
        """create_mirror_push_benchmark returns valid benchmark."""
        benchmark = create_mirror_push_benchmark(
            coil_separation=2.0,
            coil_radius=0.5,
            coil_current=10000.0,
            initial_offset=0.1,
        )

        assert benchmark.name == "mirror_push_analytic"
        assert benchmark.coil_field is not None
        assert benchmark.analytic_trajectory is not None
        assert benchmark.analytic_trajectory.initial_position == 0.1

    def test_create_uniform_gradient_benchmark(self):
        """create_uniform_gradient_benchmark returns valid benchmark."""
        benchmark = create_uniform_gradient_benchmark(
            gradient=0.5,
            frc_magnetic_moment=1e-3,
            frc_mass=1e-6,
        )

        assert benchmark.name == "uniform_gradient"
        assert benchmark.analytic_trajectory is not None
        assert benchmark.analytic_trajectory.field_gradient == 0.5

    def test_create_staged_acceleration_benchmark(self):
        """create_staged_acceleration_benchmark returns valid benchmark."""
        benchmark = create_staged_acceleration_benchmark(
            n_stages=4,
            stage_spacing=0.5,
            coil_radius=0.3,
        )

        assert benchmark.name == "staged_acceleration_4"
        assert "4-stage" in benchmark.description
        assert benchmark.analytic_trajectory is None  # No analytic for staged


class TestComputeFieldGradient:
    """Tests for compute_field_gradient_at_point."""

    def test_gradient_at_coil_center(self):
        """Gradient at center of single coil should be approximately zero."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=10000.0)
        grad = compute_field_gradient_at_point(coil, r=0.0, z=0.0)
        # At coil center, gradient should be near zero by symmetry
        assert abs(grad) < 0.1  # Small gradient tolerance

    def test_gradient_away_from_coil(self):
        """Gradient away from coil should be non-zero."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=10000.0)
        grad = compute_field_gradient_at_point(coil, r=0.0, z=1.0)
        # Away from coil, field decreases so gradient is non-zero
        assert grad != 0.0


class TestTravelingWaveTiming:
    """Tests for traveling_wave_timing function."""

    def test_traveling_wave_creates_callable(self):
        """traveling_wave_timing returns a callable."""
        timing = traveling_wave_timing(
            n_coils=5,
            wave_speed=1000.0,
            coil_spacing=0.5,
            peak_current=10000.0,
        )
        assert callable(timing)

    def test_traveling_wave_at_t0(self):
        """At t=0, only first coil should be active."""
        timing = traveling_wave_timing(
            n_coils=3,
            wave_speed=1.0,  # 1 m/s
            coil_spacing=1.0,  # 1 m between coils
            peak_current=1000.0,
            rise_time=0.1,  # 0.1 s rise
        )
        currents = timing(0.0)
        # At t=0, wave is at z=0, so first coil just starting
        assert len(currents) == 3
        assert float(currents[0]) == pytest.approx(0.0, abs=1e-6)

    def test_traveling_wave_sequential_activation(self):
        """Coils activate sequentially as wave passes."""
        timing = traveling_wave_timing(
            n_coils=3,
            wave_speed=1.0,  # 1 m/s
            coil_spacing=1.0,  # 1 m between coils
            peak_current=1000.0,
            rise_time=0.1,
        )
        # At t=2s, wave at z=2m, so first two coils fully on
        currents = timing(2.0)
        assert float(currents[0]) == pytest.approx(1000.0)
        assert float(currents[1]) == pytest.approx(1000.0)
