"""Tests for diagnostics plotting module."""

import pytest
import tempfile
import os
from pathlib import Path


class TestStyles:
    """Tests for styles module."""

    def test_apply_style_sets_seaborn_theme(self):
        """Verify apply_style configures seaborn."""
        from jax_frc.diagnostics.styles import apply_style
        import matplotlib.pyplot as plt

        apply_style()

        # Check that figure size was set
        assert plt.rcParams['figure.figsize'] == [10.0, 6.0]

    def test_field_colors_has_required_keys(self):
        """Verify all expected field colormaps are defined."""
        from jax_frc.diagnostics.styles import FIELD_COLORS

        required = ['psi', 'B', 'n', 'p']
        for key in required:
            assert key in FIELD_COLORS

    def test_labels_has_required_keys(self):
        """Verify all expected labels are defined."""
        from jax_frc.diagnostics.styles import LABELS

        required = ['psi', 'B', 'n', 'p', 't']
        for key in required:
            assert key in LABELS


class TestFileOutput:
    """Tests for file output management."""

    def test_create_output_dir_creates_timestamped_directory(self):
        """Verify output directory is created with model name and timestamp."""
        from jax_frc.diagnostics.file_output import create_output_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = create_output_dir(
                model_name="resistive_mhd",
                base_dir=tmpdir
            )

            assert os.path.isdir(result_dir)
            assert "resistive_mhd" in result_dir
            # Should contain date pattern YYYY-MM-DD
            assert "2026" in result_dir or "202" in result_dir

    def test_is_notebook_returns_bool(self):
        """Verify notebook detection returns boolean."""
        from jax_frc.diagnostics.file_output import is_notebook

        result = is_notebook()
        assert isinstance(result, bool)
        # In pytest, we're not in a notebook
        assert result is False

    def test_save_figure_creates_file(self):
        """Verify save_figure writes PNG file."""
        from jax_frc.diagnostics.file_output import save_figure
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, "test_plot", tmpdir, format='png')

            assert os.path.isfile(path)
            assert path.endswith('.png')

        plt.close(fig)


import jax.numpy as jnp
from jax_frc.results import SimulationResult


def make_mock_result(model_name: str = "resistive_mhd") -> SimulationResult:
    """Create a mock SimulationResult for testing."""
    nr, nz = 32, 64
    n_steps = 10

    # Create mock history as dict with time series
    history = {
        'time': jnp.linspace(0, 1e-3, n_steps),
        'magnetic_energy': jnp.linspace(1.0, 0.8, n_steps),
        'kinetic_energy': jnp.linspace(0.0, 0.2, n_steps),
        'max_B': jnp.linspace(0.5, 0.4, n_steps),
    }

    return SimulationResult(
        model_name=model_name,
        final_time=1e-3,
        n_steps=n_steps,
        grid_shape=(nr, nz),
        grid_spacing=(0.01, 0.01),
        psi=jnp.zeros((nr, nz)),
        density=jnp.ones((nr, nz)) * 1e20,
        pressure=jnp.ones((nr, nz)) * 1000,
        history=history,
    )


class TestPlotTimeTraces:
    """Tests for plot_time_traces function."""

    def test_plot_time_traces_returns_figure(self):
        """Verify plot_time_traces returns matplotlib figure."""
        from jax_frc.diagnostics.plotting import plot_time_traces
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_time_traces(result, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1  # At least one subplot
        plt.close(fig)

    def test_plot_time_traces_saves_file(self):
        """Verify plot_time_traces saves to directory when specified."""
        from jax_frc.diagnostics.plotting import plot_time_traces
        import matplotlib.pyplot as plt

        result = make_mock_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = plot_time_traces(result, show=False, save_dir=tmpdir)

            files = os.listdir(tmpdir)
            assert any('time_traces' in f for f in files)

        plt.close(fig)


class TestPlotFields:
    """Tests for plot_fields function."""

    def test_plot_fields_returns_figure(self):
        """Verify plot_fields returns matplotlib figure."""
        from jax_frc.diagnostics.plotting import plot_fields
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_fields(result, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fields_with_time_index(self):
        """Verify plot_fields accepts time index."""
        from jax_frc.diagnostics.plotting import plot_fields
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_fields(result, t=0, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fields_saves_files(self):
        """Verify plot_fields saves PNG files."""
        from jax_frc.diagnostics.plotting import plot_fields
        import matplotlib.pyplot as plt

        result = make_mock_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = plot_fields(result, show=False, save_dir=tmpdir)
            files = os.listdir(tmpdir)
            # Should have at least one field plot
            assert len(files) >= 1

        plt.close(fig)


class TestPlotProfiles:
    """Tests for plot_profiles function."""

    def test_plot_profiles_returns_figure(self):
        """Verify plot_profiles returns matplotlib figure."""
        from jax_frc.diagnostics.plotting import plot_profiles
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_profiles(result, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_profiles_radial_axis(self):
        """Verify plot_profiles works with axis='r'."""
        from jax_frc.diagnostics.plotting import plot_profiles
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_profiles(result, axis='r', show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_profiles_axial_axis(self):
        """Verify plot_profiles works with axis='z'."""
        from jax_frc.diagnostics.plotting import plot_profiles
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_profiles(result, axis='z', show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def make_hybrid_result() -> SimulationResult:
    """Create a mock SimulationResult for hybrid kinetic model."""
    nr, nz = 32, 64
    n_particles = 100

    particles = {
        'x': jnp.column_stack([
            jnp.linspace(0.1, 0.3, n_particles),  # r
            jnp.zeros(n_particles),                # theta
            jnp.linspace(-0.5, 0.5, n_particles), # z
        ]),
        'v': jnp.column_stack([
            jnp.linspace(-1e5, 1e5, n_particles),  # vr
            jnp.zeros(n_particles),                 # vtheta
            jnp.linspace(-1e5, 1e5, n_particles),  # vz
        ]),
        'w': jnp.ones(n_particles) * 0.1,
    }

    return SimulationResult(
        model_name="hybrid_kinetic",
        final_time=1e-6,
        n_steps=10,
        grid_shape=(nr, nz),
        grid_spacing=(0.01, 0.01),
        particles=particles,
        history={'time': jnp.linspace(0, 1e-6, 10)},
    )


class TestPlotParticles:
    """Tests for plot_particles function."""

    def test_plot_particles_returns_figure(self):
        """Verify plot_particles returns matplotlib figure."""
        from jax_frc.diagnostics.plotting import plot_particles
        import matplotlib.pyplot as plt

        result = make_hybrid_result()
        fig = plot_particles(result, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_particles_skips_non_hybrid(self):
        """Verify plot_particles returns None for non-hybrid models."""
        from jax_frc.diagnostics.plotting import plot_particles
        import matplotlib.pyplot as plt

        result = make_mock_result()  # resistive_mhd, no particles
        fig = plot_particles(result, show=False, save_dir=None)

        assert fig is None
