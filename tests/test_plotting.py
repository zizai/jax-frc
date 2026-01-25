"""Tests for diagnostics plotting module."""

import pytest


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
