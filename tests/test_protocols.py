"""Tests for physics model protocols."""

import jax.numpy as jnp
from jax_frc.models.protocols import SplitRHS, SourceTerms


def test_split_rhs_protocol_exists():
    """SplitRHS protocol is importable and has required methods."""
    assert hasattr(SplitRHS, 'explicit_rhs')
    assert hasattr(SplitRHS, 'implicit_rhs')
    assert hasattr(SplitRHS, 'apply_implicit_operator')


def test_source_terms_protocol_exists():
    """SourceTerms protocol is importable and has required method."""
    assert hasattr(SourceTerms, 'compute_sources')
