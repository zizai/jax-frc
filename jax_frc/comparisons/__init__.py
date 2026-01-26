"""Comparison frameworks for physics model validation.

This module provides tools for comparing different simulation models
against each other and against published reference results.
"""

from jax_frc.comparisons.belova_merging import (
    MergingResult,
    ComparisonReport,
    BelovaComparisonSuite,
)

__all__ = [
    "MergingResult",
    "ComparisonReport",
    "BelovaComparisonSuite",
]
