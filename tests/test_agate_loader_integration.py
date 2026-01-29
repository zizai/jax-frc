# tests/test_agate_loader_integration.py
"""Tests for AgateDataLoader integration with AGATE runner."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from validation.utils.agate_data import AgateDataLoader


def test_ensure_files_calls_agate_runner_when_cache_invalid():
    """ensure_files should call AGATE runner when cache is invalid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AgateDataLoader(cache_dir=Path(tmpdir))
        output_dir = Path(tmpdir) / "orszag_tang" / "64"

        # Create mock HDF5 files that would be generated
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_file = output_dir / "orszag_tang_64_grid.h5"
        state_file = output_dir / "orszag_tang_64_state.h5"
        grid_file.touch()
        state_file.touch()

        with patch("validation.utils.agate_runner.is_cache_valid", return_value=False) as mock_valid, \
             patch("validation.utils.agate_runner.run_agate_simulation") as mock_run:

            paths = loader.ensure_files("ot", 64)

            # Verify AGATE runner was called
            mock_valid.assert_called_once_with("orszag_tang", 64, output_dir)
            mock_run.assert_called_once_with("orszag_tang", 64, output_dir, overwrite=True)

            # Verify HDF5 files are returned
            assert len(paths) == 2
            assert grid_file in paths
            assert state_file in paths


def test_ensure_files_skips_generation_when_cache_valid():
    """ensure_files should skip generation when cache is valid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AgateDataLoader(cache_dir=Path(tmpdir))
        output_dir = Path(tmpdir) / "orszag_tang" / "64"

        # Create mock HDF5 files
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_file = output_dir / "orszag_tang_64_grid.h5"
        grid_file.touch()

        with patch("validation.utils.agate_runner.is_cache_valid", return_value=True) as mock_valid, \
             patch("validation.utils.agate_runner.run_agate_simulation") as mock_run:

            paths = loader.ensure_files("ot", 64)

            # Verify cache was checked but simulation was NOT run
            mock_valid.assert_called_once()
            mock_run.assert_not_called()

            # Verify existing files are returned
            assert len(paths) == 1
            assert grid_file in paths


def test_ensure_files_falls_back_to_zenodo_on_import_error():
    """ensure_files should fall back to Zenodo when AGATE import fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AgateDataLoader(cache_dir=Path(tmpdir))

        # Mock the import to fail
        with patch.dict("sys.modules", {"validation.utils.agate_runner": None}):
            with patch.object(loader, "_ensure_files_zenodo", return_value=[]) as mock_zenodo:
                paths = loader.ensure_files("ot", 64)

                # Verify fallback was called
                mock_zenodo.assert_called_once_with("ot", 64)


def test_ensure_files_falls_back_to_zenodo_on_runtime_error():
    """ensure_files should fall back to Zenodo when AGATE raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AgateDataLoader(cache_dir=Path(tmpdir))

        with patch("validation.utils.agate_runner.is_cache_valid", return_value=False), \
             patch("validation.utils.agate_runner.run_agate_simulation",
                   side_effect=RuntimeError("AGATE not installed")):
            with patch.object(loader, "_ensure_files_zenodo", return_value=[]) as mock_zenodo:
                paths = loader.ensure_files("ot", 64)

                # Verify fallback was called
                mock_zenodo.assert_called_once_with("ot", 64)


def test_ensure_files_maps_case_names_correctly():
    """ensure_files should map short case names to full names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AgateDataLoader(cache_dir=Path(tmpdir))

        # Test "ot" -> "orszag_tang"
        with patch("validation.utils.agate_runner.is_cache_valid", return_value=True) as mock_valid:
            loader.ensure_files("ot", 64)
            call_args = mock_valid.call_args[0]
            assert call_args[0] == "orszag_tang"

        # Test "gem" -> "gem_reconnection"
        with patch("validation.utils.agate_runner.is_cache_valid", return_value=True) as mock_valid:
            loader.ensure_files("gem", 128)
            call_args = mock_valid.call_args[0]
            assert call_args[0] == "gem_reconnection"


@pytest.mark.slow
def test_ensure_files_generates_missing_data():
    """ensure_files should generate data if missing (requires AGATE)."""
    pytest.importorskip("agate", reason="AGATE not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AgateDataLoader(cache_dir=Path(tmpdir))

        # This should trigger AGATE simulation
        paths = loader.ensure_files("ot", 64)

        # Check files were generated
        assert len(paths) > 0
        assert any("grid" in str(p) for p in paths)
