"""AGATE reference data loader for validation cases."""
from __future__ import annotations

import json
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ZENODO_RECORD = "https://zenodo.org/api/records/15084058"


@dataclass
class AgateDataLoader:
    """Download and cache AGATE reference files from Zenodo."""

    cache_dir: Path = Path("validation/references/agate")
    record_url: str = ZENODO_RECORD

    def _fetch_record(self) -> dict:
        with urllib.request.urlopen(self.record_url) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _select_files(self, case: str, resolution: int) -> list[dict]:
        record = self._fetch_record()
        files = record.get("files", [])
        case = case.lower()
        res_token = str(resolution)
        if case == "gem":
            return [
                f
                for f in files
                if "gem" in f["key"].lower() and res_token in f["key"]
            ]
        if case == "ot":
            return [
                f
                for f in files
                if "ot" in f["key"].lower() and res_token in f["key"]
            ]
        raise ValueError(f"Unknown case: {case}")

    def _target_path(self, case: str, resolution: int, filename: str) -> Path:
        return Path(self.cache_dir) / case / str(resolution) / filename

    def _download_file(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp:
            dest.write_bytes(resp.read())
        return dest

    def _is_archive(self, path: Path) -> bool:
        suffix = "".join(path.suffixes).lower()
        return suffix.endswith(".zip") or suffix.endswith(".tar.gz") or suffix.endswith(".tgz")

    def _extract_archive(self, archive_path: Path, dest_dir: Path) -> list[Path]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest_dir)
                extracted = [dest_dir / name for name in zf.namelist()]
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                tf.extractall(dest_dir)
                extracted = [dest_dir / name for name in tf.getnames()]
        return extracted

    def ensure_files(self, case: str, resolution: int) -> list[Path]:
        """Ensure required files are present locally; generate if missing.

        Priority:
        1. Check if local data exists with valid config
        2. If missing/invalid -> run AGATE to generate
        3. Return paths to data files
        """
        # Map short names to full names
        case_map = {"ot": "orszag_tang", "gem": "gem_reconnection"}
        full_case = case_map.get(case.lower(), case.lower())

        output_dir = Path(self.cache_dir) / full_case / str(resolution)

        # Try local generation first
        try:
            from validation.utils.agate_runner import is_cache_valid, run_agate_simulation

            if not is_cache_valid(full_case, resolution, output_dir):
                print(f"Generating AGATE reference data for {full_case} at {resolution}...")
                run_agate_simulation(full_case, resolution, output_dir, overwrite=True)

            # Return all HDF5 files in the directory
            return list(output_dir.glob("*.h5"))

        except (ImportError, RuntimeError):
            # Fall back to Zenodo download if AGATE not available
            print("AGATE not installed, falling back to Zenodo download...")
            return self._ensure_files_zenodo(case, resolution)

    def _ensure_files_zenodo(self, case: str, resolution: int) -> list[Path]:
        """Original Zenodo download logic (fallback)."""
        files = self._select_files(case, resolution)
        local_paths: list[Path] = []
        for file_meta in files:
            key = file_meta["key"]
            url = file_meta["links"]["self"]
            filename = Path(key).name
            target = self._target_path(case, resolution, filename)
            if self._is_archive(target):
                if not target.exists():
                    self._download_file(url, target)
                extracted = self._extract_archive(target, target.parent)
                local_paths.extend(extracted)
                continue
            if not target.exists():
                self._download_file(url, target)
            local_paths.append(target)
        return local_paths
