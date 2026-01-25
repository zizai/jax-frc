"""Reference data management for validation."""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import jax.numpy as jnp


@dataclass
class ReferenceData:
    """Container for loaded reference data."""
    data: dict
    source_type: str
    source_path: Optional[str] = None


@dataclass
class ReferenceManager:
    """Manages loading and caching of reference data."""

    base_dir: Path = field(default_factory=lambda: Path("validation"))
    cache: dict = field(default_factory=dict)

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)

    def load(self, ref_config: dict, params: Optional[dict] = None) -> ReferenceData:
        """Load reference based on config type."""
        ref_type = ref_config.get('type', 'file')

        if ref_type == 'analytic':
            data = self.evaluate_analytic(ref_config['formula'], params or {})
            return ReferenceData(data={'result': data}, source_type='analytic')
        elif ref_type == 'file':
            data = self.load_file(ref_config)
            return ReferenceData(data=data, source_type='file', source_path=ref_config['path'])
        elif ref_type == 'external':
            raise NotImplementedError("External reference loading not yet implemented")
        else:
            raise ValueError(f"Unknown reference type: {ref_type}")

    def evaluate_analytic(self, formula: str, params: dict) -> jnp.ndarray:
        """Safely evaluate analytic formula.

        Args:
            formula: Python expression using jnp functions
            params: Variables to substitute (including arrays like 'x', 't')

        Returns:
            Evaluated array
        """
        # Build safe namespace with JAX numpy
        safe_namespace = {'jnp': jnp}
        safe_namespace.update(params)

        # Evaluate formula
        return eval(formula, {"__builtins__": {}}, safe_namespace)

    def load_file(self, ref_config: dict) -> dict:
        """Load reference data from CSV file.

        Args:
            ref_config: Config with 'path' and optional 'columns' mapping

        Returns:
            dict mapping column names to arrays
        """
        path = self.base_dir / ref_config['path']

        if str(path) in self.cache:
            return self.cache[str(path)]

        data = {}
        columns = ref_config.get('columns', {})

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Get column names from mapping or use file headers
            for csv_col in rows[0].keys():
                col_name = columns.get(csv_col, csv_col)
                values = [float(row[csv_col]) for row in rows]
                data[col_name] = jnp.array(values)

        self.cache[str(path)] = data
        return data
