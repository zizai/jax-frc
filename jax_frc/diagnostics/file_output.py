"""File output management for diagnostic plots.

Handles directory creation, figure saving, and environment detection.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


def is_notebook() -> bool:
    """Detect if running in a Jupyter notebook.

    Returns:
        True if in notebook environment, False otherwise.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if 'ZMQInteractiveShell' in shell_name:
            return True
        return False
    except (ImportError, NameError):
        return False


def create_output_dir(
    model_name: str,
    base_dir: str = "outputs"
) -> str:
    """Create timestamped output directory for a simulation run.

    Args:
        model_name: Name of the physics model (e.g., "resistive_mhd")
        base_dir: Base directory for outputs (default: "outputs")

    Returns:
        Full path to created directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{model_name}_{timestamp}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_figure(
    fig: plt.Figure,
    name: str,
    directory: str,
    format: str = 'png',
    dpi: int = 150
) -> str:
    """Save a matplotlib figure with consistent settings.

    Args:
        fig: Matplotlib figure to save
        name: Base name for the file (without extension)
        directory: Directory to save in
        format: Output format ('png', 'pdf', or 'both')
        dpi: Resolution for raster formats

    Returns:
        Path to saved file (or first file if 'both').
    """
    os.makedirs(directory, exist_ok=True)
    paths = []

    if format in ('png', 'both'):
        path = os.path.join(directory, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        paths.append(path)

    if format in ('pdf', 'both'):
        path = os.path.join(directory, f"{name}.pdf")
        fig.savefig(path, bbox_inches='tight')
        paths.append(path)

    return paths[0] if paths else ""


def generate_index_html(directory: str) -> str:
    """Generate an index.html file listing all plots in directory.

    Args:
        directory: Directory containing plot files

    Returns:
        Path to generated index.html
    """
    images = sorted([
        f for f in os.listdir(directory)
        if f.endswith(('.png', '.pdf'))
    ])

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Simulation Plots</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { color: #333; }
        .gallery { display: flex; flex-wrap: wrap; gap: 20px; }
        .plot { border: 1px solid #ddd; padding: 10px; }
        .plot img { max-width: 400px; height: auto; }
        .plot p { margin: 5px 0; font-size: 14px; }
    </style>
</head>
<body>
    <h1>Simulation Results</h1>
    <div class="gallery">
"""

    for img in images:
        if img.endswith('.png'):
            html_content += f"""        <div class="plot">
            <img src="{img}" alt="{img}">
            <p>{img}</p>
        </div>
"""

    html_content += """    </div>
</body>
</html>
"""

    index_path = os.path.join(directory, "index.html")
    with open(index_path, 'w') as f:
        f.write(html_content)

    return index_path
