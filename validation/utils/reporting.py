"""HTML report generation for validation cases."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime
import base64
import io


@dataclass
class ValidationReport:
    """Container for validation results that generates HTML reports.

    Attributes:
        name: Identifier for the validation case (used in directory names).
        description: Short description of what this validation tests.
        docstring: Physics background and methodology documentation.
        configuration: Dictionary of configuration parameters used.
        metrics: Dictionary of metric results, each containing:
            - value: The computed metric value
            - threshold: The pass/fail threshold (optional)
            - passed: Whether the metric passed (optional)
            - description: Human-readable description (optional)
        overall_pass: Whether the validation passed overall.
        plots: List of embedded plots (added via add_plot method).
        animations: List of embedded animations (added via add_animation method).
        timing: Dictionary of timing measurements in seconds.
        warnings: List of warning messages.
        summary: Optional summary text for the report.
    """

    name: str
    description: str
    docstring: str
    configuration: dict
    metrics: dict
    overall_pass: bool
    plots: list = field(default_factory=list)
    animations: list = field(default_factory=list)
    timing: Optional[dict] = None
    warnings: Optional[list] = None
    summary: Optional[str] = None

    def add_plot(self, fig, name: str = None, caption: str = None):
        """Add matplotlib figure, converts to base64 PNG for embedding.

        Args:
            fig: A matplotlib Figure object.
            name: Optional name for the plot. If not provided, generates
                  an automatic name like 'plot_0', 'plot_1', etc.
            caption: Optional caption describing the plot.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        self.plots.append({
            'name': name or f'plot_{len(self.plots)}',
            'data': img_base64,
            'caption': caption,
        })

    def add_animation(self, anim, name: str = None, caption: str = None, fps: int = 10):
        """Add matplotlib animation, converts to base64 GIF for embedding.

        Args:
            anim: A matplotlib FuncAnimation object.
            name: Optional name for the animation.
            caption: Optional caption describing the animation.
            fps: Frames per second for the GIF.
        """
        import tempfile
        import os

        # Save to temporary file (pillow writer doesn't support BytesIO)
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            anim.save(tmp_path, writer='pillow', fps=fps)
            with open(tmp_path, 'rb') as f:
                gif_base64 = base64.b64encode(f.read()).decode('utf-8')
        finally:
            os.unlink(tmp_path)

        self.animations.append({
            'name': name or f'animation_{len(self.animations)}',
            'data': gif_base64,
            'caption': caption,
        })

    def add_animation_from_file(self, filepath: Path, name: str = None, caption: str = None):
        """Add animation from existing GIF file.

        Args:
            filepath: Path to the GIF file.
            name: Optional name for the animation.
            caption: Optional caption describing the animation.
        """
        with open(filepath, 'rb') as f:
            gif_base64 = base64.b64encode(f.read()).decode('utf-8')
        self.animations.append({
            'name': name or filepath.stem,
            'data': gif_base64,
            'caption': caption,
        })

    def add_plot_from_file(self, filepath: Path, name: str = None, caption: str = None):
        """Add plot from existing PNG file.

        Args:
            filepath: Path to the PNG file.
            name: Optional name for the plot.
            caption: Optional caption describing the plot.
        """
        with open(filepath, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        self.plots.append({
            'name': name or filepath.stem,
            'data': img_base64,
            'caption': caption,
        })

    def save(self, base_dir: Path = None) -> Path:
        """Write report.html to timestamped directory.

        Args:
            base_dir: Base directory for reports. Defaults to 'validation/reports'.

        Returns:
            Path to the created report directory.
        """
        if base_dir is None:
            base_dir = Path("validation/reports")

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = base_dir / f"{timestamp}_{self.name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        html = self._generate_html()
        (output_dir / "report.html").write_text(html)

        return output_dir

    def _generate_html(self) -> str:
        """Generate self-contained HTML report.

        Returns:
            Complete HTML document as a string.
        """
        pass_badge = "PASS" if self.overall_pass else "FAIL"
        badge_color = "#28a745" if self.overall_pass else "#dc3545"

        # Summary section
        summary_html = ""
        if self.summary:
            summary_html = f"""
            <div class="summary">
                <h2>Summary</h2>
                <p>{self.summary}</p>
            </div>"""

        # Configuration table
        config_rows = "\n".join(
            f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
            for k, v in self.configuration.items()
        )

        # Metrics table
        metrics_rows = ""
        for name, data in self.metrics.items():
            status = "PASS" if data.get('passed', True) else "FAIL"
            status_color = "#28a745" if data.get('passed', True) else "#dc3545"
            # Support both old format (value) and new format (jax_value, agate_value, l2_error, relative_error)
            if 'jax_value' in data:
                jax_val = data.get('jax_value', 'N/A')
                agate_val = data.get('agate_value', 'N/A')
                l2_err = data.get('l2_error', 'N/A')
                rel_err = data.get('relative_error', l2_err)  # fallback to l2_error if not present
                if isinstance(jax_val, float):
                    jax_val = f"{jax_val:.4g}"
                if isinstance(agate_val, float):
                    agate_val = f"{agate_val:.4g}"
                if isinstance(l2_err, float):
                    l2_err = f"{l2_err:.4g}"
                if isinstance(rel_err, float):
                    rel_err = f"{rel_err:.4g}"
                value = f"JAX: {jax_val}, AGATE: {agate_val}, L2: {l2_err}, Rel: {rel_err}"
            else:
                value = data.get('value', 'N/A')
                if isinstance(value, float):
                    value = f"{value:.4g}"
            threshold = data.get('threshold', 'N/A')
            threshold_type = data.get('threshold_type', '')
            if isinstance(threshold, float):
                threshold = f"{threshold:.4g}"
            if threshold_type:
                threshold = f"{threshold} ({threshold_type})"
            desc = data.get('description', '')
            metrics_rows += f"""<tr>
                <td>{name}</td>
                <td>{value}</td>
                <td>{threshold}</td>
                <td style="color: {status_color}; font-weight: bold;">{status}</td>
                <td>{desc}</td>
            </tr>"""

        # Animations
        animations_html = ""
        if self.animations:
            animations_html = "<h2>Animations</h2>"
            for anim in self.animations:
                caption_html = f"<p class='caption'>{anim['caption']}</p>" if anim.get('caption') else ""
                animations_html += f"""
                <div class="animation">
                    <h3>{anim['name']}</h3>
                    <img src="data:image/gif;base64,{anim['data']}" alt="{anim['name']}">
                    {caption_html}
                </div>"""

        # Plots
        plots_html = ""
        if self.plots:
            plots_html = "<h2>Plots</h2>"
            for plot in self.plots:
                caption_html = f"<p class='caption'>{plot['caption']}</p>" if plot.get('caption') else ""
                plots_html += f"""
                <div class="plot">
                    <h3>{plot['name']}</h3>
                    <img src="data:image/png;base64,{plot['data']}" alt="{plot['name']}">
                    {caption_html}
                </div>"""

        # Timing
        timing_html = ""
        if self.timing:
            timing_rows = "\n".join(
                f"<tr><td>{k}</td><td>{v:.2f}s</td></tr>"
                for k, v in self.timing.items()
            )
            timing_html = f"""
            <h2>Timing</h2>
            <table><tbody>{timing_rows}</tbody></table>"""

        # Warnings
        warnings_html = ""
        if self.warnings:
            warning_items = "\n".join(f"<li>{w}</li>" for w in self.warnings)
            warnings_html = f"""
            <h2>Warnings</h2>
            <ul class="warnings">{warning_items}</ul>"""

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.name} - Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .badge {{ display: inline-block; padding: 5px 15px; border-radius: 4px;
                  color: white; font-weight: bold; margin-left: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .plot, .animation {{ margin: 20px 0; text-align: center; }}
        .plot img, .animation img {{ max-width: 100%; border: 1px solid #ddd; }}
        .caption {{ font-style: italic; color: #666; margin-top: 8px; }}
        .physics {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; }}
        .summary {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #17a2b8; margin: 20px 0; }}
        .warnings {{ color: #856404; background: #fff3cd; padding: 15px; border-radius: 4px; }}
        code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; }}
    </style>
</head>
<body>
    <h1>{self.name} <span class="badge" style="background: {badge_color}">{pass_badge}</span></h1>
    <p><strong>{self.description}</strong></p>

    {summary_html}

    <div class="physics">
        <h2>Physics Background</h2>
        <pre>{self.docstring}</pre>
    </div>

    <h2>Configuration</h2>
    <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>{config_rows}</tbody>
    </table>

    <h2>Results</h2>
    <table>
        <thead><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th><th>Description</th></tr></thead>
        <tbody>{metrics_rows}</tbody>
    </table>

    {animations_html}
    {plots_html}

    {timing_html}
    {warnings_html}

    <footer style="margin-top: 40px; color: #666; font-size: 0.9em;">
        Generated: {datetime.now().isoformat()}
    </footer>
</body>
</html>"""


def print_field_l2_table(field_errors: dict, threshold: float) -> None:
    """Print formatted table of field L2 errors to console."""
    print("  Field L2 Errors:")
    print(f"    {'Field':<18} {'L2 Error':<12} {'Threshold':<12} {'Status'}")
    print("    " + "-" * 54)
    for field, error in field_errors.items():
        status = "PASS" if error <= threshold else "FAIL"
        print(f"    {field:<18} {error:<12.4g} {threshold:<12.4g} {status}")


def print_scalar_metrics_table(metrics: dict) -> None:
    """Print formatted table of scalar metric comparisons to console."""
    print("  Scalar Metrics:")
    print(f"    {'Metric':<18} {'JAX Value':<12} {'AGATE Value':<12} "
          f"{'Rel Error':<10} {'Threshold':<10} {'Status'}")
    print("    " + "-" * 74)
    for name, data in metrics.items():
        jax_val = data['jax_value']
        agate_val = data['agate_value']
        rel_err = data['relative_error']
        threshold = data['threshold']
        status = "PASS" if data['passed'] else "FAIL"
        print(f"    {name:<18} {jax_val:<12.4g} {agate_val:<12.4g} "
              f"{rel_err:<10.2%} {threshold:<10.2%} {status}")
