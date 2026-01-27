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
        timing: Dictionary of timing measurements in seconds.
        warnings: List of warning messages.
    """

    name: str
    description: str
    docstring: str
    configuration: dict
    metrics: dict
    overall_pass: bool
    plots: list = field(default_factory=list)
    timing: Optional[dict] = None
    warnings: Optional[list] = None

    def add_plot(self, fig, name: str = None):
        """Add matplotlib figure, converts to base64 PNG for embedding.

        Args:
            fig: A matplotlib Figure object.
            name: Optional name for the plot. If not provided, generates
                  an automatic name like 'plot_0', 'plot_1', etc.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        self.plots.append({
            'name': name or f'plot_{len(self.plots)}',
            'data': img_base64,
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
            value = data.get('value', 'N/A')
            if isinstance(value, float):
                value = f"{value:.4g}"
            threshold = data.get('threshold', 'N/A')
            desc = data.get('description', '')
            metrics_rows += f"""<tr>
                <td>{name}</td>
                <td>{value}</td>
                <td>{threshold}</td>
                <td style="color: {status_color}; font-weight: bold;">{status}</td>
                <td>{desc}</td>
            </tr>"""

        # Plots
        plots_html = ""
        for plot in self.plots:
            plots_html += f"""
            <div class="plot">
                <h3>{plot['name']}</h3>
                <img src="data:image/png;base64,{plot['data']}" alt="{plot['name']}">
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
        .plot {{ margin: 20px 0; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
        .physics {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; }}
        .warnings {{ color: #856404; background: #fff3cd; padding: 15px; border-radius: 4px; }}
        code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>{self.name} <span class="badge" style="background: {badge_color}">{pass_badge}</span></h1>
    <p><strong>{self.description}</strong></p>

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

    <h2>Plots</h2>
    {plots_html}

    {timing_html}
    {warnings_html}

    <footer style="margin-top: 40px; color: #666; font-size: 0.9em;">
        Generated: {datetime.now().isoformat()}
    </footer>
</body>
</html>"""
