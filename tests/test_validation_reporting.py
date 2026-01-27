"""Tests for validation report generation."""
import pytest
from pathlib import Path
import tempfile


@pytest.fixture(autouse=True)
def use_agg_backend():
    """Use non-interactive Agg backend for matplotlib tests."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use('Agg')


def test_validation_report_creation():
    """ValidationReport can be created with required fields."""
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case",
        docstring="Test docstring",
        configuration={'param': 1.0},
        metrics={'l2_error': {'value': 0.05, 'threshold': 0.1, 'passed': True}},
        overall_pass=True,
    )

    assert report.name == "test_case"
    assert report.overall_pass is True


def test_validation_report_default_fields():
    """ValidationReport has sensible defaults for optional fields."""
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case",
        docstring="Test docstring",
        configuration={'param': 1.0},
        metrics={},
        overall_pass=True,
    )

    assert report.plots == []
    assert report.timing is None
    assert report.warnings is None


def test_add_plot():
    """add_plot converts matplotlib figure to base64 and stores it."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case",
        docstring="Test docstring",
        configuration={},
        metrics={},
        overall_pass=True,
    )

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    report.add_plot(fig, name="test_plot")
    plt.close(fig)

    assert len(report.plots) == 1
    assert report.plots[0]['name'] == "test_plot"
    assert 'data' in report.plots[0]
    # Base64 PNG starts with iVBOR
    assert report.plots[0]['data'].startswith('iVBOR')


def test_add_plot_auto_name():
    """add_plot generates automatic names when not provided."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case",
        docstring="Test docstring",
        configuration={},
        metrics={},
        overall_pass=True,
    )

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    report.add_plot(fig)
    plt.close(fig)

    assert report.plots[0]['name'] == "plot_0"


def test_generate_html_contains_key_elements():
    """Generated HTML contains all expected sections."""
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case description",
        docstring="Physics background\nwith multiple lines",
        configuration={'param1': 1.0, 'param2': 'value'},
        metrics={
            'l2_error': {
                'value': 0.05,
                'threshold': 0.1,
                'passed': True,
                'description': 'L2 norm error'
            }
        },
        overall_pass=True,
        timing={'simulation': 1.5, 'analysis': 0.3},
        warnings=['Warning 1', 'Warning 2'],
    )

    html = report._generate_html()

    # Check title and badge
    assert 'test_case' in html
    assert 'PASS' in html
    assert '#28a745' in html  # Green color for pass

    # Check description
    assert 'A test case description' in html

    # Check docstring/physics section
    assert 'Physics background' in html

    # Check configuration table
    assert 'param1' in html
    assert '1.0' in html
    assert 'param2' in html
    assert 'value' in html

    # Check metrics table
    assert 'l2_error' in html
    assert '0.05' in html
    assert '0.1' in html
    assert 'L2 norm error' in html

    # Check timing section
    assert 'Timing' in html
    assert 'simulation' in html
    assert '1.50s' in html

    # Check warnings section
    assert 'Warnings' in html
    assert 'Warning 1' in html
    assert 'Warning 2' in html


def test_generate_html_fail_badge():
    """Generated HTML shows FAIL badge with red color when overall_pass=False."""
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="failing_test",
        description="A failing test",
        docstring="",
        configuration={},
        metrics={},
        overall_pass=False,
    )

    html = report._generate_html()

    assert 'FAIL' in html
    assert '#dc3545' in html  # Red color for fail


def test_save_creates_report_file():
    """save() creates a timestamped directory with report.html."""
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case",
        docstring="",
        configuration={},
        metrics={},
        overall_pass=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        output_dir = report.save(base_dir=base_dir)

        assert output_dir.exists()
        assert 'test_case' in output_dir.name
        assert (output_dir / 'report.html').exists()

        html_content = (output_dir / 'report.html').read_text()
        assert 'test_case' in html_content


def test_save_with_plots():
    """save() includes embedded plots in the HTML."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="plot_test",
        description="Test with plots",
        docstring="",
        configuration={},
        metrics={},
        overall_pass=True,
    )

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    report.add_plot(fig, name="my_plot")
    plt.close(fig)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        output_dir = report.save(base_dir=base_dir)

        html_content = (output_dir / 'report.html').read_text()
        assert 'my_plot' in html_content
        assert 'data:image/png;base64,' in html_content


def test_import_from_utils_package():
    """ValidationReport can be imported from validation.utils."""
    from validation.utils import ValidationReport

    assert ValidationReport is not None


def test_plot_comparison():
    """plot_comparison creates overlay of simulation vs expected."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from validation.utils.plotting import plot_comparison

    x = np.linspace(-1, 1, 50)
    actual = x**2 + 0.01 * np.random.randn(50)
    expected = x**2

    fig = plot_comparison(
        x, actual, expected,
        labels=['Simulation', 'Analytic'],
        title='Test Comparison',
        xlabel='x',
        ylabel='y'
    )

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_title() == 'Test Comparison'
    plt.close(fig)


def test_plot_error():
    """plot_error creates error plot with zero reference line."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from validation.utils.plotting import plot_error

    x = np.linspace(-1, 1, 50)
    actual = x**2 + 0.05
    expected = x**2

    fig = plot_error(x, actual, expected, title='Test Error')

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # error line + zero reference
    assert ax.get_title() == 'Test Error'
    plt.close(fig)
