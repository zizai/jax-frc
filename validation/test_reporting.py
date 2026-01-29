"""Quick verification of full-mode reporting output."""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Temporarily override resolutions and t_end for faster testing
import validation.cases.regression.orszag_tang as ot

# Save original values
orig_resolutions = ot.RESOLUTIONS

# Override for fast test
ot.RESOLUTIONS = (256,)

# Patch setup_configuration to use shorter t_end
orig_setup = ot.setup_configuration
def fast_setup(quick_test, resolution):
    cfg = orig_setup(quick_test, resolution)
    cfg['t_end'] = 0.001  # Very short simulation
    return cfg
ot.setup_configuration = fast_setup

# Run full mode (not quick)
try:
    success = ot.main(quick_test=False)
finally:
    # Restore
    ot.RESOLUTIONS = orig_resolutions
    ot.setup_configuration = orig_setup

sys.exit(0 if success else 1)
