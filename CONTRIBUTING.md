# Contributing to jax-frc

Thank you for your interest in contributing to jax-frc!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/jax-frc.git
cd jax-frc

# Install dependencies
pip install jax jaxlib numpy matplotlib pytest

# Verify installation
py -m pytest tests/ -v -k "not slow"
```

## Code Style

- **Formatting**: Use Black with default settings
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for classes and public methods

```python
def compute_rhs(self, state: State, geometry: Geometry) -> State:
    """Compute time derivatives for state variables.

    Args:
        state: Current simulation state
        geometry: Computational grid

    Returns:
        State with time derivatives in each field
    """
```

## Pull Request Workflow

1. **Create a branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow patterns in existing code
3. **Add tests**: All new functionality needs tests
4. **Run tests**: `py -m pytest tests/ -v`
5. **Commit**: Use conventional commits (`feat:`, `fix:`, `docs:`)
6. **Push**: `git push -u origin feature/your-feature`
7. **Open PR**: Fill in the template

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add D-He3 reaction channel
fix: correct boundary condition at r=0
docs: update Extended MHD equations
refactor: extract resistivity models
test: add conservation law checks
```

## Testing Requirements

### For Bug Fixes

- Add a test that fails without the fix
- Verify the test passes with the fix

### For New Features

- Unit tests for new functions/classes
- Integration test if feature affects simulation flow
- Property-based tests for physics (conservation, bounds)

### For Physics Models

- Conservation law tests (energy, momentum, flux)
- Boundedness tests (positive density, valid temperatures)
- Known solution tests where analytical solutions exist

See [Testing Guide](docs/testing/index.md) for details.

## Architecture

See [Architecture Overview](docs/developer/architecture.md) for:
- Module structure
- Key abstractions (PhysicsModel, Solver, State)
- Extension points

## JAX-Specific Guidelines

See [JAX Patterns](docs/developer/jax-patterns.md) for:
- Using `lax.scan` and `lax.cond`
- Static arguments for JIT
- Debugging tracing issues

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
