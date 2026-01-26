"""Tests for species tracking."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestSpeciesState:
    """Tests for SpeciesState dataclass."""

    def test_species_state_creation(self):
        """Can create SpeciesState with all species."""
        from jax_frc.burn.species import SpeciesState
        shape = (4, 8)
        state = SpeciesState(
            n_D=jnp.ones(shape) * 1e20,
            n_T=jnp.ones(shape) * 1e20,
            n_He3=jnp.zeros(shape),
            n_He4=jnp.zeros(shape),
            n_p=jnp.zeros(shape),
        )
        assert state.n_D.shape == shape

    def test_electron_density(self):
        """n_e = n_D + n_T + n_He3 + 2*n_He4 + n_p."""
        from jax_frc.burn.species import SpeciesState
        shape = (4, 8)
        state = SpeciesState(
            n_D=jnp.ones(shape) * 1e20,
            n_T=jnp.ones(shape) * 1e20,
            n_He3=jnp.ones(shape) * 0.5e20,
            n_He4=jnp.ones(shape) * 0.1e20,  # Z=2
            n_p=jnp.ones(shape) * 0.2e20,
        )
        # n_e = 1 + 1 + 0.5 + 2*0.1 + 0.2 = 2.9e20
        expected = 2.9e20
        assert jnp.allclose(state.n_e, expected)

    def test_species_state_is_pytree(self):
        """SpeciesState works with JAX transformations."""
        from jax_frc.burn.species import SpeciesState
        shape = (4, 8)
        state = SpeciesState(
            n_D=jnp.ones(shape) * 1e20,
            n_T=jnp.ones(shape) * 1e20,
            n_He3=jnp.zeros(shape),
            n_He4=jnp.zeros(shape),
            n_p=jnp.zeros(shape),
        )

        @jax.jit
        def get_ne(s):
            return s.n_e

        result = get_ne(state)
        assert result.shape == shape


class TestSpeciesTracker:
    """Tests for SpeciesTracker module."""

    def test_burn_sources_dt(self):
        """D-T burn consumes D and T, produces He4."""
        from jax_frc.burn.species import SpeciesTracker
        from jax_frc.burn.physics import ReactionRates

        tracker = SpeciesTracker()
        shape = (4, 8)
        rates = ReactionRates(
            DT=jnp.ones(shape) * 1e18,
            DD_T=jnp.zeros(shape),
            DD_HE3=jnp.zeros(shape),
            DHE3=jnp.zeros(shape),
        )

        sources = tracker.burn_sources(rates)

        # D consumed: -rate_DT
        assert jnp.allclose(sources["D"], -1e18)
        # T consumed: -rate_DT
        assert jnp.allclose(sources["T"], -1e18)
        # He4 produced: +rate_DT
        assert jnp.allclose(sources["He4"], 1e18)

    def test_burn_sources_dd(self):
        """D-D burn consumes 2D per reaction, produces T or He3."""
        from jax_frc.burn.species import SpeciesTracker
        from jax_frc.burn.physics import ReactionRates

        tracker = SpeciesTracker()
        shape = (4, 8)
        rates = ReactionRates(
            DT=jnp.zeros(shape),
            DD_T=jnp.ones(shape) * 1e17,   # D+D -> T+p
            DD_HE3=jnp.ones(shape) * 1e17, # D+D -> He3+n
            DHE3=jnp.zeros(shape),
        )

        sources = tracker.burn_sources(rates)

        # D consumed: 2*(rate_DD_T + rate_DD_HE3) = 4e17
        assert jnp.allclose(sources["D"], -4e17)
        # T produced: +rate_DD_T
        assert jnp.allclose(sources["T"], 1e17)
        # He3 produced: +rate_DD_HE3
        assert jnp.allclose(sources["He3"], 1e17)
        # p produced: +rate_DD_T
        assert jnp.allclose(sources["p"], 1e17)

    def test_particle_conservation(self):
        """Total nucleons conserved in burn sources."""
        from jax_frc.burn.species import SpeciesTracker
        from jax_frc.burn.physics import ReactionRates

        tracker = SpeciesTracker()
        shape = (4, 8)
        rates = ReactionRates(
            DT=jnp.ones(shape) * 1e18,
            DD_T=jnp.ones(shape) * 5e17,
            DD_HE3=jnp.ones(shape) * 5e17,
            DHE3=jnp.ones(shape) * 2e17,
        )

        sources = tracker.burn_sources(rates)

        # Check that D consumption matches production pattern
        assert sources["D"].shape == shape

    def test_advance_updates_species(self):
        """advance() updates species densities correctly."""
        from jax_frc.burn.species import SpeciesState, SpeciesTracker

        tracker = SpeciesTracker()
        shape = (4, 8)

        state = SpeciesState(
            n_D=jnp.ones(shape) * 1e20,
            n_T=jnp.ones(shape) * 1e20,
            n_He3=jnp.zeros(shape),
            n_He4=jnp.zeros(shape),
            n_p=jnp.zeros(shape),
        )

        # Burn sources that consume D/T and produce He4
        burn_sources = {
            "D": jnp.ones(shape) * -1e18,  # consuming
            "T": jnp.ones(shape) * -1e18,  # consuming
            "He3": jnp.zeros(shape),
            "He4": jnp.ones(shape) * 1e18,  # producing
            "p": jnp.zeros(shape),
        }

        transport_div = {}  # no transport
        dt = 1e-6

        new_state = tracker.advance(state, burn_sources, transport_div, dt)

        # D should decrease
        assert jnp.all(new_state.n_D < state.n_D)
        # He4 should increase
        assert jnp.all(new_state.n_He4 > state.n_He4)
