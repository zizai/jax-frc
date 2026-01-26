"""Tests for electromagnetic coil field models."""

import pytest
import jax.numpy as jnp
from jax_frc.fields.coils import Solenoid, MirrorCoil, ThetaPinchArray
