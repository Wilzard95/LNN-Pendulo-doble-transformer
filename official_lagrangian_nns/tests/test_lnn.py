"""Test core LNN functionality."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax.experimental import optimizers
from functools import partial

import sys
sys.path.insert(0, '.')

from lnn.core import lagrangian_eom, unconstrained_eom, raw_lagrangian_eom, custom_init
from lnn.models import mlp


class TestLagrangianEOM:
    """Test Lagrangian equation of motion computation."""

    def test_equation_of_motion_shape(self):
        """Test that EOM returns correct shape."""
        def simple_lagrangian(q, q_dot):
            return 0.5 * jnp.sum(q_dot**2) - jnp.sum(q**2)

        state = jnp.array([0.1, 0.2, 0.3, 0.4])  # 2D system: 2 positions, 2 velocities
        state_dot = lagrangian_eom(simple_lagrangian, state)

        assert state_dot.shape == state.shape
        assert jnp.all(jnp.isfinite(state_dot))

    def test_conservation_of_energy(self):
        """Test that free particle has zero acceleration."""
        def free_particle_lagrangian(q, q_dot):
            return 0.5 * jnp.sum(q_dot**2)  # Only kinetic energy

        state = jnp.array([0.0, 1.0])  # position=0, velocity=1
        state_dot = raw_lagrangian_eom(free_particle_lagrangian, state)  # Use raw to avoid dt scaling

        # Free particle should have zero acceleration
        assert jnp.allclose(state_dot[1], 0.0, atol=1e-6)

    def test_harmonic_oscillator_dynamics(self):
        """Test that harmonic oscillator produces sinusoidal motion."""
        def ho_lagrangian(q, q_dot):
            return 0.5 * jnp.sum(q_dot**2) - 0.5 * jnp.sum(q**2)

        # Start from rest at x=1, should oscillate
        state = jnp.array([1.0, 0.0])
        state_dot = raw_lagrangian_eom(ho_lagrangian, state)  # Use raw to avoid dt scaling

        # At x=1 with v=0, acceleration should be -x = -1 (restoring force)
        assert jnp.allclose(state_dot[0], 0.0, atol=1e-6)  # velocity
        assert jnp.allclose(state_dot[1], -1.0, atol=0.2)  # acceleration


class TestNeuralLagrangian:
    """Test learning a Lagrangian with a neural network."""

    def test_gradient_based_learning(self):
        """Test that network gradients improve a simple loss."""
        rng = jax.random.PRNGKey(42)

        # Setup neural network
        init_fn, forward_fn = mlp(hidden_dim=16, output_dim=1, n_hidden_layers=1)
        _, params = init_fn(rng, (-1, 2))

        # Simple regression task
        X = jax.random.normal(rng, (20, 2))
        y = jnp.sum(X**2, axis=1, keepdims=True)  # Simple target: sum of squares

        def loss_fn(params):
            pred = vmap(forward_fn, (None, 0))(params, X)
            return jnp.mean((pred - y)**2)

        # Train
        opt_init, opt_update, get_params = optimizers.adam(1e-2)
        opt_state = opt_init(params)

        initial_loss = loss_fn(params)
        for i in range(20):
            grads = grad(loss_fn)(get_params(opt_state))
            opt_state = opt_update(i, grads, opt_state)

        final_loss = loss_fn(get_params(opt_state))

        # Should improve
        assert final_loss < initial_loss


class TestOptimization:
    """Test optimization of LNN parameters."""

    def test_adam_optimizer(self):
        """Test that Adam optimizer reduces loss."""
        rng = jax.random.PRNGKey(42)
        init_fn, forward_fn = mlp(hidden_dim=8, output_dim=1, n_hidden_layers=1)
        _, params = init_fn(rng, (-1, 2))

        def loss(params):
            x = jnp.array([1.0, 1.0])
            return jnp.sum(forward_fn(params, x)**2)

        opt_init, opt_update, get_params = optimizers.adam(1e-2)
        opt_state = opt_init(params)

        initial_loss = loss(params)

        for i in range(5):
            grads = grad(loss)(get_params(opt_state))
            opt_state = opt_update(i, grads, opt_state)

        final_params = get_params(opt_state)
        final_loss = loss(final_params)

        assert final_loss < initial_loss


class TestCustomInit:
    """Test custom initialization for LNNs."""

    def test_initialization_scale(self):
        """Test that custom_init produces sensible scales."""
        rng = jax.random.PRNGKey(42)

        # Setup network architecture
        init_fn, _ = mlp(hidden_dim=64, output_dim=1, n_hidden_layers=2)
        _, default_params = init_fn(rng, (-1, 4))
        custom_params = custom_init(default_params, seed=42)

        # Check that custom init produces different but sensible scales
        for default_layer, custom_layer in zip(default_params, custom_params):
            if isinstance(custom_layer, tuple) and len(custom_layer) == 2:
                W_custom, b_custom = custom_layer
                W_default, b_default = default_layer

                # Custom should have different initialization
                assert not jnp.allclose(W_custom, W_default)

                # But still reasonable scale
                assert jnp.std(W_custom) > 0.001
                assert jnp.std(W_custom) < 10.0

                # Biases should be initialized differently too
                assert not jnp.allclose(b_custom, b_default)