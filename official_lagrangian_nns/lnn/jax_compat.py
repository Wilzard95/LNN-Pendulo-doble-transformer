from __future__ import annotations

try:
    from jax.experimental import optimizers as optimizers  # type: ignore[attr-defined]
except ImportError:
    from jax.example_libraries import optimizers as optimizers  # type: ignore[no-redef]

try:
    from jax.experimental import stax as stax  # type: ignore[attr-defined]
except ImportError:
    from jax.example_libraries import stax as stax  # type: ignore[no-redef]

try:
    from jax.experimental.stax import Dense, Relu, Softplus, Tanh, elementwise, serial  # type: ignore[attr-defined]
except ImportError:
    from jax.example_libraries.stax import Dense, Relu, Softplus, Tanh, elementwise, serial  # type: ignore[no-redef]


def index_update(x, idx, y):
    try:
        from jax.ops import index_update as legacy_index_update  # type: ignore[attr-defined]

        return legacy_index_update(x, idx, y)
    except ImportError:
        return x.at[idx].set(y)
