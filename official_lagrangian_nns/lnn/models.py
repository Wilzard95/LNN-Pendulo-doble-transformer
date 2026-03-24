# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

from .jax_compat import stax

def mlp(args=None, input_dim=None, hidden_dim=None, output_dim=None, n_hidden_layers=None):
    """Create a multi-layer perceptron.

    Can be called either with an args object or with keyword arguments.
    """
    if args is not None:
        # Legacy mode: use args object
        return stax.serial(
            stax.Dense(args.hidden_dim),
            stax.Softplus,
            stax.Dense(args.hidden_dim),
            stax.Softplus,
            stax.Dense(args.output_dim),
        )
    else:
        # Keyword argument mode
        if hidden_dim is None or output_dim is None:
            raise ValueError("Must provide hidden_dim and output_dim")

        layers = []
        n_layers = n_hidden_layers if n_hidden_layers is not None else 2

        for i in range(n_layers):
            layers.append(stax.Dense(hidden_dim))
            layers.append(stax.Softplus)

        layers.append(stax.Dense(output_dim))
        return stax.serial(*layers)

def pixel_encoder(args):
    return stax.serial(
        stax.Dense(args.ae_hidden_dim),
        stax.Softplus,
        stax.Dense(args.ae_latent_dim),
    )

def pixel_decoder(args):
    return stax.serial(
        stax.Dense(args.ae_hidden_dim),
        stax.Softplus,
        stax.Dense(args.ae_input_dim),
    )
