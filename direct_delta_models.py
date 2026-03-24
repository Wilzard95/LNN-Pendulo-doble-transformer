from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


Array = jnp.ndarray


def _split_rng(rng: Array, n: int) -> list[Array]:
    return list(jax.random.split(rng, n))


def _xavier_normal(rng: Array, shape: tuple[int, ...], fan_in: int, fan_out: int) -> Array:
    scale = jnp.sqrt(2.0 / float(fan_in + fan_out))
    return jax.random.normal(rng, shape) * scale


def _dense_init(rng: Array, fan_in: int, fan_out: int) -> dict[str, Array]:
    return {
        "w": _xavier_normal(rng, (fan_in, fan_out), fan_in=fan_in, fan_out=fan_out),
        "b": jnp.zeros((fan_out,)),
    }


def _dense_apply(params: dict[str, Array], x: Array) -> Array:
    return x @ params["w"] + params["b"]


def _activation_fn(name: str) -> Callable[[Array], Array]:
    if name == "relu":
        return jax.nn.relu
    if name == "softplus":
        return jax.nn.softplus
    if name == "soft_relu":
        return lambda x: jax.nn.relu(jax.nn.softplus(x))
    if name == "tanh":
        return jnp.tanh
    raise ValueError(f"Unsupported activation: {name}")


def init_baseline_mlp(
    rng: Array,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
) -> dict[str, object]:
    keys = _split_rng(rng, int(layers) + 1)
    widths = [int(input_dim)] + [int(hidden_dim)] * int(layers) + [int(output_dim)]
    dense_layers = []
    for idx in range(len(widths) - 1):
        dense_layers.append(_dense_init(keys[idx], widths[idx], widths[idx + 1]))
    return {"layers": dense_layers}


def apply_baseline_mlp(params: dict[str, object], x: Array, activation: str = "soft_relu") -> Array:
    act = _activation_fn(activation)
    y = x
    layers = params["layers"]
    for layer in layers[:-1]:
        y = act(_dense_apply(layer, y))
    return _dense_apply(layers[-1], y)


def _layer_norm(x: Array, scale: Array, bias: Array, eps: float = 1e-5) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * scale + bias


def init_baseline_attention(
    rng: Array,
    input_dim: int,
    d_model: int,
    output_dim: int,
    num_heads: int = 4,
    ff_multiplier: int = 2,
    num_blocks: int = 1,
) -> dict[str, object]:
    if int(input_dim) != 4:
        raise ValueError("The attention baseline expects 4 input tokens: theta1, theta2, omega1, omega2.")
    if int(d_model) % int(num_heads) != 0:
        raise ValueError("d_model must be divisible by num_heads.")

    keys = _split_rng(rng, 3 + int(num_blocks) * 6)
    key_iter = iter(keys)

    params: dict[str, object] = {
        "token_w": jax.random.normal(next(key_iter), (int(input_dim), int(d_model))) * 0.02,
        "token_b": jnp.zeros((int(input_dim), int(d_model))),
        "pos_embed": jax.random.normal(next(key_iter), (int(input_dim), int(d_model))) * 0.02,
        "blocks": [],
    }

    ff_dim = int(d_model) * int(ff_multiplier)
    for _ in range(int(num_blocks)):
        block = {
            "ln1_scale": jnp.ones((int(d_model),)),
            "ln1_bias": jnp.zeros((int(d_model),)),
            "ln2_scale": jnp.ones((int(d_model),)),
            "ln2_bias": jnp.zeros((int(d_model),)),
            "wq": _dense_init(next(key_iter), int(d_model), int(d_model)),
            "wk": _dense_init(next(key_iter), int(d_model), int(d_model)),
            "wv": _dense_init(next(key_iter), int(d_model), int(d_model)),
            "wo": _dense_init(next(key_iter), int(d_model), int(d_model)),
            "ff1": _dense_init(next(key_iter), int(d_model), ff_dim),
            "ff2": _dense_init(next(key_iter), ff_dim, int(d_model)),
        }
        params["blocks"].append(block)

    params["head"] = _dense_init(next(key_iter), int(input_dim) * int(d_model), int(output_dim))
    return params


def apply_baseline_attention(
    params: dict[str, object],
    x: Array,
    activation: str = "relu",
    num_heads: int = 4,
) -> Array:
    act = _activation_fn(activation)
    batch_size, num_tokens = x.shape
    d_model = int(params["token_w"].shape[-1])
    head_dim = d_model // int(num_heads)

    tokens = x[..., None] * params["token_w"][None, :, :] + params["token_b"][None, :, :]
    tokens = tokens + params["pos_embed"][None, :, :]

    for block in params["blocks"]:
        y = _layer_norm(tokens, block["ln1_scale"], block["ln1_bias"])
        q = _dense_apply(block["wq"], y)
        k = _dense_apply(block["wk"], y)
        v = _dense_apply(block["wv"], y)

        q = q.reshape(batch_size, num_tokens, int(num_heads), head_dim)
        k = k.reshape(batch_size, num_tokens, int(num_heads), head_dim)
        v = v.reshape(batch_size, num_tokens, int(num_heads), head_dim)

        attn_logits = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(float(head_dim))
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.einsum("bhts,bshd->bthd", attn_weights, v).reshape(batch_size, num_tokens, d_model)
        tokens = tokens + _dense_apply(block["wo"], attn_out)

        z = _layer_norm(tokens, block["ln2_scale"], block["ln2_bias"])
        ff = _dense_apply(block["ff2"], act(_dense_apply(block["ff1"], z)))
        tokens = tokens + ff

    flat = tokens.reshape(batch_size, num_tokens * d_model)
    return _dense_apply(params["head"], flat)


def build_model(model_kind: str, config: dict[str, object]) -> tuple[Callable[[Array], dict[str, object]], Callable[..., Array]]:
    kind = str(model_kind)
    if kind == "baseline_mlp":
        def init_fn(rng: Array) -> dict[str, object]:
            return init_baseline_mlp(
                rng=rng,
                input_dim=int(config["input_dim"]),
                hidden_dim=int(config["hidden_dim"]),
                output_dim=int(config["output_dim"]),
                layers=int(config["layers"]),
            )

        def apply_fn(params: dict[str, object], x: Array) -> Array:
            return apply_baseline_mlp(params, x, activation=str(config.get("activation", "soft_relu")))

        return init_fn, apply_fn

    if kind == "baseline_attn":
        def init_fn(rng: Array) -> dict[str, object]:
            return init_baseline_attention(
                rng=rng,
                input_dim=int(config["input_dim"]),
                d_model=int(config["d_model"]),
                output_dim=int(config["output_dim"]),
                num_heads=int(config.get("num_heads", 4)),
                ff_multiplier=int(config.get("ff_multiplier", 2)),
                num_blocks=int(config.get("num_blocks", 1)),
            )

        def apply_fn(params: dict[str, object], x: Array) -> Array:
            return apply_baseline_attention(
                params,
                x,
                activation=str(config.get("activation", "relu")),
                num_heads=int(config.get("num_heads", 4)),
            )

        return init_fn, apply_fn

    raise ValueError(f"Unsupported model_kind: {model_kind}")
