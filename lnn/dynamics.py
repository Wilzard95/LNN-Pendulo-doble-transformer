from __future__ import annotations

import torch
from torch.func import functional_call, grad, jacrev, vmap

from .integrators import rk4_step
from .utils import wrap_coords_torch


def _validate_inputs(q: torch.Tensor, qdot: torch.Tensor) -> None:
    if q.ndim != 2 or qdot.ndim != 2:
        raise ValueError(f"Expected q and qdot as [B, D], got {q.shape} and {qdot.shape}")
    if q.shape != qdot.shape:
        raise ValueError(f"q and qdot must have same shape, got {q.shape} and {qdot.shape}")
    if q.shape[1] != 2:
        raise ValueError(f"This implementation expects 2 DoF, got {q.shape[1]}")


def _named_state(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    return params, buffers


def _lagrangian_single(
    model: torch.nn.Module,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    q_single: torch.Tensor,
    qdot_single: torch.Tensor,
) -> torch.Tensor:
    state_single = torch.cat([q_single, qdot_single], dim=0).unsqueeze(0)
    state_single = wrap_coords_torch(state_single)
    return functional_call(model, (params, buffers), (state_single,)).squeeze()


def _single_eom(
    model: torch.nn.Module,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    q_single: torch.Tensor,
    qdot_single: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lagrangian = lambda q_val, qdot_val: _lagrangian_single(model, params, buffers, q_val, qdot_val)
    dldq_fn = grad(lagrangian, argnums=0)
    dldqdot_fn = grad(lagrangian, argnums=1)

    dldq = dldq_fn(q_single, qdot_single)
    dldqdot = dldqdot_fn(q_single, qdot_single)
    hess_qdot = jacrev(dldqdot_fn, argnums=1)(q_single, qdot_single)
    mixed = jacrev(dldqdot_fn, argnums=0)(q_single, qdot_single)
    rhs = dldq - mixed @ qdot_single
    qddot = torch.linalg.pinv(hess_qdot) @ rhs
    return qddot, dldq, dldqdot, hess_qdot, mixed, rhs


def qddot_from_lagrangian(
    model: torch.nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    for_training: bool = True,
    return_details: bool = False,
):
    _validate_inputs(q, qdot)
    params, buffers = _named_state(model)

    batched = vmap(lambda q_i, qdot_i: _single_eom(model, params, buffers, q_i, qdot_i))(q, qdot)
    qddot, dldq, dldqdot, hess_qdot, mixed, rhs = batched
    if not return_details:
        return qddot

    details = {
        "dldq": dldq,
        "dldqdot": dldqdot,
        "H": hess_qdot,
        "C": mixed,
        "rhs": rhs,
    }
    return qddot, details


def xdot_from_lagrangian(
    model: torch.nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    for_training: bool = True,
) -> torch.Tensor:
    qddot = qddot_from_lagrangian(model=model, q=q, qdot=qdot, for_training=for_training)
    return torch.cat([qdot, qddot], dim=1)


def state_delta_from_lagrangian(
    model: torch.nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    dt: float,
    n_updates: int = 1,
    for_training: bool = True,
) -> torch.Tensor:
    _validate_inputs(q, qdot)
    q_cur = q
    qdot_cur = qdot
    sub_dt = float(dt) / max(1, int(n_updates))

    accel_fn = lambda q_val, qdot_val: qddot_from_lagrangian(
        model=model,
        q=q_val,
        qdot=qdot_val,
        for_training=for_training,
    )

    for _ in range(max(1, int(n_updates))):
        q_cur, qdot_cur = rk4_step(q_cur, qdot_cur, sub_dt, accel_fn)

    return torch.cat([q_cur - q, qdot_cur - qdot], dim=1)


def lagrangian_energy(
    model: torch.nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
    for_training: bool = False,
) -> torch.Tensor:
    _validate_inputs(q, qdot)
    params, buffers = _named_state(model)

    lagrangian = lambda q_val, qdot_val: _lagrangian_single(model, params, buffers, q_val, qdot_val)
    dldqdot_fn = grad(lagrangian, argnums=1)
    dldqdot = vmap(dldqdot_fn)(q, qdot)
    lag_values = vmap(lagrangian)(q, qdot)
    return torch.sum(qdot * dldqdot, dim=1) - lag_values
