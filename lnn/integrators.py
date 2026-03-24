from __future__ import annotations

from typing import Callable

import torch


AccelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def euler_step(q: torch.Tensor, qdot: torch.Tensor, dt: float, accel_fn: AccelFn) -> tuple[torch.Tensor, torch.Tensor]:
    qddot = accel_fn(q, qdot)
    q_next = q + dt * qdot
    qdot_next = qdot + dt * qddot
    return q_next, qdot_next


def rk4_step(q: torch.Tensor, qdot: torch.Tensor, dt: float, accel_fn: AccelFn) -> tuple[torch.Tensor, torch.Tensor]:
    dt = float(dt)

    k1_q = qdot
    k1_v = accel_fn(q, qdot)

    k2_q = qdot + 0.5 * dt * k1_v
    k2_v = accel_fn(q + 0.5 * dt * k1_q, qdot + 0.5 * dt * k1_v)

    k3_q = qdot + 0.5 * dt * k2_v
    k3_v = accel_fn(q + 0.5 * dt * k2_q, qdot + 0.5 * dt * k2_v)

    k4_q = qdot + dt * k3_v
    k4_v = accel_fn(q + dt * k3_q, qdot + dt * k3_v)

    q_next = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
    qdot_next = qdot + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return q_next, qdot_next


def step_dynamics(
    q: torch.Tensor,
    qdot: torch.Tensor,
    dt: float,
    accel_fn: AccelFn,
    integrator: str = "rk4",
) -> tuple[torch.Tensor, torch.Tensor]:
    integrator = integrator.lower()
    if integrator == "rk4":
        return rk4_step(q, qdot, dt, accel_fn)
    if integrator == "euler":
        return euler_step(q, qdot, dt, accel_fn)
    raise ValueError(f"Unsupported integrator: {integrator}")
