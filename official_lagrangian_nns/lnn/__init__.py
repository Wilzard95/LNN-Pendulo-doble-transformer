"""Lagrangian Neural Networks package."""

# Core functionality
from .core import (
    lagrangian_eom,
    unconstrained_eom,
    raw_lagrangian_eom,
    lagrangian_eom_rk4,
    solve_dynamics,
    custom_init,
)
from .models import mlp, pixel_encoder, pixel_decoder
from .utils import wrap_coords, rk4_step
try:
    from .plotting import plot_dblpend, fig2image, get_dblpend_images
except ModuleNotFoundError:
    plot_dblpend = None
    fig2image = None
    get_dblpend_images = None

__version__ = "0.1.0"

__all__ = [
    "lagrangian_eom",
    "unconstrained_eom",
    "raw_lagrangian_eom",
    "lagrangian_eom_rk4",
    "solve_dynamics",
    "custom_init",
    "mlp",
    "pixel_encoder",
    "pixel_decoder",
    "wrap_coords",
    "rk4_step",
    "plot_dblpend",
    "fig2image",
    "get_dblpend_images",
]
