from __future__ import annotations

import math

import torch
import torch.nn as nn


def custom_init_lnn(module: nn.Module, seed: int | None = None) -> None:
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return

    devices = []
    for p in module.parameters():
        if p.is_cuda:
            idx = p.device.index
            if idx is not None and idx not in devices:
                devices.append(idx)

    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(int(seed))

        last_idx = len(linear_layers) - 1
        for layer_idx, layer in enumerate(linear_layers):
            n = float(max(layer.weight.shape))
            if layer_idx == 0:
                mult = 2.2
            elif layer_idx == last_idx:
                mult = n
            else:
                mult = 0.58 * float(layer_idx)
            std = (1.0 / math.sqrt(n)) * mult
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)


def stax_like_init(module: nn.Module, seed: int | None = None) -> None:
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return

    devices = []
    for p in module.parameters():
        if p.is_cuda:
            idx = p.device.index
            if idx is not None and idx not in devices:
                devices.append(idx)

    with torch.random.fork_rng(devices=devices, enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(int(seed))

        for layer in linear_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias, mean=0.0, std=1e-2)


class LagrangianMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 500,
        num_hidden_layers: int = 4,
        activation: str = "softplus",
        output_dim: int = 1,
        init_seed: int | None = None,
        init_mode: str = "custom",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.activation_name = str(activation)
        self.output_dim = int(output_dim)
        self.init_seed = init_seed
        self.init_mode = str(init_mode)

        act_cls = self._get_activation(self.activation_name)
        modules: list[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(self.num_hidden_layers):
            modules.append(nn.Linear(in_dim, self.hidden_dim))
            modules.append(act_cls())
            in_dim = self.hidden_dim
        modules.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*modules)

        self._initialize()

    @staticmethod
    def _get_activation(name: str) -> type[nn.Module]:
        key = name.lower()
        if key == "softplus":
            return nn.Softplus
        raise ValueError(f"Unsupported activation: {name}")

    def _initialize(self) -> None:
        key = self.init_mode.lower()
        if key == "custom":
            custom_init_lnn(self.net, seed=self.init_seed)
            return
        if key == "stax":
            stax_like_init(self.net, seed=self.init_seed)
            return
        if key == "torch_default":
            return
        raise ValueError(f"Unsupported init_mode: {self.init_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_config(self) -> dict[str, int | str]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "activation": self.activation_name,
            "output_dim": self.output_dim,
            "init_mode": self.init_mode,
        }
