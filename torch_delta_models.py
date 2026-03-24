from __future__ import annotations

import torch
from torch import nn


class StateTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = 4
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(1, self.d_model)
        self.token_embed = nn.Parameter(torch.zeros(self.num_tokens, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(ff_multiplier) * self.d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_tokens * self.d_model),
            nn.Linear(self.num_tokens * self.d_model, 4),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.input_proj:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != 4:
            raise ValueError(f"Expected input shape [batch, 4], got {tuple(x.shape)}")
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self.token_embed.unsqueeze(0)
        encoded = self.encoder(tokens)
        flat = encoded.reshape(encoded.shape[0], -1)
        return self.head(flat)


class TransformerLagrangian(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = 4
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ff_multiplier = int(ff_multiplier)
        self.dropout = float(dropout)

        self.input_proj = nn.Linear(1, self.d_model)
        self.token_embed = nn.Parameter(torch.zeros(self.num_tokens, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ff_multiplier * self.d_model,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_tokens * self.d_model),
            nn.Linear(self.num_tokens * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.input_proj:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != 4:
            raise ValueError(f"Expected input shape [batch, 4], got {tuple(x.shape)}")
        tokens = self.input_proj(x.unsqueeze(-1))
        tokens = tokens + self.token_embed.unsqueeze(0)
        encoded = self.encoder(tokens)
        flat = encoded.reshape(encoded.shape[0], -1)
        return self.head(flat)

    def get_config(self) -> dict[str, int | float | str]:
        return {
            "architecture": "transformer_lagrangian",
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_multiplier": self.ff_multiplier,
            "dropout": self.dropout,
            "num_tokens": self.num_tokens,
            "output_dim": 1,
        }


class StructuredTransformerLagrangian(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
        mass_eps: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.num_tokens = 2
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ff_multiplier = int(ff_multiplier)
        self.dropout = float(dropout)
        self.mass_eps = float(mass_eps)

        self.input_proj = nn.Linear(1, self.d_model)
        self.token_embed = nn.Parameter(torch.zeros(self.num_tokens, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.ff_multiplier * self.d_model,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        flat_dim = self.num_tokens * self.d_model
        self.mass_head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 3),
        )
        self.potential_head = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

        self.reset_parameters()

    @staticmethod
    def _softplus_inverse(y: float) -> float:
        y_tensor = torch.tensor(float(y))
        return float(torch.log(torch.expm1(y_tensor)).item())

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.input_proj:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Start close to a simple physical prior: near-identity mass matrix and zero potential.
        mass_last = self.mass_head[-1]
        potential_last = self.potential_head[-1]
        nn.init.zeros_(mass_last.weight)
        nn.init.zeros_(potential_last.weight)
        mass_bias = self._softplus_inverse(1.0)
        mass_last.bias.data = torch.tensor([mass_bias, 0.0, mass_bias], dtype=mass_last.bias.dtype)
        nn.init.zeros_(potential_last.bias)

    def _encode_q(self, q: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(q.unsqueeze(-1))
        tokens = tokens + self.token_embed.unsqueeze(0)
        encoded = self.encoder(tokens)
        return encoded.reshape(encoded.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != 4:
            raise ValueError(f"Expected input shape [batch, 4], got {tuple(x.shape)}")
        q = x[:, :2]
        qdot = x[:, 2:]

        flat = self._encode_q(q)
        mass_raw = self.mass_head(flat)
        potential = self.potential_head(flat).squeeze(-1)

        l11 = torch.nn.functional.softplus(mass_raw[:, 0]) + self.mass_eps
        l21 = mass_raw[:, 1]
        l22 = torch.nn.functional.softplus(mass_raw[:, 2]) + self.mass_eps

        zero = torch.zeros_like(l11)
        chol = torch.stack(
            [
                torch.stack([l11, zero], dim=-1),
                torch.stack([l21, l22], dim=-1),
            ],
            dim=-2,
        )
        mass = chol @ chol.transpose(-1, -2)
        mass = mass + self.mass_eps * torch.eye(2, dtype=x.dtype, device=x.device).unsqueeze(0)

        qdot_vec = qdot.unsqueeze(-1)
        kinetic = 0.5 * torch.matmul(qdot.unsqueeze(1), torch.matmul(mass, qdot_vec)).squeeze(-1).squeeze(-1)
        lagrangian = kinetic - potential
        return lagrangian.unsqueeze(-1)

    def get_config(self) -> dict[str, int | float | str]:
        return {
            "architecture": "structured_transformer_lagrangian_tv",
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_multiplier": self.ff_multiplier,
            "dropout": self.dropout,
            "num_tokens": self.num_tokens,
            "output_dim": 1,
            "mass_eps": self.mass_eps,
        }
