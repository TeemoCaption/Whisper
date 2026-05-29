from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LanguageClassifierSpec:
    labels: tuple[str, ...]
    hidden_size: int
    pooling: str = "mean_max"
    hidden_ratio: float = 0.5
    num_hidden_layers: int = 2
    dropout: float = 0.1


def pool_encoder_states(
    hidden_states: torch.Tensor,
    *,
    pooling: str = "mean_max",
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError("encoder hidden states must have shape [batch, time, hidden].")

    mode = str(pooling or "mean_max").lower()
    if mode == "mean":
        return hidden_states.mean(dim=1)
    if mode == "max":
        return hidden_states.max(dim=1).values
    if mode == "mean_max":
        return torch.cat(
            [hidden_states.mean(dim=1), hidden_states.max(dim=1).values],
            dim=-1,
        )
    raise ValueError("pooling must be one of: mean, max, mean_max.")


class LanguageClassifierHead(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        num_labels: int,
        hidden_ratio: float = 0.5,
        num_hidden_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_size = max(num_labels * 4, int(input_size * float(hidden_ratio)))
        depth = max(1, int(num_hidden_layers))

        layers: list[nn.Module] = [nn.LayerNorm(input_size)]
        current_size = input_size
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current_size = hidden_size
        layers.append(nn.Linear(current_size, num_labels))
        self.net = nn.Sequential(*layers)

    def forward(self, pooled_states: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_states)


class WhisperLanguageClassifier(nn.Module):
    def __init__(self, spec: LanguageClassifierSpec) -> None:
        super().__init__()
        self.spec = spec
        input_size = spec.hidden_size * 2 if spec.pooling == "mean_max" else spec.hidden_size
        self.head = LanguageClassifierHead(
            input_size=input_size,
            num_labels=len(spec.labels),
            hidden_ratio=spec.hidden_ratio,
            num_hidden_layers=spec.num_hidden_layers,
            dropout=spec.dropout,
        )

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = pool_encoder_states(
            encoder_hidden_states,
            pooling=self.spec.pooling,
        )
        return self.head(pooled)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "labels": list(self.spec.labels),
            "hidden_size": int(self.spec.hidden_size),
            "pooling": self.spec.pooling,
            "hidden_ratio": float(self.spec.hidden_ratio),
            "num_hidden_layers": int(self.spec.num_hidden_layers),
            "dropout": float(self.spec.dropout),
            "state_dict": self.state_dict(),
        }
