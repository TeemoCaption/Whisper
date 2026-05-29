from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LanguageClassifierSpec:
    labels: tuple[str, ...]
    hidden_size: int
    pooling: str = "attention"
    attention_hidden_size: int = 256
    hidden_ratio: float = 0.5
    num_hidden_layers: int = 2
    dropout: float = 0.1


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, attention_hidden_size: int = 256) -> None:
        super().__init__()
        attn_size = max(1, int(attention_hidden_size))
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, attn_size),
            nn.Tanh(),
            nn.Linear(attn_size, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("encoder hidden states must have shape [batch, time, hidden].")
        scores = self.scorer(hidden_states).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


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
        pooling = str(spec.pooling or "attention").lower()
        if pooling != "attention":
            raise ValueError("語言分類頭目前只支援 attention 池化。")
        self.attention_pooling = AttentionPooling(
            spec.hidden_size,
            spec.attention_hidden_size,
        )
        input_size = spec.hidden_size
        self.head = LanguageClassifierHead(
            input_size=input_size,
            num_labels=len(spec.labels),
            hidden_ratio=spec.hidden_ratio,
            num_hidden_layers=spec.num_hidden_layers,
            dropout=spec.dropout,
        )

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = self.attention_pooling(encoder_hidden_states)
        return self.head(pooled)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "labels": list(self.spec.labels),
            "hidden_size": int(self.spec.hidden_size),
            "pooling": self.spec.pooling,
            "attention_hidden_size": int(self.spec.attention_hidden_size),
            "hidden_ratio": float(self.spec.hidden_ratio),
            "num_hidden_layers": int(self.spec.num_hidden_layers),
            "dropout": float(self.spec.dropout),
            "state_dict": self.state_dict(),
        }
