from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ContrastiveRouterSpec:
    labels: tuple[str, ...]
    hidden_size: int
    pooling: str = "attention"
    attention_hidden_size: int = 256
    embedding_size: int = 256
    hidden_ratio: float = 0.5
    dropout: float = 0.1
    temperature: float = 0.07
    label_smoothing: float = 0.0
    margin: float = 0.0
    margin_loss_weight: float = 0.0


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


class QueryProjection(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        hidden_ratio: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_size = max(embedding_size, int(input_size * float(hidden_ratio)))
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, pooled_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(pooled_states), dim=-1)


class ContrastiveAdapterRouter(nn.Module):
    def __init__(self, spec: ContrastiveRouterSpec) -> None:
        super().__init__()
        if len(spec.labels) < 2:
            raise ValueError("對比式路由至少需要兩個語言標籤。")
        pooling = str(spec.pooling or "attention").lower()
        if pooling != "attention":
            raise ValueError("對比式路由目前只支援 attention 池化。")

        self.spec = spec
        self.attention_pooling = AttentionPooling(
            spec.hidden_size,
            spec.attention_hidden_size,
        )
        self.query_projection = QueryProjection(
            input_size=spec.hidden_size,
            embedding_size=spec.embedding_size,
            hidden_ratio=spec.hidden_ratio,
            dropout=spec.dropout,
        )
        self.adapter_keys = nn.Parameter(
            torch.empty(len(spec.labels), spec.embedding_size)
        )
        nn.init.xavier_uniform_(self.adapter_keys)

    def forward(self, encoder_hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.attention_pooling(encoder_hidden_states)
        queries = self.query_projection(pooled.float())
        keys = F.normalize(self.adapter_keys, dim=-1)
        temperature = max(float(self.spec.temperature), 1e-6)
        logits = queries @ keys.transpose(0, 1) / temperature
        return {
            "logits": logits,
            "queries": queries,
            "keys": keys,
        }

    def compute_loss(
        self,
        encoder_hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = self(encoder_hidden_states)
        ce_loss = F.cross_entropy(
            outputs["logits"],
            labels,
            label_smoothing=max(0.0, float(self.spec.label_smoothing)),
        )
        similarities = outputs["queries"] @ outputs["keys"].transpose(0, 1)
        row_ids = torch.arange(labels.size(0), device=labels.device)
        positive = similarities[row_ids, labels]
        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask[row_ids, labels] = False
        negative = similarities.masked_fill(~mask, float("-inf")).max(dim=-1).values

        margin = max(0.0, float(self.spec.margin))
        margin_loss = F.relu(margin - (positive - negative)).mean()
        loss = ce_loss + max(0.0, float(self.spec.margin_loss_weight)) * margin_loss
        outputs["ce_loss"] = ce_loss
        outputs["margin_loss"] = margin_loss
        outputs["similarity_gap"] = positive - negative
        return loss, outputs

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "labels": list(self.spec.labels),
            "hidden_size": int(self.spec.hidden_size),
            "pooling": self.spec.pooling,
            "attention_hidden_size": int(self.spec.attention_hidden_size),
            "embedding_size": int(self.spec.embedding_size),
            "hidden_ratio": float(self.spec.hidden_ratio),
            "dropout": float(self.spec.dropout),
            "temperature": float(self.spec.temperature),
            "label_smoothing": float(self.spec.label_smoothing),
            "margin": float(self.spec.margin),
            "margin_loss_weight": float(self.spec.margin_loss_weight),
            "state_dict": self.state_dict(),
        }
