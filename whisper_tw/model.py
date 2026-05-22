from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import WhisperModel


@dataclass
class WhisperTWOutput:
    loss: torch.Tensor | None
    text_ctc_loss: torch.Tensor | None
    correction_loss: torch.Tensor | None
    bopomofo_ctc_loss: torch.Tensor | None
    text_ctc_logits: torch.Tensor
    correction_logits: torch.Tensor | None
    bopomofo_logits: torch.Tensor


class TemporalCompressor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        stride: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stride = max(int(stride), 1)
        self.downsample = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.stride,
            stride=self.stride,
            groups=1,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _build_transformer_encoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        compressed = self.downsample(states.transpose(1, 2)).transpose(1, 2)
        compressed = self.dropout(self.norm(compressed))
        return self.encoder(compressed)


class CtcDraftCorrector(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        input_dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size + vocab_size, hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.encoder = _build_transformer_encoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        compressed_states: torch.Tensor,
        compressed_ctc_probs: torch.Tensor,
    ) -> torch.Tensor:
        states = torch.cat([compressed_states, compressed_ctc_probs], dim=-1)
        states = self.input_dropout(self.input_projection(states))
        states = self.encoder(states)
        return self.output_head(states)


def _build_transformer_encoder(
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    if int(num_layers) <= 0:
        return nn.Identity()
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_size,
        nhead=int(num_heads),
        dim_feedforward=hidden_size * 4,
        dropout=float(dropout),
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=int(num_layers))


def _ctc_target_lengths(labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (labels != pad_id).sum(dim=1).to(dtype=torch.long)


def _ctc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    pad_id: int,
    blank_id: int,
    target_lengths: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if labels is None:
        return None
    if target_lengths is None:
        target_lengths = _ctc_target_lengths(labels, pad_id)
    input_lengths = torch.full(
        size=(logits.size(0),),
        fill_value=logits.size(1),
        dtype=torch.long,
        device=logits.device,
    )
    return nn.functional.ctc_loss(
        log_probs=logits.log_softmax(dim=-1).transpose(0, 1),
        targets=labels.to(logits.device),
        input_lengths=input_lengths,
        target_lengths=target_lengths.to(logits.device),
        blank=blank_id,
        zero_infinity=True,
    )


def _downsample_probabilities(
    probabilities: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    if probabilities.size(1) == target_length:
        return probabilities
    probs = probabilities.transpose(1, 2)
    probs = nn.functional.interpolate(
        probs,
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return probs.transpose(1, 2)


class WhisperTWModel(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        text_ctc_vocab_size: int,
        text_ctc_pad_id: int,
        bopomofo_vocab_size: int,
    ) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.text_ctc_pad_id = text_ctc_pad_id
        self.text_ctc_blank_id = 0
        self.bopomofo_blank_id = 0
        self.bopomofo_pad_id = 1
        self.text_ctc_weight = float(
            model_cfg.get("text_ctc", {}).get("loss_weight", 1.0)
        )
        self.correction_weight = float(
            model_cfg.get("context_corrector", {}).get("loss_weight", 0.0)
        )
        self.bopomofo_ctc_weight = float(
            model_cfg.get("bopomofo_ctc", {}).get("loss_weight", 0.0)
        )

        self.whisper = WhisperModel.from_pretrained(model_cfg["whisper_name"])
        self._configure_whisper_encoder(
            freeze=bool(model_cfg.get("freeze_whisper_encoder", True)),
            unfreeze_last_n_layers=int(
                model_cfg.get("unfreeze_encoder_last_n_layers", 0)
            ),
        )
        for param in self.whisper.decoder.parameters():
            param.requires_grad = False

        acoustic_cfg = model_cfg["acoustic_encoder"]
        whisper_hidden = self.whisper.config.d_model
        acoustic_hidden = int(acoustic_cfg["hidden_size"])
        self.encoder_projection = nn.Linear(whisper_hidden, acoustic_hidden)
        self.acoustic_encoder = _build_transformer_encoder(
            hidden_size=acoustic_hidden,
            num_layers=int(acoustic_cfg["num_layers"]),
            num_heads=int(acoustic_cfg["num_heads"]),
            dropout=float(acoustic_cfg.get("dropout", 0.1)),
        )
        text_ctc_dropout = float(model_cfg.get("text_ctc", {}).get("dropout", 0.0))
        self.text_ctc_head = nn.Sequential(
            nn.Dropout(text_ctc_dropout),
            nn.Linear(acoustic_hidden, text_ctc_vocab_size),
        )
        self.bopomofo_head = nn.Linear(acoustic_hidden, bopomofo_vocab_size)

        compressor_cfg = model_cfg["acoustic_compressor"]
        self.compressor = TemporalCompressor(
            hidden_size=acoustic_hidden,
            stride=int(compressor_cfg.get("stride", 4)),
            num_layers=int(compressor_cfg.get("num_layers", 1)),
            num_heads=int(compressor_cfg.get("num_heads", acoustic_cfg["num_heads"])),
            dropout=float(compressor_cfg.get("dropout", 0.1)),
        )

        corrector_cfg = model_cfg.get("context_corrector", {})
        self.corrector_enabled = self.correction_weight > 0.0 and int(
            corrector_cfg.get("num_layers", 0)
        ) > 0
        if self.corrector_enabled:
            self.context_corrector = CtcDraftCorrector(
                input_size=acoustic_hidden,
                hidden_size=int(corrector_cfg.get("hidden_size", acoustic_hidden)),
                vocab_size=text_ctc_vocab_size,
                num_layers=int(corrector_cfg.get("num_layers", 1)),
                num_heads=int(
                    corrector_cfg.get("num_heads", acoustic_cfg["num_heads"])
                ),
                dropout=float(corrector_cfg.get("dropout", 0.1)),
                input_dropout=float(corrector_cfg.get("input_dropout", 0.0)),
            )
        else:
            self.context_corrector = None

    def _configure_whisper_encoder(
        self,
        freeze: bool,
        unfreeze_last_n_layers: int,
    ) -> None:
        if freeze:
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
        layers = getattr(self.whisper.encoder, "layers", None)
        if not freeze or layers is None or unfreeze_last_n_layers <= 0:
            return
        for layer in layers[-unfreeze_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_features: torch.Tensor,
        text_ctc_labels: torch.Tensor | None = None,
        bopomofo_labels: torch.Tensor | None = None,
        bopomofo_label_lengths: torch.Tensor | None = None,
    ) -> WhisperTWOutput:
        encoder_hidden = self.whisper.encoder(input_features).last_hidden_state
        acoustic_states = self.encoder_projection(encoder_hidden)
        acoustic_states = self.acoustic_encoder(acoustic_states)

        text_ctc_logits = self.text_ctc_head(acoustic_states)
        bopomofo_logits = self.bopomofo_head(acoustic_states)

        text_ctc_loss = _ctc_loss(
            logits=text_ctc_logits,
            labels=text_ctc_labels,
            pad_id=self.text_ctc_pad_id,
            blank_id=self.text_ctc_blank_id,
        )
        bopomofo_ctc_loss = _ctc_loss(
            logits=bopomofo_logits,
            labels=bopomofo_labels,
            pad_id=self.bopomofo_pad_id,
            blank_id=self.bopomofo_blank_id,
            target_lengths=bopomofo_label_lengths,
        )

        compressed_states = self.compressor(acoustic_states)
        correction_logits = None
        correction_loss = None
        if self.context_corrector is not None:
            ctc_probs = text_ctc_logits.detach().softmax(dim=-1)
            compressed_probs = _downsample_probabilities(
                ctc_probs,
                target_length=compressed_states.size(1),
            )
            correction_logits = self.context_corrector(
                compressed_states=compressed_states,
                compressed_ctc_probs=compressed_probs,
            )
            correction_loss = _ctc_loss(
                logits=correction_logits,
                labels=text_ctc_labels,
                pad_id=self.text_ctc_pad_id,
                blank_id=self.text_ctc_blank_id,
            )

        loss = None
        weighted_losses: list[torch.Tensor] = []
        if text_ctc_loss is not None and self.text_ctc_weight > 0.0:
            weighted_losses.append(self.text_ctc_weight * text_ctc_loss)
        if correction_loss is not None and self.correction_weight > 0.0:
            weighted_losses.append(self.correction_weight * correction_loss)
        if bopomofo_ctc_loss is not None and self.bopomofo_ctc_weight > 0.0:
            weighted_losses.append(self.bopomofo_ctc_weight * bopomofo_ctc_loss)
        if weighted_losses:
            loss = torch.stack(weighted_losses).sum()

        return WhisperTWOutput(
            loss=loss,
            text_ctc_loss=text_ctc_loss,
            correction_loss=correction_loss,
            bopomofo_ctc_loss=bopomofo_ctc_loss,
            text_ctc_logits=text_ctc_logits,
            correction_logits=correction_logits,
            bopomofo_logits=bopomofo_logits,
        )

    @torch.no_grad()
    def generate_ctc(self, input_features: torch.Tensor, use_corrector: bool = True) -> torch.Tensor:
        self.eval()
        output = self(input_features=input_features)
        logits = (
            output.correction_logits
            if use_corrector and output.correction_logits is not None
            else output.text_ctc_logits
        )
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_features: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        del bos_id, eos_id, max_new_tokens
        return self.generate_ctc(input_features=input_features)
