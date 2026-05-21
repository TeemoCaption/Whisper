from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import WhisperModel


@dataclass
class WhisperTWOutput:
    loss: torch.Tensor | None
    text_loss: torch.Tensor | None
    bopomofo_ctc_loss: torch.Tensor | None
    logits: torch.Tensor
    bopomofo_logits: torch.Tensor


class AcousticCompressor(nn.Module):
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
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.stride * 2 + 1,
            stride=self.stride,
            padding=self.stride,
            groups=1,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        compressed = self.conv(sequence.transpose(1, 2)).transpose(1, 2)
        compressed = self.encoder(compressed)
        return self.norm(compressed)


class GatedCorrectionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.audio_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.bopomofo_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.audio_norm = nn.LayerNorm(hidden_size)
        self.bopomofo_norm = nn.LayerNorm(hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_states: torch.Tensor,
        audio_memory: torch.Tensor,
        bopomofo_memory: torch.Tensor,
    ) -> torch.Tensor:
        residual = token_states
        hidden = self.audio_norm(token_states)
        attended, _ = self.audio_attention(
            hidden,
            audio_memory,
            audio_memory,
            need_weights=False,
        )
        token_states = residual + self.dropout(attended)

        residual = token_states
        hidden = self.bopomofo_norm(token_states)
        bopomofo_attended, _ = self.bopomofo_attention(
            hidden,
            bopomofo_memory,
            bopomofo_memory,
            need_weights=False,
        )
        correction_gate = torch.sigmoid(
            self.gate(torch.cat([hidden, bopomofo_attended], dim=-1))
        )
        token_states = residual + self.dropout(correction_gate * bopomofo_attended)

        residual = token_states
        hidden = self.ffn_norm(token_states)
        return residual + self.dropout(self.ffn(hidden))


class ContextCorrector(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_target_length: int,
    ) -> None:
        super().__init__()
        self.max_target_length = max_target_length
        self.query_embedding = nn.Embedding(max_target_length, hidden_size)
        self.layers = nn.ModuleList(
            [
                GatedCorrectionLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        audio_memory: torch.Tensor,
        bopomofo_memory: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        target_length = min(target_length, self.max_target_length)
        positions = torch.arange(
            target_length,
            device=audio_memory.device,
        )
        token_states = self.query_embedding(positions).unsqueeze(0).expand(
            audio_memory.size(0),
            -1,
            -1,
        )
        for layer in self.layers:
            token_states = layer(
                token_states=token_states,
                audio_memory=audio_memory,
                bopomofo_memory=bopomofo_memory,
            )
        return self.output_head(self.norm(token_states))


class WhisperTWModel(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        vocab_size: int,
        bopomofo_vocab_size: int,
        text_pad_id: int,
    ) -> None:
        super().__init__()
        model_cfg = config["model"]
        train_cfg = config.get("training", {})
        acoustic_cfg = model_cfg["acoustic_encoder"]
        compressor_cfg = model_cfg["acoustic_compressor"]
        corrector_cfg = model_cfg["context_corrector"]
        ctc_cfg = model_cfg["bopomofo_ctc"]

        self.text_pad_id = text_pad_id
        self.ctc_enabled = bool(ctc_cfg.get("enabled", True))
        self.ctc_weight = float(ctc_cfg.get("loss_weight", 0.3))
        self.text_loss_weight = float(train_cfg.get("text_loss_weight", 1.0))
        self.text_label_smoothing = float(train_cfg.get("text_label_smoothing", 0.0))
        self.correction_weight = float(corrector_cfg.get("loss_weight", 0.2))

        self.whisper = WhisperModel.from_pretrained(model_cfg["whisper_name"])
        whisper_hidden = self.whisper.config.d_model
        unfreeze_last_n = int(model_cfg.get("unfreeze_encoder_last_n_layers", 0))
        if model_cfg.get("freeze_whisper_encoder", True):
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
            if unfreeze_last_n > 0:
                encoder_layers = self.whisper.encoder.layers
                for layer in encoder_layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        acoustic_hidden = int(acoustic_cfg.get("hidden_size", 768))
        corrector_hidden = int(corrector_cfg["hidden_size"])
        self.encoder_projection = nn.Linear(whisper_hidden, acoustic_hidden)

        acoustic_layer = nn.TransformerEncoderLayer(
            d_model=acoustic_hidden,
            nhead=int(acoustic_cfg["num_heads"]),
            dim_feedforward=acoustic_hidden * 4,
            dropout=float(acoustic_cfg.get("dropout", 0.1)),
            batch_first=True,
            norm_first=True,
        )
        self.acoustic_encoder = nn.TransformerEncoder(
            acoustic_layer,
            num_layers=int(acoustic_cfg["num_layers"]),
        )
        self.acoustic_norm = nn.LayerNorm(acoustic_hidden)

        self.text_ctc_head = nn.Linear(acoustic_hidden, vocab_size)
        self.bopomofo_head = nn.Linear(acoustic_hidden, bopomofo_vocab_size)
        self.bopomofo_token_embedding = nn.Embedding(
            bopomofo_vocab_size,
            acoustic_hidden,
        )
        compressor_dropout = float(
            compressor_cfg.get("dropout", acoustic_cfg.get("dropout", 0.1))
        )
        self.bopomofo_compressor = AcousticCompressor(
            hidden_size=acoustic_hidden,
            stride=int(compressor_cfg.get("stride", 4)),
            num_layers=int(compressor_cfg.get("num_layers", 1)),
            num_heads=int(compressor_cfg.get("num_heads", acoustic_cfg["num_heads"])),
            dropout=compressor_dropout,
        )
        self.audio_compressor = AcousticCompressor(
            hidden_size=acoustic_hidden,
            stride=int(compressor_cfg.get("stride", 4)),
            num_layers=int(compressor_cfg.get("num_layers", 1)),
            num_heads=int(compressor_cfg.get("num_heads", acoustic_cfg["num_heads"])),
            dropout=compressor_dropout,
        )
        self.audio_memory_projection = nn.Linear(acoustic_hidden, corrector_hidden)
        self.bopomofo_memory_projection = nn.Linear(acoustic_hidden, corrector_hidden)
        self.corrector = ContextCorrector(
            vocab_size=vocab_size,
            hidden_size=corrector_hidden,
            num_layers=int(corrector_cfg["num_layers"]),
            num_heads=int(corrector_cfg["num_heads"]),
            dropout=float(corrector_cfg.get("dropout", 0.1)),
            max_target_length=int(corrector_cfg["max_target_length"]),
        )

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        bopomofo_labels: torch.Tensor | None = None,
        bopomofo_label_lengths: torch.Tensor | None = None,
    ) -> WhisperTWOutput:
        acoustic_hidden = self._encode_feature_sequence(input_features)
        ctc_input_lengths = torch.full(
            size=(acoustic_hidden.size(0),),
            fill_value=acoustic_hidden.size(1),
            dtype=torch.long,
            device=acoustic_hidden.device,
        )

        bopomofo_logits = self.bopomofo_head(acoustic_hidden)
        text_ctc_logits = self.text_ctc_head(acoustic_hidden)
        bopomofo_probs = bopomofo_logits.softmax(dim=-1)
        bopomofo_context = torch.matmul(
            bopomofo_probs,
            self.bopomofo_token_embedding.weight,
        )
        audio_memory = self.audio_memory_projection(
            self.audio_compressor(acoustic_hidden)
        )
        bopomofo_memory = self.bopomofo_memory_projection(
            self.bopomofo_compressor(bopomofo_context)
        )
        target_length = decoder_input_ids.size(1)
        if labels is not None:
            target_length = labels.size(1)
        logits = self.corrector(
            audio_memory=audio_memory,
            bopomofo_memory=bopomofo_memory,
            target_length=target_length,
        )

        text_loss = None
        if labels is not None:
            target_lengths = (labels != self.text_pad_id).sum(dim=1)
            text_ctc_loss = nn.functional.ctc_loss(
                log_probs=text_ctc_logits.log_softmax(dim=-1).transpose(0, 1),
                targets=labels,
                input_lengths=ctc_input_lengths,
                target_lengths=target_lengths,
                blank=self.text_pad_id,
                zero_infinity=True,
            )
            correction_labels = labels[:, : logits.size(1)]
            correction_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                correction_labels.reshape(-1),
                ignore_index=self.text_pad_id,
                label_smoothing=self.text_label_smoothing,
            )
            text_loss = text_ctc_loss + self.correction_weight * correction_loss

        ctc_loss = None
        if (
            self.ctc_enabled
            and bopomofo_labels is not None
            and bopomofo_label_lengths is not None
        ):
            log_probs = bopomofo_logits.log_softmax(dim=-1).transpose(0, 1)
            ctc_loss = nn.functional.ctc_loss(
                log_probs=log_probs,
                targets=bopomofo_labels,
                input_lengths=ctc_input_lengths,
                target_lengths=bopomofo_label_lengths.to(bopomofo_logits.device),
                blank=0,
                zero_infinity=True,
            )

        loss = None
        if text_loss is not None:
            loss = self.text_loss_weight * text_loss
            if ctc_loss is not None:
                loss = loss + self.ctc_weight * ctc_loss

        return WhisperTWOutput(
            loss=loss,
            text_loss=text_loss,
            bopomofo_ctc_loss=ctc_loss,
            logits=logits,
            bopomofo_logits=bopomofo_logits,
        )

    def _encode_feature_sequence(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden = self.whisper.encoder(input_features).last_hidden_state
        hidden = self.encoder_projection(hidden)
        hidden = self.acoustic_encoder(hidden)
        return self.acoustic_norm(hidden)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_features: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        self.eval()
        acoustic_hidden = self._encode_feature_sequence(input_features)
        token_ids = self.text_ctc_head(acoustic_hidden).argmax(dim=-1)
        decoded: list[list[int]] = []
        for sample_ids in token_ids.tolist():
            collapsed: list[int] = [bos_id]
            previous_id: int | None = None
            for token_id in sample_ids:
                if token_id != self.text_pad_id and token_id != previous_id:
                    collapsed.append(token_id)
                    if len(collapsed) >= max_new_tokens + 1:
                        break
                previous_id = token_id
            if collapsed[-1] != eos_id and len(collapsed) < max_new_tokens + 1:
                collapsed.append(eos_id)
            decoded.append(collapsed)

        output_length = max(len(ids) for ids in decoded)
        generated = torch.full(
            (len(decoded), output_length),
            self.text_pad_id,
            dtype=torch.long,
            device=input_features.device,
        )
        for index, ids in enumerate(decoded):
            generated[index, : len(ids)] = torch.tensor(
                ids,
                dtype=torch.long,
                device=input_features.device,
            )
        return generated
