#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config, resolve_common_voice_split_source
from whisper_tw.data import load_audio_waveform, read_common_voice_split
from whisper_tw.metrics import character_error_rate
from whisper_tw.text_normalization import build_text_normalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="以目前資料集微調 Whisper 基線模型。")
    parser.add_argument(
        "--config",
        default="configs/whisper_finetune.yaml",
        help="Whisper 微調設定檔路徑。",
    )
    return parser.parse_args()


def sanitize_model_name(model_name_or_path: str) -> str:
    text = str(model_name_or_path or "whisper").strip().replace("\\", "/")
    name = text.rstrip("/").split("/")[-1] or "whisper"
    safe = "".join(char if char.isalnum() else "_" for char in name.lower())
    return "_".join(part for part in safe.split("_") if part) or "whisper"


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def resolve_output_dir(config: dict[str, Any], model_name_or_path: str) -> Path:
    output_dir = get_nested(config, "whisper_finetune", "training", "output_dir")
    if output_dir:
        return Path(str(output_dir))
    return Path("artifacts") / "baselines" / f"{sanitize_model_name(model_name_or_path)}_ft"


class WhisperFineTuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        base_config: dict[str, Any],
        split: str,
        processor,
        max_samples: int | None = None,
    ) -> None:
        data_cfg = base_config["data"]
        split_source = resolve_common_voice_split_source(data_cfg, split)
        self.samples = read_common_voice_split(data_cfg["root"], split_source)
        if max_samples is not None:
            self.samples = self.samples[: max(0, int(max_samples))]
        self.processor = processor
        self.sample_rate = int(data_cfg.get("sample_rate", 16000))
        self.max_audio_samples = int(
            self.sample_rate * float(data_cfg.get("max_audio_seconds", 30.0))
        )
        self.normalizer = build_text_normalizer(data_cfg.get("text_normalization"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        waveform = load_audio_waveform(sample.audio_path, self.sample_rate)
        waveform = waveform[: self.max_audio_samples]
        text = sample.text
        if self.normalizer.enabled:
            text = self.normalizer(text)
        if not text:
            raise ValueError(f"樣本缺少文字: {sample.rel_path}")

        input_features = self.processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
        ).input_features[0]
        labels = self.processor.tokenizer(text).input_ids
        return {"input_features": input_features, "labels": labels}


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )
        decoder_start_token_id = self.processor.tokenizer.bos_token_id
        if (
            decoder_start_token_id is not None
            and labels.size(1) > 0
            and (labels[:, 0] == decoder_start_token_id).all()
        ):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def build_compute_metrics(processor, base_config: dict[str, Any]):
    normalizer = build_text_normalizer(
        base_config.get("data", {}).get("text_normalization")
    )

    def normalize(text: str) -> str:
        value = str(text or "")
        if not normalizer.enabled:
            return value.strip()
        return normalizer(value)

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)
        predictions = [normalize(text) for text in pred_texts]
        references = [normalize(text) for text in label_texts]
        return {"cer": character_error_rate(predictions, references)}

    return compute_metrics


def get_whisper_encoder(model):
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder
    if hasattr(model, "encoder"):
        return model.encoder
    return None


def configure_encoder_trainability(
    model,
    *,
    freeze_encoder: bool,
    unfreeze_last_n_layers: int,
    train_decoder: bool,
) -> dict[str, int]:
    encoder = get_whisper_encoder(model)
    if freeze_encoder and not train_decoder:
        for param in model.parameters():
            param.requires_grad = False
    if encoder is None:
        return {
            "encoder_layers": 0,
            "unfreeze_encoder_last_n_layers": 0,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
        }

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    if not train_decoder:
        decoder = getattr(getattr(model, "model", None), "decoder", None)
        if decoder is not None:
            for param in decoder.parameters():
                param.requires_grad = False
        projection = getattr(model, "proj_out", None)
        if projection is not None:
            for param in projection.parameters():
                param.requires_grad = False

    layers = getattr(encoder, "layers", None)
    layer_count = len(layers) if layers is not None else 0
    requested_layers = max(0, int(unfreeze_last_n_layers))
    if freeze_encoder and layers is not None and requested_layers > 0:
        for layer in layers[-requested_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    return {
        "encoder_layers": layer_count,
        "unfreeze_encoder_last_n_layers": min(requested_layers, layer_count),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }


def build_training_arguments(training_cls, training_cfg: dict[str, Any], output_dir: Path):
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(
            training_cfg.get("per_device_train_batch_size", 4)
        ),
        "per_device_eval_batch_size": int(
            training_cfg.get("per_device_eval_batch_size", 4)
        ),
        "gradient_accumulation_steps": int(
            training_cfg.get("gradient_accumulation_steps", 4)
        ),
        "learning_rate": float(training_cfg.get("learning_rate", 1.0e-5)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 500)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 10.0)),
        "fp16": bool(training_cfg.get("fp16", False)),
        "save_steps": int(training_cfg.get("save_steps", 500)),
        "logging_steps": int(training_cfg.get("logging_steps", 25)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "predict_with_generate": True,
        "generation_max_length": int(training_cfg.get("generation_max_length", 192)),
        "generation_num_beams": int(training_cfg.get("generation_num_beams", 1)),
        "load_best_model_at_end": True,
        "metric_for_best_model": str(training_cfg.get("metric_for_best_model", "cer")),
        "greater_is_better": bool(training_cfg.get("greater_is_better", False)),
        "report_to": list(training_cfg.get("report_to", [])),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 2)),
    }

    signature = inspect.signature(training_cls.__init__)
    eval_strategy_key = (
        "eval_strategy"
        if "eval_strategy" in signature.parameters
        else "evaluation_strategy"
    )
    kwargs[eval_strategy_key] = str(training_cfg.get("eval_strategy", "steps"))
    kwargs["eval_steps"] = int(training_cfg.get("eval_steps", 500))

    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return training_cls(**supported_kwargs)


def main() -> None:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    args = parse_args()
    config = load_config(args.config)
    base_config_path = Path(str(config.get("base_config") or "configs/config.yaml"))
    base_config = load_config(base_config_path)

    fine_cfg = config.get("whisper_finetune", {}) or {}
    model_cfg = fine_cfg.get("model", {}) or {}
    data_cfg = fine_cfg.get("data", {}) or {}
    training_cfg = fine_cfg.get("training", {}) or {}
    early_stopping_cfg = fine_cfg.get("early_stopping", {}) or {}

    model_name_or_path = str(
        model_cfg.get("model_name_or_path") or "openai/whisper-medium"
    )
    language = str(model_cfg.get("language") or "zh")
    task = str(model_cfg.get("task") or "transcribe")
    freeze_encoder = bool(model_cfg.get("freeze_encoder", False))
    unfreeze_encoder_last_n_layers = int(
        model_cfg.get("unfreeze_encoder_last_n_layers", 0)
    )
    train_decoder = bool(model_cfg.get("train_decoder", False))
    gradient_checkpointing = bool(model_cfg.get("gradient_checkpointing", True))

    train_split = str(
        data_cfg.get("train_split") or base_config.get("data", {}).get("train_split", "train")
    )
    eval_split = str(
        data_cfg.get("eval_split") or base_config.get("data", {}).get("dev_split", "dev")
    )
    max_train_samples = training_cfg.get("max_train_samples")
    max_eval_samples = training_cfg.get("max_eval_samples")
    output_dir = resolve_output_dir(config, model_name_or_path)

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        language=language,
        task=task,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    trainability = configure_encoder_trainability(
        model,
        freeze_encoder=freeze_encoder,
        unfreeze_last_n_layers=unfreeze_encoder_last_n_layers,
        train_decoder=train_decoder,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = WhisperFineTuneDataset(
        base_config=base_config,
        split=train_split,
        processor=processor,
        max_samples=None if max_train_samples is None else int(max_train_samples),
    )
    eval_dataset = WhisperFineTuneDataset(
        base_config=base_config,
        split=eval_split,
        processor=processor,
        max_samples=None if max_eval_samples is None else int(max_eval_samples),
    )

    training_args = build_training_arguments(
        Seq2SeqTrainingArguments,
        training_cfg,
        output_dir,
    )
    callbacks = []
    if bool(early_stopping_cfg.get("enabled", True)):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_cfg.get("patience", 3)),
                early_stopping_threshold=float(
                    early_stopping_cfg.get("threshold", 0.0)
                ),
            )
        )

    trainer_kwargs: dict[str, Any] = {
        "args": training_args,
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorSpeechSeq2SeqWithPadding(processor),
        "compute_metrics": build_compute_metrics(processor, base_config),
        "callbacks": callbacks,
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    print(
        json.dumps(
            {
                "config": args.config,
                "base_config": str(base_config_path),
                "model_name_or_path": model_name_or_path,
                "train_split": train_split,
                "eval_split": eval_split,
                "output_dir": str(output_dir),
                "final_dir": str(output_dir / "final"),
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "metric_for_best_model": training_args.metric_for_best_model,
                "load_best_model_at_end": training_args.load_best_model_at_end,
                "freeze_encoder": freeze_encoder,
                "train_decoder": train_decoder,
                "encoder_layers": trainability["encoder_layers"],
                "unfreeze_encoder_last_n_layers": trainability[
                    "unfreeze_encoder_last_n_layers"
                ],
                "trainable_parameters": trainability["trainable_parameters"],
                "total_parameters": trainability["total_parameters"],
                "save_best_to_final": True,
                "early_stopping": {
                    "enabled": bool(early_stopping_cfg.get("enabled", True)),
                    "patience": int(early_stopping_cfg.get("patience", 3)),
                    "threshold": float(early_stopping_cfg.get("threshold", 0.0)),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    summary = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "metric_for_best_model": training_args.metric_for_best_model,
        "final_dir": str(final_dir),
    }
    (output_dir / "best_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "已保存 Whisper 微調 BEST 權重: "
        f"{final_dir} best_checkpoint={trainer.state.best_model_checkpoint} "
        f"best_metric={trainer.state.best_metric}",
        flush=True,
    )


if __name__ == "__main__":
    main()
