#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
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
from whisper_tw.text_norm import build_text_normalizer

try:
    from scripts.lora_adapters import (
        apply_peft_adapters,
        count_parameters,
        is_peft_enabled,
        save_peft_artifacts,
        update_adalora_rank_allocation,
    )
except ImportError:
    from lora_adapters import (
        apply_peft_adapters,
        count_parameters,
        is_peft_enabled,
        save_peft_artifacts,
        update_adalora_rank_allocation,
    )


def parse_args(
    *,
    default_config: str = "configs/baseline.yaml",
    description: str = "訓練 Whisper-medium 基線模型。",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_config,
        help="Whisper 訓練設定檔路徑。",
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


def get_whisper_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("whisper_train")
    if isinstance(section, dict):
        return section
    section = config.get("whisper_baseline")
    if isinstance(section, dict):
        return section
    return {}


def get_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data")
    if isinstance(data_cfg, dict):
        return data_cfg
    experiment_cfg = get_whisper_experiment_config(config)
    data_cfg = experiment_cfg.get("data") if isinstance(experiment_cfg, dict) else None
    if isinstance(data_cfg, dict):
        return data_cfg
    return {}


def resolve_output_dir(config: dict[str, Any], model_name_or_path: str) -> Path:
    experiment_cfg = get_whisper_experiment_config(config)
    output_dir = get_nested(experiment_cfg, "training", "output_dir")
    if output_dir:
        return Path(str(output_dir))
    return Path("artifacts") / "baselines" / f"{sanitize_model_name(model_name_or_path)}_ft"


def resolve_split_source(
    base_config: dict[str, Any],
    train_data_cfg: dict[str, Any],
    split: str,
) -> str | Path:
    if split == str(train_data_cfg.get("train_split", "train")) and train_data_cfg.get("train_tsv"):
        return train_data_cfg["train_tsv"]
    if split == str(train_data_cfg.get("eval_split", "dev")) and train_data_cfg.get("eval_tsv"):
        return train_data_cfg["eval_tsv"]
    base_data_cfg = get_data_config(base_config)
    if not base_data_cfg:
        return split
    return resolve_common_voice_split_source(base_data_cfg, split)


class WhisperTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        base_config: dict[str, Any],
        train_data_cfg: dict[str, Any],
        split: str,
        processor,
        max_samples: int | None = None,
        language_filter: str | None = None,
    ) -> None:
        data_cfg = get_data_config(base_config)
        if not data_cfg:
            data_cfg = train_data_cfg
        split_source = resolve_split_source(base_config, train_data_cfg, split)
        self.samples = read_common_voice_split(
            data_cfg.get("root", "data"),
            split_source,
            language_filter=language_filter,
        )
        if max_samples is not None:
            self.samples = self.samples[: max(0, int(max_samples))]
        self.processor = processor
        self.sample_rate = int(data_cfg.get("sample_rate", 16000))
        self.max_audio_samples = int(
            self.sample_rate * float(data_cfg.get("max_audio_seconds", 30.0))
        )
        self.normalizer = build_text_normalizer(data_cfg.get("text_normalization"))
        self.language_filter = language_filter

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
    data_cfg = get_data_config(base_config)
    normalizer = build_text_normalizer(
        data_cfg.get("text_normalization")
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
    candidates = [model]
    for attr in ("base_model", "model"):
        nested = getattr(model, attr, None)
        if nested is not None:
            candidates.append(nested)
            nested_model = getattr(nested, "model", None)
            if nested_model is not None:
                candidates.append(nested_model)

    for candidate in candidates:
        if hasattr(candidate, "model") and hasattr(candidate.model, "encoder"):
            return candidate.model.encoder
        if hasattr(candidate, "encoder"):
            return candidate.encoder
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
        "bf16": bool(training_cfg.get("bf16", False)),
        "save_steps": int(training_cfg.get("save_steps", 500)),
        "logging_steps": int(training_cfg.get("logging_steps", 25)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "predict_with_generate": bool(training_cfg.get("predict_with_generate", True)),
        "generation_max_length": int(training_cfg.get("generation_max_length", 192)),
        "generation_num_beams": int(training_cfg.get("generation_num_beams", 1)),
        "load_best_model_at_end": True,
        "metric_for_best_model": str(training_cfg.get("metric_for_best_model", "cer")),
        "greater_is_better": bool(training_cfg.get("greater_is_better", False)),
        "report_to": list(training_cfg.get("report_to", [])),
        "remove_unused_columns": False,
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 2)),
        "disable_tqdm": bool(training_cfg.get("disable_tqdm", False)),
    }

    signature = inspect.signature(training_cls.__init__)
    eval_strategy_key = (
        "eval_strategy"
        if "eval_strategy" in signature.parameters
        else "evaluation_strategy"
    )
    kwargs[eval_strategy_key] = str(training_cfg.get("eval_strategy", "steps"))
    kwargs["eval_steps"] = int(training_cfg.get("eval_steps", 500))
    kwargs["logging_strategy"] = str(training_cfg.get("logging_strategy", "steps"))
    kwargs["run_name"] = str(training_cfg.get("run_name") or output_dir.name)
    optional_int_args = (
        "eval_accumulation_steps",
        "torch_empty_cache_steps",
    )
    for key in optional_int_args:
        value = training_cfg.get(key)
        if value is not None:
            kwargs[key] = int(value)
    optional_bool_args = (
        "batch_eval_metrics",
        "eval_do_concat_batches",
        "include_inputs_for_metrics",
    )
    for key in optional_bool_args:
        value = training_cfg.get(key)
        if value is not None:
            kwargs[key] = bool(value)

    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return training_cls(**supported_kwargs)


def configure_wandb_environment(training_cfg: dict[str, Any], output_dir: Path) -> None:
    report_to = training_cfg.get("report_to", [])
    if isinstance(report_to, str):
        enabled = report_to.lower() == "wandb"
    else:
        enabled = "wandb" in [str(item).lower() for item in report_to]
    if not enabled:
        return

    project = str(training_cfg.get("wandb_project") or "whisper-tw")
    run_name = str(training_cfg.get("run_name") or output_dir.name)
    log_model = str(training_cfg.get("wandb_log_model", "false")).lower()
    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_NAME", run_name)
    os.environ.setdefault("WANDB_LOG_MODEL", log_model)


def get_process_memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def build_adalora_callback(trainer_callback_cls, peft_info: dict[str, Any]):
    if str(peft_info.get("method") or "").lower() != "adalora":
        return None

    class AdaLoraRankAllocatorCallback(trainer_callback_cls):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if model is not None:
                update_adalora_rank_allocation(model, int(state.global_step) + 1)
            return control

    return AdaLoraRankAllocatorCallback()


def configure_gradient_checkpointing(model, model_cfg: dict[str, Any], peft_enabled: bool) -> None:
    if not bool(model_cfg.get("gradient_checkpointing", True)):
        return

    if peft_enabled and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    use_reentrant = bool(model_cfg.get("gradient_checkpointing_use_reentrant", False))
    kwargs = {"use_reentrant": use_reentrant}
    signature = inspect.signature(model.gradient_checkpointing_enable)
    if "gradient_checkpointing_kwargs" in signature.parameters:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    else:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False


def maybe_warmup_first_batch(
    train_dataset: WhisperTrainingDataset,
    eval_dataset: WhisperTrainingDataset,
    training_cfg: dict[str, Any],
) -> None:
    if not bool(training_cfg.get("warmup_first_batch", True)):
        return

    print("預熱第一筆訓練與驗證音訊，確認音訊解碼與特徵抽取可正常執行。", flush=True)
    train_dataset[0]
    eval_dataset[0]
    print("第一筆音訊預熱完成。", flush=True)


def main(
    *,
    default_config: str = "configs/baseline.yaml",
    description: str = "訓練 Whisper-medium 基線模型。",
) -> None:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )

    args = parse_args(default_config=default_config, description=description)
    config = load_config(args.config)
    base_config_path = Path(str(config.get("base_config") or "configs/config.yaml"))
    base_config = load_config(base_config_path)

    train_cfg = get_whisper_experiment_config(config)
    model_cfg = train_cfg.get("model", {}) or {}
    data_cfg = train_cfg.get("data", {}) or {}
    training_cfg = train_cfg.get("training", {}) or {}
    early_stopping_cfg = train_cfg.get("early_stopping", {}) or {}
    peft_cfg = train_cfg.get("peft", {}) or {}
    language_filter = data_cfg.get("language_filter")
    if language_filter is None and str(peft_cfg.get("adapter_scope") or "").lower() == "language":
        language_filter = peft_cfg.get("active_language")
    language_filter = None if language_filter in (None, "") else str(language_filter)

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

    peft_info: dict[str, Any] = {"enabled": False, **count_parameters(model)}
    if is_peft_enabled(peft_cfg):
        model, peft_info = apply_peft_adapters(model, peft_cfg)
        encoder = get_whisper_encoder(model)
        layers = getattr(encoder, "layers", None) if encoder is not None else None
        trainability = {
            "encoder_layers": len(layers) if layers is not None else 0,
            "unfreeze_encoder_last_n_layers": 0,
            **count_parameters(model),
        }
    else:
        trainability = configure_encoder_trainability(
            model,
            freeze_encoder=freeze_encoder,
            unfreeze_last_n_layers=unfreeze_encoder_last_n_layers,
            train_decoder=train_decoder,
        )
    configure_gradient_checkpointing(model, model_cfg, is_peft_enabled(peft_cfg))

    train_dataset = WhisperTrainingDataset(
        base_config=base_config,
        train_data_cfg=data_cfg,
        split=train_split,
        processor=processor,
        max_samples=None if max_train_samples is None else int(max_train_samples),
        language_filter=language_filter,
    )
    eval_dataset = WhisperTrainingDataset(
        base_config=base_config,
        train_data_cfg=data_cfg,
        split=eval_split,
        processor=processor,
        max_samples=None if max_eval_samples is None else int(max_eval_samples),
        language_filter=language_filter,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "訓練資料為空；請先執行 Common Voice 前處理，或檢查 language_filter。"
        )
    if len(eval_dataset) == 0:
        raise ValueError(
            "驗證資料為空；請先執行 Common Voice 前處理，或檢查 language_filter。"
        )
    print(
        json.dumps(
            {
                "stage": "datasets_ready",
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "first_train_audio": str(train_dataset.samples[0].audio_path),
                "first_eval_audio": str(eval_dataset.samples[0].audio_path),
                "process_memory_mb": get_process_memory_mb(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    maybe_warmup_first_batch(train_dataset, eval_dataset, training_cfg)

    configure_wandb_environment(training_cfg, output_dir)
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
    adalora_callback = build_adalora_callback(TrainerCallback, peft_info)
    if adalora_callback is not None:
        callbacks.append(adalora_callback)

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
                "language_filter": language_filter,
                "output_dir": str(output_dir),
                "final_dir": str(output_dir / "final"),
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "metric_for_best_model": training_args.metric_for_best_model,
                "load_best_model_at_end": training_args.load_best_model_at_end,
                "freeze_encoder": freeze_encoder,
                "train_decoder": train_decoder,
                "peft": peft_info,
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

    print("開始訓練；若進度停在 0%，通常是在解碼第一批音訊與建立特徵。", flush=True)
    trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    if bool(peft_info.get("enabled", False)):
        save_peft_artifacts(trainer.model, final_dir, peft_info)

    summary = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "metric_for_best_model": training_args.metric_for_best_model,
        "final_dir": str(final_dir),
        "peft": peft_info,
    }
    (output_dir / "best_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "已保存 Whisper 基線 BEST 權重: "
        f"{final_dir} best_checkpoint={trainer.state.best_model_checkpoint} "
        f"best_metric={trainer.state.best_metric}",
        flush=True,
    )


if __name__ == "__main__":
    main()
