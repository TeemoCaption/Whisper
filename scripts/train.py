#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.runtime_env import configure_runtime_environment

configure_runtime_environment()

print("初始化 AdaLoRA 訓練環境。", flush=True)
import torch
print("PyTorch 載入完成。", flush=True)

from whisper_tw.config import load_config, resolve_common_voice_split_source
from whisper_tw.data import load_audio_waveform, read_common_voice_split
from whisper_tw.metrics import character_error_rate
from whisper_tw.text_norm import build_text_normalizer

try:
    from scripts.lora_adapters import (
        apply_peft_adapters,
        count_parameters,
        is_peft_enabled,
        sanitize_adapter_name,
        save_peft_artifacts,
        update_adalora_rank_allocation,
    )
except ImportError:
    from lora_adapters import (
        apply_peft_adapters,
        count_parameters,
        is_peft_enabled,
        sanitize_adapter_name,
        save_peft_artifacts,
        update_adalora_rank_allocation,
    )


def parse_args(
    *,
    default_config: str = "configs/config.yaml",
    description: str = "訓練 Whisper-medium 語言專屬 AdaLoRA 模型。",
    include_language_arg: bool = True,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_config,
        help="Whisper 訓練設定檔路徑。",
    )
    if include_language_arg:
        parser.add_argument(
            "--language",
            required=True,
            help="訓練語言專屬 adapter，例如 zh-TW 或 nan-tw。",
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
    return {}


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_merged_experiment_config(
    config: dict[str, Any],
    base_config: dict[str, Any],
    experiment_key: str | None = None,
) -> dict[str, Any]:
    keys = (experiment_key,) if experiment_key else ("whisper_train",)
    for key in keys:
        if not key:
            continue
        section = config.get(key)
        if not isinstance(section, dict):
            continue
        base_section = base_config.get(key)
        if isinstance(base_section, dict) and config.get("base_config"):
            return deep_merge_dict(base_section, section)
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


def resolve_training_output_dir(
    experiment_cfg: dict[str, Any],
    model_name_or_path: str,
) -> Path:
    output_dir = get_nested(experiment_cfg, "training", "output_dir")
    if output_dir:
        return Path(str(output_dir))
    return Path("artifacts") / "models" / f"{sanitize_model_name(model_name_or_path)}_adalora"


def resolve_output_dir(config: dict[str, Any], model_name_or_path: str) -> Path:
    return resolve_training_output_dir(
        get_whisper_experiment_config(config),
        model_name_or_path,
    )


def _training_profile_name(output_dir: Path) -> str:
    name = output_dir.name.lower()
    if "h100" in name:
        return "h100"
    if "8gb" in name:
        return "8gb"
    return "local"


LANGUAGE_TRAINING_DEFAULTS: dict[str, dict[str, Any]] = {
    "zh-TW": {
        "learning_rate": 1.0e-5,
        "warmup_steps": 500,
        "num_train_epochs": 10.0,
    },
    "nan-tw": {
        "learning_rate": 1.5e-5,
        "warmup_steps": 500,
        "num_train_epochs": 40.0,
    },
}


def apply_language_training_override(
    train_cfg: dict[str, Any],
    language: str | None,
) -> str | None:
    if language in (None, ""):
        return None

    selected_language = str(language)
    peft_cfg = train_cfg.setdefault("peft", {})
    language_adapters = {
        str(key): sanitize_adapter_name(str(value))
        for key, value in dict(peft_cfg.get("language_adapters") or {}).items()
    }
    if selected_language not in language_adapters:
        available = ", ".join(language_adapters) or "未設定"
        raise ValueError(
            f"--language={selected_language!r} 沒有對應 adapter；可用語言: {available}。"
        )

    adapter_name = language_adapters[selected_language]
    peft_cfg["adapter_scope"] = "language"
    peft_cfg["active_language"] = selected_language

    data_cfg = train_cfg.setdefault("data", {})
    data_cfg["language_filter"] = selected_language

    training_cfg = train_cfg.setdefault("training", {})
    language_training_defaults = dict(
        LANGUAGE_TRAINING_DEFAULTS.get(selected_language, {})
    )
    configured_language_defaults = train_cfg.get("language_training_defaults") or {}
    if isinstance(configured_language_defaults, dict):
        language_training_defaults.update(
            dict(configured_language_defaults.get(selected_language) or {})
        )
    for key, value in language_training_defaults.items():
        training_cfg[key] = value
    base_output_dir = Path(
        str(training_cfg.get("output_dir") or "artifacts/models/whisper_medium_adalora")
    )
    profile = _training_profile_name(base_output_dir)
    suffix = "" if profile == "local" else f"_{profile}"
    training_cfg["output_dir"] = str(
        base_output_dir.parent / f"whisper_medium_adalora_{adapter_name}{suffix}"
    )
    training_cfg["run_name"] = f"whisper-medium-adalora-{adapter_name.replace('_', '-')}-{profile}"
    return adapter_name


def require_language_adapter_selection(train_cfg: dict[str, Any]) -> None:
    peft_cfg = train_cfg.get("peft", {}) or {}
    if not is_peft_enabled(peft_cfg):
        return
    if str(peft_cfg.get("adapter_scope") or "").lower() != "language":
        return
    if peft_cfg.get("active_language"):
        return
    languages = ", ".join(dict(peft_cfg.get("language_adapters") or {}).keys())
    raise ValueError(
        "目前設定使用語言專屬 AdaLoRA，但尚未指定訓練語言；"
        f"請在指令加入 --language，例如 --language zh-TW 或 --language nan-tw。"
        f"可用語言: {languages or '未設定'}。"
    )


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
        self.data_root = data_cfg.get("root", "data")
        self.split_source = str(split_source)
        self.samples = read_common_voice_split(
            self.data_root,
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


def configure_torch_backend(training_cfg: dict[str, Any]) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(training_cfg.get("tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(training_cfg.get("tf32", False))


def build_dataloader_kwargs(
    training_cfg: dict[str, Any],
    *,
    batch_size_key: str,
    shuffle: bool,
) -> dict[str, Any]:
    num_workers = max(0, int(training_cfg.get("dataloader_num_workers", 0)))
    kwargs: dict[str, Any] = {
        "batch_size": int(training_cfg.get(batch_size_key, 1)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("dataloader_pin_memory", False))
        and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            training_cfg.get("dataloader_persistent_workers", False)
        )
        if training_cfg.get("dataloader_prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(training_cfg["dataloader_prefetch_factor"])
    return kwargs


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def autocast_context(
    device: torch.device,
    *,
    fp16: bool,
    bf16: bool,
):
    enabled = device.type == "cuda" and (fp16 or bf16)
    dtype = torch.float16 if fp16 else torch.bfloat16
    try:
        return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.autocast(dtype=dtype, enabled=enabled)


def build_grad_scaler(device: torch.device, fp16: bool):
    enabled = device.type == "cuda" and fp16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_generation_kwargs(training_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "max_length": int(training_cfg.get("generation_max_length", 96)),
        "num_beams": int(training_cfg.get("generation_num_beams", 1)),
    }
    optional_int_keys = {
        "no_repeat_ngram_size": "generation_no_repeat_ngram_size",
    }
    optional_float_keys = {
        "repetition_penalty": "generation_repetition_penalty",
        "length_penalty": "generation_length_penalty",
    }
    for generate_key, config_key in optional_int_keys.items():
        if training_cfg.get(config_key) is not None:
            kwargs[generate_key] = int(training_cfg[config_key])
    for generate_key, config_key in optional_float_keys.items():
        if training_cfg.get(config_key) is not None:
            kwargs[generate_key] = float(training_cfg[config_key])
    return kwargs


def configure_spec_augment(model, training_cfg: dict[str, Any]) -> None:
    spec_cfg = training_cfg.get("spec_augment") or {}
    if not isinstance(spec_cfg, dict) or not spec_cfg:
        return
    enabled = bool(spec_cfg.get("enabled", False))
    values = {
        "apply_spec_augment": enabled,
        "mask_time_prob": float(spec_cfg.get("mask_time_prob", 0.0)),
        "mask_time_length": int(spec_cfg.get("mask_time_length", 10)),
        "mask_feature_prob": float(spec_cfg.get("mask_feature_prob", 0.0)),
        "mask_feature_length": int(spec_cfg.get("mask_feature_length", 10)),
    }
    candidates = [model, getattr(model, "base_model", None)]
    if hasattr(model, "get_base_model"):
        candidates.append(model.get_base_model())
    for candidate in candidates:
        config = getattr(candidate, "config", None) if candidate is not None else None
        if config is None:
            continue
        for key, value in values.items():
            if hasattr(config, key):
                setattr(config, key, value)


def configure_whisper_generation(model, *, language: str, task: str) -> None:
    candidates = [model, getattr(model, "base_model", None)]
    if hasattr(model, "get_base_model"):
        candidates.append(model.get_base_model())

    for candidate in candidates:
        if candidate is None:
            continue
        config = getattr(candidate, "config", None)
        generation_config = getattr(candidate, "generation_config", None)
        if generation_config is not None:
            generation_config.language = language
            generation_config.task = task
            generation_config.forced_decoder_ids = None
            generation_config.suppress_tokens = []
        if config is None:
            continue

        for attr in ("max_length", "suppress_tokens", "begin_suppress_tokens"):
            if generation_config is not None and hasattr(config, attr):
                value = getattr(config, attr)
                if value is not None:
                    setattr(generation_config, attr, value)
            if attr in getattr(config, "__dict__", {}):
                delattr(config, attr)
        config.forced_decoder_ids = None


def maybe_tqdm(iterable, *, total: int, desc: str, disabled: bool):
    if disabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def build_optimizer(model, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("目前模型沒有可訓練參數，請檢查低秩設定或凍結設定。")
    return torch.optim.AdamW(
        trainable_params,
        lr=float(training_cfg.get("learning_rate", 1.0e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    *,
    total_update_steps: int,
):
    scheduler_type = str(training_cfg.get("lr_scheduler_type", "linear")).lower()
    warmup_steps = max(0, int(training_cfg.get("warmup_steps", 0)))
    total_update_steps = max(1, int(total_update_steps))
    min_lr_ratio = max(0.0, min(1.0, float(training_cfg.get("min_lr_ratio", 0.0))))

    def linear_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        remaining_steps = total_update_steps - current_step
        decay_steps = max(1, total_update_steps - warmup_steps)
        return max(0.0, float(remaining_steps) / float(decay_steps))

    def linear_floor_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(min_lr_ratio, linear_lambda(current_step))

    def constant_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, constant_lambda)
    if scheduler_type in {"linear_floor", "linear_with_floor"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_floor_lambda)
    if scheduler_type != "linear":
        print(
            f"未支援的學習率排程 {scheduler_type!r}，改用線性排程。",
            flush=True,
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, linear_lambda)


def setup_wandb_run(
    training_cfg: dict[str, Any],
    output_dir: Path,
    *,
    run_config: dict[str, Any],
):
    report_to = training_cfg.get("report_to", [])
    if isinstance(report_to, str):
        enabled = report_to.lower() == "wandb"
    else:
        enabled = "wandb" in [str(item).lower() for item in report_to]
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("已要求記錄到 wandb，但目前環境未安裝 wandb；略過線上紀錄。", flush=True)
        return None
    run = wandb.init(
        project=str(training_cfg.get("wandb_project") or "whisper-tw"),
        name=str(training_cfg.get("run_name") or output_dir.name),
        config=run_config,
        dir=str(output_dir),
    )
    run.define_metric("epoch")
    run.define_metric("*", step_metric="epoch")
    return run


def save_model_artifacts(model, processor, final_dir: Path, peft_info: dict[str, Any]) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(final_dir), safe_serialization=True)
    except TypeError:
        model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    if bool(peft_info.get("enabled", False)):
        save_peft_artifacts(model, final_dir, peft_info)


def is_better_metric(
    value: float,
    best_value: float | None,
    *,
    greater_is_better: bool,
    threshold: float,
) -> bool:
    if best_value is None:
        return True
    if greater_is_better:
        return value > best_value + threshold
    return value < best_value - threshold


def train_one_epoch(
    *,
    model,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    global_step: int,
    training_cfg: dict[str, Any],
    peft_info: dict[str, Any],
    wandb_run,
) -> tuple[int, float]:
    model.train()
    gradient_accumulation_steps = max(
        1,
        int(training_cfg.get("gradient_accumulation_steps", 1)),
    )
    logging_steps = max(1, int(training_cfg.get("logging_steps", 50)))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    empty_cache_steps = int(training_cfg.get("torch_empty_cache_steps", 0) or 0)
    use_fp16 = bool(training_cfg.get("fp16", False))
    use_bf16 = bool(training_cfg.get("bf16", False)) and not use_fp16
    disable_tqdm = bool(training_cfg.get("disable_tqdm", False))

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    recent_loss = 0.0
    recent_count = 0
    progress = maybe_tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"epoch {epoch}/{total_epochs} train",
        disabled=disable_tqdm,
    )
    for batch_index, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, fp16=use_fp16, bf16=use_bf16):
            loss = model(**batch).loss
            scaled_loss = loss / gradient_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        loss_value = float(loss.detach().cpu())
        total_loss += loss_value
        recent_loss += loss_value
        recent_count += 1

        should_step = (
            batch_index % gradient_accumulation_steps == 0
            or batch_index == len(train_loader)
        )
        if not should_step:
            continue

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_grad_norm,
            )

        next_global_step = global_step + 1
        if str(peft_info.get("method") or "").lower() == "adalora":
            update_adalora_rank_allocation(model, next_global_step)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step = next_global_step

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                {
                    "loss": f"{total_loss / batch_index:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        if global_step % logging_steps == 0:
            epoch_progress = (epoch - 1) + (batch_index / max(1, len(train_loader)))
            metrics = {
                "epoch": epoch_progress,
                "train_loss": recent_loss / max(1, recent_count),
                "train_learning_rate": scheduler.get_last_lr()[0],
            }
            print(json.dumps({"stage": "train_step", **metrics}, ensure_ascii=False), flush=True)
            recent_loss = 0.0
            recent_count = 0

        if empty_cache_steps > 0 and global_step % empty_cache_steps == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

    return global_step, total_loss / max(1, len(train_loader))


@torch.no_grad()
def evaluate_loss(
    *,
    model,
    eval_loader,
    device: torch.device,
    epoch: int,
    training_cfg: dict[str, Any],
) -> float:
    model.eval()
    use_fp16 = bool(training_cfg.get("fp16", False))
    use_bf16 = bool(training_cfg.get("bf16", False)) and not use_fp16
    disable_tqdm = bool(training_cfg.get("disable_tqdm", False))
    empty_cache_steps = int(training_cfg.get("torch_empty_cache_steps", 0) or 0)

    total_loss = 0.0
    total_items = 0
    progress = maybe_tqdm(
        eval_loader,
        total=len(eval_loader),
        desc=f"epoch {epoch} eval",
        disabled=disable_tqdm,
    )
    for batch_index, batch in enumerate(progress, start=1):
        batch_size = int(batch["labels"].shape[0])
        batch = move_batch_to_device(batch, device)
        with autocast_context(device, fp16=use_fp16, bf16=use_bf16):
            loss = model(**batch).loss
        loss_value = float(loss.detach().cpu())
        total_loss += loss_value * batch_size
        total_items += batch_size
        if hasattr(progress, "set_postfix"):
            progress.set_postfix({"eval_loss": f"{total_loss / max(1, total_items):.4f}"})
        if empty_cache_steps > 0 and batch_index % empty_cache_steps == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
    return total_loss / max(1, total_items)


@torch.no_grad()
def evaluate_generation_cer(
    *,
    model,
    eval_dataset: WhisperTrainingDataset,
    processor,
    device: torch.device,
    training_cfg: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    max_samples = int(training_cfg.get("eval_cer_max_samples", 0) or 0)
    samples = eval_dataset.samples if max_samples <= 0 else eval_dataset.samples[:max_samples]
    if not samples:
        return {"eval_cer": 0.0, "eval_cer_samples": 0, "eval_cer_seconds": 0.0}

    batch_size = max(
        1,
        int(
            training_cfg.get(
                "eval_cer_batch_size",
                training_cfg.get("per_device_eval_batch_size", 1),
            )
        ),
    )
    use_fp16 = bool(training_cfg.get("fp16", False))
    use_bf16 = bool(training_cfg.get("bf16", False)) and not use_fp16
    disable_tqdm = bool(training_cfg.get("disable_tqdm", False))
    generation_kwargs = build_generation_kwargs(training_cfg)
    references: list[str] = []
    predictions: list[str] = []
    start = time.perf_counter()

    progress = maybe_tqdm(
        range(0, len(samples), batch_size),
        total=math.ceil(len(samples) / batch_size),
        desc=f"epoch {epoch} eval CER",
        disabled=disable_tqdm,
    )
    for start_index in progress:
        batch_samples = samples[start_index : start_index + batch_size]
        waveforms = []
        batch_references = []
        for sample in batch_samples:
            waveform = load_audio_waveform(sample.audio_path, eval_dataset.sample_rate)
            waveform = waveform[: eval_dataset.max_audio_samples]
            waveforms.append(waveform.numpy())
            reference = sample.text
            if eval_dataset.normalizer.enabled:
                reference = eval_dataset.normalizer(reference)
            batch_references.append(reference)

        inputs = processor.feature_extractor(
            waveforms,
            sampling_rate=eval_dataset.sample_rate,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        )
        input_features = inputs["input_features"].to(
            device=device,
            dtype=getattr(model, "dtype", inputs["input_features"].dtype),
        )
        attention_mask = (
            inputs["attention_mask"].to(device)
            if inputs.get("attention_mask") is not None
            else None
        )
        with autocast_context(device, fp16=use_fp16, bf16=use_bf16):
            generate_inputs = {"input_features": input_features, **generation_kwargs}
            if attention_mask is not None:
                generate_inputs["attention_mask"] = attention_mask
            generated_ids = model.generate(**generate_inputs)

        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        if eval_dataset.normalizer.enabled:
            decoded = [eval_dataset.normalizer(text) for text in decoded]
        references.extend(batch_references)
        predictions.extend(decoded)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                {
                    "cer": f"{character_error_rate(predictions, references):.4f}",
                    "samples": len(references),
                }
            )

    return {
        "eval_cer": character_error_rate(predictions, references),
        "eval_cer_samples": len(references),
        "eval_cer_seconds": time.perf_counter() - start,
    }


def main(
    *,
    default_config: str = "configs/config.yaml",
    description: str = "訓練 Whisper-medium 語言專屬 AdaLoRA 模型。",
    experiment_key: str | None = "whisper_train",
    include_language_arg: bool = True,
) -> None:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
    )

    args = parse_args(
        default_config=default_config,
        description=description,
        include_language_arg=include_language_arg,
    )
    print(
        f"啟動 AdaLoRA 訓練: config={args.config} "
        f"language={getattr(args, 'language', None)}",
        flush=True,
    )
    config = load_config(args.config)
    base_config_path = Path(str(config.get("base_config") or "configs/config.yaml"))
    base_config = load_config(base_config_path)

    train_cfg = get_merged_experiment_config(
        config,
        base_config,
        experiment_key=experiment_key,
    )
    if not train_cfg:
        raise ValueError(f"設定檔缺少訓練區塊: {experiment_key or 'whisper_train'}")
    selected_adapter = apply_language_training_override(
        train_cfg,
        getattr(args, "language", None),
    )
    require_language_adapter_selection(train_cfg)
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
    output_dir = resolve_training_output_dir(train_cfg, model_name_or_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_backend(training_cfg)

    print(f"載入 processor: {model_name_or_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        language=language,
        task=task,
    )
    print(f"載入 Whisper 模型: {model_name_or_path}", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)
    print("Whisper 模型載入完成。", flush=True)
    configure_whisper_generation(model, language=language, task=task)
    configure_spec_augment(model, training_cfg)

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
        max_samples=None,
        language_filter=language_filter,
    )
    eval_dataset = WhisperTrainingDataset(
        base_config=base_config,
        train_data_cfg=data_cfg,
        split=eval_split,
        processor=processor,
        max_samples=None,
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
    requested_device = str(training_cfg.get("device") or "auto")
    if requested_device and requested_device != "auto":
        device = torch.device(requested_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=data_collator,
        **build_dataloader_kwargs(
            training_cfg,
            batch_size_key="per_device_train_batch_size",
            shuffle=True,
        ),
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        **build_dataloader_kwargs(
            training_cfg,
            batch_size_key="per_device_eval_batch_size",
            shuffle=False,
        ),
    )
    total_epochs = max(1, int(float(training_cfg.get("num_train_epochs", 10.0))))
    gradient_accumulation_steps = max(
        1,
        int(training_cfg.get("gradient_accumulation_steps", 1)),
    )
    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_update_steps = max(1, updates_per_epoch * total_epochs)
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_lr_scheduler(
        optimizer,
        training_cfg,
        total_update_steps=total_update_steps,
    )
    use_fp16 = bool(training_cfg.get("fp16", False))
    use_bf16 = bool(training_cfg.get("bf16", False)) and not use_fp16
    scaler = build_grad_scaler(device, use_fp16)
    metric_for_best_model = str(training_cfg.get("metric_for_best_model", "eval_loss"))
    supported_best_metrics = {"eval_loss", "eval_cer"}
    if metric_for_best_model not in supported_best_metrics:
        print(
            f"目前手寫訓練迴圈不支援 {metric_for_best_model!r}，改用 eval_loss。",
            flush=True,
        )
        metric_for_best_model = "eval_loss"
    eval_cer_enabled = bool(training_cfg.get("eval_cer_enabled", False)) or (
        metric_for_best_model == "eval_cer"
    )
    greater_is_better = bool(training_cfg.get("greater_is_better", False))
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", True))
    early_stopping_patience = int(early_stopping_cfg.get("patience", 3))
    early_stopping_min_epochs = max(1, int(early_stopping_cfg.get("min_epochs", 1)))
    early_stopping_threshold = float(early_stopping_cfg.get("threshold", 0.0))
    final_dir = output_dir / "final"

    print(
        json.dumps(
            {
                "config": args.config,
                "base_config": str(base_config_path),
                "model_name_or_path": model_name_or_path,
                "train_split": train_split,
                "eval_split": eval_split,
                "train_split_source": train_dataset.split_source,
                "eval_split_source": eval_dataset.split_source,
                "train_data_root": str(train_dataset.data_root),
                "eval_data_root": str(eval_dataset.data_root),
                "language_filter": language_filter,
                "selected_adapter": selected_adapter,
                "output_dir": str(output_dir),
                "final_dir": str(final_dir),
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "train_batches_per_epoch": len(train_loader),
                "eval_batches_per_epoch": len(eval_loader),
                "updates_per_epoch": updates_per_epoch,
                "total_update_steps": total_update_steps,
                "device": str(device),
                "fp16": use_fp16,
                "bf16": use_bf16,
                "training_hyperparameters": {
                    "num_train_epochs": total_epochs,
                    "learning_rate": float(training_cfg.get("learning_rate", 1.0e-5)),
                    "lr_scheduler_type": str(
                        training_cfg.get("lr_scheduler_type", "linear")
                    ),
                    "min_lr_ratio": float(training_cfg.get("min_lr_ratio", 0.0)),
                    "warmup_steps": int(training_cfg.get("warmup_steps", 0)),
                    "generation_max_length": int(
                        training_cfg.get("generation_max_length", 96)
                    ),
                    "generation_no_repeat_ngram_size": int(
                        training_cfg.get("generation_no_repeat_ngram_size", 0) or 0
                    ),
                    "generation_repetition_penalty": float(
                        training_cfg.get("generation_repetition_penalty", 1.0)
                    ),
                    "spec_augment": training_cfg.get("spec_augment") or {},
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "per_device_train_batch_size": int(
                        training_cfg.get("per_device_train_batch_size", 1)
                    ),
                    "per_device_eval_batch_size": int(
                        training_cfg.get("per_device_eval_batch_size", 1)
                    ),
                },
                "metric_for_best_model": metric_for_best_model,
                "eval_cer": {
                    "enabled": eval_cer_enabled,
                    "max_samples": int(training_cfg.get("eval_cer_max_samples", 0) or 0),
                    "batch_size": int(
                        training_cfg.get(
                            "eval_cer_batch_size",
                            training_cfg.get("per_device_eval_batch_size", 1),
                        )
                    ),
                },
                "load_best_model_at_end": bool(
                    training_cfg.get("load_best_model_at_end", True)
                ),
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
                    "min_epochs": early_stopping_min_epochs,
                    "threshold": float(early_stopping_cfg.get("threshold", 0.0)),
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    wandb_run = setup_wandb_run(
        training_cfg,
        output_dir,
        run_config={
            "config": args.config,
            "language": getattr(args, "language", None),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "total_epochs": total_epochs,
            "total_update_steps": total_update_steps,
            "peft": peft_info,
        },
    )

    best_metric: float | None = None
    best_epoch: int | None = None
    best_global_step: int | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, total_epochs + 1):
        global_step, train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=total_epochs,
            global_step=global_step,
            training_cfg=training_cfg,
            peft_info=peft_info,
            wandb_run=wandb_run,
        )
        eval_loss = evaluate_loss(
            model=model,
            eval_loader=eval_loader,
            device=device,
            epoch=epoch,
            training_cfg=training_cfg,
        )
        eval_cer_metrics: dict[str, Any] = {}
        if eval_cer_enabled:
            eval_cer_metrics = evaluate_generation_cer(
                model=model,
                eval_dataset=eval_dataset,
                processor=processor,
                device=device,
                training_cfg=training_cfg,
                epoch=epoch,
            )
        current_metric = (
            float(eval_cer_metrics["eval_cer"])
            if metric_for_best_model == "eval_cer"
            else eval_loss
        )
        improved = is_better_metric(
            current_metric,
            best_metric,
            greater_is_better=greater_is_better,
            threshold=early_stopping_threshold,
        )
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_global_step = global_step
            stale_epochs = 0
            save_model_artifacts(model, processor, final_dir, peft_info)
        else:
            stale_epochs += 1

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            **eval_cer_metrics,
            "best_metric_name": metric_for_best_model,
            "best_metric": best_metric,
            "best_eval_loss": best_metric
            if metric_for_best_model == "eval_loss"
            else None,
            "best_eval_cer": best_metric if metric_for_best_model == "eval_cer" else None,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "learning_rate": scheduler.get_last_lr()[0],
            "saved_best": improved,
        }
        history.append(epoch_metrics)
        print(json.dumps({"stage": "epoch_end", **epoch_metrics}, ensure_ascii=False), flush=True)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    **eval_cer_metrics,
                    "eval_best_metric": best_metric,
                    "eval_best_loss": best_metric
                    if metric_for_best_model == "eval_loss"
                    else None,
                    "eval_best_cer": best_metric
                    if metric_for_best_model == "eval_cer"
                    else None,
                },
            )

        if (
            early_stopping_enabled
            and epoch >= early_stopping_min_epochs
            and stale_epochs >= early_stopping_patience
        ):
            print(
                json.dumps(
                    {
                        "stage": "early_stopping",
                        "epoch": epoch,
                        "metric": metric_for_best_model,
                        "best_metric": best_metric,
                        "stale_epochs": stale_epochs,
                        "patience": early_stopping_patience,
                        "min_epochs": early_stopping_min_epochs,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    if best_metric is None:
        save_model_artifacts(model, processor, final_dir, peft_info)

    summary = {
        "best_model_checkpoint": str(final_dir),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_global_step": best_global_step,
        "metric_for_best_model": metric_for_best_model,
        "final_dir": str(final_dir),
        "model_name_or_path": model_name_or_path,
        "language_filter": language_filter,
        "trainable_parameters": trainability["trainable_parameters"],
        "total_parameters": trainability["total_parameters"],
        "freeze_encoder": freeze_encoder,
        "train_decoder": train_decoder,
        "unfreeze_encoder_last_n_layers": trainability[
            "unfreeze_encoder_last_n_layers"
        ],
        "peft": peft_info,
        "history": history,
    }
    (output_dir / "best_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(
        "已保存 Whisper AdaLoRA BEST 權重: "
        f"{final_dir} best_epoch={best_epoch} best_metric={best_metric}",
        flush=True,
    )


if __name__ == "__main__":
    main()
