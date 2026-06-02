#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.runtime_env import configure_runtime_environment

configure_runtime_environment()

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from whisper_tw.config import load_config, resolve_common_voice_split_source
from whisper_tw.contrastive_router import (
    ContrastiveAdapterRouter,
    ContrastiveRouterSpec,
)
from whisper_tw.data import load_audio_waveform, read_common_voice_split
from whisper_tw.metrics import character_error_rate
from whisper_tw.text_norm import build_text_normalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="評估語言專屬 AdaLoRA 與路由模型。")
    parser.add_argument("--config", required=True, help="訓練設定檔路徑。")
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "router", "router_metrics"],
        help="single 評估單一語言 adapter；router_metrics 評估路由器指標；router 評估完整路由。",
    )
    parser.add_argument(
        "--language",
        help="single 模式要評估的語言，例如 zh-TW 或 nan-tw。",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="評估資料切分。",
    )
    parser.add_argument("--max-samples", type=int, help="只評估前 N 筆樣本。")
    parser.add_argument("--batch-size", type=int, help="覆蓋評估批次大小。")
    parser.add_argument("--device", help="覆蓋評估裝置，例如 cuda 或 cpu。")
    parser.add_argument("--output-dir", default="artifacts/eval", help="評估輸出資料夾。")
    parser.add_argument("--adapter-dir", help="single 模式覆蓋 adapter 權重資料夾。")
    parser.add_argument("--router-checkpoint", help="router 模式覆蓋路由權重路徑。")
    return parser.parse_args()


def get_train_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("whisper_train")
    if isinstance(section, dict):
        return section
    return config


def get_model_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(train_cfg.get("model") or {})


def get_data_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(train_cfg.get("data") or {})


def get_training_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(train_cfg.get("training") or {})


def get_router_config(train_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(train_cfg.get("contrastive_router") or {})


def sanitize_name(value: str) -> str:
    text = str(value or "").strip().replace("-", "_")
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    safe = safe.strip("._-")
    return safe or "model"


def sanitize_adapter_name(value: str) -> str:
    text = str(value or "").strip().replace("-", "_")
    safe = "".join(char.lower() if char.isalnum() else "_" for char in text)
    return "_".join(part for part in safe.split("_") if part) or "shared"


def resolve_device(train_cfg: dict[str, Any], requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    configured = get_training_config(train_cfg).get("device", "auto")
    if configured and configured != "auto":
        return torch.device(str(configured))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_split_source(data_cfg: dict[str, Any], split: str) -> str | Path:
    if split == str(data_cfg.get("train_split", "train")) and data_cfg.get("train_tsv"):
        return data_cfg["train_tsv"]
    if split == str(data_cfg.get("eval_split", "dev")) and data_cfg.get("eval_tsv"):
        return data_cfg["eval_tsv"]
    if split == str(data_cfg.get("test_split", "test")) and data_cfg.get("test_tsv"):
        return data_cfg["test_tsv"]
    return resolve_common_voice_split_source(data_cfg, split)


def resolve_model_dtype(device: torch.device, training_cfg: dict[str, Any]) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if bool(training_cfg.get("bf16", False)) and not bool(training_cfg.get("fp16", False)):
        return torch.bfloat16
    if bool(training_cfg.get("fp16", False)):
        return torch.float16
    return torch.float32


def build_autocast(device: torch.device, training_cfg: dict[str, Any]):
    if device.type != "cuda":
        return nullcontext()
    if bool(training_cfg.get("bf16", False)) and not bool(training_cfg.get("fp16", False)):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if bool(training_cfg.get("fp16", False)):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_torch_backend(training_cfg: dict[str, Any]) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(training_cfg.get("tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(training_cfg.get("tf32", False))


def configure_generation(model, model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> dict[str, Any]:
    language = str(model_cfg.get("language") or "zh")
    task = str(model_cfg.get("task") or "transcribe")
    for candidate in (model, getattr(model, "base_model", None), get_base_model(model)):
        if candidate is None:
            continue
        config = getattr(candidate, "config", None)
        if config is not None:
            config.forced_decoder_ids = None
            config.suppress_tokens = []
        generation_config = getattr(candidate, "generation_config", None)
        if generation_config is not None:
            generation_config.language = language
            generation_config.task = task
            generation_config.forced_decoder_ids = None
            generation_config.suppress_tokens = []
    return {
        "max_length": int(training_cfg.get("generation_max_length", 96)),
        "num_beams": int(training_cfg.get("generation_num_beams", 1)),
    }


def language_adapter_map(train_cfg: dict[str, Any]) -> dict[str, str]:
    peft_cfg = dict(train_cfg.get("peft") or {})
    adapters = dict(peft_cfg.get("language_adapters") or {})
    return {str(language): sanitize_adapter_name(name) for language, name in adapters.items()}


def training_profile_name(output_dir: Path) -> str:
    name = output_dir.name.lower()
    if "h100" in name:
        return "h100"
    if "8gb" in name:
        return "8gb"
    return "local"


def resolve_language_output_dir(train_cfg: dict[str, Any], language: str) -> Path:
    adapters = language_adapter_map(train_cfg)
    if language not in adapters:
        available = ", ".join(adapters) or "未設定"
        raise ValueError(f"找不到 {language!r} 對應的 adapter；可用語言: {available}。")
    training_cfg = get_training_config(train_cfg)
    base_output_dir = Path(
        str(training_cfg.get("output_dir") or "artifacts/models/whisper_medium_adalora")
    )
    profile = training_profile_name(base_output_dir)
    suffix = "" if profile == "local" else f"_{profile}"
    return base_output_dir.parent / f"whisper_medium_adalora_{adapters[language]}{suffix}"


def resolve_adapter_dir(
    train_cfg: dict[str, Any],
    language: str,
    explicit_adapter_dir: str | None = None,
) -> Path:
    if explicit_adapter_dir:
        adapter_dir = Path(explicit_adapter_dir)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"找不到 adapter 權重資料夾: {adapter_dir}")
        return adapter_dir

    adapter_name = language_adapter_map(train_cfg).get(language)
    if not adapter_name:
        raise ValueError(f"設定檔沒有 {language!r} 的 adapter 名稱。")
    final_dir = resolve_language_output_dir(train_cfg, language) / "final"
    candidates = [
        final_dir / "adapters" / adapter_name,
        final_dir / adapter_name,
    ]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate
    raise FileNotFoundError(
        f"找不到 {language} adapter 權重；已檢查: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def resolve_router_checkpoint(train_cfg: dict[str, Any], explicit_path: str | None = None) -> Path:
    if explicit_path:
        router_path = Path(explicit_path)
    else:
        router_cfg = get_router_config(train_cfg)
        router_path = Path(
            str(router_cfg.get("output_dir") or "artifacts/models/contrastive_router")
        ) / "contrastive_router.pt"
    if not router_path.exists():
        raise FileNotFoundError(f"找不到對比式路由權重: {router_path}")
    return router_path


class SpeechEvalDataset(Dataset):
    def __init__(
        self,
        *,
        data_cfg: dict[str, Any],
        split: str,
        max_samples: int | None,
        language_filter: str | None = None,
        allowed_languages: set[str] | None = None,
    ) -> None:
        split_source = resolve_split_source(data_cfg, split)
        samples = read_common_voice_split(
            data_cfg.get("root", "data"),
            split_source,
            language_filter=language_filter,
        )
        if allowed_languages is not None:
            samples = [sample for sample in samples if sample.language_label in allowed_languages]
        if max_samples is not None:
            samples = samples[: max(0, int(max_samples))]
        self.samples = samples
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
        reference = sample.text
        if self.normalizer.enabled:
            reference = self.normalizer(reference)
        duration = float(waveform.numel()) / float(self.sample_rate)
        return {
            "audio": waveform,
            "reference": reference,
            "audio_path": str(sample.audio_path),
            "rel_path": sample.rel_path,
            "language_label": sample.language_label,
            "duration_seconds": duration,
        }


class SpeechEvalCollator:
    def __init__(self, processor, sample_rate: int) -> None:
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self.processor.feature_extractor(
            [feature["audio"].numpy() for feature in features],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        batch = {
            "input_features": inputs["input_features"],
            "references": [feature["reference"] for feature in features],
            "audio_paths": [feature["audio_path"] for feature in features],
            "rel_paths": [feature["rel_path"] for feature in features],
            "language_labels": [feature["language_label"] for feature in features],
            "duration_seconds": [float(feature["duration_seconds"]) for feature in features],
        }
        if inputs.get("attention_mask") is not None:
            batch["attention_mask"] = inputs["attention_mask"]
        return batch


def build_dataloader(
    *,
    dataset: SpeechEvalDataset,
    processor,
    data_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    num_workers = max(0, int(training_cfg.get("dataloader_num_workers", 0)))
    kwargs: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": False,
        "collate_fn": SpeechEvalCollator(processor, int(data_cfg.get("sample_rate", 16000))),
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("dataloader_pin_memory", False))
        and device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(
            training_cfg.get("dataloader_persistent_workers", False)
        )
        if training_cfg.get("dataloader_prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(training_cfg["dataloader_prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def default_eval_batch_size(training_cfg: dict[str, Any], cli_batch_size: int | None = None) -> int:
    if cli_batch_size is not None:
        return max(1, int(cli_batch_size))
    return max(1, int(training_cfg.get("per_device_eval_batch_size", 1)))


def get_base_model(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def get_whisper_encoder(model):
    base = get_base_model(model)
    if hasattr(base, "model") and hasattr(base.model, "encoder"):
        return base.model.encoder
    if hasattr(base, "encoder"):
        return base.encoder
    raise ValueError("無法從模型取得 Whisper encoder。")


def disabled_adapter_context(model):
    method = getattr(model, "disable_adapter", None)
    if callable(method):
        return method()
    return nullcontext()


def find_processor_source(adapter_dir: Path) -> Path | None:
    candidates = [
        adapter_dir,
        adapter_dir.parent,
        adapter_dir.parent.parent,
        adapter_dir.parent.parent.parent,
    ]
    for candidate in candidates:
        if (candidate / "processor_config.json").exists() and (candidate / "tokenizer.json").exists():
            return candidate
    return None


def load_processor(model_cfg: dict[str, Any], local_source: Path | None = None):
    from transformers import AutoProcessor, AutoTokenizer, WhisperFeatureExtractor, WhisperProcessor

    model_name_or_path = str(model_cfg.get("model_name_or_path") or "openai/whisper-medium")
    if local_source is not None:
        processor_config = local_source / "processor_config.json"
        try:
            return AutoProcessor.from_pretrained(str(local_source), local_files_only=True)
        except Exception:
            if processor_config.exists():
                payload = json.loads(processor_config.read_text(encoding="utf-8"))
                feature_extractor = WhisperFeatureExtractor(
                    **dict(payload.get("feature_extractor") or {})
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    str(local_source),
                    local_files_only=True,
                )
                return WhisperProcessor(
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                )
            raise
    try:
        return AutoProcessor.from_pretrained(
            model_name_or_path,
            language=str(model_cfg.get("language") or "zh"),
            task=str(model_cfg.get("task") or "transcribe"),
            local_files_only=True,
        )
    except Exception:
        pass
    return AutoProcessor.from_pretrained(
        model_name_or_path,
        language=str(model_cfg.get("language") or "zh"),
        task=str(model_cfg.get("task") or "transcribe"),
    )


def load_base_model(model_cfg: dict[str, Any], device: torch.device, training_cfg: dict[str, Any]):
    from transformers import AutoModelForSpeechSeq2Seq

    model_name_or_path = str(model_cfg.get("model_name_or_path") or "openai/whisper-medium")
    kwargs = {"dtype": resolve_model_dtype(device, training_cfg)}
    try:
        return AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path,
            local_files_only=True,
            **kwargs,
        )
    except Exception:
        return AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, **kwargs)


def load_adapted_model(
    *,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    device: torch.device,
    adapter_dirs: dict[str, Path],
) -> Any:
    from peft import PeftModel

    if not adapter_dirs:
        raise ValueError("至少需要一個 adapter 才能載入語言專屬模型。")

    base_model = load_base_model(model_cfg, device, training_cfg)
    first_adapter_name, first_adapter_dir = next(iter(adapter_dirs.items()))
    model = PeftModel.from_pretrained(
        base_model,
        str(first_adapter_dir),
        adapter_name=first_adapter_name,
        is_trainable=False,
    )
    for adapter_name, adapter_dir in list(adapter_dirs.items())[1:]:
        model.load_adapter(
            str(adapter_dir),
            adapter_name=adapter_name,
            is_trainable=False,
        )
    model.to(device)
    model.eval()
    configure_generation(model, model_cfg, training_cfg)
    return model


def normalize_predictions(texts: list[str], data_cfg: dict[str, Any]) -> list[str]:
    normalizer = build_text_normalizer(data_cfg.get("text_normalization"))
    if not normalizer.enabled:
        return [str(text or "").strip() for text in texts]
    return [normalizer(str(text or "")) for text in texts]


def decode_generated(processor, generated_ids: torch.Tensor, data_cfg: dict[str, Any]) -> list[str]:
    texts = processor.batch_decode(
        generated_ids.detach().cpu(),
        skip_special_tokens=True,
        decode_with_timestamps=False,
        clean_up_tokenization_spaces=False,
    )
    return normalize_predictions(texts, data_cfg)


def generate_for_language_groups(
    *,
    model,
    processor,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None,
    languages: list[str],
    language_to_adapter: dict[str, str],
    data_cfg: dict[str, Any],
    device: torch.device,
    training_cfg: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> tuple[list[str], float]:
    predictions: list[str | None] = [None for _ in languages]
    total_seconds = 0.0
    groups: dict[str, list[int]] = defaultdict(list)
    for index, language in enumerate(languages):
        if language not in language_to_adapter:
            raise ValueError(f"找不到 {language!r} 對應的 adapter。")
        groups[language].append(index)

    for language, indices in groups.items():
        adapter_name = language_to_adapter[language]
        model.set_adapter(adapter_name)
        index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        selected_features = input_features.index_select(0, index_tensor)
        selected_attention_mask = (
            attention_mask.index_select(0, index_tensor)
            if attention_mask is not None
            else None
        )
        synchronize_for_timing(device)
        start = time.perf_counter()
        with build_autocast(device, training_cfg):
            generate_inputs = {"input_features": selected_features, **generation_kwargs}
            if selected_attention_mask is not None:
                generate_inputs["attention_mask"] = selected_attention_mask
            generated_ids = model.generate(
                **generate_inputs,
            )
        synchronize_for_timing(device)
        total_seconds += time.perf_counter() - start
        decoded = decode_generated(processor, generated_ids, data_cfg)
        for original_index, prediction in zip(indices, decoded):
            predictions[original_index] = prediction

    return [str(prediction or "") for prediction in predictions], total_seconds


def compute_classification_metrics(
    predictions: list[int],
    references: list[int],
    *,
    labels: list[str],
) -> dict[str, Any]:
    num_labels = len(labels)
    total = len(references)
    correct = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for pred, ref in zip(predictions, references):
        confusion[ref][pred] += 1

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []
    for label_id in range(num_labels):
        tp = confusion[label_id][label_id]
        fp = sum(confusion[row][label_id] for row in range(num_labels) if row != label_id)
        fn = sum(confusion[label_id][col] for col in range(num_labels) if col != label_id)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "router_accuracy": correct / max(1, total),
        "router_macro_precision": sum(precision_scores) / max(1, len(precision_scores)),
        "router_macro_recall": sum(recall_scores) / max(1, len(recall_scores)),
        "router_macro_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "router_per_label_precision": precision_scores,
        "router_per_label_recall": recall_scores,
        "router_per_label_f1": f1_scores,
        "confusion_matrix": confusion,
    }


def build_asr_payload(
    *,
    mode: str,
    split: str,
    references: list[str],
    predictions: list[str],
    records: list[dict[str, Any]],
    inference_seconds: float,
    elapsed_seconds: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_audio_seconds = sum(float(record["duration_seconds"]) for record in records)
    payload = {
        "mode": mode,
        "split": split,
        "samples": len(references),
        "cer": character_error_rate(predictions, references),
        "inference_seconds": inference_seconds,
        "inference_seconds_per_sample": inference_seconds / max(1, len(references)),
        "total_audio_seconds": total_audio_seconds,
        "realtime_factor": (
            inference_seconds / total_audio_seconds
            if total_audio_seconds > 0.0
            else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / max(1, len(references)),
        "records": [
            {
                **record,
                "prediction": prediction,
                "char_error_rate": character_error_rate([prediction], [reference]),
            }
            for record, prediction, reference in zip(records, predictions, references)
        ],
    }
    if extra:
        payload.update(extra)
    return payload


def write_eval_json(output_dir: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / filename
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def evaluate_single_adapter(
    *,
    config: dict[str, Any],
    language: str,
    split: str,
    max_samples: int | None,
    batch_size: int | None,
    device_name: str | None,
    adapter_dir: str | None = None,
) -> dict[str, Any]:
    train_cfg = get_train_config(config)
    model_cfg = get_model_config(train_cfg)
    data_cfg = get_data_config(train_cfg)
    training_cfg = get_training_config(train_cfg)
    device = resolve_device(train_cfg, device_name)
    configure_torch_backend(training_cfg)

    language_to_adapter = language_adapter_map(train_cfg)
    if language not in language_to_adapter:
        available = ", ".join(language_to_adapter) or "未設定"
        raise ValueError(f"--language={language!r} 不在設定檔中；可用語言: {available}。")
    resolved_adapter_dir = resolve_adapter_dir(train_cfg, language, adapter_dir)
    adapter_name = language_to_adapter[language]
    processor = load_processor(model_cfg, find_processor_source(resolved_adapter_dir))
    model = load_adapted_model(
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        device=device,
        adapter_dirs={adapter_name: resolved_adapter_dir},
    )
    generation_kwargs = configure_generation(model, model_cfg, training_cfg)
    dataset = SpeechEvalDataset(
        data_cfg=data_cfg,
        split=split,
        max_samples=max_samples,
        language_filter=language,
    )
    if len(dataset) == 0:
        raise ValueError(f"{split} 切分中找不到 language_label={language} 的樣本。")
    dataloader = build_dataloader(
        dataset=dataset,
        processor=processor,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        batch_size=default_eval_batch_size(training_cfg, batch_size),
        device=device,
    )

    print(
        f"eval_start mode=single language={language} split={split} "
        f"samples={len(dataset)} adapter={resolved_adapter_dir} device={device}",
        flush=True,
    )
    references: list[str] = []
    predictions: list[str] = []
    records: list[dict[str, Any]] = []
    inference_seconds = 0.0
    start = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(dataloader, desc=f"eval {language} adapter", dynamic_ncols=True)
        for batch in progress:
            input_features = batch["input_features"].to(device)
            batch_predictions, batch_seconds = generate_for_language_groups(
                model=model,
                processor=processor,
                input_features=input_features,
                attention_mask=batch.get("attention_mask").to(device)
                if batch.get("attention_mask") is not None
                else None,
                languages=[language for _ in batch["references"]],
                language_to_adapter=language_to_adapter,
                data_cfg=data_cfg,
                device=device,
                training_cfg=training_cfg,
                generation_kwargs=generation_kwargs,
            )
            inference_seconds += batch_seconds
            references.extend(batch["references"])
            predictions.extend(batch_predictions)
            for index, reference in enumerate(batch["references"]):
                records.append(
                    {
                        "audio_path": batch["audio_paths"][index],
                        "rel_path": batch["rel_paths"][index],
                        "language_label": batch["language_labels"][index],
                        "reference": reference,
                        "duration_seconds": batch["duration_seconds"][index],
                    }
                )
            progress.set_postfix(
                samples=len(references),
                cer=f"{character_error_rate(predictions, references):.4f}",
            )

    return build_asr_payload(
        mode="single",
        split=split,
        references=references,
        predictions=predictions,
        records=records,
        inference_seconds=inference_seconds,
        elapsed_seconds=time.perf_counter() - start,
        extra={
            "language": language,
            "adapter_name": adapter_name,
            "adapter_dir": str(resolved_adapter_dir),
        },
    )


def load_router(router_path: Path, device: torch.device) -> ContrastiveAdapterRouter:
    payload = torch.load(router_path, map_location=device, weights_only=False)
    labels = tuple(str(label) for label in payload["labels"])
    spec = ContrastiveRouterSpec(
        labels=labels,
        hidden_size=int(payload["hidden_size"]),
        pooling=str(payload.get("pooling") or "attention"),
        attention_hidden_size=int(payload.get("attention_hidden_size", 256)),
        embedding_size=int(payload.get("embedding_size", 256)),
        hidden_ratio=float(payload.get("hidden_ratio", 0.5)),
        dropout=float(payload.get("dropout", 0.1)),
        temperature=float(payload.get("temperature", 0.07)),
    )
    router = ContrastiveAdapterRouter(spec).to(device)
    router.load_state_dict(payload["state_dict"])
    router.eval()
    return router


def route_batch(
    *,
    model,
    router: ContrastiveAdapterRouter,
    input_features: torch.Tensor,
    device: torch.device,
    training_cfg: dict[str, Any],
) -> tuple[list[int], float]:
    encoder = get_whisper_encoder(model)
    synchronize_for_timing(device)
    start = time.perf_counter()
    with torch.no_grad():
        with disabled_adapter_context(model):
            with build_autocast(device, training_cfg):
                hidden = encoder(input_features).last_hidden_state
        logits = router(hidden.float())["logits"]
    synchronize_for_timing(device)
    return logits.argmax(dim=-1).detach().cpu().tolist(), time.perf_counter() - start


def route_batch_with_scores(
    *,
    model,
    router: ContrastiveAdapterRouter,
    input_features: torch.Tensor,
    reference_ids: torch.Tensor,
    device: torch.device,
    training_cfg: dict[str, Any],
) -> tuple[list[int], float, list[float], list[float], float]:
    encoder = get_whisper_encoder(model)
    synchronize_for_timing(device)
    start = time.perf_counter()
    with torch.no_grad():
        with build_autocast(device, training_cfg):
            hidden = encoder(input_features).last_hidden_state
        loss, outputs = router.compute_loss(hidden.float(), reference_ids)
        logits = outputs["logits"]
        similarities = outputs["queries"] @ outputs["keys"].transpose(0, 1)
    synchronize_for_timing(device)

    row_ids = torch.arange(reference_ids.size(0), device=device)
    positive = similarities[row_ids, reference_ids]
    mask = torch.ones_like(similarities, dtype=torch.bool)
    mask[row_ids, reference_ids] = False
    negative = similarities.masked_fill(~mask, float("-inf")).max(dim=-1).values
    return (
        logits.argmax(dim=-1).detach().cpu().tolist(),
        time.perf_counter() - start,
        positive.detach().cpu().tolist(),
        negative.detach().cpu().tolist(),
        float(loss.detach().cpu()),
    )


def wrong_languages_for(labels: list[str], references: list[str]) -> list[str]:
    wrong: list[str] = []
    for reference in references:
        candidates = [label for label in labels if label != reference]
        if not candidates:
            raise ValueError("wrong adapter 評估至少需要兩個語言標籤。")
        wrong.append(candidates[0])
    return wrong


def evaluate_router(
    *,
    config: dict[str, Any],
    split: str,
    max_samples: int | None,
    batch_size: int | None,
    device_name: str | None,
    router_checkpoint: str | None = None,
) -> dict[str, Any]:
    train_cfg = get_train_config(config)
    model_cfg = get_model_config(train_cfg)
    data_cfg = get_data_config(train_cfg)
    training_cfg = get_training_config(train_cfg)
    device = resolve_device(train_cfg, device_name)
    configure_torch_backend(training_cfg)

    router_path = resolve_router_checkpoint(train_cfg, router_checkpoint)
    router = load_router(router_path, device)
    labels = list(router.spec.labels)
    language_to_adapter = language_adapter_map(train_cfg)
    missing_labels = [label for label in labels if label not in language_to_adapter]
    if missing_labels:
        raise ValueError(f"路由標籤缺少 adapter 設定: {missing_labels}")
    adapter_dirs = {
        language_to_adapter[label]: resolve_adapter_dir(train_cfg, label)
        for label in labels
    }

    first_adapter_dir = next(iter(adapter_dirs.values()))
    processor = load_processor(model_cfg, find_processor_source(first_adapter_dir))
    model = load_adapted_model(
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        device=device,
        adapter_dirs=adapter_dirs,
    )
    generation_kwargs = configure_generation(model, model_cfg, training_cfg)
    dataset = SpeechEvalDataset(
        data_cfg=data_cfg,
        split=split,
        max_samples=max_samples,
        allowed_languages=set(labels),
    )
    if len(dataset) == 0:
        raise ValueError(f"{split} 切分中沒有符合路由標籤 {labels} 的樣本。")
    dataloader = build_dataloader(
        dataset=dataset,
        processor=processor,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        batch_size=default_eval_batch_size(training_cfg, batch_size),
        device=device,
    )

    print(
        f"eval_start mode=router split={split} samples={len(dataset)} "
        f"router={router_path} labels={labels} device={device}",
        flush=True,
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    references: list[str] = []
    reference_label_ids: list[int] = []
    predicted_label_ids: list[int] = []
    predicted_labels: list[str] = []
    selected_predictions: list[str] = []
    oracle_predictions: list[str] = []
    wrong_predictions: list[str] = []
    records: list[dict[str, Any]] = []
    router_seconds = 0.0
    selected_seconds = 0.0
    oracle_seconds = 0.0
    wrong_seconds = 0.0
    start = time.perf_counter()

    with torch.inference_mode():
        progress = tqdm(dataloader, desc="eval router", dynamic_ncols=True)
        for batch in progress:
            input_features = batch["input_features"].to(device)
            batch_route_ids, batch_router_seconds = route_batch(
                model=model,
                router=router,
                input_features=input_features,
                device=device,
                training_cfg=training_cfg,
            )
            router_seconds += batch_router_seconds
            batch_predicted_labels = [labels[index] for index in batch_route_ids]
            batch_reference_labels = [str(label) for label in batch["language_labels"]]
            batch_wrong_labels = wrong_languages_for(labels, batch_reference_labels)

            batch_selected_predictions, batch_selected_seconds = generate_for_language_groups(
                model=model,
                processor=processor,
                input_features=input_features,
                attention_mask=batch.get("attention_mask").to(device)
                if batch.get("attention_mask") is not None
                else None,
                languages=batch_predicted_labels,
                language_to_adapter=language_to_adapter,
                data_cfg=data_cfg,
                device=device,
                training_cfg=training_cfg,
                generation_kwargs=generation_kwargs,
            )
            batch_oracle_predictions, batch_oracle_seconds = generate_for_language_groups(
                model=model,
                processor=processor,
                input_features=input_features,
                attention_mask=batch.get("attention_mask").to(device)
                if batch.get("attention_mask") is not None
                else None,
                languages=batch_reference_labels,
                language_to_adapter=language_to_adapter,
                data_cfg=data_cfg,
                device=device,
                training_cfg=training_cfg,
                generation_kwargs=generation_kwargs,
            )
            batch_wrong_predictions, batch_wrong_seconds = generate_for_language_groups(
                model=model,
                processor=processor,
                input_features=input_features,
                attention_mask=batch.get("attention_mask").to(device)
                if batch.get("attention_mask") is not None
                else None,
                languages=batch_wrong_labels,
                language_to_adapter=language_to_adapter,
                data_cfg=data_cfg,
                device=device,
                training_cfg=training_cfg,
                generation_kwargs=generation_kwargs,
            )
            selected_seconds += batch_selected_seconds
            oracle_seconds += batch_oracle_seconds
            wrong_seconds += batch_wrong_seconds

            references.extend(batch["references"])
            selected_predictions.extend(batch_selected_predictions)
            oracle_predictions.extend(batch_oracle_predictions)
            wrong_predictions.extend(batch_wrong_predictions)
            predicted_label_ids.extend(batch_route_ids)
            predicted_labels.extend(batch_predicted_labels)
            reference_label_ids.extend(label_to_id[label] for label in batch_reference_labels)
            for index, reference in enumerate(batch["references"]):
                records.append(
                    {
                        "audio_path": batch["audio_paths"][index],
                        "rel_path": batch["rel_paths"][index],
                        "language_label": batch_reference_labels[index],
                        "router_prediction": batch_predicted_labels[index],
                        "wrong_adapter_language": batch_wrong_labels[index],
                        "reference": reference,
                        "duration_seconds": batch["duration_seconds"][index],
                    }
                )
            progress.set_postfix(
                samples=len(references),
                router_acc=(
                    f"{sum(int(p == r) for p, r in zip(predicted_label_ids, reference_label_ids)) / max(1, len(reference_label_ids)):.4f}"
                ),
                cer=f"{character_error_rate(selected_predictions, references):.4f}",
            )

    router_metrics = compute_classification_metrics(
        predicted_label_ids,
        reference_label_ids,
        labels=labels,
    )
    total_audio_seconds = sum(float(record["duration_seconds"]) for record in records)
    elapsed_seconds = time.perf_counter() - start
    payload_records: list[dict[str, Any]] = []
    for record, selected, oracle, wrong, reference in zip(
        records,
        selected_predictions,
        oracle_predictions,
        wrong_predictions,
        references,
    ):
        payload_records.append(
            {
                **record,
                "prediction": selected,
                "router_selected_prediction": selected,
                "oracle_adapter_prediction": oracle,
                "wrong_adapter_prediction": wrong,
                "char_error_rate": character_error_rate([selected], [reference]),
                "oracle_char_error_rate": character_error_rate([oracle], [reference]),
                "wrong_char_error_rate": character_error_rate([wrong], [reference]),
            }
        )

    selected_total_seconds = router_seconds + selected_seconds
    return {
        "mode": "router",
        "split": split,
        "samples": len(references),
        "labels": labels,
        "router_checkpoint": str(router_path),
        **router_metrics,
        "cer": character_error_rate(selected_predictions, references),
        "cer_router_selected": character_error_rate(selected_predictions, references),
        "cer_oracle_adapter": character_error_rate(oracle_predictions, references),
        "cer_wrong_adapter": character_error_rate(wrong_predictions, references),
        "router_inference_seconds": router_seconds,
        "router_selected_generation_seconds": selected_seconds,
        "oracle_generation_seconds": oracle_seconds,
        "wrong_generation_seconds": wrong_seconds,
        "inference_seconds": selected_total_seconds,
        "inference_seconds_per_sample": selected_total_seconds / max(1, len(references)),
        "total_audio_seconds": total_audio_seconds,
        "realtime_factor": (
            selected_total_seconds / total_audio_seconds
            if total_audio_seconds > 0.0
            else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / max(1, len(references)),
        "records": payload_records,
    }


def evaluate_router_metrics(
    *,
    config: dict[str, Any],
    split: str,
    max_samples: int | None,
    batch_size: int | None,
    device_name: str | None,
    router_checkpoint: str | None = None,
) -> dict[str, Any]:
    train_cfg = get_train_config(config)
    model_cfg = get_model_config(train_cfg)
    data_cfg = get_data_config(train_cfg)
    training_cfg = get_training_config(train_cfg)
    device = resolve_device(train_cfg, device_name)
    configure_torch_backend(training_cfg)

    router_path = resolve_router_checkpoint(train_cfg, router_checkpoint)
    router = load_router(router_path, device)
    labels = list(router.spec.labels)
    label_to_id = {label: index for index, label in enumerate(labels)}

    processor = load_processor(model_cfg)
    model = load_base_model(model_cfg, device, training_cfg).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = SpeechEvalDataset(
        data_cfg=data_cfg,
        split=split,
        max_samples=max_samples,
        allowed_languages=set(labels),
    )
    if len(dataset) == 0:
        raise ValueError(f"{split} 切分中沒有符合路由標籤 {labels} 的樣本。")
    dataloader = build_dataloader(
        dataset=dataset,
        processor=processor,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        batch_size=default_eval_batch_size(training_cfg, batch_size),
        device=device,
    )

    print(
        f"eval_start mode=router_metrics split={split} samples={len(dataset)} "
        f"router={router_path} labels={labels} device={device}",
        flush=True,
    )
    reference_ids: list[int] = []
    prediction_ids: list[int] = []
    positive_similarities: list[float] = []
    negative_similarities: list[float] = []
    losses: list[float] = []
    records: list[dict[str, Any]] = []
    inference_seconds = 0.0
    start = time.perf_counter()

    with torch.inference_mode():
        progress = tqdm(dataloader, desc="eval router metrics", dynamic_ncols=True)
        for batch in progress:
            input_features = batch["input_features"].to(device)
            batch_reference_labels = [str(label) for label in batch["language_labels"]]
            batch_reference_ids = torch.tensor(
                [label_to_id[label] for label in batch_reference_labels],
                dtype=torch.long,
                device=device,
            )
            (
                batch_prediction_ids,
                batch_seconds,
                batch_positive,
                batch_negative,
                batch_loss,
            ) = route_batch_with_scores(
                model=model,
                router=router,
                input_features=input_features,
                reference_ids=batch_reference_ids,
                device=device,
                training_cfg=training_cfg,
            )
            inference_seconds += batch_seconds
            losses.append(batch_loss)
            positive_similarities.extend(batch_positive)
            negative_similarities.extend(batch_negative)
            prediction_ids.extend(batch_prediction_ids)
            reference_ids.extend(batch_reference_ids.detach().cpu().tolist())

            for index, reference_label in enumerate(batch_reference_labels):
                predicted_label = labels[batch_prediction_ids[index]]
                records.append(
                    {
                        "audio_path": batch["audio_paths"][index],
                        "rel_path": batch["rel_paths"][index],
                        "language_label": reference_label,
                        "router_prediction": predicted_label,
                        "duration_seconds": batch["duration_seconds"][index],
                    }
                )
            progress.set_postfix(
                samples=len(reference_ids),
                acc=(
                    f"{sum(int(p == r) for p, r in zip(prediction_ids, reference_ids)) / max(1, len(reference_ids)):.4f}"
                ),
            )

    router_metrics = compute_classification_metrics(
        prediction_ids,
        reference_ids,
        labels=labels,
    )
    total_audio_seconds = sum(float(record["duration_seconds"]) for record in records)
    avg_positive = sum(positive_similarities) / max(1, len(positive_similarities))
    avg_negative = sum(negative_similarities) / max(1, len(negative_similarities))
    elapsed_seconds = time.perf_counter() - start
    return {
        "mode": "router_metrics",
        "split": split,
        "samples": len(reference_ids),
        "labels": labels,
        "router_checkpoint": str(router_path),
        **router_metrics,
        "router_loss": sum(losses) / max(1, len(losses)),
        "avg_positive_similarity": avg_positive,
        "avg_max_negative_similarity": avg_negative,
        "avg_similarity_gap": avg_positive - avg_negative,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_sample": inference_seconds / max(1, len(reference_ids)),
        "total_audio_seconds": total_audio_seconds,
        "realtime_factor": (
            inference_seconds / total_audio_seconds
            if total_audio_seconds > 0.0
            else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / max(1, len(reference_ids)),
        "records": records,
    }


def evaluate_hf_whisper(
    *,
    config: dict[str, Any],
    model_name_or_path: str,
    split: str,
    max_samples: int | None,
    batch_size: int | None,
    device_name: str | None,
    language_filter: str | None = None,
    decode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_cfg = get_train_config(config)
    model_cfg = get_model_config(train_cfg)
    data_cfg = get_data_config(train_cfg)
    training_cfg = get_training_config(train_cfg)
    device = resolve_device(train_cfg, device_name)
    configure_torch_backend(training_cfg)

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        dtype=resolve_model_dtype(device, training_cfg),
    )
    model.to(device)
    model.eval()
    generation_kwargs = configure_generation(model, model_cfg, training_cfg)
    generation_kwargs.update(dict(decode or {}))
    generation_kwargs.setdefault("task", str(model_cfg.get("task") or "transcribe"))
    generation_kwargs.setdefault("language", str(model_cfg.get("language") or "zh"))

    dataset = SpeechEvalDataset(
        data_cfg=data_cfg,
        split=split,
        max_samples=max_samples,
        language_filter=language_filter,
    )
    if len(dataset) == 0:
        raise ValueError(f"{split} 切分沒有符合 language_filter={language_filter!r} 的樣本。")
    dataloader = build_dataloader(
        dataset=dataset,
        processor=processor,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        batch_size=default_eval_batch_size(training_cfg, batch_size),
        device=device,
    )

    references: list[str] = []
    predictions: list[str] = []
    records: list[dict[str, Any]] = []
    inference_seconds = 0.0
    start = time.perf_counter()
    print(
        f"eval_start mode=hf_whisper model={model_name_or_path} split={split} "
        f"samples={len(dataset)} device={device}",
        flush=True,
    )
    with torch.inference_mode():
        progress = tqdm(dataloader, desc=f"eval {model_name_or_path}", dynamic_ncols=True)
        for batch in progress:
            input_features = batch["input_features"].to(
                device=device,
                dtype=getattr(model, "dtype", batch["input_features"].dtype),
            )
            attention_mask = (
                batch["attention_mask"].to(device)
                if batch.get("attention_mask") is not None
                else None
            )
            synchronize_for_timing(device)
            batch_start = time.perf_counter()
            with build_autocast(device, training_cfg):
                generate_inputs = {"input_features": input_features, **generation_kwargs}
                if attention_mask is not None:
                    generate_inputs["attention_mask"] = attention_mask
                generated_ids = model.generate(
                    **generate_inputs,
                )
            synchronize_for_timing(device)
            inference_seconds += time.perf_counter() - batch_start
            batch_predictions = decode_generated(processor, generated_ids, data_cfg)
            references.extend(batch["references"])
            predictions.extend(batch_predictions)
            for index, reference in enumerate(batch["references"]):
                records.append(
                    {
                        "audio_path": batch["audio_paths"][index],
                        "rel_path": batch["rel_paths"][index],
                        "language_label": batch["language_labels"][index],
                        "reference": reference,
                        "duration_seconds": batch["duration_seconds"][index],
                    }
                )
            progress.set_postfix(
                samples=len(references),
                cer=f"{character_error_rate(predictions, references):.4f}",
            )

    return build_asr_payload(
        mode="hf_whisper",
        split=split,
        references=references,
        predictions=predictions,
        records=records,
        inference_seconds=inference_seconds,
        elapsed_seconds=time.perf_counter() - start,
        extra={
            "model_name_or_path": model_name_or_path,
            "language_filter": language_filter,
        },
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "single":
        if not args.language:
            raise ValueError("single 模式必須指定 --language，例如 --language zh-TW。")
        payload = evaluate_single_adapter(
            config=config,
            language=str(args.language),
            split=str(args.split),
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device_name=args.device,
            adapter_dir=args.adapter_dir,
        )
        filename = f"eval_adalora_{sanitize_name(str(args.language)).lower()}_{args.split}.json"
    elif args.mode == "router":
        payload = evaluate_router(
            config=config,
            split=str(args.split),
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device_name=args.device,
            router_checkpoint=args.router_checkpoint,
        )
        filename = f"eval_router_full_{args.split}.json"
    else:
        payload = evaluate_router_metrics(
            config=config,
            split=str(args.split),
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device_name=args.device,
            router_checkpoint=args.router_checkpoint,
        )
        filename = f"eval_router_metrics_{args.split}.json"

    output_path = write_eval_json(args.output_dir, filename, payload)
    print(f"eval_json={output_path}", flush=True)
    print(
        json.dumps(
            {
                "mode": payload["mode"],
                "split": payload["split"],
                "samples": payload["samples"],
                "cer": payload.get("cer"),
                "router_accuracy": payload.get("router_accuracy"),
                "router_macro_f1": payload.get("router_macro_f1"),
                "realtime_factor": payload.get("realtime_factor"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
