#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.char_vocab import CharacterVocab
from whisper_tw.config import (
    load_config,
    resolve_common_voice_split_source,
    resolve_device,
)
from whisper_tw.data import load_audio_waveform, read_common_voice_split
from whisper_tw.metrics import character_error_rate
from whisper_tw.text_norm import build_text_normalizer
from whisper_tw.training import (
    build_components,
    build_dataloader_kwargs,
    configure_training_runtime,
    get_amp_dtype,
    use_mixed_precision,
)


def decode_ctc(ids: list[int], character_vocab: CharacterVocab) -> str:
    collapsed: list[int] = []
    previous_id: int | None = None
    for token_id in ids:
        if token_id != previous_id and token_id != character_vocab.blank_id:
            collapsed.append(token_id)
        previous_id = token_id
    return character_vocab.decode(collapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="評估 Whisper-TW 並與基線語音辨識模型比較。"
    )
    parser.add_argument("--config", required=True, help="專案設定檔路徑。")
    parser.add_argument(
        "--baselines-config",
        default="configs/baselines.yaml",
        help="基線模型設定檔路徑。",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="評估資料切分。",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Whisper-TW checkpoint 路徑。",
    )
    parser.add_argument(
        "--current-name",
        default="whisper_tw",
        help="目前專案模型的輸出名稱。",
    )
    parser.add_argument("--max-samples", type=int, help="只評估前 N 筆樣本。")
    parser.add_argument("--batch-size", type=int, help="覆蓋評估批次大小。")
    parser.add_argument("--device", help="覆蓋所有模型的裝置。")
    parser.add_argument("--output-dir", help="覆蓋評估輸出目錄。")
    return parser.parse_args()


def load_structured_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() == ".json":
            return json.load(f) or {}

        import yaml

        return yaml.safe_load(f) or {}


def sanitize_name(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    safe = safe.strip("._-")
    return safe or "model"


def synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_eval_device(
    config: dict[str, Any],
    requested: str | None,
) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device(resolve_device(config))


def default_batch_size(
    config: dict[str, Any],
    cli_batch_size: int | None,
    baseline_cfg: dict[str, Any] | None = None,
) -> int:
    train_cfg = config.get("training", {})
    if cli_batch_size is not None:
        return max(1, int(cli_batch_size))
    if baseline_cfg and baseline_cfg.get("batch_size") is not None:
        return max(1, int(baseline_cfg["batch_size"]))
    return max(1, int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])))


def audio_duration_seconds(
    path: str | Path,
    *,
    fallback_sample_rate: int,
    max_audio_seconds: float,
) -> float:
    try:
        info = torchaudio.info(str(path))
        if info.sample_rate > 0:
            duration = float(info.num_frames) / float(info.sample_rate)
            return min(duration, max_audio_seconds)
    except Exception:
        pass
    waveform = load_audio_waveform(path, fallback_sample_rate)
    duration = float(waveform.numel()) / float(fallback_sample_rate)
    return min(duration, max_audio_seconds)


def total_audio_seconds_for_split(
    config: dict[str, Any],
    split: str,
    max_samples: int | None,
) -> float:
    data_cfg = config["data"]
    split_source = resolve_common_voice_split_source(data_cfg, split)
    samples = read_common_voice_split(data_cfg["root"], split_source)
    if max_samples is not None:
        samples = samples[:max_samples]
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    max_audio_seconds = float(data_cfg.get("max_audio_seconds", 30.0))
    return sum(
        audio_duration_seconds(
            sample.audio_path,
            fallback_sample_rate=sample_rate,
            max_audio_seconds=max_audio_seconds,
        )
        for sample in samples
    )


def build_result_payload(
    *,
    split: str,
    references: list[str],
    predictions: list[str],
    total_inference_seconds: float,
    total_audio_seconds: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    samples = len(references)
    return {
        "split": split,
        "samples": samples,
        "cer": character_error_rate(predictions, references),
        "inference_seconds": total_inference_seconds,
        "inference_seconds_per_sample": total_inference_seconds / max(samples, 1),
        "total_audio_seconds": total_audio_seconds,
        "realtime_factor": (
            total_inference_seconds / total_audio_seconds
            if total_audio_seconds > 0.0
            else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / max(samples, 1),
        "records": [
            {
                "reference": reference,
                "prediction": prediction,
                "char_error_rate": character_error_rate(
                    [prediction],
                    [reference],
                ),
            }
            for reference, prediction in zip(references, predictions)
        ],
    }


def build_error_payload(
    *,
    split: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "split": split,
        "samples": 0,
        "cer": None,
        "inference_seconds": 0.0,
        "inference_seconds_per_sample": None,
        "total_audio_seconds": 0.0,
        "realtime_factor": None,
        "elapsed_seconds": 0.0,
        "seconds_per_sample": None,
        "status": "error",
        "error_type": type(error).__name__,
        "error": str(error),
        "records": [],
    }


def write_eval_json(
    output_dir: Path,
    name: str,
    payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eval_{sanitize_name(name)}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def normalize_prediction_for_current_project(
    text: str,
    config: dict[str, Any],
) -> str:
    normalizer = build_text_normalizer(config.get("data", {}).get("text_normalization"))
    if not normalizer.enabled:
        return str(text or "").strip()
    return normalizer(str(text or ""))


def evaluate_current_model(
    *,
    name: str,
    config: dict[str, Any],
    split: str,
    checkpoint: str,
    max_samples: int | None,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    train_cfg = config.get("training", {})
    configure_training_runtime(train_cfg, device)
    _tokenizer, dataset, collator, model = build_components(
        config,
        split,
        max_samples,
    )
    character_vocab = CharacterVocab.build_from_config(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    dataloader_kwargs = build_dataloader_kwargs(train_cfg, device, split="eval")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        **dataloader_kwargs,
    )
    predictions: list[str] = []
    references: list[str] = []
    total_inference_seconds = 0.0
    amp_enabled = use_mixed_precision(train_cfg, device)
    amp_dtype = get_amp_dtype(train_cfg, device)
    print(
        f"eval_start name={name} split={split} samples={len(dataset)} "
        f"batch_size={batch_size} device={device} "
        f"num_workers={dataloader_kwargs.get('num_workers', 0)}",
        flush=True,
    )
    start = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(
            dataloader,
            desc=f"eval {split} {name}",
            dynamic_ncols=True,
        )
        for batch in progress:
            input_features = batch["input_features"].to(device)
            synchronize_for_timing(device)
            batch_start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled and amp_dtype is not None,
            ):
                generated = model.generate_ctc(
                    input_features=input_features,
                    use_corrector=False,
                )
                batch_predictions = [
                    decode_ctc(row.tolist(), character_vocab)
                    for row in generated.cpu()
                ]
            synchronize_for_timing(device)
            total_inference_seconds += time.perf_counter() - batch_start
            predictions.extend(batch_predictions)
            references.extend(batch["texts"])
            progress.set_postfix(
                samples=len(references),
                cer=f"{character_error_rate(predictions, references):.4f}",
                infer_s_per_sample=(
                    f"{total_inference_seconds / max(len(references), 1):.3f}"
                ),
            )

    elapsed_seconds = time.perf_counter() - start
    return build_result_payload(
        split=split,
        references=references,
        predictions=predictions,
        total_inference_seconds=total_inference_seconds,
        total_audio_seconds=total_audio_seconds_for_split(config, split, max_samples),
        elapsed_seconds=elapsed_seconds,
    )


class RawAudioDataset(Dataset):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        split: str,
        max_samples: int | None,
    ) -> None:
        data_cfg = config["data"]
        split_source = resolve_common_voice_split_source(data_cfg, split)
        self.samples = read_common_voice_split(data_cfg["root"], split_source)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
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
        return {
            "audio": waveform,
            "text": text,
        }


def collate_raw_audio(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio": [item["audio"] for item in batch],
        "texts": [item["text"] for item in batch],
    }


@dataclass
class BaselineContext:
    config: dict[str, Any]
    split: str
    max_samples: int | None
    checkpoint: str
    batch_size: int
    device: torch.device
    train_cfg: dict[str, Any]
    amp_enabled: bool
    amp_dtype: torch.dtype | None


def build_raw_audio_dataloader(context: BaselineContext) -> DataLoader:
    dataset = RawAudioDataset(
        config=context.config,
        split=context.split,
        max_samples=context.max_samples,
    )
    dataloader_kwargs = build_dataloader_kwargs(
        context.train_cfg,
        context.device,
        split="eval",
    )
    return DataLoader(
        dataset,
        batch_size=context.batch_size,
        shuffle=False,
        collate_fn=collate_raw_audio,
        **dataloader_kwargs,
    )


def split_whisper_decode_kwargs(
    decode_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    generate_kwargs = dict(decode_cfg)
    generate_kwargs.setdefault("task", "transcribe")
    generate_kwargs.setdefault("language", "zh")
    generate_kwargs.setdefault("num_beams", 1)
    batch_decode_kwargs = {
        "skip_special_tokens": bool(
            generate_kwargs.pop("skip_special_tokens", True)
        )
    }
    for key in ("clean_up_tokenization_spaces", "decode_with_timestamps"):
        if key in generate_kwargs:
            batch_decode_kwargs[key] = generate_kwargs.pop(key)
    batch_decode_kwargs.setdefault("decode_with_timestamps", False)
    return generate_kwargs, batch_decode_kwargs


def resolve_model_dtype(device: torch.device, amp_dtype: torch.dtype | None) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if amp_dtype in {torch.float16, torch.bfloat16}:
        return amp_dtype
    return torch.float16


def evaluate_whisper_tw(
    baseline_cfg: dict[str, Any],
    context: BaselineContext,
) -> dict[str, Any]:
    checkpoint = str(baseline_cfg.get("checkpoint") or context.checkpoint)
    return evaluate_current_model(
        name=str(baseline_cfg.get("name") or "whisper_tw"),
        config=context.config,
        split=context.split,
        checkpoint=checkpoint,
        max_samples=context.max_samples,
        batch_size=context.batch_size,
        device=context.device,
    )


def evaluate_hf_whisper(
    baseline_cfg: dict[str, Any],
    context: BaselineContext,
) -> dict[str, Any]:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_name_or_path = str(baseline_cfg["model_name_or_path"])
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=resolve_model_dtype(context.device, context.amp_dtype),
    )
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    if hasattr(model, "config"):
        model.config.forced_decoder_ids = None
    model.to(context.device)
    model.eval()

    generate_kwargs, batch_decode_kwargs = split_whisper_decode_kwargs(
        dict(baseline_cfg.get("decode") or {})
    )
    dataloader = build_raw_audio_dataloader(context)
    predictions: list[str] = []
    references: list[str] = []
    total_inference_seconds = 0.0
    sample_rate = int(context.config.get("data", {}).get("sample_rate", 16000))
    baseline_name = str(baseline_cfg.get("name") or model_name_or_path)
    print(
        f"eval_start name={baseline_name} split={context.split} "
        f"samples={len(dataloader.dataset)} batch_size={context.batch_size} "
        f"device={context.device}",
        flush=True,
    )
    start = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(
            dataloader,
            desc=f"eval {context.split} {baseline_name}",
            dynamic_ncols=True,
        )
        for batch in progress:
            waveforms = [audio.numpy() for audio in batch["audio"]]
            inputs = processor(
                waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_features = inputs["input_features"].to(
                device=context.device,
                dtype=getattr(model, "dtype", inputs["input_features"].dtype),
            )
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(context.device)
            synchronize_for_timing(context.device)
            batch_start = time.perf_counter()
            with torch.autocast(
                device_type=context.device.type,
                dtype=context.amp_dtype,
                enabled=context.amp_enabled and context.amp_dtype is not None,
            ):
                generated_ids = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    **generate_kwargs,
                )
                batch_predictions = processor.batch_decode(
                    generated_ids.detach().cpu(),
                    **batch_decode_kwargs,
                )
            synchronize_for_timing(context.device)
            total_inference_seconds += time.perf_counter() - batch_start
            normalized_predictions = [
                normalize_prediction_for_current_project(text, context.config)
                for text in batch_predictions
            ]
            predictions.extend(normalized_predictions)
            references.extend(batch["texts"])
            progress.set_postfix(
                samples=len(references),
                cer=f"{character_error_rate(predictions, references):.4f}",
                infer_s_per_sample=(
                    f"{total_inference_seconds / max(len(references), 1):.3f}"
                ),
            )

    elapsed_seconds = time.perf_counter() - start
    return build_result_payload(
        split=context.split,
        references=references,
        predictions=predictions,
        total_inference_seconds=total_inference_seconds,
        total_audio_seconds=total_audio_seconds_for_split(
            context.config,
            context.split,
            context.max_samples,
        ),
        elapsed_seconds=elapsed_seconds,
    )


def split_ctc_decode_kwargs(decode_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        key: decode_cfg[key]
        for key in ("skip_special_tokens", "clean_up_tokenization_spaces")
        if key in decode_cfg
    }


def evaluate_hf_ctc(
    baseline_cfg: dict[str, Any],
    context: BaselineContext,
) -> dict[str, Any]:
    from transformers import AutoModelForCTC, AutoProcessor

    model_name_or_path = str(baseline_cfg["model_name_or_path"])
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForCTC.from_pretrained(model_name_or_path)
    model.to(context.device)
    model.eval()

    decode_kwargs = split_ctc_decode_kwargs(dict(baseline_cfg.get("decode") or {}))
    dataloader = build_raw_audio_dataloader(context)
    predictions: list[str] = []
    references: list[str] = []
    total_inference_seconds = 0.0
    sample_rate = int(context.config.get("data", {}).get("sample_rate", 16000))
    baseline_name = str(baseline_cfg.get("name") or model_name_or_path)
    print(
        f"eval_start name={baseline_name} split={context.split} "
        f"samples={len(dataloader.dataset)} batch_size={context.batch_size} "
        f"device={context.device}",
        flush=True,
    )
    start = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(
            dataloader,
            desc=f"eval {context.split} {baseline_name}",
            dynamic_ncols=True,
        )
        for batch in progress:
            waveforms = [audio.numpy() for audio in batch["audio"]]
            inputs = processor(
                waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(context.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(context.device)
            synchronize_for_timing(context.device)
            batch_start = time.perf_counter()
            with torch.autocast(
                device_type=context.device.type,
                dtype=context.amp_dtype,
                enabled=context.amp_enabled and context.amp_dtype is not None,
            ):
                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                ).logits
                predicted_ids = logits.argmax(dim=-1)
                batch_predictions = processor.batch_decode(
                    predicted_ids.detach().cpu(),
                    **decode_kwargs,
                )
            synchronize_for_timing(context.device)
            total_inference_seconds += time.perf_counter() - batch_start
            normalized_predictions = [
                normalize_prediction_for_current_project(text, context.config)
                for text in batch_predictions
            ]
            predictions.extend(normalized_predictions)
            references.extend(batch["texts"])
            progress.set_postfix(
                samples=len(references),
                cer=f"{character_error_rate(predictions, references):.4f}",
                infer_s_per_sample=(
                    f"{total_inference_seconds / max(len(references), 1):.3f}"
                ),
            )

    elapsed_seconds = time.perf_counter() - start
    return build_result_payload(
        split=context.split,
        references=references,
        predictions=predictions,
        total_inference_seconds=total_inference_seconds,
        total_audio_seconds=total_audio_seconds_for_split(
            context.config,
            context.split,
            context.max_samples,
        ),
        elapsed_seconds=elapsed_seconds,
    )


RUNNER_TYPES = {
    "whisper_tw": evaluate_whisper_tw,
    "hf_whisper": evaluate_hf_whisper,
    "hf_ctc": evaluate_hf_ctc,
}


def load_enabled_baselines(path: str | Path) -> list[dict[str, Any]]:
    baselines = load_baseline_entries(path)
    enabled: list[dict[str, Any]] = []
    for item in baselines:
        if not bool(item.get("enabled", True)):
            continue
        enabled.append(dict(item))
    return enabled


def load_baseline_entries(path: str | Path) -> list[dict[str, Any]]:
    config = load_structured_config(path)
    baselines = config.get("baselines", [])
    if not isinstance(baselines, list):
        raise ValueError("baselines config must contain a list named 'baselines'.")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(baselines):
        if not isinstance(item, dict):
            raise ValueError(f"baseline #{index + 1} must be a mapping.")
        baseline_type = str(item.get("type") or "").strip()
        if baseline_type not in RUNNER_TYPES:
            raise ValueError(f"Unsupported baseline type: {baseline_type}")
        if not item.get("name"):
            item["name"] = f"{baseline_type}_{index + 1}"
        if baseline_type != "whisper_tw" and not item.get("model_name_or_path"):
            raise ValueError(f"baseline {item['name']} missing model_name_or_path.")
        entries.append(dict(item))
    return entries


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    output_dir = Path(
        args.output_dir
        or config.get("evaluation", {}).get("output_dir", "artifacts/eval")
    )
    all_baselines = load_baseline_entries(args.baselines_config)
    baselines = [
        item for item in all_baselines if bool(item.get("enabled", True))
    ]
    has_configured_whisper_tw = any(
        str(item.get("type")) == "whisper_tw" for item in all_baselines
    )
    if not has_configured_whisper_tw:
        baselines.insert(
            0,
            {
                "name": args.current_name,
                "enabled": True,
                "type": "whisper_tw",
                "checkpoint": args.checkpoint,
            },
        )
    for baseline_cfg in baselines:
        baseline_device = resolve_eval_device(
            config,
            args.device or baseline_cfg.get("device"),
        )
        configure_training_runtime(train_cfg, baseline_device)
        baseline_batch_size = default_batch_size(
            config,
            args.batch_size,
            baseline_cfg,
        )
        context = BaselineContext(
            config=config,
            split=args.split,
            max_samples=args.max_samples,
            checkpoint=args.checkpoint,
            batch_size=baseline_batch_size,
            device=baseline_device,
            train_cfg=train_cfg,
            amp_enabled=use_mixed_precision(train_cfg, baseline_device),
            amp_dtype=get_amp_dtype(train_cfg, baseline_device),
        )
        runner = RUNNER_TYPES[str(baseline_cfg["type"])]
        try:
            payload = runner(baseline_cfg, context)
        except Exception as exc:
            payload = build_error_payload(split=args.split, error=exc)
            print(
                f"eval_error name={baseline_cfg['name']} "
                f"type={type(exc).__name__}: {exc}",
                flush=True,
            )
        baseline_path = write_eval_json(output_dir, str(baseline_cfg["name"]), payload)
        print(f"eval_json={baseline_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
