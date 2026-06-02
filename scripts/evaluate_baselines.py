#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.runtime_env import configure_runtime_environment

configure_runtime_environment()

from scripts.evaluate import (
    evaluate_hf_whisper,
    evaluate_router,
    evaluate_single_adapter,
    load_config,
    sanitize_name,
    write_eval_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用目前 AdaLoRA 架構與 Hugging Face Whisper 基線做比較。"
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
    parser.add_argument("--max-samples", type=int, help="只評估前 N 筆樣本。")
    parser.add_argument("--batch-size", type=int, help="覆蓋評估批次大小。")
    parser.add_argument("--device", help="覆蓋所有模型的裝置。")
    parser.add_argument("--output-dir", default="artifacts/eval", help="評估輸出資料夾。")
    parser.add_argument(
        "--language",
        default="zh-TW",
        help="未在 baselines.yaml 指定語言時，adapter 評估使用的語言。",
    )
    parser.add_argument(
        "--language-filter",
        help="覆蓋 Hugging Face Whisper 基線的資料語言篩選。",
    )
    return parser.parse_args()


def load_structured_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() == ".json":
            return json.load(f) or {}

        import yaml

        return yaml.safe_load(f) or {}


def load_baseline_entries(path: str | Path) -> list[dict[str, Any]]:
    config = load_structured_config(path)
    baselines = config.get("baselines", [])
    if not isinstance(baselines, list):
        raise ValueError("baselines 設定必須包含 baselines 清單。")

    allowed_types = {"adalora_adapter", "adalora_router", "hf_whisper"}
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(baselines):
        if not isinstance(item, dict):
            raise ValueError(f"baseline #{index + 1} 必須是 mapping。")
        baseline_type = str(item.get("type") or "").strip()
        if baseline_type not in allowed_types:
            raise ValueError(
                f"不支援的 baseline type: {baseline_type}；"
                f"可用類型: {', '.join(sorted(allowed_types))}"
            )
        if not item.get("name"):
            item["name"] = f"{baseline_type}_{index + 1}"
        entries.append(dict(item))
    return entries


def build_error_payload(*, split: str, error: Exception) -> dict[str, Any]:
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


def run_entry(
    *,
    baseline_cfg: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline_type = str(baseline_cfg["type"])
    if baseline_type == "adalora_adapter":
        language = str(baseline_cfg.get("language") or args.language)
        return evaluate_single_adapter(
            config=config,
            language=language,
            split=str(args.split),
            max_samples=args.max_samples,
            batch_size=int(baseline_cfg.get("batch_size") or args.batch_size)
            if baseline_cfg.get("batch_size") or args.batch_size
            else None,
            device_name=str(baseline_cfg.get("device") or args.device)
            if baseline_cfg.get("device") or args.device
            else None,
            adapter_dir=baseline_cfg.get("adapter_dir"),
        )

    if baseline_type == "adalora_router":
        return evaluate_router(
            config=config,
            split=str(args.split),
            max_samples=args.max_samples,
            batch_size=int(baseline_cfg.get("batch_size") or args.batch_size)
            if baseline_cfg.get("batch_size") or args.batch_size
            else None,
            device_name=str(baseline_cfg.get("device") or args.device)
            if baseline_cfg.get("device") or args.device
            else None,
            router_checkpoint=baseline_cfg.get("router_checkpoint"),
        )

    if baseline_type == "hf_whisper":
        model_name_or_path = baseline_cfg.get("model_name_or_path")
        if not model_name_or_path:
            raise ValueError(f"baseline {baseline_cfg['name']} 缺少 model_name_or_path。")
        language_filter = baseline_cfg.get("language_filter")
        if language_filter is None:
            language_filter = args.language_filter
        return evaluate_hf_whisper(
            config=config,
            model_name_or_path=str(model_name_or_path),
            split=str(args.split),
            max_samples=args.max_samples,
            batch_size=int(baseline_cfg.get("batch_size") or args.batch_size)
            if baseline_cfg.get("batch_size") or args.batch_size
            else None,
            device_name=str(baseline_cfg.get("device") or args.device)
            if baseline_cfg.get("device") or args.device
            else None,
            language_filter=None if language_filter in (None, "") else str(language_filter),
            decode=dict(baseline_cfg.get("decode") or {}),
        )

    raise ValueError(f"不支援的 baseline type: {baseline_type}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    baselines = [
        item for item in load_baseline_entries(args.baselines_config)
        if bool(item.get("enabled", True))
    ]
    if not baselines:
        raise ValueError("目前沒有啟用任何 baseline。")

    for baseline_cfg in baselines:
        name = str(baseline_cfg["name"])
        try:
            payload = run_entry(baseline_cfg=baseline_cfg, config=config, args=args)
        except Exception as exc:
            payload = build_error_payload(split=str(args.split), error=exc)
            print(
                f"eval_error name={name} type={type(exc).__name__}: {exc}",
                flush=True,
            )
        output_path = write_eval_json(
            args.output_dir,
            f"eval_{sanitize_name(name)}.json",
            payload,
        )
        print(f"eval_json={output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
