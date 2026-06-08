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

from whisper_tw.config import load_config


SUPPORTED_MODELS = {
    "openai/whisper-small": "whisper_small",
    "openai/whisper-medium": "whisper_medium",
    "openai/whisper-large-v3-turbo": "whisper_large_v3_turbo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微調 Whisper 監督式基線。")
    parser.add_argument("--config", required=True, help="專案設定檔路徑。")
    parser.add_argument(
        "--language",
        required=True,
        choices=["zh-TW", "nan-tw"],
        help="微調資料語言。",
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        choices=sorted(SUPPORTED_MODELS),
        help="Whisper 基線模型。",
    )
    parser.add_argument("--output-dir", help="覆寫輸出資料夾。")
    parser.add_argument("--num-train-epochs", type=float, help="覆寫訓練週期。")
    parser.add_argument("--learning-rate", type=float, help="覆寫學習率。")
    parser.add_argument("--lr-scheduler-type", help="覆寫學習率排程。")
    parser.add_argument("--min-lr-ratio", type=float, help="覆寫最低學習率比例。")
    parser.add_argument("--weight-decay", type=float, help="覆寫權重衰減。")
    parser.add_argument("--batch-size", type=int, help="覆寫訓練與驗證批次大小。")
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        help="覆寫訓練與驗證 DataLoader worker 數；共享記憶體不足時建議設為 0。",
    )
    parser.add_argument("--device", help="覆寫訓練裝置，例如 cuda 或 cpu。")
    return parser.parse_args()


def profile_name(output_dir: Path) -> str:
    name = output_dir.name.lower()
    if "h100" in name:
        return "h100"
    if "8gb" in name:
        return "8gb"
    return "local"


def language_slug(language: str) -> str:
    return str(language).replace("-", "_").lower()


def default_output_dir(config: dict[str, Any], model_name_or_path: str, language: str) -> Path:
    train_cfg = dict(config.get("whisper_train") or {})
    training_cfg = dict(train_cfg.get("training") or {})
    base_output = Path(
        str(training_cfg.get("output_dir") or "artifacts/models/whisper_medium_adalora")
    )
    suffix = profile_name(base_output)
    model_slug = SUPPORTED_MODELS[model_name_or_path]
    return Path("artifacts") / "baselines" / f"{model_slug}_ft_{language_slug(language)}_{suffix}"


def build_finetune_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    base_train_cfg = dict(config.get("whisper_train") or {})
    if not base_train_cfg:
        raise ValueError("設定檔缺少 whisper_train 區塊。")

    model_cfg = dict(base_train_cfg.get("model") or {})
    data_cfg = dict(base_train_cfg.get("data") or {})
    training_cfg = dict(base_train_cfg.get("training") or {})
    early_stopping_cfg = dict(base_train_cfg.get("early_stopping") or {})

    model_cfg.update(
        {
            "model_name_or_path": args.model_name_or_path,
            "freeze_encoder": True,
            "unfreeze_encoder_last_n_layers": 4,
            "train_decoder": False,
        }
    )
    data_cfg["language_filter"] = args.language

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(
        config,
        args.model_name_or_path,
        args.language,
    )
    training_cfg["output_dir"] = str(output_dir)
    training_cfg["run_name"] = f"{SUPPORTED_MODELS[args.model_name_or_path]}-ft-{language_slug(args.language)}"
    if args.num_train_epochs is not None:
        training_cfg["num_train_epochs"] = float(args.num_train_epochs)
    if args.learning_rate is not None:
        training_cfg["learning_rate"] = float(args.learning_rate)
    if args.lr_scheduler_type:
        training_cfg["lr_scheduler_type"] = str(args.lr_scheduler_type)
    if args.min_lr_ratio is not None:
        training_cfg["min_lr_ratio"] = float(args.min_lr_ratio)
    if args.weight_decay is not None:
        training_cfg["weight_decay"] = float(args.weight_decay)
    if args.batch_size is not None:
        training_cfg["per_device_train_batch_size"] = int(args.batch_size)
        training_cfg["per_device_eval_batch_size"] = int(args.batch_size)
    if args.dataloader_num_workers is not None:
        training_cfg["dataloader_num_workers"] = max(0, int(args.dataloader_num_workers))
        if int(args.dataloader_num_workers) <= 0:
            training_cfg["dataloader_persistent_workers"] = False
            training_cfg["dataloader_prefetch_factor"] = None
    if args.device:
        training_cfg["device"] = str(args.device)

    return {
        "base_config": str(Path(args.config)),
        "whisper_finetune": {
            "model": model_cfg,
            "peft": {"enabled": False},
            "data": data_cfg,
            "training": training_cfg,
            "early_stopping": early_stopping_cfg,
        },
    }


def main() -> int:
    args = parse_args()
    config = build_finetune_config(args)
    output_dir = Path(
        str(config["whisper_finetune"]["training"]["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_config = output_dir / "ft_config.generated.json"
    generated_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    training_cfg = config["whisper_finetune"]["training"]
    print(
        "finetune_runtime="
        + json.dumps(
            {
                "output_dir": str(output_dir),
                "per_device_train_batch_size": training_cfg.get(
                    "per_device_train_batch_size"
                ),
                "per_device_eval_batch_size": training_cfg.get(
                    "per_device_eval_batch_size"
                ),
                "learning_rate": training_cfg.get("learning_rate"),
                "lr_scheduler_type": training_cfg.get("lr_scheduler_type"),
                "min_lr_ratio": training_cfg.get("min_lr_ratio"),
                "weight_decay": training_cfg.get("weight_decay"),
                "dataloader_num_workers": training_cfg.get(
                    "dataloader_num_workers"
                ),
                "dataloader_persistent_workers": training_cfg.get(
                    "dataloader_persistent_workers"
                ),
                "dataloader_prefetch_factor": training_cfg.get(
                    "dataloader_prefetch_factor"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    old_argv = sys.argv[:]
    try:
        from scripts import train as train_script

        sys.argv = [old_argv[0], "--config", str(generated_config)]
        train_script.main(
            default_config=str(generated_config),
            description="微調 Whisper 監督式基線。",
            experiment_key="whisper_finetune",
            include_language_arg=False,
        )
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
