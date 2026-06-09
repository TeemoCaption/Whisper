#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TABLE_FIELDS = [
    "表格",
    "模型",
    "底座模型",
    "訓練方式",
    "測試資料",
    "可訓練參數量",
    "字元錯誤率",
    "推論時間",
    "即時率",
    "路由準確率",
    "巨平均準確率",
    "巨平均精確率",
    "巨平均召回率",
    "巨平均F1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一鍵執行論文基線與路由器評估。")
    parser.add_argument("--config", required=True, help="專案設定檔路徑。")
    parser.add_argument(
        "--baselines-config",
        default="configs/baselines.yaml",
        help="基線設定檔路徑。",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="評估資料切分。",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/eval/baselines",
        help="輸出 JSON 與 CSV 的資料夾。",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        help="覆蓋微調與評估 DataLoader worker 數；共享記憶體不足時建議設為 0。",
    )
    parser.add_argument(
        "--redo-ft",
        action="store_true",
        help="重新執行所有 Whisper 基線微調，即使輸出資料夾已存在。",
    )
    parser.add_argument(
        "--redo-adalora",
        action="store_true",
        help="重新執行所有本研究 AdaLoRA 訓練，即使輸出資料夾已存在。",
    )
    parser.add_argument(
        "--redo-router",
        action="store_true",
        help="重新執行所有對比式路由器訓練，即使輸出資料夾已存在。",
    )
    return parser.parse_args()


def enabled_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items if bool(item.get("enabled", True))]


def load_structured_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix.lower() == ".json":
            return json.load(f) or {}

        import yaml

        return yaml.safe_load(f) or {}


def sanitize_name(value: str) -> str:
    text = str(value or "").strip().replace("-", "_")
    safe = "".join(char if char.isalnum() or char in "._" else "_" for char in text)
    return "_".join(part for part in safe.split("_") if part) or "model"


def sanitize_model_name(model_name_or_path: str) -> str:
    text = str(model_name_or_path or "whisper").strip().replace("\\", "/")
    name = text.rstrip("/").split("/")[-1] or "whisper"
    safe = "".join(char if char.isalnum() else "_" for char in name.lower())
    return "_".join(part for part in safe.split("_") if part) or "whisper"


def display_model_name(model_name_or_path: str | None) -> str:
    if not model_name_or_path:
        return ""
    slug = sanitize_model_name(model_name_or_path)
    return {
        "whisper_small": "Whisper-small",
        "whisper_medium": "Whisper-medium",
        "whisper_large_v3_turbo": "Whisper-large-v3-turbo",
    }.get(slug, str(model_name_or_path))


def configure_entry_model(config: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    scoped_config = copy.deepcopy(config)
    model_name_or_path = entry.get("model_name_or_path")
    output_dir = entry.get("training_output_dir")
    if model_name_or_path or output_dir:
        train_cfg = scoped_config.setdefault("whisper_train", {})
        if model_name_or_path:
            model_cfg = train_cfg.setdefault("model", {})
            model_cfg["model_name_or_path"] = str(model_name_or_path)
        if output_dir:
            training_cfg = train_cfg.setdefault("training", {})
            training_cfg["output_dir"] = str(output_dir)
    return scoped_config


def write_eval_json(output_dir: str | Path, filename: str, payload: dict[str, Any]) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / filename
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary_trainable_parameters(entry: dict[str, Any]) -> int | None:
    if entry.get("trainable_parameters") is not None:
        return int(entry["trainable_parameters"])
    summary_path = entry.get("summary_path")
    if summary_path and Path(summary_path).exists():
        summary = read_json(summary_path)
        value = summary.get("trainable_parameters")
        return None if value is None else int(value)
    return None


def adapter_trainable_parameters(payload: dict[str, Any]) -> int | None:
    adapter_dir = payload.get("adapter_dir")
    if not adapter_dir:
        return None
    adapter_path = Path(str(adapter_dir))
    candidates = [
        adapter_path.parent.parent / "adapter_manifest.json",
        adapter_path.parent / "adapter_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            manifest = read_json(candidate)
            value = manifest.get("trainable_parameters")
            return None if value is None else int(value)
    summary_path = adapter_path.parent.parent.parent / "best_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        value = summary.get("trainable_parameters")
        return None if value is None else int(value)
    return None


def finetune_done(job: dict[str, Any]) -> bool:
    output_dir = Path(str(job["output_dir"]))
    final_dir = output_dir / "final"
    summary_path = output_dir / "best_summary.json"
    return summary_path.exists() and (final_dir / "config.json").exists()


def adalora_done(job: dict[str, Any]) -> bool:
    output_dir = Path(str(job["output_dir"]))
    final_dir = output_dir / "final"
    summary_path = output_dir / "best_summary.json"
    adapter_name = str(job.get("adapter_name") or str(job["language"]).replace("-", "_").lower())
    candidates = [
        final_dir / "adapters" / adapter_name / "adapter_config.json",
        final_dir / adapter_name / "adapter_config.json",
    ]
    return summary_path.exists() and any(candidate.exists() for candidate in candidates)


def router_done(job: dict[str, Any]) -> bool:
    output_dir = Path(str(job["output_dir"]))
    return (output_dir / "contrastive_router.pt").exists()


def run_finetune_job(
    *,
    job: dict[str, Any],
    defaults: dict[str, Any],
    config_path: str,
    dataloader_num_workers: int | None,
) -> None:
    finetune_cfg = {**defaults, **job}
    command = [
        sys.executable,
        str(ROOT / "scripts" / "ft_whisper.py"),
        "--config",
        config_path,
        "--language",
        str(job["language"]),
        "--model-name-or-path",
        str(job["model_name_or_path"]),
        "--output-dir",
        str(job["output_dir"]),
    ]
    if finetune_cfg.get("num_train_epochs") is not None:
        command.extend(["--num-train-epochs", str(finetune_cfg["num_train_epochs"])])
    if finetune_cfg.get("learning_rate") is not None:
        command.extend(["--learning-rate", str(finetune_cfg["learning_rate"])])
    if finetune_cfg.get("lr_scheduler_type") is not None:
        command.extend(["--lr-scheduler-type", str(finetune_cfg["lr_scheduler_type"])])
    if finetune_cfg.get("min_lr_ratio") is not None:
        command.extend(["--min-lr-ratio", str(finetune_cfg["min_lr_ratio"])])
    if finetune_cfg.get("warmup_steps") is not None:
        command.extend(["--warmup-steps", str(finetune_cfg["warmup_steps"])])
    if finetune_cfg.get("weight_decay") is not None:
        command.extend(["--weight-decay", str(finetune_cfg["weight_decay"])])
    if finetune_cfg.get("unfreeze_encoder_last_n_layers") is not None:
        command.extend(
            [
                "--unfreeze-encoder-last-n-layers",
                str(finetune_cfg["unfreeze_encoder_last_n_layers"]),
            ]
        )
    if finetune_cfg.get("batch_size") is not None:
        command.extend(["--batch-size", str(finetune_cfg["batch_size"])])
    if dataloader_num_workers is not None:
        command.extend(["--dataloader-num-workers", str(dataloader_num_workers)])
    if finetune_cfg.get("device"):
        command.extend(["--device", str(finetune_cfg["device"])])
    print(f"finetune_start name={job['name']} output_dir={job['output_dir']}", flush=True)
    print(f"finetune_command={shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def run_adalora_job(
    *,
    job: dict[str, Any],
    defaults: dict[str, Any],
    config_path: str,
    dataloader_num_workers: int | None,
) -> None:
    adalora_cfg = {**defaults, **job}
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--config",
        config_path,
        "--language",
        str(job["language"]),
        "--model-name-or-path",
        str(job["model_name_or_path"]),
        "--output-dir",
        str(job["output_dir"]),
    ]
    if adalora_cfg.get("run_name"):
        command.extend(["--run-name", str(adalora_cfg["run_name"])])
    if adalora_cfg.get("batch_size") is not None:
        command.extend(["--batch-size", str(adalora_cfg["batch_size"])])
    print(f"adalora_start name={job['name']} output_dir={job['output_dir']}", flush=True)
    if dataloader_num_workers is not None:
        command.extend(["--dataloader-num-workers", str(dataloader_num_workers)])
    if adalora_cfg.get("device"):
        command.extend(["--device", str(adalora_cfg["device"])])
    print(f"adalora_command={shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def run_router_job(
    *,
    job: dict[str, Any],
    defaults: dict[str, Any],
    config_path: str,
    dataloader_num_workers: int | None,
) -> None:
    router_cfg = {**defaults, **job}
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_contrastive_router.py"),
        "--config",
        config_path,
        "--model-name-or-path",
        str(job["model_name_or_path"]),
        "--output-dir",
        str(job["output_dir"]),
    ]
    if router_cfg.get("feature_cache_dir"):
        command.extend(["--feature-cache-dir", str(router_cfg["feature_cache_dir"])])
    if router_cfg.get("num_train_epochs") is not None:
        command.extend(["--num-epochs", str(router_cfg["num_train_epochs"])])
    if dataloader_num_workers is not None:
        command.extend(["--dataloader-num-workers", str(dataloader_num_workers)])
    if router_cfg.get("device"):
        command.extend(["--device", str(router_cfg["device"])])
    print(f"router_start name={job['name']} output_dir={job['output_dir']}", flush=True)
    print(f"router_command={shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def ensure_finetune_jobs(
    *,
    suite_cfg: dict[str, Any],
    config_path: str,
    dataloader_num_workers: int | None,
    redo_finetune: bool,
) -> None:
    defaults = dict(suite_cfg.get("finetune_defaults") or {})
    for job in suite_cfg.get("finetune_jobs", []) or []:
        if not isinstance(job, dict):
            raise ValueError("finetune_jobs 每個項目都必須是 mapping。")
        if finetune_done(job) and not redo_finetune:
            print(f"finetune_skip_existing name={job['name']}", flush=True)
            continue
        if redo_finetune and finetune_done(job):
            print(f"finetune_redo_existing name={job['name']}", flush=True)
        run_finetune_job(
            job=job,
            defaults=defaults,
            config_path=config_path,
            dataloader_num_workers=dataloader_num_workers,
        )


def ensure_adalora_jobs(
    *,
    suite_cfg: dict[str, Any],
    config_path: str,
    dataloader_num_workers: int | None,
    redo_finetune: bool,
) -> None:
    defaults = dict(suite_cfg.get("adalora_defaults") or {})
    for job in suite_cfg.get("adalora_jobs", []) or []:
        if not isinstance(job, dict):
            raise ValueError("adalora_jobs 每個項目都必須是 mapping。")
        if adalora_done(job) and not redo_finetune:
            print(f"adalora_skip_existing name={job['name']}", flush=True)
            continue
        if redo_finetune and adalora_done(job):
            print(f"adalora_redo_existing name={job['name']}", flush=True)
        run_adalora_job(
            job=job,
            defaults=defaults,
            config_path=config_path,
            dataloader_num_workers=dataloader_num_workers,
        )


def ensure_router_jobs(
    *,
    suite_cfg: dict[str, Any],
    config_path: str,
    dataloader_num_workers: int | None,
    redo_finetune: bool,
) -> None:
    defaults = dict(suite_cfg.get("router_defaults") or {})
    for job in suite_cfg.get("router_jobs", []) or []:
        if not isinstance(job, dict):
            raise ValueError("router_jobs 每個項目都必須是 mapping。")
        if router_done(job) and not redo_finetune:
            print(f"router_skip_existing name={job['name']}", flush=True)
            continue
        if redo_finetune and router_done(job):
            print(f"router_redo_existing name={job['name']}", flush=True)
        run_router_job(
            job=job,
            defaults=defaults,
            config_path=config_path,
            dataloader_num_workers=dataloader_num_workers,
        )


def check_required_artifacts(config: dict[str, Any], baselines: list[dict[str, Any]]) -> None:
    from scripts.evaluate import resolve_adapter_dir, resolve_router_checkpoint

    missing: list[str] = []
    for item in baselines:
        baseline_type = str(item.get("type") or "")
        scoped_config = configure_entry_model(config, item)
        train_cfg = scoped_config.get("whisper_train") or scoped_config
        if baseline_type in {"adalora_router", "router_metrics"}:
            try:
                resolve_router_checkpoint(train_cfg, item.get("router_checkpoint"))
            except Exception as exc:
                missing.append(
                    f"{item.get('name')} 路由器權重缺少；請先執行對應 router_jobs "
                    f"({exc})"
                )
        if baseline_type == "adalora_adapter":
            language = str(item.get("language"))
            try:
                resolve_adapter_dir(train_cfg, language, item.get("adapter_dir"))
            except Exception as exc:
                missing.append(
                    f"{item.get('name')} 的 {language} AdaLoRA 權重缺少；"
                    f"請先執行對應 adalora_jobs ({exc})"
                )
    if missing:
        raise FileNotFoundError("\n".join(missing))


def evaluate_entry(
    *,
    entry: dict[str, Any],
    config: dict[str, Any],
    split: str,
    dataloader_num_workers: int | None,
) -> dict[str, Any]:
    from scripts.evaluate import (
        evaluate_hf_whisper,
        evaluate_router,
        evaluate_router_metrics,
        evaluate_single_adapter,
    )

    baseline_type = str(entry["type"])
    scoped_config = configure_entry_model(config, entry)
    if dataloader_num_workers is not None:
        train_cfg = scoped_config.setdefault("whisper_train", {})
        training_cfg = train_cfg.setdefault("training", {})
        training_cfg["dataloader_num_workers"] = max(0, int(dataloader_num_workers))
        if int(dataloader_num_workers) <= 0:
            training_cfg["dataloader_persistent_workers"] = False
            training_cfg["dataloader_prefetch_factor"] = None
    device_name = str(entry.get("device")) if entry.get("device") else None
    batch_size = int(entry["batch_size"]) if entry.get("batch_size") else None
    if baseline_type == "adalora_adapter":
        return evaluate_single_adapter(
            config=scoped_config,
            language=str(entry["language"]),
            split=split,
            max_samples=None,
            batch_size=batch_size,
            device_name=device_name,
            adapter_dir=entry.get("adapter_dir"),
        )
    if baseline_type == "adalora_router":
        return evaluate_router(
            config=scoped_config,
            split=split,
            max_samples=None,
            batch_size=batch_size,
            device_name=device_name,
            router_checkpoint=entry.get("router_checkpoint"),
        )
    if baseline_type == "router_metrics":
        return evaluate_router_metrics(
            config=scoped_config,
            split=split,
            max_samples=None,
            batch_size=batch_size,
            device_name=device_name,
            router_checkpoint=entry.get("router_checkpoint"),
        )
    if baseline_type == "hf_whisper":
        return evaluate_hf_whisper(
            config=config,
            model_name_or_path=str(entry["model_name_or_path"]),
            split=split,
            max_samples=None,
            batch_size=batch_size,
            device_name=device_name,
            language_filter=entry.get("language_filter"),
            decode=dict(entry.get("decode") or {}),
        )
    raise ValueError(f"不支援的 baseline type: {baseline_type}")


def base_row(
    *,
    table: str,
    entry: dict[str, Any],
    payload: dict[str, Any],
    trainable_parameters: int | None,
    cer: float | None,
    inference_seconds: float | None,
    realtime_factor: float | None,
) -> dict[str, Any]:
    return {
        "表格": table,
        "模型": entry.get("display_name") or entry["name"],
        "底座模型": entry.get("base_model")
        or display_model_name(payload.get("model_name_or_path") or entry.get("model_name_or_path")),
        "訓練方式": entry.get("training_method") or "",
        "測試資料": entry.get("test_data") or payload.get("split") or "",
        "可訓練參數量": "" if trainable_parameters is None else trainable_parameters,
        "字元錯誤率": "" if cer is None else cer,
        "推論時間": "" if inference_seconds is None else inference_seconds,
        "即時率": "" if realtime_factor is None else realtime_factor,
        "路由準確率": "",
        "巨平均準確率": "",
        "巨平均精確率": "",
        "巨平均召回率": "",
        "巨平均F1": "",
    }


def router_rows(entry: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    row = base_row(
        table="router_test",
        entry={
            **entry,
            "display_name": entry.get("display_name") or "對比式查詢路由器",
            "training_method": entry.get("training_method") or "單獨路由器驗證",
            "test_data": "zh-TW + nan-tw test",
        },
        payload=payload,
        trainable_parameters=None,
        cer=payload.get("cer_router_selected") if payload.get("mode") == "router" else None,
        inference_seconds=payload.get("inference_seconds"),
        realtime_factor=payload.get("realtime_factor"),
    )
    row["路由準確率"] = payload.get("router_accuracy", "")
    row["巨平均準確率"] = payload.get("router_macro_accuracy", "")
    row["巨平均精確率"] = payload.get("router_macro_precision", "")
    row["巨平均召回率"] = payload.get("router_macro_recall", "")
    row["巨平均F1"] = payload.get("router_macro_f1", "")
    return [row]


def rows_for_payload(entry: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    table = str(entry.get("table") or "")
    if table == "router_test":
        return router_rows(entry, payload)
    trainable = summary_trainable_parameters(entry)
    if entry.get("type") == "adalora_adapter":
        trainable = adapter_trainable_parameters(payload)
    return [
        base_row(
            table=table,
            entry=entry,
            payload=payload,
            trainable_parameters=trainable,
            cer=payload.get("cer"),
            inference_seconds=payload.get("inference_seconds"),
            realtime_factor=payload.get("realtime_factor"),
        )
    ]


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    suite_cfg = load_structured_config(args.baselines_config)
    baselines = enabled_items(suite_cfg.get("baselines", []) or [])
    if not baselines:
        raise ValueError("目前沒有啟用任何基線。")

    ensure_finetune_jobs(
        suite_cfg=suite_cfg,
        config_path=args.config,
        dataloader_num_workers=args.dataloader_num_workers,
        redo_finetune=args.redo_ft,
    )
    ensure_adalora_jobs(
        suite_cfg=suite_cfg,
        config_path=args.config,
        dataloader_num_workers=args.dataloader_num_workers,
        redo_finetune=args.redo_adalora,
    )
    ensure_router_jobs(
        suite_cfg=suite_cfg,
        config_path=args.config,
        dataloader_num_workers=args.dataloader_num_workers,
        redo_finetune=args.redo_router,
    )

    from whisper_tw.config import load_config

    config = load_config(args.config)
    check_required_artifacts(config, baselines)

    output_dir = Path(args.output_dir)
    rows_by_table: dict[str, list[dict[str, Any]]] = {
        "router_test": [],
        "zh_tw_baselines": [],
        "nan_tw_baselines": [],
    }
    for entry in baselines:
        name = str(entry["name"])
        payload = evaluate_entry(
            entry=entry,
            config=config,
            split=str(args.split),
            dataloader_num_workers=args.dataloader_num_workers,
        )
        payload["baseline_name"] = name
        payload["display_name"] = entry.get("display_name")
        payload["training_method"] = entry.get("training_method")
        payload["test_data"] = entry.get("test_data")
        if payload.get("mode") == "router_metrics" and payload.get("confusion_matrix"):
            from scripts.evaluate import write_confusion_matrix_artifacts

            payload["confusion_matrix_paths"] = write_confusion_matrix_artifacts(
                output_dir=output_dir,
                name=f"{sanitize_name(name)}_{args.split}",
                labels=[str(label) for label in payload["labels"]],
                confusion_matrix=payload["confusion_matrix"],
            )
        output_path = write_eval_json(
            output_dir,
            f"eval_{sanitize_name(name)}.json",
            payload,
        )
        print(f"eval_json={output_path}", flush=True)
        for row in rows_for_payload(entry, payload):
            rows_by_table.setdefault(str(row["表格"]), []).append(row)

    for table_name, rows in rows_by_table.items():
        write_table(output_dir / f"{table_name}.csv", rows)
        (output_dir / f"{table_name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"table_csv={output_dir / f'{table_name}.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
