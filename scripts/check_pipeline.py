#!/usr/bin/env python3
"""靜態檢查 Whisper-medium 雙語低秩適應流程。

這個檢查不下載模型、不讀取真實音訊內容，只驗證目前程式碼與設定是否
能支撐 zh-TW -> 華語文字、nan-tw -> 台語文字的前處理與路由設計。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_cv import inspect_split
from scripts.lora_adapters import activate_routed_adapters, build_peft_config
from scripts.prepare_cv import prepare_datasets
from scripts.route_confidence import build_routing_config, route_adapters
from whisper_tw.config import load_config
from whisper_tw.data import read_common_voice_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="檢查 Whisper-medium 雙語低秩適應流程。")
    parser.add_argument("--json", action="store_true", help="輸出 JSON 報告。")
    parser.add_argument(
        "--require-real-data",
        action="store_true",
        help="要求真實前處理資料存在並通過欄位與音檔路徑檢查。",
    )
    parser.add_argument("--data-root", default="data", help="真實資料根目錄。")
    parser.add_argument(
        "--prepared-dir",
        default="data/processed/common_voice",
        help="真實前處理 TSV 資料夾。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="真實資料檢查的切分檔名，不含 .tsv。",
    )
    return parser.parse_args()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_common_voice_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "client_id",
        "path",
        "sentence",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accent",
        "accents",
        "variant",
        "segment",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_tiny_common_voice_tree(root: Path) -> None:
    zh_rows = [
        {
            "client_id": "zh-1",
            "path": "zh_sample.mp3",
            "sentence": "需要認真考慮",
        }
    ]
    nan_rows = [
        {
            "client_id": "nan-1",
            "path": "nan_sample.mp3",
            "sentence": "竹仔籃（Tik-á-nâ）",
        },
        {
            "client_id": "nan-2",
            "path": "nan_romanized.mp3",
            "sentence": "Tik-á-nâ",
        },
    ]
    for locale, rows in (("zh-TW", zh_rows), ("nan-tw", nan_rows)):
        locale_root = root / locale
        (locale_root / "clips").mkdir(parents=True, exist_ok=True)
        for row in rows:
            (locale_root / "clips" / row["path"]).write_bytes(b"")
        for split in ("train", "dev", "test", "validated"):
            write_common_voice_split(locale_root / f"{split}.tsv", rows)


def validate_configs(failures: list[str]) -> None:
    requirements = read_text(ROOT / "requirements.txt")
    baseline_cfg = read_text(ROOT / "configs" / "baseline.yaml")
    lora_cfg = read_text(ROOT / "configs" / "config.yaml")
    lora_h100_cfg = read_text(ROOT / "configs" / "config_h100.yaml")
    docs = read_text(ROOT / "docs" / "lora_confidence_method.md")
    lora = load_config(ROOT / "configs" / "config.yaml")["whisper_train"]
    lora_h100 = load_config(ROOT / "configs" / "config_h100.yaml")["whisper_train"]

    require("peft" in requirements, "requirements.txt 缺少 peft 依賴。", failures)
    train_script = read_text(ROOT / "scripts" / "train_baseline.py")
    require(
        "def on_pre_optimizer_step" in train_script,
        "AdaLoRA 秩分配應在最佳化器更新前執行，避免梯度已清空。",
        failures,
    )
    require(
        "def on_step_end" not in train_script
        or "update_adalora_rank_allocation(model" not in train_script.split("def on_step_end", 1)[1],
        "AdaLoRA 不可在 on_step_end 更新秩分配，否則可能遇到 grad=None。",
        failures,
    )
    require(
        "model_name_or_path: openai/whisper-medium" in baseline_cfg,
        "baseline.yaml 未固定 Whisper-medium。",
        failures,
    )
    require(
        "peft:" not in baseline_cfg,
        "baseline.yaml 不應啟用低秩適應，這份應是純基線。",
        failures,
    )
    baseline = load_config(ROOT / "configs" / "baseline.yaml")["whisper_baseline"]
    require(
        baseline.get("training", {}).get("disable_tqdm") is False,
        "baseline.yaml 未明確開啟訓練進度條。",
        failures,
    )
    require(
        baseline.get("model", {}).get("gradient_checkpointing_use_reentrant") is False,
        "baseline.yaml 應使用非重入梯度檢查點，避免凍結 Whisper 時梯度中斷。",
        failures,
    )
    require(
        "model_name_or_path: openai/whisper-medium" in lora_cfg,
        "config.yaml 未固定 Whisper-medium。",
        failures,
    )
    require(
        "model_name_or_path: openai/whisper-medium" in lora_h100_cfg,
        "config_h100.yaml 未固定 Whisper-medium。",
        failures,
    )
    require("whisper_train:" in lora_cfg, "config.yaml 缺少 whisper_train 區塊。", failures)
    require("method: adalora" in lora_cfg, "config.yaml 未預設自適應低秩方法。", failures)
    require("lora:" in lora_cfg, "config.yaml 缺少 lora 固定秩對照設定。", failures)
    require("adalora:" in lora_cfg, "config.yaml 缺少 adalora 設定。", failures)
    require("adapter_scope:" in lora_cfg, "config.yaml 缺少 adapter_scope。", failures)
    require("zh-TW: zh_tw" in lora_cfg, "config.yaml 缺少 zh-TW adapter。", failures)
    require("nan-tw: nan_tw" in lora_cfg, "config.yaml 缺少 nan-tw adapter。", failures)
    require(
        "train_tsv: data/processed/common_voice/train.tsv" in lora_cfg,
        "config.yaml 未指向前處理後 train.tsv。",
        failures,
    )
    require("remove_punctuation: false" in lora_cfg, "config.yaml 可能會清掉台語標記符號。", failures)
    require("adalora" in docs, "方法文件未對齊自適應低秩適應。", failures)
    require("adapter_scope: language" in docs, "方法文件未對齊語言專屬 adapter。", failures)
    require("信心" in docs, "方法文件缺少信心閥值說明。", failures)
    for key in ("model", "peft", "data", "routing"):
        require(
            lora.get(key) == lora_h100.get(key),
            f"8GB 與 H100 低秩訓練的 {key} 設定不一致。",
            failures,
        )
    require(
        lora.get("training", {}).get("per_device_train_batch_size") == 1,
        "config.yaml 不是 8GB 小批次設定。",
        failures,
    )
    require(
        lora.get("training", {}).get("eval_steps") >= 1000,
        "config.yaml 的驗證頻率過高，8GB 低 CPU 設定應降低驗證頻率。",
        failures,
    )
    require(
        lora.get("training", {}).get("logging_steps") >= 100,
        "config.yaml 的紀錄頻率過高，會增加 CPU 與 wandb 負擔。",
        failures,
    )
    require(
        lora.get("training", {}).get("disable_tqdm") is True,
        "config.yaml 8GB 低 CPU 設定應關閉 tqdm 終端進度條。",
        failures,
    )
    require(
        lora.get("training", {}).get("warmup_first_batch") is True,
        "config.yaml 應啟用第一批資料預熱，方便定位 0% 卡住原因。",
        failures,
    )
    require(
        lora.get("training", {}).get("max_eval_samples") == 1000,
        "config.yaml 8GB 設定應限制驗證樣本數，避免評估時 RAM 持續上升。",
        failures,
    )
    require(
        lora.get("training", {}).get("eval_accumulation_steps") == 16,
        "config.yaml 8GB 設定應啟用評估累積釋放，降低 RAM 峰值。",
        failures,
    )
    require(
        lora.get("training", {}).get("eval_do_concat_batches") is False,
        "config.yaml 8GB 設定應避免評估批次長時間累積在 RAM。",
        failures,
    )
    require(
        lora.get("training", {}).get("torch_empty_cache_steps") == 100,
        "config.yaml 8GB 設定應定期釋放框架快取。",
        failures,
    )
    require(
        lora_h100.get("training", {}).get("bf16") is True,
        "H100 設定未啟用 bf16。",
        failures,
    )
    for label, cfg in (
        ("config.yaml", lora),
        ("config_h100.yaml", lora_h100),
    ):
        require(
            cfg.get("model", {}).get("gradient_checkpointing_use_reentrant") is False,
            f"{label} 應使用非重入梯度檢查點，避免低秩參數梯度為 None。",
            failures,
        )
    require(
        lora_h100.get("training", {}).get("disable_tqdm") is False,
        "config_h100.yaml 未明確開啟訓練進度條。",
        failures,
    )
    require(
        "scripts\\train_baseline.py" in read_text(ROOT / "README.md"),
        "README.md 缺少基線訓練腳本指令。",
        failures,
    )
    require(
        "scripts\\train_lora.py" in read_text(ROOT / "README.md"),
        "README.md 缺少低秩適應訓練腳本指令。",
        failures,
    )


def validate_preprocessing_and_inspection(failures: list[str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="whisper_tw_validate_") as temp_dir:
        temp_root = Path(temp_dir)
        data_root = temp_root / "data"
        output_dir = data_root / "processed" / "common_voice"
        build_tiny_common_voice_tree(data_root)
        args = SimpleNamespace(
            data_root=str(data_root),
            output_dir=str(output_dir),
            locales=["zh-TW", "nan-tw"],
            splits=["train", "dev", "test", "validated"],
            min_duration_sec=0.0,
            max_duration_sec=30.0,
            keep_romanized_only=False,
            no_progress=True,
        )
        summary = prepare_datasets(args)
        train_path = output_dir / "train.tsv"
        split_report, errors = inspect_split(
            train_path,
            5,
            data_root=data_root,
            check_audio=True,
        )
        failures.extend(f"前處理檢查失敗: {error}" for error in errors)

        with train_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        zh_rows = [row for row in rows if row.get("language_label") == "zh-TW"]
        nan_rows = [row for row in rows if row.get("language_label") == "nan-tw"]
        require(bool(zh_rows), "前處理後缺少 zh-TW 樣本。", failures)
        require(bool(nan_rows), "前處理後缺少 nan-tw 樣本。", failures)
        if nan_rows:
            nan = nan_rows[0]
            require(nan.get("target_text") == "竹仔籃", "nan-tw 主目標未移除括號標音。", failures)
            require(nan.get("romanization_text") == "Tik-á-nâ", "nan-tw 台羅輔助欄位不正確。", failures)
            require(nan.get("target_script") == "han_with_romanization", "nan-tw target_script 不正確。", failures)
        require(
            summary["locales"]["nan-tw"]["splits"]["train"]["filter_reason_counts"].get("romanized_only") == 1,
            "romanized_only 樣本未依預設排除。",
            failures,
        )
        loaded_samples = read_common_voice_split(data_root, train_path)
        require(len(loaded_samples) == 2, "前處理 TSV 無法被訓練資料讀取器正確載入。", failures)
        require(
            {sample.language_label for sample in loaded_samples} == {"zh-TW", "nan-tw"},
            "訓練資料讀取器未保留 zh-TW / nan-tw 語言標籤。",
            failures,
        )
        require(
            all(sample.audio_path.exists() for sample in loaded_samples),
            "訓練資料讀取器解析出的音檔路徑不存在。",
            failures,
        )
        return {"summary": summary, "inspect": split_report}


def validate_real_prepared_data(
    failures: list[str],
    *,
    data_root: Path,
    prepared_dir: Path,
    splits: list[str],
    require_real_data: bool,
) -> dict:
    report: dict[str, object] = {
        "data_root": str(data_root),
        "prepared_dir": str(prepared_dir),
        "required": require_real_data,
        "status": "not_checked",
        "splits": {},
    }
    if not prepared_dir.exists():
        report["status"] = "missing"
        if require_real_data:
            failures.append(f"找不到真實前處理資料夾: {prepared_dir}")
        return report

    found_any = False
    for split in splits:
        split_path = prepared_dir / f"{split}.tsv"
        split_status: dict[str, object] = {
            "path": str(split_path),
            "exists": split_path.exists(),
        }
        report["splits"][split] = split_status
        if not split_path.exists():
            if require_real_data:
                failures.append(f"找不到真實前處理檔: {split_path}")
            continue

        found_any = True
        split_report, errors = inspect_split(
            split_path,
            3,
            data_root=data_root,
            check_audio=True,
        )
        split_status["inspect"] = split_report
        split_status["errors"] = errors
        failures.extend(f"真實資料 {split}: {error}" for error in errors)

        samples = read_common_voice_split(data_root, split_path)
        language_counts: dict[str, int] = {}
        script_counts: dict[str, int] = {}
        for sample in samples:
            language_counts[sample.language_label] = (
                language_counts.get(sample.language_label, 0) + 1
            )
            script_counts[sample.target_script] = (
                script_counts.get(sample.target_script, 0) + 1
            )
        split_status["loaded_samples"] = len(samples)
        split_status["language_counts"] = language_counts
        split_status["target_script_counts"] = script_counts

        if require_real_data:
            require(bool(samples), f"真實資料 {split} 無可訓練樣本。", failures)
            require(
                "zh-TW" in language_counts,
                f"真實資料 {split} 缺少 zh-TW 樣本。",
                failures,
            )
            require(
                "nan-tw" in language_counts,
                f"真實資料 {split} 缺少 nan-tw 樣本。",
                failures,
            )

    if found_any:
        report["status"] = "checked"
    else:
        report["status"] = "missing"
        if require_real_data:
            failures.append(f"{prepared_dir} 內沒有任何指定切分 TSV。")
    return report


class FakePeftModel:
    def __init__(self) -> None:
        self.active_adapter = ""

    def set_adapter(self, name: str) -> None:
        self.active_adapter = name


def validate_routing(failures: list[str]) -> dict:
    project_config = load_config(ROOT / "configs" / "config.yaml")
    routing_config = build_routing_config(project_config)
    high = route_adapters({"zh-TW": 0.9, "nan-tw": 0.1}, config=routing_config)
    middle = route_adapters({"zh-TW": 0.62, "nan-tw": 0.38}, config=routing_config)
    low = route_adapters(
        {"zh-TW": 0.51, "nan-tw": 0.49},
        config=routing_config,
        target_language="nan-tw",
    )

    require(high.routing_mode == "single", "高信心未使用單一 adapter。", failures)
    require(middle.routing_mode == "mixed", "中信心未使用混合 adapter。", failures)
    require(low.routing_mode == "shared", "低信心未使用共享 adapter。", failures)
    require(low.target_language == "nan-tw", "低信心時錯誤改變 nan-tw 目標語言。", failures)

    fake_model = FakePeftModel()
    activation = activate_routed_adapters(fake_model, high)
    require(
        fake_model.active_adapter == "zh_tw",
        "路由結果無法實際切換 adapter。",
        failures,
    )
    return {
        "high": high.to_dict(),
        "middle": middle.to_dict(),
        "low": low.to_dict(),
        "activation": activation,
    }


def validate_peft_config(failures: list[str]) -> dict:
    project_config = load_config(ROOT / "configs" / "config.yaml")
    peft_cfg = project_config["whisper_train"]["peft"]
    peft_config = build_peft_config(peft_cfg)
    task_type = getattr(peft_config, "task_type", None)
    require(
        task_type is None,
        "Whisper 低秩適應不可使用文字序列任務類型，否則訓練會傳入 input_ids。",
        failures,
    )
    return {
        "method": str(peft_cfg.get("method") or ""),
        "task_type": task_type,
        "target_modules": list(getattr(peft_config, "target_modules", []) or []),
    }


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    validate_configs(failures)
    preprocessing_report = validate_preprocessing_and_inspection(failures)
    real_data_report = validate_real_prepared_data(
        failures,
        data_root=Path(args.data_root),
        prepared_dir=Path(args.prepared_dir),
        splits=list(args.splits),
        require_real_data=bool(args.require_real_data),
    )
    routing_report = validate_routing(failures)
    peft_report = validate_peft_config(failures)

    report = {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "preprocessing": preprocessing_report,
        "real_data": real_data_report,
        "routing": routing_report,
        "peft": peft_report,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if failures:
            print("整合檢查失敗：")
            for failure in failures:
                print(f"- {failure}")
        else:
            print("Whisper-medium 雙語低秩適應流程整合檢查通過。")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
