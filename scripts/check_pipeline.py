#!/usr/bin/env python3
"""靜態檢查 Whisper-medium 雙語低秩適應流程。

這個檢查不下載模型、不讀取真實音訊內容，只驗證目前程式碼與設定是否
能支撐 zh-TW -> 華語文字、nan-tw -> 台語文字的前處理與對比式路由設計。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_cv import inspect_split
from scripts.lora_adapters import build_peft_config
from scripts.prepare_cv import prepare_datasets
from scripts.train import apply_language_training_override
from whisper_tw.config import load_config
from whisper_tw.contrastive_router import (
    ContrastiveAdapterRouter,
    ContrastiveRouterSpec,
)
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
    lora_cfg = read_text(ROOT / "configs" / "config.yaml")
    lora_h100_cfg = read_text(ROOT / "configs" / "config_h100.yaml")
    docs = read_text(ROOT / "docs" / "language_adalora_method.md")
    evaluate_script = read_text(ROOT / "scripts" / "evaluate.py")
    evaluate_baselines_script = read_text(ROOT / "scripts" / "evaluate_baselines.py")
    baselines_cfg = load_config(ROOT / "configs" / "baselines.yaml")
    config_8gb = load_config(ROOT / "configs" / "config.yaml")
    config_h100 = load_config(ROOT / "configs" / "config_h100.yaml")
    lora = config_8gb["whisper_train"]
    lora_h100 = config_h100["whisper_train"]

    require("peft" in requirements, "requirements.txt 缺少 peft 依賴。", failures)
    train_script = read_text(ROOT / "scripts" / "train.py")
    require(
        "Seq2SeqTrainer" not in train_script
        and "Seq2SeqTrainingArguments" not in train_script,
        "主訓練腳本不應再使用 Hugging Face Seq2SeqTrainer。",
        failures,
    )
    require(
        "update_adalora_rank_allocation(model, next_global_step)" in train_script,
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
        "whisper_baseline" not in config_8gb and "whisper_baseline" not in config_h100,
        "訓練設定不應再保留 whisper_baseline 區塊。",
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
    require("adalora:" in lora_cfg, "config.yaml 缺少 adalora 設定。", failures)
    require("adapter_scope:" in lora_cfg, "config.yaml 缺少 adapter_scope。", failures)
    require(
        "lora" not in lora.get("peft", {}),
        "config.yaml 不應再保留固定秩低秩設定。",
        failures,
    )
    require(
        "lora" not in lora_h100.get("peft", {}),
        "config_h100.yaml 不應再保留固定秩低秩設定。",
        failures,
    )
    require(
        lora.get("peft", {}).get("adapter_scope") == "language",
        "config.yaml 應作為語言專屬 adapter 的共用基底，而不是 shared adapter 訓練目標。",
        failures,
    )
    require(
        not lora.get("peft", {}).get("active_language"),
        "config.yaml 共用基底不應直接指定 active_language。",
        failures,
    )
    require("zh-TW: zh_tw" in lora_cfg, "config.yaml 缺少 zh-TW adapter。", failures)
    require("nan-tw: nan_tw" in lora_cfg, "config.yaml 缺少 nan-tw adapter。", failures)
    require(
        not (ROOT / "configs" / "baseline.yaml").exists(),
        "不應再保留 baseline.yaml。",
        failures,
    )
    require(
        not (ROOT / "configs" / "config_zh_tw.yaml").exists(),
        "不應再保留 config_zh_tw.yaml；語言請由 --language 指定。",
        failures,
    )
    require(
        not (ROOT / "configs" / "config_nan_tw.yaml").exists(),
        "不應再保留 config_nan_tw.yaml；語言請由 --language 指定。",
        failures,
    )
    for expected_language in ("zh-TW", "nan-tw"):
        train_cfg = copy.deepcopy(lora)
        adapter_name = apply_language_training_override(train_cfg, expected_language)
        require(
            train_cfg.get("peft", {}).get("active_language") == expected_language,
            f"--language {expected_language} 未正確設定 active_language。",
            failures,
        )
        require(
            train_cfg.get("data", {}).get("language_filter") == expected_language,
            f"--language {expected_language} 未正確限制訓練語言。",
            failures,
        )
        require(
            bool(adapter_name) and adapter_name in str(train_cfg.get("training", {}).get("output_dir")),
            f"--language {expected_language} 未產生語言專屬輸出資料夾。",
            failures,
        )
    zh_cfg = copy.deepcopy(lora)
    nan_cfg = copy.deepcopy(lora)
    apply_language_training_override(zh_cfg, "zh-TW")
    apply_language_training_override(nan_cfg, "nan-tw")
    require(
        zh_cfg.get("training", {}).get("learning_rate")
        == nan_cfg.get("training", {}).get("learning_rate"),
        "--language zh-TW 與 --language nan-tw 應套用相同學習率。",
        failures,
    )
    require(
        zh_cfg.get("training", {}).get("warmup_steps")
        == nan_cfg.get("training", {}).get("warmup_steps"),
        "--language zh-TW 與 --language nan-tw 應套用相同 warmup_steps。",
        failures,
    )
    require(
        "language_training_overrides" not in lora,
        "config.yaml 不應新增 language_training_overrides 區塊。",
        failures,
    )
    require(
        "train_tsv: data/processed/common_voice/train.tsv" in lora_cfg,
        "config.yaml 未指向前處理後 train.tsv。",
        failures,
    )
    require("remove_punctuation: false" in lora_cfg, "config.yaml 可能會清掉台語標記符號。", failures)
    require("adalora" in docs.lower(), "方法文件未對齊自適應低秩適應。", failures)
    require("adapter_scope: language" in docs, "方法文件未對齊語言專屬 adapter。", failures)
    require("AdaLoRA" in docs, "方法文件未說明自適應低秩容量分配。", failures)
    require("對比式鑰匙查詢路由" in docs, "方法文件未說明對比式鑰匙查詢路由。", failures)
    require(
        "contrastive_router:" in lora_cfg,
        "config.yaml 缺少對比式路由設定。",
        failures,
    )
    require(
        "language_classifier" not in lora and "language_classifier" not in lora_h100,
        "訓練設定不應再保留語言分類頭區塊。",
        failures,
    )
    combined_eval = evaluate_script + "\n" + evaluate_baselines_script
    require(
        "generate_ctc" not in combined_eval and "CharacterVocab" not in combined_eval,
        "評估腳本不應再保留舊 CTC 評估路徑。",
        failures,
    )
    require(
        "PeftModel.from_pretrained" in evaluate_script,
        "evaluate.py 應載入 Whisper 底座加語言專屬 AdaLoRA adapter。",
        failures,
    )
    require(
        'choices=["single", "router", "router_metrics"]' in evaluate_script,
        "evaluate.py 應提供 single、router_metrics 與 router 三種評估模式。",
        failures,
    )
    require(
        "def evaluate_router_metrics" in evaluate_script
        and "avg_similarity_gap" in evaluate_script,
        "evaluate.py 應提供不依賴 LoRA adapter 的路由器單獨評估。",
        failures,
    )
    require(
        "cer_router_selected" in evaluate_script
        and "cer_oracle_adapter" in evaluate_script
        and "cer_wrong_adapter" in evaluate_script,
        "完整路由評估應輸出 router-selected、oracle 與 wrong adapter CER。",
        failures,
    )
    baseline_types = {
        str(item.get("type"))
        for item in baselines_cfg.get("baselines", [])
        if isinstance(item, dict)
    }
    require(
        "hf_ctc" not in baseline_types and "whisper_tw" not in baseline_types,
        "baselines.yaml 不應再保留舊 CTC 或舊 Whisper-TW runner 類型。",
        failures,
    )
    require(
        {"adalora_adapter", "hf_whisper"}.issubset(baseline_types),
        "baselines.yaml 應包含新 adapter 評估與 Hugging Face Whisper 基線類型。",
        failures,
    )
    router_cfg = lora.get("contrastive_router", {}) or {}
    router_h100_cfg = lora_h100.get("contrastive_router", {}) or {}
    require(
        router_cfg.get("num_train_epochs") == 5.0,
        "config.yaml 對比式路由應訓練 5 epochs。",
        failures,
    )
    require(
        router_cfg.get("early_stopping", {}).get("enabled") is True,
        "config.yaml 對比式路由應啟用早停。",
        failures,
    )
    require(
        router_cfg.get("test_split") == "test",
        "config.yaml 對比式路由應設定 test_split 供評估腳本重跑測試。",
        failures,
    )
    require(
        router_cfg.get("pooling") == "attention",
        "config.yaml 對比式路由應使用注意力池化。",
        failures,
    )
    require(
        router_cfg.get("attention_hidden_size") == 128,
        "config.yaml 對比式路由應設定 attention_hidden_size。",
        failures,
    )
    require(
        router_cfg.get("embedding_size") == 128,
        "config.yaml 對比式路由應設定 embedding_size。",
        failures,
    )
    require(
        router_cfg.get("hidden_ratio") == 0.25,
        "config.yaml 對比式路由應降低 hidden_ratio 以抑制過擬合。",
        failures,
    )
    require(
        router_cfg.get("temperature") is not None,
        "config.yaml 對比式路由應設定 temperature。",
        failures,
    )
    require(
        router_cfg.get("label_smoothing") == 0.05,
        "config.yaml 對比式路由應啟用標籤平滑。",
        failures,
    )
    require(
        router_cfg.get("margin") == 0.2,
        "config.yaml 對比式路由應設定相似度邊界。",
        failures,
    )
    require(
        router_cfg.get("margin_loss_weight") == 0.1,
        "config.yaml 對比式路由應設定邊界損失權重。",
        failures,
    )
    require(
        router_cfg.get("labels") == router_h100_cfg.get("labels"),
        "8GB 與 H100 對比式路由語言標籤不一致。",
        failures,
    )
    for key in ("model", "peft", "data"):
        require(
            lora.get(key) == lora_h100.get(key),
            f"8GB 與 H100 低秩訓練的 {key} 設定不一致。",
            failures,
        )
    require(
        lora.get("training", {}).get("per_device_train_batch_size") == 4,
        "config.yaml 不是 8GB 加速批次設定。",
        failures,
    )
    require(
        lora.get("training", {}).get("lr_scheduler_type") == "linear",
        "config.yaml 未明確設定線性學習率排程。",
        failures,
    )
    require(
        lora.get("training", {}).get("gradient_accumulation_steps") == 4,
        "config.yaml 不是 8GB 加速梯度累積設定。",
        failures,
    )
    require(
        lora.get("training", {}).get("eval_strategy") == "epoch",
        "config.yaml 應以 epoch 為單位驗證。",
        failures,
    )
    require(
        lora.get("training", {}).get("save_strategy") == "epoch",
        "config.yaml 應以 epoch 為單位存檔。",
        failures,
    )
    require(
        lora.get("training", {}).get("logging_steps") >= 100,
        "config.yaml 的紀錄頻率過高，會增加 CPU 與 wandb 負擔。",
        failures,
    )
    require(
        lora.get("training", {}).get("warmup_first_batch") is True,
        "config.yaml 應啟用第一批資料預熱，方便定位 0% 卡住原因。",
        failures,
    )
    require(
        "max_train_samples" not in lora.get("training", {})
        and "max_eval_samples" not in lora.get("training", {}),
        "config.yaml 主訓練不應設定樣本數上限。",
        failures,
    )
    require(
        "max_train_samples" not in lora_h100.get("training", {})
        and "max_eval_samples" not in lora_h100.get("training", {}),
        "config_h100.yaml 主訓練不應設定樣本數上限。",
        failures,
    )
    require(
        lora.get("training", {}).get("predict_with_generate") is False,
        "config.yaml 完整驗證資料設定下應關閉訓練期生成式評估。",
        failures,
    )
    require(
        lora.get("training", {}).get("metric_for_best_model") == "eval_loss",
        "config.yaml 關閉生成式評估時應用 eval_loss 選最佳模型。",
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
        lora.get("training", {}).get("dataloader_num_workers") == 1,
        "config.yaml 8GB 設定應使用 1 個資料載入 worker 提高 GPU 餵資料速度。",
        failures,
    )
    require(
        lora.get("training", {}).get("dataloader_pin_memory") is True,
        "config.yaml 8GB 設定應啟用 pin memory 加速資料搬移。",
        failures,
    )
    require(
        lora.get("training", {}).get("tf32") is True,
        "config.yaml 8GB 設定應啟用 TF32 提高 NVIDIA GPU 訓練速度。",
        failures,
    )
    require(
        lora_h100.get("training", {}).get("bf16") is True,
        "H100 設定未啟用 bf16。",
        failures,
    )
    require(
        lora_h100.get("training", {}).get("lr_scheduler_type") == "linear",
        "config_h100.yaml 未明確設定線性學習率排程。",
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
        "scripts\\train.py" in read_text(ROOT / "README.md")
        or "scripts/train.py" in read_text(ROOT / "README.md"),
        "README.md 缺少主訓練腳本指令。",
        failures,
    )
    readme = read_text(ROOT / "README.md")
    require(
        "--language zh-TW" in readme
        and "--language nan-tw" in readme,
        "README.md 缺少語言專屬 AdaLoRA 訓練指令。",
        failures,
    )
    require(
        "configs\\config_zh_tw.yaml" not in readme
        and "configs\\config_nan_tw.yaml" not in readme,
        "README.md 不應再引用語言專屬設定檔。",
        failures,
    )
    require(
        "train_contrastive_router.py" in readme,
        "README.md 缺少對比式路由訓練指令。",
        failures,
    )
    require(
        "train_lang_classifier.py" not in readme,
        "README.md 不應再引用語言分類頭訓練指令。",
        failures,
    )
    require(
        not (ROOT / "scripts" / "train_lang_classifier.py").exists(),
        "不應再保留語言分類頭訓練腳本。",
        failures,
    )
    require(
        not (ROOT / "whisper_tw" / "lang_classifier.py").exists(),
        "不應再保留語言分類頭模組。",
        failures,
    )
    require(
        "configs\\baseline.yaml" not in readme and "configs/baseline.yaml" not in readme,
        "README.md 不應再引用 baseline.yaml。",
        failures,
    )
    require(
        "train_baseline.py" not in readme and "train_lora.py" not in readme,
        "README.md 不應再引用舊訓練腳本名稱。",
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


def validate_contrastive_router(failures: list[str]) -> dict:
    project_config = load_config(ROOT / "configs" / "config.yaml")
    router_cfg = project_config["whisper_train"]["contrastive_router"]
    labels = tuple(str(label) for label in router_cfg.get("labels", []))
    spec = ContrastiveRouterSpec(
        labels=labels,
        hidden_size=8,
        attention_hidden_size=4,
        embedding_size=6,
        hidden_ratio=0.5,
        dropout=0.0,
        temperature=float(router_cfg.get("temperature", 0.07)),
    )
    router = ContrastiveAdapterRouter(spec)
    hidden = torch.randn(3, 5, 8)
    labels_tensor = torch.tensor([0, 1, 0], dtype=torch.long)
    loss, outputs = router.compute_loss(hidden, labels_tensor)
    require(
        outputs["logits"].shape == (3, len(labels)),
        "對比式路由 logits 維度不正確。",
        failures,
    )
    require(
        torch.isfinite(loss).item(),
        "對比式路由損失不是有限值。",
        failures,
    )
    return {
        "labels": list(labels),
        "logits_shape": list(outputs["logits"].shape),
        "loss": float(loss.detach().cpu()),
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
    peft_report = validate_peft_config(failures)
    router_report = validate_contrastive_router(failures)

    report = {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "preprocessing": preprocessing_report,
        "real_data": real_data_report,
        "peft": peft_report,
        "contrastive_router": router_report,
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
