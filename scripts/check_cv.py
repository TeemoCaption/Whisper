#!/usr/bin/env python3
"""檢查 prepare_cv.py 產出的欄位與標註。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {
    "path",
    "audio_path",
    "sentence",
    "target_text",
    "language_label",
    "target_script",
    "romanization_text",
    "raw_sentence",
    "filter_reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="檢查整理後 Common Voice TSV 欄位。")
    parser.add_argument(
        "--prepared-dir",
        default="data/processed/common_voice",
        help="prepare_cv.py 輸出的整併 TSV 資料夾。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="要檢查的切分檔名，不含 .tsv。",
    )
    parser.add_argument("--max-examples", type=int, default=5, help="每種語言列出的範例數。")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="任一切分缺檔或欄位錯誤時，以非 0 狀態結束。",
    )
    parser.add_argument("--data-root", default="data", help="搭配 --check-audio 使用的資料根目錄。")
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="檢查 audio_path 是否能從資料根目錄找到音檔。",
    )
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def validate_rows(
    rows: list[dict[str, str]],
    *,
    data_root: Path,
    check_audio: bool,
) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows, start=2):
        label = row.get("language_label", "")
        target = row.get("target_text", "")
        sentence = row.get("sentence", "")
        raw = row.get("raw_sentence", "")
        script = row.get("target_script", "")
        romanization = row.get("romanization_text", "")
        reason = row.get("filter_reason", "")
        audio_path = row.get("audio_path", "")

        if reason:
            errors.append(f"第 {index} 列不應出現在整併訓練檔，filter_reason={reason}")
        if not target:
            errors.append(f"第 {index} 列 target_text 為空")
        if sentence != target:
            errors.append(f"第 {index} 列 sentence 與 target_text 不一致")
        if label == "zh-TW" and romanization:
            errors.append(f"第 {index} 列 zh-TW 不應有 romanization_text")
        if label == "nan-tw":
            if script == "han_with_romanization" and not romanization:
                errors.append(f"第 {index} 列 nan-tw 缺少 romanization_text")
            if "（" in target or "）" in target or "(" in target or ")" in target:
                errors.append(f"第 {index} 列 nan-tw target_text 仍含括號標音")
            if target == raw and romanization:
                errors.append(f"第 {index} 列 nan-tw 未從 raw_sentence 拆出主目標")
        if label not in {"zh-TW", "nan-tw"}:
            errors.append(f"第 {index} 列 language_label 非預期值: {label}")
        if check_audio and audio_path and not (data_root / audio_path).exists():
            errors.append(f"第 {index} 列找不到音檔: {data_root / audio_path}")
    return errors


def inspect_split(
    path: Path,
    max_examples: int,
    *,
    data_root: Path,
    check_audio: bool,
) -> tuple[dict, list[str]]:
    fieldnames, rows = read_rows(path)
    missing_fields = sorted(REQUIRED_FIELDS - set(fieldnames))
    errors = [f"缺少欄位: {', '.join(missing_fields)}"] if missing_fields else []
    if not missing_fields:
        errors.extend(
            validate_rows(rows, data_root=data_root, check_audio=check_audio)
        )

    by_label = Counter(row.get("language_label", "") for row in rows)
    by_script = Counter(row.get("target_script", "") for row in rows)
    examples: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        label = row.get("language_label", "")
        bucket = examples.setdefault(label, [])
        if len(bucket) < max_examples:
            bucket.append(
                {
                    "raw_sentence": row.get("raw_sentence", ""),
                    "target_text": row.get("target_text", ""),
                    "romanization_text": row.get("romanization_text", ""),
                    "target_script": row.get("target_script", ""),
                }
            )

    return (
        {
            "path": str(path),
            "rows": len(rows),
            "language_label_counts": dict(by_label),
            "target_script_counts": dict(by_script),
            "examples": examples,
        },
        errors,
    )


def main() -> int:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir)
    report = {"prepared_dir": str(prepared_dir), "splits": {}}
    all_errors: list[str] = []

    for split in args.splits:
        path = prepared_dir / f"{split}.tsv"
        if not path.exists():
            message = f"{split}: 找不到檔案 {path}"
            all_errors.append(message)
            report["splits"][split] = {"path": str(path), "missing": True}
            continue
        split_report, errors = inspect_split(
            path,
            args.max_examples,
            data_root=Path(args.data_root),
            check_audio=bool(args.check_audio),
        )
        report["splits"][split] = split_report
        all_errors.extend(f"{split}: {error}" for error in errors)

    report["errors"] = all_errors
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.strict and all_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
