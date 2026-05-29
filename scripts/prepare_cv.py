#!/usr/bin/env python3
"""整理 Common Voice zh-TW 與 nan-tw 資料欄位。

輸出資料會保留原始句子，並另外建立模型訓練用目標文字。nan-tw 會把
括號內台羅或白話字抽出為輔助欄位，主目標保留台語漢字。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("data/processed/common_voice")
DEFAULT_LOCALES = ("zh-TW", "nan-tw")
DEFAULT_SPLITS = ("train", "dev", "test", "validated")
REQUIRED_OUTPUT_FIELDS = (
    "client_id",
    "path",
    "audio_path",
    "sentence",
    "target_text",
    "language_label",
    "target_script",
    "romanization_text",
    "raw_sentence",
    "filter_reason",
    "source_locale",
    "source_split",
    "duration_ms",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accent",
    "accents",
    "variant",
    "segment",
)
PAREN_ANNOTATION_RE = re.compile(r"\(([^()]*)\)")
UNBALANCED_PAREN_RE = re.compile(r"\([^()]*$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ\u0300-\u036f]")
WHITESPACE_RE = re.compile(r"\s+")
ROMANIZATION_SEPARATORS = (",", "，")


@dataclass(frozen=True)
class PreparedRow:
    row: dict[str, str]
    keep: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清理 Common Voice zh-TW 與 nan-tw，輸出可訓練與可檢查的 TSV。",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="資料根目錄。")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="整併後 TSV 輸出資料夾。",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=list(DEFAULT_LOCALES),
        help="要整理的語言資料夾名稱。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="要整理的切分檔名，不含 .tsv。",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=0.2,
        help="若 clip_durations.tsv 可用，低於此秒數的樣本會被排除。",
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=30.0,
        help="若 clip_durations.tsv 可用，高於此秒數的樣本會被排除。",
    )
    parser.add_argument(
        "--keep-romanized-only",
        action="store_true",
        help="保留只有台羅或白話字、沒有漢字的 nan-tw 樣本。",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只執行文字清理自我檢查，不讀取資料集。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="關閉 tqdm 進度條。",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = WHITESPACE_RE.sub(" ", text)
    return text


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def contains_latin(text: str) -> bool:
    return bool(LATIN_RE.search(normalize_text(text)))


def looks_like_romanization(text: str) -> bool:
    value = normalize_text(text)
    if not value or has_cjk(value):
        return False
    return contains_latin(value)


def split_trailing_romanization(text: str) -> tuple[str, str]:
    value = normalize_text(text)
    separator_positions = [value.rfind(sep) for sep in ROMANIZATION_SEPARATORS]
    split_at = max(separator_positions)
    if split_at < 0:
        return value, ""

    prefix = normalize_text(value[:split_at])
    suffix = normalize_text(value[split_at + 1 :]).rstrip(")")
    if prefix and has_cjk(prefix) and suffix and contains_latin(suffix):
        return prefix, suffix
    return value, ""


def split_nan_tw_sentence(raw_sentence: str) -> tuple[str, str, str]:
    raw = normalize_text(raw_sentence)
    romanizations: list[str] = []

    def replace_match(match: re.Match[str]) -> str:
        content = normalize_text(match.group(1))
        if content and contains_latin(content):
            romanizations.append(content)
        return ""

    target = PAREN_ANNOTATION_RE.sub(replace_match, raw)
    target = UNBALANCED_PAREN_RE.sub("", target)
    target = normalize_text(target)
    target, trailing_romanization = split_trailing_romanization(target)
    if trailing_romanization:
        romanizations.append(trailing_romanization)
    target = target.strip(" ,，.。;；:：、")
    romanization_text = " | ".join(romanizations)

    if has_cjk(target) and romanization_text:
        target_script = "han_with_romanization"
    elif has_cjk(target):
        target_script = "han"
    elif looks_like_romanization(raw):
        target = raw
        target_script = "romanized_only"
    else:
        target_script = "unknown"

    return target, romanization_text, target_script


def prepare_zh_tw_sentence(raw_sentence: str) -> tuple[str, str]:
    target = normalize_text(raw_sentence)
    return target, "traditional_han" if has_cjk(target) else "unknown"


def find_dataset_content_root(locale_root: Path, split_names: Iterable[str]) -> Path:
    if any((locale_root / f"{split}.tsv").exists() for split in split_names):
        return locale_root
    candidates = sorted(
        {
            path.parent
            for split in split_names
            for path in locale_root.rglob(f"{split}.tsv")
        },
        key=lambda path: (len(path.relative_to(locale_root).parts), str(path)),
    )
    return candidates[0] if candidates else locale_root


def read_duration_index(content_root: Path) -> dict[str, int]:
    durations_path = content_root / "clip_durations.tsv"
    if not durations_path.exists():
        return {}
    durations: dict[str, int] = {}
    with durations_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "clip" not in (reader.fieldnames or []) or "duration[ms]" not in (
            reader.fieldnames or []
        ):
            return {}
        for row in reader:
            clip = normalize_text(row.get("clip", ""))
            if not clip:
                continue
            try:
                durations[clip] = int(float(normalize_text(row.get("duration[ms]", "0"))))
            except ValueError:
                continue
    return durations


def choose_sentence(row: dict[str, str]) -> str:
    return row.get("sentence") or row.get("text") or ""


def build_audio_path(locale: str, rel_path: str) -> str:
    return str(Path(locale) / "clips" / rel_path).replace("\\", "/")


def prepare_row(
    *,
    locale: str,
    split: str,
    row: dict[str, str],
    duration_ms: int | None,
    min_duration_sec: float,
    max_duration_sec: float,
    keep_romanized_only: bool,
) -> PreparedRow:
    raw_sentence = str(choose_sentence(row) or "").strip()
    normalized_sentence = normalize_text(raw_sentence)
    rel_path = normalize_text(row.get("path", ""))
    filter_reason = ""

    if locale == "nan-tw":
        target_text, romanization_text, target_script = split_nan_tw_sentence(
            normalized_sentence
        )
        if target_script == "romanized_only" and not keep_romanized_only:
            filter_reason = "romanized_only"
    elif locale == "zh-TW":
        target_text, target_script = prepare_zh_tw_sentence(normalized_sentence)
        romanization_text = ""
    else:
        target_text = normalized_sentence
        romanization_text = ""
        target_script = "unknown"

    if not rel_path:
        filter_reason = filter_reason or "missing_path"
    if not raw_sentence:
        filter_reason = filter_reason or "missing_sentence"
    if not target_text:
        filter_reason = filter_reason or "empty_target_text"
    if target_script == "unknown":
        filter_reason = filter_reason or "unknown_script"
    if duration_ms is not None:
        seconds = duration_ms / 1000.0
        if seconds < min_duration_sec:
            filter_reason = filter_reason or "too_short_audio"
        elif seconds > max_duration_sec:
            filter_reason = filter_reason or "too_long_audio"

    prepared = {
        "client_id": row.get("client_id", ""),
        "path": rel_path,
        "audio_path": build_audio_path(locale, rel_path) if rel_path else "",
        "sentence": target_text,
        "target_text": target_text,
        "language_label": locale,
        "target_script": target_script,
        "romanization_text": romanization_text,
        "raw_sentence": raw_sentence,
        "filter_reason": filter_reason,
        "source_locale": locale,
        "source_split": split,
        "duration_ms": "" if duration_ms is None else str(duration_ms),
        "up_votes": row.get("up_votes", ""),
        "down_votes": row.get("down_votes", ""),
        "age": row.get("age", ""),
        "gender": row.get("gender", ""),
        "accent": row.get("accent", ""),
        "accents": row.get("accents", ""),
        "variant": row.get("variant", ""),
        "segment": row.get("segment", ""),
    }
    return PreparedRow(row=prepared, keep=not filter_reason)


def read_split_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def resolve_output_audio_path(
    *,
    data_root: Path,
    content_root: Path,
    locale: str,
    rel_path: str,
) -> str:
    if not rel_path:
        return ""
    source = content_root / "clips" / rel_path
    try:
        return str(source.resolve().relative_to(data_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(source).replace("\\", "/")


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(REQUIRED_OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_datasets(args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    split_names = tuple(str(split) for split in args.splits)
    show_progress = not bool(getattr(args, "no_progress", False))
    merged_rows = {split: [] for split in split_names}
    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "locales": {},
        "splits": {},
        "required_fields": list(REQUIRED_OUTPUT_FIELDS),
    }

    locale_iter = tqdm(
        args.locales,
        desc="處理語言資料集",
        unit="locale",
        disable=not show_progress,
    )
    for locale in locale_iter:
        locale_iter.set_postfix_str(str(locale))
        locale_root = data_root / locale
        content_root = find_dataset_content_root(locale_root, split_names)
        durations = read_duration_index(content_root)
        locale_summary = {
            "root": str(locale_root),
            "content_root": str(content_root),
            "duration_index": str(content_root / "clip_durations.tsv")
            if durations
            else "",
            "splits": {},
        }
        split_iter = tqdm(
            split_names,
            desc=f"{locale} 切分",
            unit="split",
            leave=False,
            disable=not show_progress,
        )
        for split in split_iter:
            split_iter.set_postfix_str(split)
            split_path = content_root / f"{split}.tsv"
            raw_rows = read_split_rows(split_path)
            kept: list[dict[str, str]] = []
            rejected: list[dict[str, str]] = []
            filter_counts: dict[str, int] = {}
            script_counts: dict[str, int] = {}

            row_iter = tqdm(
                raw_rows,
                desc=f"{locale}/{split} 樣本",
                unit="row",
                leave=False,
                disable=not show_progress or not raw_rows,
            )
            for row in row_iter:
                rel_path = normalize_text(row.get("path", ""))
                prepared = prepare_row(
                    locale=locale,
                    split=split,
                    row=row,
                    duration_ms=durations.get(rel_path),
                    min_duration_sec=float(args.min_duration_sec),
                    max_duration_sec=float(args.max_duration_sec),
                    keep_romanized_only=bool(args.keep_romanized_only),
                )
                if rel_path:
                    prepared.row["audio_path"] = resolve_output_audio_path(
                        data_root=data_root,
                        content_root=content_root,
                        locale=locale,
                        rel_path=rel_path,
                    )
                script = prepared.row["target_script"]
                script_counts[script] = script_counts.get(script, 0) + 1
                reason = prepared.row["filter_reason"]
                if reason:
                    filter_counts[reason] = filter_counts.get(reason, 0) + 1
                    rejected.append(prepared.row)
                else:
                    kept.append(prepared.row)
                    merged_rows[split].append(prepared.row)

            if raw_rows:
                write_tsv(content_root / f"{split}_prepared.tsv", kept)
                write_tsv(content_root / f"{split}_rejected.tsv", rejected)

            locale_summary["splits"][split] = {
                "source": str(split_path),
                "rows": len(raw_rows),
                "kept": len(kept),
                "rejected": len(rejected),
                "target_script_counts": script_counts,
                "filter_reason_counts": filter_counts,
                "prepared_tsv": str(content_root / f"{split}_prepared.tsv")
                if raw_rows
                else "",
                "rejected_tsv": str(content_root / f"{split}_rejected.tsv")
                if raw_rows
                else "",
            }
        summary["locales"][locale] = locale_summary

    merge_iter = tqdm(
        merged_rows.items(),
        desc="寫出整併 TSV",
        unit="split",
        disable=not show_progress,
    )
    for split, rows in merge_iter:
        merge_iter.set_postfix_str(str(split))
        write_tsv(output_dir / f"{split}.tsv", rows)
        summary["splits"][split] = {
            "path": str(output_dir / f"{split}.tsv"),
            "rows": len(rows),
        }

    write_summary(output_dir / "prepare_summary.json", summary)
    return summary


def run_self_test() -> None:
    examples = {
        "竹仔籃（Tik-á-nâ）": ("竹仔籃", "Tik-á-nâ", "han_with_romanization"),
        "竹坑口（Tik-khinn-kháu | Tek-khi-kháu）": (
            "竹坑口",
            "Tik-khinn-kháu | Tek-khi-kháu",
            "han_with_romanization",
        ),
        "公雞（雞公仔）": ("公雞", "", "han"),
        "浮筒仔（）": ("浮筒仔", "", "han"),
        "瀧（たき）（thá-khih）": ("瀧", "thá-khih", "han_with_romanization"),
        "摳門（kho̍k-仔頭）": ("摳門", "kho̍k-仔頭", "han_with_romanization"),
        "咱這月日的生產目標（Lán tsit gue̍h-ji̍t ê sing-sán bo̍k-piau": (
            "咱這月日的生產目標",
            "",
            "han",
        ),
        "啞口興講話，é-káu hìng kóng-uē tshenn-mê hìng pua̍h-pue）": (
            "啞口興講話",
            "é-káu hìng kóng-uē tshenn-mê hìng pua̍h-pue",
            "han_with_romanization",
        ),
        "需要認真考慮": ("需要認真考慮", "", "traditional_han"),
    }
    for raw, expected in examples.items():
        if raw == "需要認真考慮":
            actual = (*prepare_zh_tw_sentence(raw),)
            expected_zh = (expected[0], expected[2])
            if actual != expected_zh:
                raise AssertionError(f"zh-TW 清理不符合預期: {raw} -> {actual}")
            continue
        actual = split_nan_tw_sentence(raw)
        if actual != expected:
            raise AssertionError(f"nan-tw 清理不符合預期: {raw} -> {actual}")
    print("前處理文字規則自我檢查通過。")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    summary = prepare_datasets(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
