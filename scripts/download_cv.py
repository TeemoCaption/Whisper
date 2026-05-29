#!/usr/bin/env python3
"""透過 Mozilla Data Collective 官方介面下載 Common Voice 資料集。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
import sys
from pathlib import Path
from urllib import error, parse, request

DEFAULT_BASE_URL = "https://mozilladatacollective.com/api"
DEFAULT_DATASET_ID = "cmn2g7eaj01fio10769r1m96n"
DEFAULT_OUTPUT_ROOT = Path("data")
DATASET_SPECS = (
    {
        "name": "zh-TW",
        "dataset_id": DEFAULT_DATASET_ID,
        "output_subdir": "zh-TW",
    },
    {
        "name": "nan-tw",
        "dataset_id": "cmn2cyd8901jemm0738nubysq",
        "output_subdir": "nan-tw",
    },
)
CHUNK_SIZE = 1024 * 1024
STAT_SPLITS = ("train", "dev", "test")
STAT_SUMMARY_FILENAME = "duration_summary.json"
STAT_LOG_FILENAME = "duration_summary.log"
DATASET_RECORD_SEPARATOR = "-" * 72
AUDIO_STATS_CACHE_FILENAME = "audio_duration_cache.json"


class DownloadError(RuntimeError):
    """下載流程失敗。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="透過 Mozilla Data Collective API 下載 Common Voice 資料集。",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MDC_API_KEY")
        or os.environ.get("MOZILLA_DATA_COLLECTIVE_API_KEY"),
        help="API 金鑰。未提供時會先讀 MDC_API_KEY 或 MOZILLA_DATA_COLLECTIVE_API_KEY，仍沒有則會提示輸入。",
    )
    parser.add_argument(
        "--api-key-file",
        help="讀取 API 金鑰的本機文字檔路徑；建議放在不提交到版本控制的位置。",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API 基底網址。預設值符合官方文件。",
    )
    parser.add_argument(
        "--output-dir",
        help="輸出根資料夾。批次模式下會自動建立各資料集子資料夾。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若輸出檔已存在，直接覆蓋。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="單次請求逾時秒數。",
    )
    return parser.parse_args()


def build_download_request(base_url: str, dataset_id: str, api_key: str) -> request.Request:
    endpoint = (
        f"{base_url.rstrip('/')}/datasets/"
        f"{parse.quote(dataset_id, safe='')}/download"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Whisper-TW-Downloader/1.0",
    }
    return request.Request(endpoint, data=b"", headers=headers, method="POST")


def get_download_info(
    base_url: str,
    dataset_id: str,
    api_key: str,
    timeout: int,
) -> dict:
    req = build_download_request(base_url, dataset_id, api_key)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 401:
            message = "驗證失敗，請確認金鑰是否正確。"
        elif exc.code == 403:
            message = (
                "伺服器拒絕下載，通常代表尚未在網站上同意資料集條款，"
                "或目前帳號沒有下載權限。"
            )
        elif exc.code == 404:
            message = "找不到指定的資料集。"
        elif exc.code == 429:
            message = "已達到請求限制，請稍後再試。"
        else:
            message = f"取得下載連結失敗，HTTP 狀態碼為 {exc.code}。"
        raise DownloadError(f"{message}\n伺服器回應：{body}") from exc
    except error.URLError as exc:
        raise DownloadError(f"無法連上 API：{exc.reason}") from exc

    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DownloadError(f"API 回傳不是有效的 JSON：{raw}") from exc

    download_url = info.get("downloadUrl")
    if not download_url:
        raise DownloadError("API 回傳內容缺少 downloadUrl。")
    return info


def resolve_output_path(
    info: dict,
    dataset_id: str,
    output_dir: Path,
) -> Path:
    filename = info.get("filename") or f"{dataset_id}.tar.gz"
    return output_dir / filename


def resolve_extraction_dir(destination: Path) -> Path:
    return destination.parent


def resolve_api_key(api_key: str | None, api_key_file: str | None = None) -> str:
    if api_key:
        key = api_key.strip()
        if key:
            return key

    if api_key_file:
        key_path = Path(api_key_file)
        try:
            key = key_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DownloadError(f"無法讀取 API 金鑰檔案：{key_path}") from exc
        if key:
            return key
        raise DownloadError(f"API 金鑰檔案是空的：{key_path}")

    if not sys.stdin.isatty():
        raise DownloadError(
            "缺少 API 金鑰，且目前不是互動式終端機，無法提示輸入。"
            "請加上 --api-key、--api-key-file，或設定環境變數 MDC_API_KEY。"
        )

    try:
        key = input("請輸入 API 金鑰：").strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise DownloadError("未輸入 API 金鑰。") from exc

    if not key:
        raise DownloadError("未輸入 API 金鑰。")

    return key


def format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "未知"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def parse_expected_checksum(checksum: str | None) -> tuple[str | None, str | None]:
    if not checksum or ":" not in checksum:
        return None, None
    algorithm, digest = checksum.split(":", 1)
    return algorithm.lower(), digest.lower()


def get_tar_members_with_optional_root_strip(archive: tarfile.TarFile) -> list[tuple[tarfile.TarInfo, Path]]:
    members = archive.getmembers()
    normalized_parts: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
    for member in members:
        member_path = Path(member.name)
        if member_path.is_absolute():
            raise DownloadError(f"壓縮檔內含絕對路徑，已拒絕解壓：{member.name}")
        parts = tuple(part for part in member_path.parts if part not in ("", "."))
        if any(part == ".." for part in parts):
            raise DownloadError(f"壓縮檔內含可疑路徑，已拒絕解壓：{member.name}")
        normalized_parts.append((member, parts))

    root_name: str | None = None
    has_root_file = False
    for _, parts in normalized_parts:
        if not parts:
            continue
        if len(parts) == 1:
            has_root_file = True
            break
        if root_name is None:
            root_name = parts[0]
        elif root_name != parts[0]:
            root_name = ""
            break

    strip_root = bool(root_name) and not has_root_file

    resolved: list[tuple[tarfile.TarInfo, Path]] = []
    for member, parts in normalized_parts:
        if not parts:
            continue
        if strip_root and parts[0] == root_name:
            parts = parts[1:]
        if not parts:
            continue
        resolved.append((member, Path(*parts)))
    return resolved


def extract_archive(archive_path: Path, target_dir: Path, overwrite: bool) -> None:
    if not archive_path.exists():
        raise DownloadError(f"找不到要解壓的壓縮檔：{archive_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = get_tar_members_with_optional_root_strip(archive)
            total_bytes = sum(member.size for member, _ in members if member.isfile())
            extracted_bytes = 0
            last_percent = -1

            def report_progress() -> None:
                nonlocal last_percent
                if total_bytes <= 0:
                    return
                percent = int(extracted_bytes * 100 / total_bytes)
                if percent != last_percent:
                    print(
                        f"\r  解壓進度：{format_size(extracted_bytes)} / {format_size(total_bytes)} "
                        f"({percent}%)",
                        end="",
                        flush=True,
                    )
                    last_percent = percent

            print("開始解壓縮...")
            for member, relative_path in members:
                destination = target_dir / relative_path
                resolved_destination = destination.resolve(strict=False)
                resolved_target = target_dir.resolve(strict=False)
                if resolved_target != resolved_destination and resolved_target not in resolved_destination.parents:
                    raise DownloadError(f"解壓路徑超出目標資料夾：{member.name}")

                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue

                if member.issym() or member.islnk():
                    raise DownloadError(f"壓縮檔內含連結檔案，已拒絕解壓：{member.name}")

                if member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        if not overwrite:
                            raise DownloadError(f"檔案已存在，請加上 --overwrite：{destination}")
                        if destination.is_dir():
                            raise DownloadError(f"目標位置已存在且是資料夾，無法覆蓋：{destination}")
                        destination.unlink()
                    with archive.extractfile(member) as source, destination.open("wb") as out:
                        if source is None:
                            raise DownloadError(f"無法讀取壓縮檔內的檔案：{member.name}")
                        while True:
                            chunk = source.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            out.write(chunk)
                            extracted_bytes += len(chunk)
                            report_progress()
                    continue

                raise DownloadError(f"不支援的壓縮檔項目：{member.name}")
    except tarfile.TarError as exc:
        raise DownloadError(f"解壓縮失敗：{exc}") from exc

    if total_bytes > 0:
        print()
    archive_path.unlink()


def download_file(
    download_url: str,
    destination: Path,
    expected_size: int | None,
    expected_checksum: str | None,
    timeout: int,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = destination.with_name(destination.name + ".part")
    if temp_path.exists():
        temp_path.unlink()

    req = request.Request(download_url, headers={"User-Agent": "Whisper-TW-Downloader/1.0"})
    sha256 = hashlib.sha256()
    downloaded = 0
    total = expected_size
    last_percent = -1

    try:
        with request.urlopen(req, timeout=timeout) as resp, temp_path.open("wb") as out:
            if total is None:
                content_length = resp.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    total = int(content_length)

            print("開始下載...")
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                sha256.update(chunk)
                downloaded += len(chunk)

                if total:
                    percent = int(downloaded * 100 / total)
                    if percent != last_percent:
                        print(
                            f"\r  進度：{format_size(downloaded)} / {format_size(total)} "
                            f"({percent}%)",
                            end="",
                            flush=True,
                        )
                        last_percent = percent
                else:
                    print(f"\r  已下載：{format_size(downloaded)}", end="", flush=True)

            print()
    except error.URLError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise DownloadError(f"下載檔案時發生網路錯誤：{exc.reason}") from exc

    if expected_size is not None and downloaded != expected_size:
        if temp_path.exists():
            temp_path.unlink()
        raise DownloadError(
            f"下載大小不符，預期 {expected_size} bytes，實際 {downloaded} bytes。"
        )

    algorithm, expected_digest = parse_expected_checksum(expected_checksum)
    if algorithm == "sha256" and expected_digest:
        actual_digest = sha256.hexdigest().lower()
        if actual_digest != expected_digest:
            if temp_path.exists():
                temp_path.unlink()
            raise DownloadError(
                "檢查碼不符，"
                f"預期 sha256:{expected_digest}，實際 sha256:{actual_digest}。"
            )

    if destination.exists() and overwrite:
        destination.unlink()
    temp_path.replace(destination)


def download_or_reuse_archive(
    download_url: str,
    destination: Path,
    expected_size: int | None,
    expected_checksum: str | None,
    timeout: int,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        print(f"已存在壓縮檔，略過下載：{destination}")
        return

    download_file(
        download_url=download_url,
        destination=destination,
        expected_size=expected_size,
        expected_checksum=expected_checksum,
        timeout=timeout,
        overwrite=overwrite,
    )


def read_cached_duration_summary(target_dir: Path) -> dict | None:
    summary_path = target_dir / STAT_SUMMARY_FILENAME
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(summary, dict):
        return None
    normalized = normalize_cached_summary_paths(target_dir, summary)
    if normalized != summary:
        write_duration_summary(target_dir, normalized)
    return normalized


def find_dataset_content_root(target_dir: Path) -> Path:
    split_names = {f"{split}.tsv" for split in STAT_SPLITS}
    if any((target_dir / name).exists() for name in split_names):
        return target_dir

    candidates = sorted(
        {
            path.parent
            for name in split_names
            for path in target_dir.rglob(name)
        },
        key=lambda path: (len(path.relative_to(target_dir).parts), str(path)),
    )
    if candidates:
        return candidates[0]
    return target_dir


def flatten_dataset_root_if_needed(target_dir: Path, expected_root_name: str) -> Path:
    nested_root = target_dir / expected_root_name
    if not nested_root.is_dir():
        return target_dir
    if any((target_dir / f"{split}.tsv").exists() for split in STAT_SPLITS):
        return target_dir
    for item in nested_root.iterdir():
        destination = target_dir / item.name
        if destination.exists():
            continue
        item.rename(destination)
    nested_root.rmdir()
    return target_dir


def write_duration_summary(target_dir: Path, summary: dict) -> None:
    summary_path = target_dir / STAT_SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_display_name(target_dir: Path, fallback_name: str) -> str:
    content_root = find_dataset_content_root(target_dir)
    if content_root != target_dir:
        return content_root.name
    return fallback_name


def normalize_cached_summary_paths(target_dir: Path, summary: dict) -> dict:
    normalized = dict(summary)
    content_root = find_dataset_content_root(target_dir)
    normalized["dataset_dir"] = str(target_dir)
    normalized["content_root"] = str(content_root)
    duration_source = content_root / "clip_durations.tsv"
    if duration_source.exists():
        normalized["duration_source"] = str(duration_source)
    else:
        normalized["duration_source"] = str(content_root / "clips")
    return normalized


def read_audio_duration_cache(target_dir: Path) -> dict[str, float]:
    cache_path = target_dir / AUDIO_STATS_CACHE_FILENAME
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    durations = payload.get("durations_seconds")
    if not isinstance(durations, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in durations.items()
        if isinstance(key, str) and isinstance(value, (int, float))
    }


def write_audio_duration_cache(target_dir: Path, durations: dict[str, float]) -> None:
    cache_path = target_dir / AUDIO_STATS_CACHE_FILENAME
    payload = {
        "durations_seconds": durations,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_split_clip_refs(split_path: Path) -> list[str]:
    if not split_path.exists():
        return []
    with split_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "path" not in (reader.fieldnames or []):
            return []
        return [
            (row.get("path") or "").strip()
            for row in reader
            if (row.get("path") or "").strip()
        ]


def build_clip_split_index(target_dir: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    clip_to_splits: dict[str, list[str]] = {}
    row_counts: dict[str, int] = {}
    for split in STAT_SPLITS:
        clips = read_split_clip_refs(target_dir / f"{split}.tsv")
        row_counts[split] = len(clips)
        for clip in clips:
            clip_to_splits.setdefault(clip, []).append(split)
    return clip_to_splits, row_counts


def summarize_from_duration_index(
    clip_to_splits: dict[str, list[str]],
    row_counts: dict[str, int],
    durations_path: Path,
) -> dict | None:
    duration_ms = {split: 0 for split in STAT_SPLITS}
    matched_counts = {split: 0 for split in STAT_SPLITS}

    with durations_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "clip" not in (reader.fieldnames or []) or "duration[ms]" not in (reader.fieldnames or []):
            print(f"時長索引欄位不符合預期，改用音檔資訊統計：{durations_path}")
            return None
        for row in reader:
            clip = (row.get("clip") or "").strip()
            splits = clip_to_splits.get(clip)
            if not splits:
                continue
            try:
                clip_duration_ms = int(float((row.get("duration[ms]") or "0").strip()))
            except ValueError:
                continue
            for split in splits:
                duration_ms[split] += clip_duration_ms
                matched_counts[split] += 1

    return {
        split: {
            "clips": row_counts[split],
            "matched_clips": matched_counts[split],
            "missing_duration_clips": max(row_counts[split] - matched_counts[split], 0),
            "seconds": duration_ms[split] / 1000.0,
            "hours": duration_ms[split] / 1000.0 / 3600.0,
        }
        for split in STAT_SPLITS
    }


def summarize_from_audio_files(
    summary_target_dir: Path,
    content_root: Path,
    clip_to_splits: dict[str, list[str]],
    row_counts: dict[str, int],
) -> dict:
    try:
        import torchaudio
    except ImportError as exc:
        raise DownloadError("缺少 torchaudio，無法從音檔中繼資訊統計時長。") from exc

    clips_dir = content_root / "clips"
    cached_seconds = read_audio_duration_cache(summary_target_dir)
    duration_seconds = {split: 0.0 for split in STAT_SPLITS}
    matched_counts = {split: 0 for split in STAT_SPLITS}
    updated_cache = dict(cached_seconds)

    for clip, splits in clip_to_splits.items():
        seconds = updated_cache.get(clip)
        if seconds is None:
            audio_path = clips_dir / clip
            if not audio_path.exists():
                continue
            info = torchaudio.info(str(audio_path))
            if not info.sample_rate:
                continue
            seconds = float(info.num_frames) / float(info.sample_rate)
            updated_cache[clip] = seconds
        for split in splits:
            duration_seconds[split] += seconds
            matched_counts[split] += 1

    write_audio_duration_cache(summary_target_dir, updated_cache)
    return {
        split: {
            "clips": row_counts[split],
            "matched_clips": matched_counts[split],
            "missing_duration_clips": max(row_counts[split] - matched_counts[split], 0),
            "seconds": duration_seconds[split],
            "hours": duration_seconds[split] / 3600.0,
        }
        for split in STAT_SPLITS
    }


def summarize_split_durations(target_dir: Path) -> dict | None:
    content_root = find_dataset_content_root(target_dir)

    clip_to_splits, row_counts = build_clip_split_index(content_root)
    if not clip_to_splits:
        print(f"找不到 train/dev/test split，略過時長統計：{target_dir}")
        return None

    durations_path = content_root / "clip_durations.tsv"
    splits = None
    if durations_path.exists():
        splits = summarize_from_duration_index(clip_to_splits, row_counts, durations_path)
    if splits is None:
        splits = summarize_from_audio_files(target_dir, content_root, clip_to_splits, row_counts)

    summary = {
        "dataset_dir": str(target_dir),
        "content_root": str(content_root),
        "duration_source": str(durations_path if durations_path.exists() else content_root / "clips"),
        "splits": splits,
    }
    write_duration_summary(target_dir, summary)
    return summary


def print_duration_summary(dataset_name: str, summary: dict) -> None:
    print(f"時長統計：{dataset_name}")
    for split in STAT_SPLITS:
        item = summary.get("splits", {}).get(split, {})
        hours = float(item.get("hours", 0.0))
        clips = int(item.get("clips", 0))
        missing = int(item.get("missing_duration_clips", 0))
        warning = f"，缺少時長 {missing} 筆" if missing else ""
        print(f"  {split}: {hours:.2f} 小時，{clips} 筆{warning}")
    print(f"  摘要檔：{Path(summary['dataset_dir']) / STAT_SUMMARY_FILENAME}")


def report_dataset_duration(dataset_name: str, target_dir: Path) -> None:
    summary = read_cached_duration_summary(target_dir)
    if summary is None:
        summary = summarize_split_durations(target_dir)
    if summary is not None:
        print_duration_summary(resolve_display_name(target_dir, dataset_name), summary)


def dataset_ready(target_dir: Path) -> bool:
    if not target_dir.exists():
        return False
    content_root = find_dataset_content_root(target_dir)
    return any((content_root / f"{split}.tsv").exists() for split in STAT_SPLITS)


def resolve_output_root(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return DEFAULT_OUTPUT_ROOT


def get_legacy_target_dirs(dataset_spec: dict[str, str], output_root: Path) -> list[Path]:
    if dataset_spec["dataset_id"] == "cmn2cyd8901jemm0738nubysq":
        return [output_root / "taigi"]
    return []


def resolve_dataset_target_dir(dataset_spec: dict[str, str], output_root: Path) -> Path:
    preferred = output_root / dataset_spec["output_subdir"]
    if preferred.exists():
        return preferred
    for legacy_dir in get_legacy_target_dirs(dataset_spec, output_root):
        if legacy_dir.exists():
            return legacy_dir
    return preferred


def migrate_dataset_dir(dataset_spec: dict[str, str], output_root: Path) -> Path:
    preferred = output_root / dataset_spec["output_subdir"]
    if preferred.exists():
        return preferred
    for legacy_dir in get_legacy_target_dirs(dataset_spec, output_root):
        if legacy_dir.exists():
            legacy_dir.rename(preferred)
            return preferred
    return preferred


def append_duration_log(log_path: Path, dataset_name: str, summary: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{DATASET_RECORD_SEPARATOR}\n")
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"dataset_dir: {summary.get('dataset_dir', '')}\n")
        f.write(f"duration_source: {summary.get('duration_source', '')}\n")
        for split in STAT_SPLITS:
            item = summary.get("splits", {}).get(split, {})
            f.write(
                f"{split}: {float(item.get('hours', 0.0)):.4f} hours, "
                f"{int(item.get('clips', 0))} clips, "
                f"{int(item.get('missing_duration_clips', 0))} missing_duration\n"
            )
        f.write("\n")


def download_dataset(
    dataset_spec: dict[str, str],
    api_key: str | None,
    base_url: str,
    output_root: Path,
    timeout: int,
    overwrite: bool,
    duration_log_path: Path,
    api_key_file: str | None,
) -> Path | None:
    target_dir = migrate_dataset_dir(dataset_spec, output_root)
    flatten_dataset_root_if_needed(target_dir, dataset_spec["output_subdir"])
    if dataset_ready(target_dir) and not overwrite:
        print(f"已存在資料集，略過：{resolve_display_name(target_dir, dataset_spec['name'])} ({target_dir})")
        report_dataset_duration(dataset_spec["name"], target_dir)
        summary = read_cached_duration_summary(target_dir)
        if summary is not None:
            append_duration_log(duration_log_path, resolve_display_name(target_dir, dataset_spec["name"]), summary)
        return None

    resolved_api_key = resolve_api_key(api_key, api_key_file)
    info = get_download_info(base_url, dataset_spec["dataset_id"], resolved_api_key, timeout)
    download_url = info["downloadUrl"]
    expected_size = info.get("sizeBytes")
    if isinstance(expected_size, str) and expected_size.isdigit():
        expected_size = int(expected_size)
    elif not isinstance(expected_size, int):
        expected_size = None

    target_dir.mkdir(parents=True, exist_ok=True)
    destination = resolve_output_path(
        info,
        dataset_spec["dataset_id"],
        target_dir,
    )

    print(f"資料集：{resolve_display_name(target_dir, dataset_spec['name'])}")
    print(f"輸出位置：{destination}")
    if expected_size is not None:
        print(f"預估大小：{format_size(expected_size)}")

    download_or_reuse_archive(
        download_url=download_url,
        destination=destination,
        expected_size=expected_size,
        expected_checksum=info.get("checksum"),
        timeout=timeout,
        overwrite=overwrite,
    )

    extraction_dir = resolve_extraction_dir(destination)
    extract_archive(destination, extraction_dir, overwrite)
    print(f"完成：{extraction_dir}")
    report_dataset_duration(dataset_spec["name"], extraction_dir)
    summary = read_cached_duration_summary(extraction_dir)
    if summary is not None:
        append_duration_log(duration_log_path, resolve_display_name(extraction_dir, dataset_spec["name"]), summary)
    return extraction_dir


def main() -> int:
    args = parse_args()

    try:
        output_root = resolve_output_root(args.output_dir)
        duration_log_path = output_root / STAT_LOG_FILENAME
        if duration_log_path.exists():
            duration_log_path.unlink()
        for index, dataset_spec in enumerate(DATASET_SPECS):
            if index > 0:
                print(f"\n{DATASET_RECORD_SEPARATOR}")
            download_dataset(
                dataset_spec=dataset_spec,
                api_key=args.api_key,
                base_url=args.base_url,
                output_root=output_root,
                timeout=args.timeout,
                overwrite=args.overwrite,
                duration_log_path=duration_log_path,
                api_key_file=args.api_key_file,
            )
        print(
            "\n資料集下載或確認完成後，可執行前處理："
            f"\n  python scripts/prepare_cv.py --data-root {output_root}"
            "\n並檢查欄位："
            "\n  python scripts/check_cv.py --strict"
        )
    except DownloadError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
