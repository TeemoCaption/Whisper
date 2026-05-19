#!/usr/bin/env python3
"""透過 Mozilla Data Collective 官方介面下載 Common Voice 資料集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib import error, parse, request

DEFAULT_BASE_URL = "https://mozilladatacollective.com/api"
DEFAULT_DATASET_ID = "cmn2g7eaj01fio10769r1m96n"
CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """下載流程失敗。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="透過 Mozilla Data Collective API 下載 Common Voice 資料集。",
    )
    parser.add_argument(
        "dataset_id",
        nargs="?",
        default=DEFAULT_DATASET_ID,
        help="資料集識別碼，預設為 cmn2g7eaj01fio10769r1m96n。",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MDC_API_KEY")
        or os.environ.get("MOZILLA_DATA_COLLECTIVE_API_KEY"),
        help="API 金鑰。未提供時會先讀 MDC_API_KEY 或 MOZILLA_DATA_COLLECTIVE_API_KEY，仍沒有則會提示輸入。",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API 基底網址。預設值符合官方文件。",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-file",
        help="輸出檔案完整路徑。若未指定，會使用 API 回傳的檔名。",
    )
    output_group.add_argument(
        "--output-dir",
        help="輸出資料夾。檔名會使用 API 回傳的檔名。",
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
    output_file: str | None,
    output_dir: str | None,
) -> Path:
    filename = info.get("filename") or f"{dataset_id}.tar.gz"
    if output_file:
        return Path(output_file)
    if output_dir:
        return Path(output_dir) / filename
    return Path(filename)


def resolve_api_key(api_key: str | None) -> str:
    if api_key:
        key = api_key.strip()
        if key:
            return key

    if not sys.stdin.isatty():
        raise DownloadError(
            "缺少 API 金鑰，且目前不是互動式終端機，無法提示輸入。"
            "請加上 --api-key 或設定環境變數 MDC_API_KEY。"
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


def download_file(
    download_url: str,
    destination: Path,
    expected_size: int | None,
    expected_checksum: str | None,
    timeout: int,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise DownloadError(f"輸出檔已存在，請加上 --overwrite 或改用其他路徑：{destination}")

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


def main() -> int:
    args = parse_args()

    try:
        api_key = resolve_api_key(args.api_key)
        info = get_download_info(args.base_url, args.dataset_id, api_key, args.timeout)
        download_url = info["downloadUrl"]
        expected_size = info.get("sizeBytes")
        if isinstance(expected_size, str) and expected_size.isdigit():
            expected_size = int(expected_size)
        elif not isinstance(expected_size, int):
            expected_size = None

        destination = resolve_output_path(
            info,
            args.dataset_id,
            args.output_file,
            args.output_dir,
        )

        print(f"資料集：{info.get('filename', args.dataset_id)}")
        print(f"輸出位置：{destination}")
        if expected_size is not None:
            print(f"預估大小：{format_size(expected_size)}")

        download_file(
            download_url=download_url,
            destination=destination,
            expected_size=expected_size,
            expected_checksum=info.get("checksum"),
            timeout=args.timeout,
            overwrite=args.overwrite,
        )
    except DownloadError as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    print(f"完成：{destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
