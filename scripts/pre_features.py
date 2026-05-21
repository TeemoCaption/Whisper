#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config
from whisper_tw.data import build_audio_augmentor, precompute_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="預先計算 Whisper 特徵快取。")
    parser.add_argument("--config", required=True, help="設定檔路徑。")
    parser.add_argument(
        "--splits",
        nargs="+",
        help="要建立快取的資料切分，預設使用 train/dev/test。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆寫既有快取。")
    parser.add_argument("--max-samples", type=int, help="每個 split 只處理前 N 筆。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    cache_cfg = data_cfg.get("feature_cache", {})
    cache_root = cache_cfg.get("root")
    if not cache_root:
        raise RuntimeError("data.feature_cache.root 未設定。")

    splits = args.splits or [
        data_cfg.get("train_split", "train"),
        data_cfg.get("dev_split", "dev"),
        data_cfg.get("test_split", "test"),
    ]
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config["model"]["whisper_name"]
    )
    train_split = data_cfg.get("train_split", "train")
    train_augmentor = build_audio_augmentor(
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        config=data_cfg.get("audio_augmentation"),
    )
    total_written = 0
    total_skipped = 0
    for split in splits:
        num_variants = (
            int(cache_cfg.get("train_variants", 1))
            if split == train_split
            else 1
        )
        written, skipped = precompute_feature_cache(
            data_root=data_cfg["root"],
            split=split,
            cache_root=cache_root,
            feature_extractor=feature_extractor,
            sample_rate=int(data_cfg.get("sample_rate", 16000)),
            max_audio_seconds=float(data_cfg.get("max_audio_seconds", 30.0)),
            audio_augmentor=train_augmentor if split == train_split else None,
            num_variants=num_variants,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
        )
        print(
            f"split={split} variants={num_variants} "
            f"written={written} skipped={skipped}"
        )
        total_written += written
        total_skipped += skipped
    print(f"cache_root={cache_root}")
    print(f"total_written={total_written}")
    print(f"total_skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
