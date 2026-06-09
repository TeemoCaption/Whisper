#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_baselines import ensure_router_jobs, load_structured_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批次訓練不同 Whisper 型號的對比式查詢路由器。"
    )
    parser.add_argument("--config", required=True, help="專案設定檔路徑。")
    parser.add_argument(
        "--baselines-config",
        default="configs/baselines.yaml",
        help="包含 router_jobs 的設定檔路徑。",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        help="覆寫訓練 DataLoader worker 數；共享記憶體不足時建議設為 0。",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="即使已存在權重也重新訓練全部路由器工作。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_cfg = load_structured_config(args.baselines_config)
    ensure_router_jobs(
        suite_cfg=suite_cfg,
        config_path=args.config,
        dataloader_num_workers=args.dataloader_num_workers,
        redo_finetune=args.redo,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
