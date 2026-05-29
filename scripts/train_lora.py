#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_baseline import main


if __name__ == "__main__":
    main(
        default_config="configs/config.yaml",
        description="訓練 Whisper-medium 低秩適應模型。",
    )
