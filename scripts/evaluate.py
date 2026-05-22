#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config, resolve_device
from whisper_tw.char_vocab import CharacterVocab
from whisper_tw.metrics import character_error_rate
from whisper_tw.training import build_components, configure_training_runtime


def decode_ctc(ids: list[int], character_vocab: CharacterVocab) -> str:
    collapsed: list[int] = []
    previous_id: int | None = None
    for token_id in ids:
        if token_id != previous_id and token_id != character_vocab.blank_id:
            collapsed.append(token_id)
        previous_id = token_id
    return character_vocab.decode(collapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="評估 Whisper-TW 模型。")
    parser.add_argument("--config", required=True, help="設定檔路徑。")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="評估資料切分。",
    )
    parser.add_argument("--checkpoint", help="模型 checkpoint 路徑。")
    parser.add_argument("--max-samples", type=int, help="只評估前 N 筆樣本。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(resolve_device(config))
    configure_training_runtime(config.get("training", {}), device)
    _tokenizer, dataset, collator, model = build_components(
        config, args.split, args.max_samples
    )
    character_vocab = CharacterVocab.build_from_config(config)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=int(
            config["training"].get("eval_batch_size", config["training"]["batch_size"])
        ),
        shuffle=False,
        collate_fn=collator,
    )
    predictions: list[str] = []
    references: list[str] = []
    total_audio = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            generated = model.generate_ctc(input_features=input_features)
            predictions.extend(
                decode_ctc(row.tolist(), character_vocab) for row in generated.cpu()
            )
            references.extend(batch["texts"])
            total_audio += input_features.size(0)

    elapsed = time.perf_counter() - start
    cer = character_error_rate(predictions, references)
    print(f"split={args.split}")
    print(f"samples={len(references)}")
    print(f"cer={cer:.4f}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"seconds_per_sample={elapsed / max(total_audio, 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
