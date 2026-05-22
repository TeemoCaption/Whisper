#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config, resolve_device
from whisper_tw.char_vocab import CharacterVocab
from whisper_tw.metrics import character_error_rate
from whisper_tw.training import (
    build_components,
    build_dataloader_kwargs,
    configure_training_runtime,
    get_amp_dtype,
    use_mixed_precision,
)


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
    parser.add_argument("--batch-size", type=int, help="覆蓋設定檔的評估批次大小。")
    return parser.parse_args()


def synchronize_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(resolve_device(config))
    train_cfg = config.get("training", {})
    configure_training_runtime(train_cfg, device)
    _tokenizer, dataset, collator, model = build_components(
        config, args.split, args.max_samples
    )
    character_vocab = CharacterVocab.build_from_config(config)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    batch_size = int(
        args.batch_size
        or train_cfg.get("eval_batch_size", train_cfg["batch_size"])
    )
    dataloader_kwargs = build_dataloader_kwargs(train_cfg, device, split="eval")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        **dataloader_kwargs,
    )
    predictions: list[str] = []
    references: list[str] = []
    total_audio = 0
    total_inference_seconds = 0.0
    amp_enabled = use_mixed_precision(train_cfg, device)
    amp_dtype = get_amp_dtype(train_cfg, device)
    print(
        f"eval_start split={args.split} samples={len(dataset)} "
        f"batch_size={batch_size} device={device} "
        f"num_workers={dataloader_kwargs.get('num_workers', 0)}",
        flush=True,
    )
    start = time.perf_counter()
    with torch.inference_mode():
        progress = tqdm(
            dataloader,
            desc=f"eval {args.split}",
            dynamic_ncols=True,
        )
        for batch in progress:
            input_features = batch["input_features"].to(device)
            synchronize_for_timing(device)
            batch_start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled and amp_dtype is not None,
            ):
                generated = model.generate_ctc(
                    input_features=input_features,
                    use_corrector=False,
                )
                batch_predictions = [
                    decode_ctc(row.tolist(), character_vocab)
                    for row in generated.cpu()
                ]
            synchronize_for_timing(device)
            batch_elapsed = time.perf_counter() - batch_start
            total_inference_seconds += batch_elapsed
            predictions.extend(batch_predictions)
            references.extend(batch["texts"])
            total_audio += input_features.size(0)
            running_cer = character_error_rate(predictions, references)
            progress.set_postfix(
                samples=total_audio,
                cer=f"{running_cer:.4f}",
                infer_s_per_sample=f"{total_inference_seconds / max(total_audio, 1):.3f}",
            )

    elapsed = time.perf_counter() - start
    cer = character_error_rate(predictions, references)
    print(f"split={args.split}")
    print(f"samples={len(references)}")
    print(f"cer={cer:.4f}")
    print(f"inference_seconds={total_inference_seconds:.3f}")
    print(f"inference_seconds_per_sample={total_inference_seconds / max(total_audio, 1):.3f}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"seconds_per_sample={elapsed / max(total_audio, 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
