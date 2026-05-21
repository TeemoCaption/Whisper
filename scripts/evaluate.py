#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config, resolve_device
from whisper_tw.metrics import character_error_rate
from whisper_tw.text_normalization import build_text_normalizer
from whisper_tw.training import build_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="評估 Whisper-TW 模型。")
    parser.add_argument("--config", required=True, help="設定檔路徑。")
    parser.add_argument("--checkpoint", help="模型 checkpoint 路徑。")
    parser.add_argument("--max-samples", type=int, help="只評估前 N 筆樣本。")
    parser.add_argument("--audio", help="辨識單一音訊檔，支援超過 30 秒的長音訊。")
    return parser.parse_args()


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def transcribe_audio(args: argparse.Namespace, config: dict) -> int:
    device = torch.device(resolve_device(config))
    tokenizer, _, collator, model = build_components(
        config, "test", None
    )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    sample_rate = int(config["data"].get("sample_rate", 16000))
    waveform = load_audio(args.audio, sample_rate)
    with torch.no_grad():
        features = collator.feature_extractor(
            [waveform.numpy()],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        generated = model.generate_greedy(
            input_features=features.input_features.to(device),
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
        )
        print(tokenizer.decode(generated[0].cpu().tolist()))
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.audio:
        return transcribe_audio(args, config)
    device = torch.device(resolve_device(config))
    test_split = config["data"].get("test_split", "test")
    tokenizer, dataset, collator, model = build_components(
        config, test_split, args.max_samples
    )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )
    text_normalizer = build_text_normalizer(config["data"].get("text_normalization"))
    predictions: list[str] = []
    references: list[str] = []
    total_audio = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            input_features = batch["input_features"].to(device)
            generated = model.generate_greedy(
                input_features=input_features,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
            )
            predictions.extend(
                (
                    text_normalizer(tokenizer.decode(row.tolist()))
                    if text_normalizer.enabled
                    else tokenizer.decode(row.tolist())
                )
                for row in generated.cpu()
            )
            references.extend(batch["texts"])
            total_audio += input_features.size(0)

    elapsed = time.perf_counter() - start
    cer = character_error_rate(predictions, references)
    print(f"split={test_split}")
    print(f"samples={len(references)}")
    print(f"cer={cer:.4f}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"seconds_per_sample={elapsed / max(total_audio, 1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
