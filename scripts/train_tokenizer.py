#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from whisper_tw.config import load_config
from whisper_tw.data import iter_tokenizer_sentences, resolve_common_voice_split_source
from whisper_tw.tokenizer import train_sentencepiece_from_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="訓練 Whisper-TW 的 SentencePiece tokenizer。")
    parser.add_argument("--config", required=True, help="設定檔路徑。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    tok_cfg = config["tokenizer"]
    model_path = Path(tok_cfg["model_path"])
    corpus_path = model_path.with_suffix(".corpus.txt")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    splits = list(config["data"]["tokenizer_text_splits"])
    split_sources = {
        split: resolve_common_voice_split_source(config["data"], split)
        for split in splits
    }
    count = 0
    with corpus_path.open("w", encoding="utf-8", newline="\n") as f:
        for sentence in iter_tokenizer_sentences(
            config["data"]["root"],
            splits,
            split_sources,
            config["data"].get("text_normalization"),
        ):
            f.write(sentence.replace("\n", " ").strip() + "\n")
            count += 1

    if count == 0:
        raise RuntimeError("沒有可用的 tokenizer 訓練句子。")

    model_prefix = model_path.with_suffix("")
    train_sentencepiece_from_corpus(
        corpus_path=corpus_path,
        model_prefix=model_prefix,
        vocab_size=int(tok_cfg["vocab_size"]),
        model_type=str(tok_cfg["model_type"]),
        character_coverage=float(tok_cfg["character_coverage"]),
        pad_id=int(tok_cfg.get("pad_id", 0)),
        unk_id=int(tok_cfg.get("unk_id", 1)),
        bos_id=int(tok_cfg.get("bos_id", 2)),
        eos_id=int(tok_cfg.get("eos_id", 3)),
    )
    print(f"sentences={count}")
    print(f"model={model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
