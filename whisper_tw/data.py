from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import WhisperFeatureExtractor

from .bopomofo import BopomofoVocab
from .text_normalization import TextNormalizer, build_text_normalizer
from .tokenizer import SentencePieceTextTokenizer


@dataclass(frozen=True)
class CommonVoiceSample:
    audio_path: Path
    rel_path: str
    text: str


@dataclass
class AudioAugmentor:
    sample_rate: int
    speed_factors: tuple[float, ...]
    speed_prob: float
    noise_prob: float
    min_snr_db: float
    max_snr_db: float
    gain_prob: float
    min_gain_db: float
    max_gain_db: float
    shift_prob: float
    max_shift_ms: float

    def __call__(
        self,
        waveform: torch.Tensor,
        rng: random.Random | None = None,
        torch_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        rng = rng or random
        augmented = waveform
        if self.speed_factors and rng.random() < self.speed_prob:
            speed = rng.choice(self.speed_factors)
            if abs(speed - 1.0) > 1e-4:
                target_rate = max(int(round(self.sample_rate * speed)), 1)
                augmented = torchaudio.functional.resample(
                    augmented,
                    self.sample_rate,
                    target_rate,
                )
        if self.gain_prob > 0.0 and rng.random() < self.gain_prob:
            gain_db = rng.uniform(self.min_gain_db, self.max_gain_db)
            augmented = augmented * (10 ** (gain_db / 20.0))
        if self.noise_prob > 0.0 and rng.random() < self.noise_prob:
            signal_rms = float(augmented.pow(2).mean().sqrt().item())
            signal_rms = max(signal_rms, 1e-4)
            snr_db = rng.uniform(self.min_snr_db, self.max_snr_db)
            noise_rms = signal_rms / (10 ** (snr_db / 20.0))
            noise = torch.randn(
                augmented.shape,
                generator=torch_generator,
                device=augmented.device,
                dtype=augmented.dtype,
            )
            augmented = augmented + noise * noise_rms
        if self.max_shift_ms > 0.0 and rng.random() < self.shift_prob:
            max_shift = int(self.sample_rate * self.max_shift_ms / 1000.0)
            shift = rng.randint(-max_shift, max_shift)
            if shift != 0:
                rolled = torch.roll(augmented, shifts=shift, dims=0)
                if shift > 0:
                    rolled[:shift] = 0
                else:
                    rolled[shift:] = 0
                augmented = rolled
        return augmented.clamp_(-1.0, 1.0)


def build_audio_augmentor(
    sample_rate: int,
    config: dict[str, Any] | None,
) -> AudioAugmentor | None:
    cfg = config or {}
    if not bool(cfg.get("enabled", False)):
        return None
    speed_factors = tuple(float(value) for value in cfg.get("speed_factors", [0.95, 1.0, 1.05]))
    return AudioAugmentor(
        sample_rate=sample_rate,
        speed_factors=speed_factors,
        speed_prob=float(cfg.get("speed_prob", 0.5)),
        noise_prob=float(cfg.get("noise_prob", 0.35)),
        min_snr_db=float(cfg.get("min_snr_db", 18.0)),
        max_snr_db=float(cfg.get("max_snr_db", 30.0)),
        gain_prob=float(cfg.get("gain_prob", 0.3)),
        min_gain_db=float(cfg.get("min_gain_db", -3.0)),
        max_gain_db=float(cfg.get("max_gain_db", 3.0)),
        shift_prob=float(cfg.get("shift_prob", 0.2)),
        max_shift_ms=float(cfg.get("max_shift_ms", 120.0)),
    )


def resolve_common_voice_split_source(
    data_cfg: dict[str, Any],
    split: str,
) -> str | Path:
    split_paths = data_cfg.get("split_paths", {})
    if isinstance(split_paths, dict):
        split_source = split_paths.get(split)
        if split_source:
            return split_source

    split_tsv_key = f"{split}_tsv"
    if data_cfg.get(split_tsv_key):
        return data_cfg[split_tsv_key]

    split_name_key = f"{split}_split"
    if data_cfg.get(split_name_key):
        return data_cfg[split_name_key]

    return split


def read_common_voice_split(
    data_root: str | Path,
    split: str | Path,
) -> list[CommonVoiceSample]:
    root = Path(data_root)
    split_path = Path(split)
    if split_path.exists() or split_path.suffix == ".tsv":
        tsv_path = split_path
    else:
        tsv_path = root / f"{split}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing Common Voice split: {tsv_path}")

    samples: list[CommonVoiceSample] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("sentence") or "").strip()
            rel_path = (row.get("path") or "").strip()
            if not text or not rel_path:
                continue
            samples.append(
                CommonVoiceSample(
                    audio_path=root / "clips" / rel_path,
                    rel_path=rel_path,
                    text=text,
                )
            )
    return samples


def build_feature_cache_path(
    cache_root: str | Path,
    split: str,
    rel_path: str,
    variant_index: int = 0,
) -> Path:
    safe_name = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()
    return Path(cache_root) / split / f"{safe_name}.v{variant_index}.pt"


def _seed_from_key(*parts: str | int) -> int:
    payload = "::".join(str(part) for part in parts)
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], 16)


def load_audio_waveform(path: str | Path, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


class CommonVoiceTaiwanDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        text_tokenizer: SentencePieceTextTokenizer,
        bopomofo_vocab: BopomofoVocab,
        sample_rate: int = 16000,
        max_audio_seconds: float = 30.0,
        max_samples: int | None = None,
        text_normalizer: TextNormalizer | None = None,
        audio_augmentor: AudioAugmentor | None = None,
        feature_cache_dir: str | Path | None = None,
        require_feature_cache: bool = False,
        feature_cache_variants: int = 1,
        sample_cached_variant: bool = False,
        split_source: str | Path | None = None,
    ) -> None:
        self.split = split
        self.samples = read_common_voice_split(data_root, split_source or split)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.text_tokenizer = text_tokenizer
        self.bopomofo_vocab = bopomofo_vocab
        self.sample_rate = sample_rate
        self.max_audio_samples = int(sample_rate * max_audio_seconds)
        self.text_normalizer = text_normalizer
        self.audio_augmentor = audio_augmentor
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.require_feature_cache = require_feature_cache
        self.feature_cache_variants = max(int(feature_cache_variants), 1)
        self.sample_cached_variant = sample_cached_variant

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        normalized_text = sample.text
        if self.text_normalizer is not None and self.text_normalizer.enabled:
            normalized_text = self.text_normalizer(normalized_text)
        item = {
            "audio": None,
            "text": normalized_text,
            "text_ids": torch.tensor(
                self.text_tokenizer.encode(normalized_text), dtype=torch.long
            ),
            "bopomofo_ids": torch.tensor(
                self.bopomofo_vocab.encode(normalized_text), dtype=torch.long
            ),
        }
        if self.feature_cache_dir is not None:
            variant_index = 0
            if self.sample_cached_variant and self.feature_cache_variants > 1:
                variant_index = random.randrange(self.feature_cache_variants)
            cache_path = build_feature_cache_path(
                self.feature_cache_dir,
                split=self.split,
                rel_path=sample.rel_path,
                variant_index=variant_index,
            )
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                item["input_features"] = cached["input_features"]
                item["audio"] = None
                return item
            if self.require_feature_cache:
                raise FileNotFoundError(f"Missing feature cache: {cache_path}")

        waveform = load_audio_waveform(sample.audio_path, self.sample_rate)
        if self.audio_augmentor is not None:
            waveform = self.audio_augmentor(waveform)
        waveform = waveform[: self.max_audio_samples]
        item["audio"] = waveform
        return item


class WhisperTWCollator:
    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        text_pad_id: int,
        bopomofo_pad_id: int,
        sample_rate: int = 16000,
        feature_padding: str = "max_length",
        feature_pad_to_multiple_of: int | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.text_pad_id = text_pad_id
        self.bopomofo_pad_id = bopomofo_pad_id
        self.sample_rate = sample_rate
        self.feature_padding = feature_padding
        self.feature_pad_to_multiple_of = feature_pad_to_multiple_of

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if all(item.get("input_features") is not None for item in batch):
            input_features = torch.stack(
                [item["input_features"] for item in batch],
                dim=0,
            )
        else:
            pad_to_multiple_of = self.feature_pad_to_multiple_of
            if self.feature_padding == "max_length":
                pad_to_multiple_of = None
            features = self.feature_extractor(
                [item["audio"].numpy() for item in batch],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=self.feature_padding,
                truncation=True,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            input_features = features.input_features

        text_ids = torch.nn.utils.rnn.pad_sequence(
            [item["text_ids"] for item in batch],
            batch_first=True,
            padding_value=self.text_pad_id,
        )
        bopomofo_ids = torch.nn.utils.rnn.pad_sequence(
            [item["bopomofo_ids"] for item in batch],
            batch_first=True,
            padding_value=self.bopomofo_pad_id,
        )
        return {
            "input_features": input_features,
            "decoder_input_ids": text_ids[:, :-1],
            "labels": text_ids[:, 1:],
            "bopomofo_labels": bopomofo_ids,
            "bopomofo_label_lengths": torch.tensor(
                [
                    int((item["bopomofo_ids"] != self.bopomofo_pad_id).sum())
                    for item in batch
                ],
                dtype=torch.long,
            ),
            "texts": [item["text"] for item in batch],
        }


def precompute_feature_cache(
    data_root: str | Path,
    split: str,
    cache_root: str | Path,
    feature_extractor: WhisperFeatureExtractor,
    sample_rate: int,
    max_audio_seconds: float,
    audio_augmentor: AudioAugmentor | None = None,
    num_variants: int = 1,
    overwrite: bool = False,
    max_samples: int | None = None,
    split_source: str | Path | None = None,
) -> tuple[int, int]:
    samples = read_common_voice_split(data_root, split_source or split)
    if max_samples is not None:
        samples = samples[:max_samples]
    max_audio_samples = int(sample_rate * max_audio_seconds)
    written = 0
    skipped = 0
    for sample in tqdm(samples, desc=f"cache {split}", dynamic_ncols=True):
        base_waveform = load_audio_waveform(sample.audio_path, sample_rate)
        base_waveform = base_waveform[:max_audio_samples]
        for variant_index in range(max(int(num_variants), 1)):
            cache_path = build_feature_cache_path(
                cache_root,
                split,
                sample.rel_path,
                variant_index=variant_index,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists() and not overwrite:
                skipped += 1
                continue
            waveform = base_waveform.clone()
            if variant_index > 0 and audio_augmentor is not None:
                seed = _seed_from_key(split, sample.rel_path, variant_index)
                rng = random.Random(seed)
                torch_generator = torch.Generator(device=waveform.device)
                torch_generator.manual_seed(seed)
                waveform = audio_augmentor(
                    waveform,
                    rng=rng,
                    torch_generator=torch_generator,
                )
            features = feature_extractor(
                [waveform.numpy()],
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            torch.save(
                {"input_features": features.input_features.squeeze(0).cpu()},
                cache_path,
            )
            written += 1
    return written, skipped


def iter_tokenizer_sentences(
    data_root: str | Path,
    splits: list[str],
    split_sources: dict[str, str | Path] | None = None,
    text_normalization_cfg: dict[str, Any] | None = None,
):
    normalizer = build_text_normalizer(text_normalization_cfg)
    for split in splits:
        split_source = split_sources.get(split) if split_sources else None
        for sample in read_common_voice_split(data_root, split_source or split):
            if not normalizer.enabled:
                yield sample.text
                continue
            normalized = normalizer(sample.text)
            if normalized:
                yield normalized
