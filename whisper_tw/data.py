from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass(frozen=True)
class CommonVoiceSample:
    audio_path: Path
    rel_path: str
    text: str
    language_label: str = ""
    target_script: str = ""
    raw_sentence: str = ""
    romanization_text: str = ""


def _resolve_audio_path(root: Path, tsv_path: Path, row: dict[str, str], rel_path: str) -> Path:
    explicit_audio_path = (row.get("audio_path") or "").strip()
    if explicit_audio_path:
        candidate = Path(explicit_audio_path)
        if candidate.is_absolute():
            return candidate
        root_candidate = root / candidate
        if root_candidate.exists():
            return root_candidate
        return tsv_path.parent / candidate
    return root / "clips" / rel_path


def read_common_voice_split(
    data_root: str | Path,
    split: str | Path,
    *,
    language_filter: str | None = None,
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
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if (row.get("filter_reason") or "").strip():
                continue
            language_label = (row.get("language_label") or row.get("locale") or "").strip()
            if language_filter and language_label != language_filter:
                continue
            text = (
                row.get("target_text")
                or row.get("sentence")
                or row.get("text")
                or ""
            ).strip()
            rel_path = (row.get("path") or "").strip()
            if not text or not rel_path:
                continue
            samples.append(
                CommonVoiceSample(
                    audio_path=_resolve_audio_path(root, tsv_path, row, rel_path),
                    rel_path=rel_path,
                    text=text,
                    language_label=language_label,
                    target_script=(row.get("target_script") or "").strip(),
                    raw_sentence=(row.get("raw_sentence") or row.get("sentence") or "").strip(),
                    romanization_text=(row.get("romanization_text") or "").strip(),
                )
            )
    return samples


def load_audio_waveform(path: str | Path, sample_rate: int) -> torch.Tensor:
    audio_path = Path(path)
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        waveform = waveform.mean(dim=0)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        return waveform
    except Exception as torchaudio_error:
        try:
            import librosa
        except ImportError as import_error:
            raise RuntimeError(
                "無法解碼音訊，且目前環境缺少 librosa 後備解碼。"
                "Common Voice 使用 MP3，請先執行："
                " conda install -c conda-forge ffmpeg libsndfile -y"
                " 並執行 pip install -r requirements.txt。"
            ) from import_error

        try:
            waveform_array, _ = librosa.load(
                str(audio_path),
                sr=sample_rate,
                mono=True,
            )
        except Exception as librosa_error:
            raise RuntimeError(
                f"無法解碼音訊: {audio_path}。"
                "請確認環境已安裝 ffmpeg/libsndfile，或檔案不是損毀的 MP3。"
            ) from librosa_error
        return torch.as_tensor(waveform_array, dtype=torch.float32)
