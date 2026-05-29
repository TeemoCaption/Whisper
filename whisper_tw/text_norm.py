from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextNormalizer:
    enabled: bool = True
    apply_nfkc: bool = True
    remove_punctuation: bool = True
    remove_whitespace: bool = True

    def __call__(self, text: str) -> str:
        normalized = text or ""
        if self.apply_nfkc:
            normalized = unicodedata.normalize("NFKC", normalized)
        if self.remove_whitespace:
            normalized = "".join(normalized.split())
        else:
            normalized = " ".join(normalized.split())
        if self.remove_punctuation:
            normalized = "".join(
                char
                for char in normalized
                if not unicodedata.category(char).startswith("P")
            )
        return normalized.strip()


def build_text_normalizer(config: dict[str, Any] | None) -> TextNormalizer:
    cfg = config or {}
    return TextNormalizer(
        enabled=bool(cfg.get("enabled", True)),
        apply_nfkc=bool(cfg.get("apply_nfkc", True)),
        remove_punctuation=bool(cfg.get("remove_punctuation", True)),
        remove_whitespace=bool(cfg.get("remove_whitespace", True)),
    )


def normalize_text(text: str, config: dict[str, Any] | None) -> str:
    normalizer = build_text_normalizer(config)
    if not normalizer.enabled:
        return (text or "").strip()
    return normalizer(text)
