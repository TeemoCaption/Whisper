#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_LABELS = ("zh-TW", "nan-tw")
DEFAULT_ADAPTERS = {
    "zh-TW": "adapter_zh_tw",
    "nan-tw": "adapter_nan_tw",
}


@dataclass
class RoutingConfig:
    labels: tuple[str, ...] = DEFAULT_LABELS
    adapters: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_ADAPTERS))
    shared_adapter_name: str = "adapter_shared_tw"
    high_confidence_threshold: float = 0.75
    low_confidence_threshold: float = 0.55
    mid_top_k: int = 2
    target_languages: Mapping[str, str] = field(
        default_factory=lambda: {"zh-TW": "zh-TW", "nan-tw": "nan-tw"}
    )

    def __post_init__(self) -> None:
        self.labels = tuple(str(label) for label in self.labels)
        self.adapters = {str(key): str(value) for key, value in self.adapters.items()}
        self.target_languages = {
            str(key): str(value) for key, value in self.target_languages.items()
        }
        self.shared_adapter_name = str(self.shared_adapter_name)
        self.high_confidence_threshold = float(self.high_confidence_threshold)
        self.low_confidence_threshold = float(self.low_confidence_threshold)
        self.mid_top_k = int(self.mid_top_k)

        if not self.labels:
            raise ValueError("routing labels must not be empty")
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.high_confidence_threshold <= 1.0:
            raise ValueError("high_confidence_threshold must be between 0 and 1")
        if self.low_confidence_threshold > self.high_confidence_threshold:
            raise ValueError(
                "low_confidence_threshold must not exceed high_confidence_threshold"
            )
        if self.mid_top_k < 1:
            raise ValueError("mid_top_k must be at least 1")

    def adapter_name_for(self, label: str) -> str:
        return self.adapters.get(label, f"adapter_{_safe_name(label)}")

    def target_language_for(self, label: str) -> str:
        return self.target_languages.get(label, label)


@dataclass(frozen=True)
class RoutingDecision:
    adapter_names: tuple[str, ...]
    weights: tuple[float, ...]
    routing_mode: str
    predicted_language: str
    target_language: str
    confidence: float
    candidate_languages: tuple[str, ...]
    probabilities: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_names": list(self.adapter_names),
            "weights": list(self.weights),
            "routing_mode": self.routing_mode,
            "predicted_language": self.predicted_language,
            "target_language": self.target_language,
            "confidence": self.confidence,
            "candidate_languages": list(self.candidate_languages),
            "probabilities": dict(self.probabilities),
        }


def _safe_name(label: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in label).strip("_")


def _tensor_to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _flatten_scores(values: Any) -> list[float]:
    values = _tensor_to_list(values)
    if isinstance(values, (int, float)):
        return [float(values)]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("scores must be a mapping, sequence, or tensor-like value")

    flat: list[float] = []
    for value in values:
        converted = _tensor_to_list(value)
        if isinstance(converted, Sequence) and not isinstance(converted, (str, bytes)):
            flat.extend(_flatten_scores(converted))
        else:
            flat.append(float(converted))
    return flat


def _resolve_scores(
    scores: Mapping[str, float] | Sequence[float] | Any,
    labels: Sequence[str] | None,
) -> tuple[tuple[str, ...], list[float]]:
    if isinstance(scores, Mapping):
        resolved_labels = tuple(str(label) for label in (labels or scores.keys()))
        values = [float(scores[label]) for label in resolved_labels]
        return resolved_labels, values

    if labels is None:
        raise ValueError("labels are required when scores are not a mapping")
    resolved_labels = tuple(str(label) for label in labels)
    values = _flatten_scores(scores)
    if len(resolved_labels) != len(values):
        raise ValueError(
            f"labels length ({len(resolved_labels)}) does not match scores length ({len(values)})"
        )
    return resolved_labels, values


def normalize_language_scores(
    scores: Mapping[str, float] | Sequence[float] | Any,
    *,
    labels: Sequence[str] | None = None,
    from_logits: bool | None = None,
) -> dict[str, float]:
    resolved_labels, values = _resolve_scores(scores, labels)
    if not values:
        raise ValueError("scores must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scores must be finite")

    if from_logits is None:
        total = sum(values)
        from_logits = not (
            all(value >= 0.0 for value in values) and total > 0.0 and abs(total - 1.0) <= 1.0e-3
        )

    if from_logits:
        max_value = max(values)
        exp_values = [math.exp(value - max_value) for value in values]
        total = sum(exp_values)
        probabilities = [value / total for value in exp_values]
    else:
        if any(value < 0.0 for value in values):
            raise ValueError("probabilities must not be negative")
        total = sum(values)
        if total <= 0.0:
            raise ValueError("probabilities must sum to a positive value")
        probabilities = [value / total for value in values]

    return {
        label: probability for label, probability in zip(resolved_labels, probabilities)
    }


def route_adapters(
    scores: Mapping[str, float] | Sequence[float] | Any,
    *,
    config: RoutingConfig | None = None,
    labels: Sequence[str] | None = None,
    from_logits: bool | None = None,
    target_language: str | None = None,
) -> RoutingDecision:
    config = config or RoutingConfig()
    labels = tuple(labels or config.labels)
    probabilities = normalize_language_scores(
        scores,
        labels=labels,
        from_logits=from_logits,
    )
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    predicted_language, confidence = ranked[0]
    selected_target_language = target_language or config.target_language_for(
        predicted_language
    )

    if confidence >= config.high_confidence_threshold:
        return RoutingDecision(
            adapter_names=(config.adapter_name_for(predicted_language),),
            weights=(1.0,),
            routing_mode="single",
            predicted_language=predicted_language,
            target_language=selected_target_language,
            confidence=confidence,
            candidate_languages=(predicted_language,),
            probabilities=probabilities,
        )

    if confidence < config.low_confidence_threshold:
        return RoutingDecision(
            adapter_names=(config.shared_adapter_name,),
            weights=(1.0,),
            routing_mode="shared",
            predicted_language=predicted_language,
            target_language=selected_target_language,
            confidence=confidence,
            candidate_languages=(predicted_language,),
            probabilities=probabilities,
        )

    top_k = min(config.mid_top_k, len(ranked))
    candidates = ranked[:top_k]
    total = sum(probability for _, probability in candidates)
    adapter_weights: dict[str, float] = {}
    adapter_order: list[str] = []
    candidate_languages: list[str] = []
    for label, probability in candidates:
        adapter_name = config.adapter_name_for(label)
        if adapter_name not in adapter_weights:
            adapter_order.append(adapter_name)
            adapter_weights[adapter_name] = 0.0
        adapter_weights[adapter_name] += probability / total
        candidate_languages.append(label)

    return RoutingDecision(
        adapter_names=tuple(adapter_order),
        weights=tuple(adapter_weights[name] for name in adapter_order),
        routing_mode="mixed",
        predicted_language=predicted_language,
        target_language=selected_target_language,
        confidence=confidence,
        candidate_languages=tuple(candidate_languages),
        probabilities=probabilities,
    )


def build_routing_config(config: Mapping[str, Any]) -> RoutingConfig:
    train_cfg = (
        config.get("whisper_train")
        or config.get("whisper_baseline")
        or config
    )
    routing_cfg = train_cfg.get("routing", {}) if isinstance(train_cfg, Mapping) else {}
    if not isinstance(routing_cfg, Mapping):
        routing_cfg = {}

    return RoutingConfig(
        labels=tuple(routing_cfg.get("labels", DEFAULT_LABELS)),
        adapters=routing_cfg.get("adapters", DEFAULT_ADAPTERS),
        shared_adapter_name=routing_cfg.get(
            "shared_adapter_name",
            "adapter_shared_tw",
        ),
        high_confidence_threshold=routing_cfg.get("high_confidence_threshold", 0.75),
        low_confidence_threshold=routing_cfg.get("low_confidence_threshold", 0.55),
        mid_top_k=routing_cfg.get("mid_top_k", 2),
        target_languages=routing_cfg.get(
            "target_languages",
            {"zh-TW": "zh-TW", "nan-tw": "nan-tw"},
        ),
    )


def load_routing_config(path: str | Path) -> RoutingConfig:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return build_routing_config(yaml.safe_load(handle) or {})


def _self_test() -> None:
    config = RoutingConfig()

    high = route_adapters({"zh-TW": 0.9, "nan-tw": 0.1}, config=config)
    assert high.routing_mode == "single"
    assert high.adapter_names == ("adapter_zh_tw",)
    assert high.weights == (1.0,)

    middle = route_adapters({"zh-TW": 0.62, "nan-tw": 0.38}, config=config)
    assert middle.routing_mode == "mixed"
    assert middle.adapter_names == ("adapter_zh_tw", "adapter_nan_tw")
    assert abs(sum(middle.weights) - 1.0) < 1.0e-6

    low = route_adapters(
        {"zh-TW": 0.51, "nan-tw": 0.49},
        config=config,
        target_language="nan-tw",
    )
    assert low.routing_mode == "shared"
    assert low.adapter_names == ("adapter_shared_tw",)
    assert low.target_language == "nan-tw"

    logits = route_adapters(
        [2.0, 0.0],
        config=config,
        labels=("zh-TW", "nan-tw"),
        from_logits=True,
    )
    assert logits.routing_mode == "single"
    assert logits.predicted_language == "zh-TW"

    print("confidence routing self-test passed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="語言信心閥值路由工具。")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--scores-json", help="語言分數，例如 {\"zh-TW\": 0.8, \"nan-tw\": 0.2}")
    parser.add_argument("--target-language", help="已知目標文字語言，例如 nan-tw")
    parser.add_argument("--self-test", action="store_true")
    score_mode = parser.add_mutually_exclusive_group()
    score_mode.add_argument("--from-logits", action="store_true")
    score_mode.add_argument("--from-probabilities", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return

    if not args.scores_json:
        raise SystemExit("--scores-json or --self-test is required")

    from_logits: bool | None = None
    if args.from_logits:
        from_logits = True
    elif args.from_probabilities:
        from_logits = False

    config = load_routing_config(args.config)
    scores = json.loads(args.scores_json)
    decision = route_adapters(
        scores,
        config=config,
        from_logits=from_logits,
        target_language=args.target_language,
    )
    print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
