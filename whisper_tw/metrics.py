from __future__ import annotations


def _levenshtein_distance(source: str, target: str) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            substitution_cost = 0 if source_char == target_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    total_errors = 0
    total_chars = 0
    for prediction, reference in zip(predictions, references):
        pred = str(prediction or "")
        ref = str(reference or "")
        total_errors += _levenshtein_distance(pred, ref)
        total_chars += len(ref)
    if total_chars == 0:
        return 0.0 if total_errors == 0 else 1.0
    return total_errors / total_chars
