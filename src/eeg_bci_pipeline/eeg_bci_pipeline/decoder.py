"""Deterministic baseline decoder used before hardware and model training exist."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

DEFAULT_CLASS_LABELS = ("rest", "left_hand", "right_hand")


@dataclass(frozen=True)
class IntentPrediction:
    """Small pure-Python result type used by the ROS node and unit tests."""

    label: str
    confidence: float
    class_labels: tuple[str, ...]
    probabilities: tuple[float, ...]


def decode_mock_intent(
    samples: Sequence[float] | Iterable[float],
    class_labels: Sequence[str] = DEFAULT_CLASS_LABELS,
) -> IntentPrediction:
    """Map signal energy to a deterministic mock intent.

    This is intentionally not a real BCI model. It gives the ROS graph stable
    behavior until recordings, preprocessing, and trained models are added.
    """

    labels = tuple(class_labels)
    if not labels:
        raise ValueError("class_labels must contain at least one label")

    values = tuple(float(sample) for sample in samples)
    if not values:
        probabilities = _one_hot(labels, 0, confidence=1.0)
        return IntentPrediction(labels[0], 1.0, labels, probabilities)

    # RMS estimates signal magnitude without positive and negative samples canceling out.
    rms = sqrt(sum(sample * sample for sample in values) / len(values))
    selected_index = min(int(rms // 15.0), len(labels) - 1)
    confidence = max(0.5, min(0.95, 0.5 + (rms % 15.0) / 30.0))
    probabilities = _one_hot(labels, selected_index, confidence)
    return IntentPrediction(labels[selected_index], confidence, labels, probabilities)


def _one_hot(
    labels: Sequence[str],
    selected_index: int,
    confidence: float,
) -> tuple[float, ...]:
    if len(labels) == 1:
        return (1.0,)

    remaining = max(0.0, 1.0 - confidence)
    fallback = remaining / (len(labels) - 1)
    return tuple(
        confidence if index == selected_index else fallback for index, _ in enumerate(labels)
    )
