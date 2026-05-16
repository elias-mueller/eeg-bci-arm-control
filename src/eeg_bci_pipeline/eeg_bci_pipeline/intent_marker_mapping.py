"""Pure helpers for rendering decoded intents as simple RViz markers."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class IntentMarkerStyle:
    """Visual style for a single decoded intent."""

    label: str
    yaw_rad: float
    length: float
    color_rgba: tuple[float, float, float, float]


def style_for_intent(label: str, confidence: float) -> IntentMarkerStyle:
    """Map an intent label to a compact RViz arrow style."""

    normalized = label.strip().lower()
    alpha = max(0.35, min(1.0, float(confidence)))

    if normalized == "left_hand":
        return IntentMarkerStyle("left_hand", pi, 1.0, (0.1, 0.35, 1.0, alpha))
    if normalized == "right_hand":
        return IntentMarkerStyle("right_hand", 0.0, 1.0, (0.0, 0.8, 0.35, alpha))
    if normalized == "rest":
        return IntentMarkerStyle("rest", 0.0, 0.25, (0.6, 0.6, 0.6, alpha))

    return IntentMarkerStyle(
        normalized or "unknown",
        0.0,
        0.35,
        (1.0, 0.65, 0.0, alpha),
    )
