"""Validation for EEG frames entering the ROS BCI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

DEFAULT_EEG_CHANNEL_COUNT = 16
DEFAULT_EEG_SAMPLING_RATE_HZ = 250.0
DEFAULT_SAMPLING_RATE_TOLERANCE_HZ = 0.5
DEFAULT_MAX_ABS_SAMPLE_UV = 10_000.0
SUSPICIOUS_SMALL_PEAK_UV = 0.01
EEG_SAMPLE_UNIT = "microvolts"


class EegFrameContractError(ValueError):
    """Raised when an EEG frame violates the pipeline data contract."""


@dataclass(frozen=True)
class EegFrameShape:
    """Derived shape for a validated channel-major EEG frame."""

    channel_count: int
    samples_per_channel: int
    duration_sec: float
    peak_abs_sample_uv: float
    suspiciously_small_peak: bool


def default_channel_labels(
    channel_count: int = DEFAULT_EEG_CHANNEL_COUNT,
) -> tuple[str, ...]:
    """Return stable fallback labels when acquisition metadata is unavailable."""

    if channel_count < 1:
        raise ValueError("channel_count must be at least 1")
    return tuple(f"ch_{index + 1:02d}" for index in range(channel_count))


def validate_eeg_frame_payload(
    *,
    sampling_rate_hz: float,
    channel_labels: Sequence[str],
    samples: Sequence[float],
    expected_channel_count: int = DEFAULT_EEG_CHANNEL_COUNT,
    expected_channel_labels: Sequence[str] | None = None,
    expected_sampling_rate_hz: float = DEFAULT_EEG_SAMPLING_RATE_HZ,
    sampling_rate_tolerance_hz: float = DEFAULT_SAMPLING_RATE_TOLERANCE_HZ,
    max_abs_sample_uv: float = DEFAULT_MAX_ABS_SAMPLE_UV,
) -> EegFrameShape:
    """Validate fields from an ``eeg_bci_interfaces/EegFrame`` message.

    Samples are channel-major microvolt values:
    ``[ch0_sample0, ch0_sample1, ..., ch1_sample0, ch1_sample1, ...]``.
    """

    _validate_sampling_rate(
        sampling_rate_hz,
        expected_sampling_rate_hz,
        sampling_rate_tolerance_hz,
    )
    if expected_channel_labels is not None:
        expected_labels = _validate_expected_channel_labels(expected_channel_labels)
        expected_channel_count = len(expected_labels)
    else:
        expected_labels = None

    labels = _validate_channel_labels(channel_labels, expected_channel_count)
    if expected_labels is not None:
        if labels != expected_labels:
            raise EegFrameContractError(
                f"channel_labels must remain stable and ordered exactly as {expected_labels}"
            )

    sample_count = len(samples)
    if sample_count == 0:
        raise EegFrameContractError("samples must be non-empty")
    if sample_count % len(labels) != 0:
        raise EegFrameContractError(
            "samples length must be divisible by channel count "
            f"({sample_count} samples for {len(labels)} channels)"
        )

    peak_abs_sample_uv = _validate_samples(samples, max_abs_sample_uv)
    samples_per_channel = sample_count // len(labels)
    return EegFrameShape(
        channel_count=len(labels),
        samples_per_channel=samples_per_channel,
        duration_sec=samples_per_channel / float(sampling_rate_hz),
        # Peak-based so one normal-amplitude sample prevents a frame-level unit warning.
        peak_abs_sample_uv=peak_abs_sample_uv,
        suspiciously_small_peak=0.0 < peak_abs_sample_uv < SUSPICIOUS_SMALL_PEAK_UV,
    )


def _validate_sampling_rate(
    sampling_rate_hz: float,
    expected_sampling_rate_hz: float,
    sampling_rate_tolerance_hz: float,
) -> None:
    value = _finite_float("sampling_rate_hz", sampling_rate_hz)
    expected = _finite_float("expected_sampling_rate_hz", expected_sampling_rate_hz)
    tolerance = _finite_float("sampling_rate_tolerance_hz", sampling_rate_tolerance_hz)
    if value <= 0.0:
        raise EegFrameContractError("sampling_rate_hz must be greater than 0")
    if expected <= 0.0:
        raise EegFrameContractError("expected_sampling_rate_hz must be greater than 0")
    if tolerance < 0.0:
        raise EegFrameContractError("sampling_rate_tolerance_hz must be non-negative")
    if abs(value - expected) > tolerance:
        raise EegFrameContractError(
            f"sampling_rate_hz must be {expected:g} Hz +/- {tolerance:g} Hz"
        )


def _validate_channel_labels(
    channel_labels: Sequence[str],
    expected_channel_count: int,
) -> tuple[str, ...]:
    if expected_channel_count < 1:
        raise EegFrameContractError("expected_channel_count must be at least 1")

    labels = _normalized_labels(channel_labels)
    if len(labels) != expected_channel_count:
        raise EegFrameContractError(
            f"channel_labels must contain exactly {expected_channel_count} labels"
        )
    if any(not label for label in labels):
        raise EegFrameContractError("channel_labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise EegFrameContractError("channel_labels must be unique")
    return labels


def _validate_expected_channel_labels(channel_labels: Sequence[str]) -> tuple[str, ...]:
    labels = _normalized_labels(channel_labels)
    if not labels:
        raise EegFrameContractError("expected_channel_labels must be non-empty")
    if any(not label for label in labels):
        raise EegFrameContractError("expected_channel_labels must not contain empty labels")
    if len(set(labels)) != len(labels):
        raise EegFrameContractError("expected_channel_labels must be unique")
    return labels


def _validate_samples(samples: Sequence[float], max_abs_sample_uv: float) -> float:
    max_abs = _finite_float("max_abs_sample_uv", max_abs_sample_uv)
    if max_abs <= 0.0:
        raise EegFrameContractError("max_abs_sample_uv must be greater than 0")

    peak_abs = 0.0
    for index, sample in enumerate(samples):
        value = _finite_float(f"samples[{index}]", sample)
        abs_value = abs(value)
        if abs_value > max_abs:
            raise EegFrameContractError(
                f"samples[{index}] must be less than or equal to {max_abs:g} {EEG_SAMPLE_UNIT}"
            )
        peak_abs = max(peak_abs, abs_value)
    return peak_abs


def _finite_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EegFrameContractError(f"{name} must be numeric and finite") from error
    if not isfinite(result):
        raise EegFrameContractError(f"{name} must be finite")
    return result


def _normalized_labels(channel_labels: Sequence[str]) -> tuple[str, ...]:
    """Trim labels before uniqueness and order checks."""

    return tuple(str(label).strip() for label in channel_labels)
