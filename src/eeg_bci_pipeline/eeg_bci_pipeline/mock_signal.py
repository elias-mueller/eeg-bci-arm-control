"""Pure helpers for synthetic EEG data used before hardware is available."""

from __future__ import annotations

from math import pi, sin
from typing import Sequence


DEFAULT_AMPLITUDE_CYCLE_UV = (10.0, 25.0, 45.0)


def select_cycle_value(
    values: Sequence[float],
    frame_index: int,
    frames_per_value: int,
) -> float:
    if not values:
        raise ValueError("values must contain at least one item")
    if frames_per_value < 1:
        raise ValueError("frames_per_value must be at least 1")

    cycle_index = (frame_index // frames_per_value) % len(values)
    return float(values[cycle_index])


def generate_mock_eeg_samples(
    start_sample_index: int,
    samples_per_frame: int,
    channel_count: int,
    sampling_rate_hz: float,
    amplitude_uv: float,
) -> list[float]:
    """Generate sample-major synthetic EEG data.

    Layout is [sample0_ch0, sample0_ch1, ..., sample1_ch0, ...].
    """

    samples: list[float] = []
    for sample_offset in range(samples_per_frame):
        t = (start_sample_index + sample_offset) / sampling_rate_hz
        for channel_index in range(channel_count):
            carrier = sin(2.0 * pi * 10.0 * t)
            modulation = sin(2.0 * pi * 0.5 * t + channel_index * 0.2)
            samples.append(amplitude_uv * (carrier + 0.25 * modulation))
    return samples
