"""Pure helpers for live calibration capture (no rclpy).

A calibration session presents left/right cues and records one epoch of motor
imagery per trial from the live EEG stream. These helpers own the parts that are
worth unit-testing without ROS: the balanced trial order, slicing one epoch out
of an accumulated imagery buffer, and assembling the captured windows into the
``LabeledEpochs`` container that training already consumes.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs
from eeg_bci_pipeline.data.gdf_recording import FloatArray


def build_trial_schedule(
    trials_per_class: int,
    class_labels: Sequence[str],
    *,
    seed: int = 0,
) -> tuple[str, ...]:
    """Return a balanced, deterministically shuffled per-trial cue order.

    Each class appears ``trials_per_class`` times; ``seed`` makes the order
    reproducible (so a session and its tests are deterministic).
    """

    if trials_per_class < 1:
        raise ValueError("trials_per_class must be at least 1")
    labels = tuple(str(label) for label in class_labels)
    if len(labels) < 2:
        raise ValueError("class_labels must contain at least two classes")
    if len(set(labels)) != len(labels):
        raise ValueError("class_labels must be unique")

    schedule = [label for label in labels for _ in range(trials_per_class)]
    random.Random(seed).shuffle(schedule)
    return tuple(schedule)


def reshape_frame_to_channel_major(
    samples: Sequence[float],
    channel_count: int,
) -> FloatArray | None:
    """Reshape one flat channel-major EEG frame to ``(channels, frame_len)``.

    Returns ``None`` for an empty frame or one whose length is not a multiple of
    ``channel_count`` (a malformed frame the caller should drop rather than let it
    scramble channels).
    """

    if channel_count < 1:
        raise ValueError("channel_count must be at least 1")
    flat = np.asarray(samples, dtype=np.float64)
    if flat.size == 0 or flat.size % channel_count != 0:
        return None
    return flat.reshape(channel_count, flat.size // channel_count)


def extract_epoch(
    buffer: FloatArray,
    settle_offset: int,
    samples_per_epoch: int,
) -> FloatArray | None:
    """Slice one ``(channels, samples_per_epoch)`` window from an imagery buffer.

    ``buffer`` is the accumulated ``(channels, T)`` imagery data; the window starts
    at ``settle_offset`` to skip the imagery-onset transient. Returns ``None`` when
    the buffer is too short (a dropout), so the caller can count the trial skipped.
    """

    if settle_offset < 0:
        raise ValueError("settle_offset must be non-negative")
    if samples_per_epoch < 1:
        raise ValueError("samples_per_epoch must be at least 1")
    if buffer.ndim != 2:
        raise ValueError("buffer must be 2-D (channels, samples)")

    stop = settle_offset + samples_per_epoch
    if buffer.shape[1] < stop:
        return None
    return buffer[:, settle_offset:stop].copy()


def assemble_labeled_epochs(
    *,
    source_id: str,
    sampling_rate_hz: float,
    channel_labels: Sequence[str],
    class_labels: Sequence[str],
    records: Sequence[tuple[str, FloatArray]],
    skipped_epoch_count: int = 0,
) -> LabeledEpochs:
    """Build a ``LabeledEpochs`` from captured ``(label, (channels, samples))`` records.

    Validates that every window shares the same ``(channels, samples)`` shape and
    that the channel axis matches ``channel_labels``; stacks them channel-major
    into ``epochs_uv`` of shape ``(n_epochs, channels, samples)``.
    """

    if not records:
        raise ValueError("no epochs were captured")

    channels = tuple(str(label) for label in channel_labels)
    classes = tuple(str(label) for label in class_labels)
    epoch_labels = tuple(str(label) for label, _ in records)
    unknown = sorted(set(epoch_labels) - set(classes))
    if unknown:
        raise ValueError(f"record labels not in class_labels: {unknown}")
    windows = [np.asarray(window, dtype=np.float64) for _, window in records]

    expected_shape = windows[0].shape
    for window in windows:
        if window.ndim != 2 or window.shape != expected_shape:
            raise ValueError("all captured epochs must share one (channels, samples) shape")
    if expected_shape[0] != len(channels):
        raise ValueError(
            f"captured epochs have {expected_shape[0]} channels, expected {len(channels)}"
        )

    return LabeledEpochs(
        source_id=source_id,
        sampling_rate_hz=float(sampling_rate_hz),
        channel_labels=channels,
        class_labels=classes,
        labels=epoch_labels,
        # Trial ordinals: live capture has no global recording offset (unlike the
        # GDF path, where this field holds the real per-epoch sample index).
        start_sample_indices=tuple(range(len(records))),
        epochs_uv=np.stack(windows),
        skipped_epoch_count=int(skipped_epoch_count),
    )
