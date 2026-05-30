"""Pure helpers for running a saved hand classifier on streaming EEG frames.

The ROS node (`model_intent_decoder.py`) is a thin shell over this module: the
sliding-window buffering, rest gating, and per-window decoding all live here so
they can be unit-tested without rclpy.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from eeg_bci_pipeline.data.gdf_recording import FloatArray
from eeg_bci_pipeline.decoder import DEFAULT_CLASS_LABELS, IntentPrediction
from eeg_bci_pipeline.training.hand_classifier import (
    HandClassifierArtifact,
    predict_hand_proba,
)

DEFAULT_REST_CONFIDENCE_THRESHOLD = 0.6
REST_LABEL = "rest"


class SlidingEpochBuffer:
    """Accumulate channel-major EEG frames into a rolling (channels, samples) window.

    Each pushed frame is a flat channel-major sequence
    (``[ch0_s0, ch0_s1, ..., ch1_s0, ...]``) of arbitrary per-frame length. The
    buffer keeps only the most recent ``samples_per_epoch`` samples per channel
    and returns that window once it is full, or ``None`` while warming up.
    """

    def __init__(self, channel_count: int, samples_per_epoch: int) -> None:
        if channel_count < 1:
            raise ValueError("channel_count must be at least 1")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be at least 1")
        self._channel_count = channel_count
        self._samples_per_epoch = samples_per_epoch
        self._buffer: FloatArray = np.empty((channel_count, 0), dtype=np.float64)

    @property
    def is_full(self) -> bool:
        return self._buffer.shape[1] >= self._samples_per_epoch

    def reset(self) -> None:
        self._buffer = np.empty((self._channel_count, 0), dtype=np.float64)

    def push(self, samples: Sequence[float]) -> FloatArray | None:
        """Append one channel-major frame; return the latest full window or None."""

        flat = np.asarray(samples, dtype=np.float64)
        if flat.ndim != 1:
            raise ValueError("samples must be a 1-D channel-major sequence")
        if flat.size % self._channel_count != 0:
            raise ValueError(
                "samples length must be divisible by channel count "
                f"({flat.size} samples for {self._channel_count} channels)"
            )
        frame_length = flat.size // self._channel_count
        block = flat.reshape(self._channel_count, frame_length)
        combined = np.concatenate([self._buffer, block], axis=1)
        self._buffer = combined[:, -self._samples_per_epoch :]
        if self._buffer.shape[1] < self._samples_per_epoch:
            return None
        return self._buffer.copy()


def indicator_probabilities(labels: Sequence[str], selected_index: int) -> tuple[float, ...]:
    """One-hot probability vector with all mass on ``selected_index``.

    The runtime ``Intent.probabilities`` field is part of the message contract but
    unread downstream; a hard indicator keeps it a valid distribution whose argmax
    always agrees with the chosen ``label`` for any threshold or class count.
    """

    return tuple(1.0 if index == selected_index else 0.0 for index in range(len(labels)))


def rest_intent(
    runtime_class_labels: Sequence[str] = DEFAULT_CLASS_LABELS,
    *,
    confidence: float = 1.0,
) -> IntentPrediction:
    """Build a ``rest`` intent (the robot driver holds position on rest).

    Used for warm-up and for degenerate windows that cannot be decoded.
    """

    runtime_labels = tuple(str(label) for label in runtime_class_labels)
    if REST_LABEL not in runtime_labels:
        raise ValueError("runtime_class_labels must include the rest label")
    rest_index = runtime_labels.index(REST_LABEL)
    return IntentPrediction(
        label=REST_LABEL,
        confidence=confidence,
        class_labels=runtime_labels,
        probabilities=indicator_probabilities(runtime_labels, rest_index),
    )


def gate_intent(
    probabilities: Sequence[float],
    model_class_labels: Sequence[str],
    *,
    runtime_class_labels: Sequence[str] = DEFAULT_CLASS_LABELS,
    rest_threshold: float = DEFAULT_REST_CONFIDENCE_THRESHOLD,
) -> IntentPrediction:
    """Map model class probabilities to a runtime intent, gating to rest when unsure.

    Assumes a per-class probability distribution from a binary hand model: when the
    top class probability is below ``rest_threshold`` the intent is reported as
    ``rest`` with confidence ``1 - top`` (the runner-up mass for two classes);
    otherwise the winning hand label is reported with its probability as the
    confidence. The returned ``class_labels``/``probabilities`` use the 3-class
    runtime set; ``probabilities`` is a hard indicator aligned with ``label`` so the
    two never disagree (the field is decorative — no consumer reads it).
    """

    probs = [float(value) for value in probabilities]
    labels = tuple(str(label) for label in model_class_labels)
    if not probs:
        raise ValueError("probabilities must be non-empty")
    if len(probs) != len(labels):
        raise ValueError("probabilities and model_class_labels must have equal length")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probs):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not 0.0 <= rest_threshold <= 1.0:
        raise ValueError("rest_threshold must be between 0 and 1")

    runtime_labels = tuple(str(label) for label in runtime_class_labels)
    if REST_LABEL not in runtime_labels:
        raise ValueError("runtime_class_labels must include the rest label")

    winner_index = int(np.argmax(probs))
    winner_prob = probs[winner_index]

    if winner_prob < rest_threshold:
        return rest_intent(runtime_labels, confidence=1.0 - winner_prob)

    winner_label = labels[winner_index]
    if winner_label not in runtime_labels:
        raise ValueError(f"model label {winner_label!r} is not in runtime_class_labels")
    selected_index = runtime_labels.index(winner_label)
    return IntentPrediction(
        label=winner_label,
        confidence=winner_prob,
        class_labels=runtime_labels,
        probabilities=indicator_probabilities(runtime_labels, selected_index),
    )


def decode_window(
    artifact: HandClassifierArtifact,
    window: FloatArray,
    *,
    runtime_class_labels: Sequence[str] = DEFAULT_CLASS_LABELS,
    rest_threshold: float = DEFAULT_REST_CONFIDENCE_THRESHOLD,
) -> IntentPrediction:
    """Decode one (channels, samples) window into a gated runtime intent.

    A degenerate (non-finite or flat/zero-variance) window is held at ``rest``
    rather than decoded: CSP takes the log of band power, so a flatlined window
    would yield ``-inf`` features and make the classifier raise.
    """

    epoch = np.asarray(window, dtype=np.float64)
    if epoch.ndim != 2:
        raise ValueError("window must have shape (channels, samples)")
    if not bool(np.all(np.isfinite(epoch))) or float(np.ptp(epoch)) == 0.0:
        return rest_intent(runtime_class_labels)

    batched = epoch.reshape(1, epoch.shape[0], epoch.shape[1])
    _labels, probabilities = predict_hand_proba(artifact, batched)
    return gate_intent(
        probabilities[0],
        artifact.class_labels,
        runtime_class_labels=runtime_class_labels,
        rest_threshold=rest_threshold,
    )
