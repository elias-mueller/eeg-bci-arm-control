"""Labeled epoch extraction for BCI Competition IV data set 2a."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast

import numpy as np

from eeg_bci_pipeline.data.gdf_recording import (
    FloatArray,
    MneRawLike,
    recording_from_mne_raw,
)

BCICIV2A_CUE_LABELS = {
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
}
DEFAULT_EPOCH_TMIN_SEC = 0.0
DEFAULT_EPOCH_TMAX_SEC = 4.0


class MneAnnotationsLike(Protocol):
    onset: Sequence[float]
    description: Sequence[str | bytes]


class AnnotatedMneRawLike(MneRawLike, Protocol):
    annotations: MneAnnotationsLike


@dataclass(frozen=True)
class LabeledEpochs:
    source_id: str
    sampling_rate_hz: float
    channel_labels: tuple[str, ...]
    class_labels: tuple[str, ...]
    labels: tuple[str, ...]
    start_sample_indices: tuple[int, ...]
    epochs_uv: FloatArray
    skipped_epoch_count: int = 0

    @property
    def epoch_count(self) -> int:
        return int(self.epochs_uv.shape[0])


def read_bciciv2a_epochs(
    gdf_path: str | Path,
    *,
    tmin_sec: float = DEFAULT_EPOCH_TMIN_SEC,
    tmax_sec: float = DEFAULT_EPOCH_TMAX_SEC,
    class_labels: Sequence[str] | None = None,
    channel_labels: Sequence[str] | None = None,
) -> LabeledEpochs:
    """Read a BCIC IV 2a GDF file and return labeled cue-locked epochs."""

    try:
        import mne
    except ImportError as error:
        raise RuntimeError(
            "BCIC IV 2a epoch extraction requires MNE. Install: python3-mne"
        ) from error

    path = Path(gdf_path)
    read_raw_gdf = cast(Callable[..., AnnotatedMneRawLike], mne.io.read_raw_gdf)
    raw = read_raw_gdf(path, preload=True, verbose="ERROR")
    return extract_bciciv2a_epochs(
        raw,
        source_id=path.stem,
        tmin_sec=tmin_sec,
        tmax_sec=tmax_sec,
        class_labels=class_labels,
        channel_labels=channel_labels,
    )


def extract_bciciv2a_epochs(
    raw: AnnotatedMneRawLike,
    *,
    source_id: str,
    tmin_sec: float = DEFAULT_EPOCH_TMIN_SEC,
    tmax_sec: float = DEFAULT_EPOCH_TMAX_SEC,
    class_labels: Sequence[str] | None = None,
    channel_labels: Sequence[str] | None = None,
) -> LabeledEpochs:
    """Extract channel-major epochs from cue annotations 769-772."""

    if tmax_sec <= tmin_sec:
        raise ValueError("tmax_sec must be greater than tmin_sec")

    selected_labels = _selected_class_labels(class_labels)
    selected_label_set = set(selected_labels)
    recording = recording_from_mne_raw(
        raw,
        source_id=source_id,
        channel_labels=channel_labels,
    )
    sfreq = recording.sampling_rate_hz
    sample_offset = int(round(tmin_sec * sfreq))
    epoch_samples = int(round((tmax_sec - tmin_sec) * sfreq))
    if epoch_samples < 1:
        raise ValueError("epoch duration must contain at least one sample")

    epochs: list[FloatArray] = []
    labels: list[str] = []
    start_indices: list[int] = []
    skipped = 0
    annotations = zip(raw.annotations.onset, raw.annotations.description)
    for onset_sec, description in annotations:
        label = BCICIV2A_CUE_LABELS.get(_annotation_description_key(description))
        if label is None:
            continue
        if label not in selected_label_set:
            continue

        start = int(round(float(onset_sec) * sfreq)) + sample_offset
        stop = start + epoch_samples
        if start < 0 or stop > recording.samples_per_channel:
            skipped += 1
            continue

        epochs.append(recording.samples_uv[:, start:stop])
        labels.append(label)
        start_indices.append(start)

    if not epochs:
        raise ValueError("no BCIC IV 2a cue epochs found")

    return LabeledEpochs(
        source_id=source_id,
        sampling_rate_hz=sfreq,
        channel_labels=recording.channel_labels,
        class_labels=selected_labels,
        labels=tuple(labels),
        start_sample_indices=tuple(start_indices),
        epochs_uv=np.stack(epochs),
        skipped_epoch_count=skipped,
    )


def _selected_class_labels(class_labels: Sequence[str] | None) -> tuple[str, ...]:
    available_labels = tuple(BCICIV2A_CUE_LABELS.values())
    if class_labels is None:
        return available_labels

    selected = tuple(str(label) for label in class_labels)
    unknown = sorted(set(selected) - set(available_labels))
    if unknown:
        raise ValueError(f"unknown BCIC IV 2a class labels: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("BCIC IV 2a class labels must be unique")
    return selected


def _annotation_description_key(description: str | bytes) -> str:
    if isinstance(description, bytes):
        return description.decode()
    return str(description)
