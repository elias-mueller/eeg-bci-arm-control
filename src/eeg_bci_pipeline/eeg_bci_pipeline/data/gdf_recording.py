"""Helpers for replaying EEG recordings stored as GDF files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast

import numpy as np
import numpy.typing as npt

BCICIV2A_EEG_CHANNEL_COUNT = 22
DEFAULT_REPLAY_SAMPLES_PER_FRAME = 25
VOLTS_TO_MICROVOLTS = 1_000_000.0

FloatArray = npt.NDArray[np.float64]


class MneRawLike(Protocol):
    """Small subset of the MNE Raw API used by this package."""

    ch_names: Sequence[str]
    info: Mapping[str, float]

    def get_channel_types(self) -> Sequence[str]: ...

    def get_data(self, picks: Sequence[str]) -> FloatArray: ...


@dataclass(frozen=True)
class EegRecording:
    source_id: str
    sampling_rate_hz: float
    channel_labels: tuple[str, ...]
    samples_uv: FloatArray

    @property
    def samples_per_channel(self) -> int:
        return int(self.samples_uv.shape[1])


@dataclass(frozen=True)
class EegReplayFrame:
    start_sample_index: int
    samples: list[float]


def read_gdf_recording(
    gdf_path: str | Path,
    channel_labels: Sequence[str] | None = None,
) -> EegRecording:
    """Read a GDF recording with MNE and return channel-major microvolt samples."""

    try:
        import mne
    except ImportError as error:
        raise RuntimeError(
            "GDF replay requires MNE. Install the Ubuntu package: python3-mne"
        ) from error

    path = Path(gdf_path)
    read_raw_gdf = cast(Callable[..., MneRawLike], mne.io.read_raw_gdf)
    raw = read_raw_gdf(path, preload=True, verbose="ERROR")
    return recording_from_mne_raw(raw, source_id=path.stem, channel_labels=channel_labels)


def recording_from_mne_raw(
    raw: MneRawLike,
    *,
    source_id: str,
    channel_labels: Sequence[str] | None = None,
) -> EegRecording:
    """Convert an MNE Raw-like object into the pipeline recording shape."""

    if channel_labels is not None:
        selected_labels = normalize_channel_labels(channel_labels)
        if not selected_labels:
            selected_labels = _eeg_channel_labels(raw)
    else:
        selected_labels = _eeg_channel_labels(raw)
    if not selected_labels:
        raise ValueError("recording must contain at least one EEG channel")

    data_volts = raw.get_data(picks=list(selected_labels))
    samples_uv = cast(
        FloatArray,
        np.asarray(data_volts, dtype=np.float64) * VOLTS_TO_MICROVOLTS,
    )
    return EegRecording(
        source_id=source_id,
        sampling_rate_hz=float(raw.info["sfreq"]),
        channel_labels=tuple(str(label) for label in selected_labels),
        samples_uv=samples_uv,
    )


def iter_replay_frames(
    recording: EegRecording,
    samples_per_frame: int = DEFAULT_REPLAY_SAMPLES_PER_FRAME,
) -> Iterable[EegReplayFrame]:
    """Yield channel-major frame payloads from a recording."""

    if samples_per_frame < 1:
        raise ValueError("samples_per_frame must be at least 1")

    for start in range(0, recording.samples_per_channel, samples_per_frame):
        stop = min(start + samples_per_frame, recording.samples_per_channel)
        frame_samples = recording.samples_uv[:, start:stop].reshape(-1)
        yield EegReplayFrame(
            start_sample_index=start,
            samples=frame_samples.tolist(),
        )


def replay_elapsed_sec(
    sample_index: int,
    *,
    loop_index: int,
    samples_per_channel: int,
    sampling_rate_hz: float,
) -> float:
    """Return replay elapsed time for a sample, including completed loop cycles."""

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if loop_index < 0:
        raise ValueError("loop_index must be non-negative")
    if samples_per_channel < 1:
        raise ValueError("samples_per_channel must be at least 1")
    if sample_index >= samples_per_channel:
        raise ValueError("sample_index must be less than samples_per_channel")
    if sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be greater than 0")

    absolute_sample_index = loop_index * samples_per_channel + sample_index
    return absolute_sample_index / sampling_rate_hz


def normalize_channel_labels(channel_labels: Sequence[str]) -> tuple[str, ...]:
    """Normalize explicitly configured channel labels."""

    if isinstance(channel_labels, str):
        raise ValueError("channel_labels must be a sequence of labels, not a single string")

    try:
        labels = tuple(str(label).strip() for label in channel_labels)
    except TypeError as error:
        raise ValueError("channel_labels must be a sequence of labels") from error

    if any(not label for label in labels):
        raise ValueError("channel_labels must not contain empty labels")
    return labels


def _eeg_channel_labels(raw: MneRawLike) -> tuple[str, ...]:
    channel_types = raw.get_channel_types()
    eeg_labels = [
        name
        for name, channel_type in zip(raw.ch_names, channel_types)
        if channel_type == "eeg" and "eog" not in name.lower()
    ]
    if eeg_labels:
        return tuple(eeg_labels)

    raise ValueError(
        "recording contains no channels typed as EEG; pass explicit channel_labels "
        "if this file needs a manual channel selection"
    )
