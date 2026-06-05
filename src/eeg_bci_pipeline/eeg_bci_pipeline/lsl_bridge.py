"""Pure helpers for bridging an LSL EEG stream to the ``EegFrame`` contract.

This module has no top-level ``pylsl`` or ``rclpy`` import (only a lazy
``import_pylsl`` helper), so it is fully unit-testable without Lab Streaming
Layer or ROS installed. It owns the two things that are easy to get silently
wrong at the LSL boundary:

* mapping LSL ``StreamInfo`` metadata (channel count, per-channel labels from the
  ``desc`` XML, nominal rate, unit) into the fields the pipeline expects, and
* converting between LSL's sample-major chunk layout and the channel-major
  microvolt layout used by ``EegFrame.samples`` (see ``EegFrame.msg``).

The ``LslStreamInfoLike`` / ``LslXmlElementLike`` Protocols model only the small
subset of pylsl's API used here, mirroring ``MneRawLike`` in
``data/gdf_recording.py`` so tests can substitute trivial fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from eeg_bci_pipeline.data.gdf_recording import (
    VOLTS_TO_MICROVOLTS,
    normalize_channel_labels,
)
from eeg_bci_pipeline.eeg_frame_contract import default_channel_labels

_MICROVOLT_UNITS = frozenset({"uv", "microvolt", "microvolts", "µv", "μv"})
_VOLT_UNITS = frozenset({"v", "volt", "volts"})


def import_pylsl() -> Any:
    """Import pylsl lazily, raising a clear setup error if it (or liblsl) is absent.

    Mirrors the lazy ``import mne`` pattern in ``data/gdf_recording.py``: the LSL
    nodes call this so the rest of the package stays importable without Lab
    Streaming Layer installed. Catches any import failure, including pylsl's
    RuntimeError when the native liblsl cannot be loaded.
    """

    try:
        import pylsl  # pyright: ignore[reportMissingImports]
    except Exception as error:
        raise RuntimeError(
            "LSL support requires pylsl and the native liblsl library. "
            "Set up the environment with scripts/setup-python-env."
        ) from error
    return pylsl


class LslXmlElementLike(Protocol):
    """Minimal subset of pylsl's ``XMLElement`` (a libxml node wrapper)."""

    def child(self, name: str) -> "LslXmlElementLike": ...

    def child_value(self, name: str) -> str: ...

    def next_sibling(self) -> "LslXmlElementLike": ...

    def empty(self) -> bool: ...


class LslStreamInfoLike(Protocol):
    """Minimal subset of pylsl's ``StreamInfo`` used to derive frame metadata."""

    def channel_count(self) -> int: ...

    def nominal_srate(self) -> float: ...

    def name(self) -> str: ...

    def type(self) -> str: ...

    def source_id(self) -> str: ...

    def desc(self) -> LslXmlElementLike: ...


@dataclass(frozen=True)
class LslStreamMetadata:
    """Frame-relevant metadata resolved from an LSL ``StreamInfo``."""

    source_id: str
    stream_name: str
    stream_type: str
    channel_count: int
    nominal_srate_hz: float
    channel_labels: tuple[str, ...]
    declared_unit: str
    # True when the stream declared labels but they were rejected (incomplete,
    # duplicated, or wrong count) and ch_NN fallbacks were substituted. A label
    # strict decoder will reject such frames, so the node warns on this.
    labels_downgraded: bool


def extract_lsl_metadata(
    info: LslStreamInfoLike,
    *,
    fallback_source_id: str = "lsl-eeg",
) -> LslStreamMetadata:
    """Read channel count, rate, source id, labels, and unit from a stream info.

    ``channel_labels`` is always length ``channel_count``: declared labels are
    used verbatim when complete and unique, otherwise stable ``ch_NN`` fallbacks
    are substituted (see :func:`resolve_channel_labels`). ``nominal_srate_hz`` is
    returned as declared (``0.0`` for irregular streams); the caller decides
    whether that is acceptable.
    """

    channel_count = int(info.channel_count())
    if channel_count < 1:
        raise ValueError("LSL stream must declare at least one channel")

    source_id = info.source_id() or info.name() or fallback_source_id
    declared_labels, declared_unit = _read_channel_desc(info.desc(), channel_count)
    channel_labels = resolve_channel_labels(declared_labels, channel_count)
    declared_present = any(label.strip() for label in declared_labels)
    labels_downgraded = declared_present and channel_labels == default_channel_labels(
        channel_count
    )
    return LslStreamMetadata(
        source_id=source_id,
        stream_name=info.name(),
        stream_type=info.type(),
        channel_count=channel_count,
        nominal_srate_hz=float(info.nominal_srate()),
        channel_labels=channel_labels,
        declared_unit=declared_unit,
        labels_downgraded=labels_downgraded,
    )


def _read_channel_desc(
    desc: LslXmlElementLike,
    channel_count: int,
) -> tuple[list[str], str]:
    """Walk ``desc/channels/channel`` in document order for labels and a unit.

    Single-unit assumption: the first non-empty channel unit is taken as the whole
    stream's unit. Heterogeneous per-channel units (rare for an EEG stream) are not
    modeled; every channel is scaled by that one unit.
    """

    labels: list[str] = []
    declared_unit = ""
    node = desc.child("channels").child("channel")
    while not node.empty() and len(labels) < channel_count:
        labels.append(node.child_value("label"))
        if not declared_unit:
            unit = node.child_value("unit").strip().lower()
            if unit:
                declared_unit = unit
        node = node.next_sibling()
    return labels, declared_unit


def resolve_channel_labels(
    declared_labels: Sequence[str],
    channel_count: int,
) -> tuple[str, ...]:
    """Use declared labels verbatim when complete and unique, else ``ch_NN``.

    The decoder enforces an exact, ordered, unique channel-label contract, so a
    partial, mismatched-length, or duplicate label set is downgraded to stable
    defaults rather than forwarded (which would make every frame be rejected).
    Labels are never rewritten beyond the ``.strip()`` in
    :func:`normalize_channel_labels`.
    """

    if channel_count < 1:
        raise ValueError("channel_count must be at least 1")
    try:
        labels = normalize_channel_labels(declared_labels)
    except ValueError:
        labels = ()
    if len(labels) == channel_count and len(set(labels)) == channel_count:
        return labels
    return default_channel_labels(channel_count)


def unit_scale_for(declared_unit: str, default_scale: float = 1.0) -> float:
    """Return the raw-to-microvolt multiplier for a declared LSL unit.

    Known microvolt units map to ``1.0`` and volt units to
    ``VOLTS_TO_MICROVOLTS``; an unknown or absent unit falls back to
    ``default_scale`` (the node's ``scale_to_microvolts`` policy knob).
    """

    unit = declared_unit.strip().lower()
    if unit in _MICROVOLT_UNITS:
        return 1.0
    if unit in _VOLT_UNITS:
        return VOLTS_TO_MICROVOLTS
    return default_scale


def chunk_to_channel_major_uv(
    sample_major_chunk: Sequence[Sequence[float]],
    channel_count: int,
    *,
    unit_scale: float = 1.0,
) -> list[float]:
    """Convert an LSL ``pull_chunk`` result to channel-major microvolt samples.

    ``sample_major_chunk`` is ``[n_samples][n_channels]`` (LSL's layout); the
    result is the flat channel-major ``[ch0_s0, ch0_s1, ..., ch1_s0, ...]`` list
    that ``EegFrame.samples`` requires, scaled by ``unit_scale``. An empty chunk
    yields ``[]``; a chunk whose per-sample width is not ``channel_count`` raises.
    """

    if channel_count < 1:
        raise ValueError("channel_count must be at least 1")
    if len(sample_major_chunk) == 0:
        return []

    data = np.asarray(sample_major_chunk, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != channel_count:
        raise ValueError(
            f"chunk must be sample-major with {channel_count} channels per sample, "
            f"got array of shape {data.shape}"
        )
    channel_major = np.ascontiguousarray(data.T) * unit_scale
    return channel_major.reshape(-1).tolist()


def channel_major_to_sample_major(
    channel_major_flat: Sequence[float],
    channel_count: int,
) -> list[list[float]]:
    """Inverse of :func:`chunk_to_channel_major_uv`, for feeding ``push_chunk``.

    Turns a flat channel-major frame into sample-major rows
    ``[[ch0, ch1, ...](t0), [ch0, ch1, ...](t1), ...]``. An empty input yields
    ``[]``; a length not divisible by ``channel_count`` raises.
    """

    if channel_count < 1:
        raise ValueError("channel_count must be at least 1")
    if len(channel_major_flat) == 0:
        return []
    if len(channel_major_flat) % channel_count != 0:
        raise ValueError(
            f"channel-major length {len(channel_major_flat)} is not divisible by "
            f"channel_count {channel_count}"
        )

    data = np.asarray(channel_major_flat, dtype=np.float64)
    sample_major = data.reshape(channel_count, -1).T
    return sample_major.tolist()
