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

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from eeg_bci_pipeline.data.gdf_recording import (
    VOLTS_TO_MICROVOLTS,
    normalize_channel_labels,
)
from eeg_bci_pipeline.eeg_frame_contract import default_channel_labels

_MICROVOLT_UNITS = frozenset({"uv", "microvolt", "microvolts", "µv", "μv"})
_VOLT_UNITS = frozenset({"v", "volt", "volts"})

NANOSECONDS_PER_SECOND = 1_000_000_000
# How far the LSL-derived stamp may wander from the actual ROS arrival time before
# the aligner re-anchors. 0.25 s comfortably absorbs normal pull jitter at 250 Hz
# while still bounding the offset a multi-minute session can accumulate.
DEFAULT_RESYNC_THRESHOLD_SEC = 0.25


def build_resolve_predicate(stream_name: str, stream_type: str) -> str:
    """Build an LSL resolve predicate from an optional name and/or type.

    Conjoins ``name='...'`` and ``type='...'`` so a same-named non-EEG stream is
    not matched; at least one of the two must be non-empty. Single quotes are
    rejected rather than escaped: LSL's XPath-subset predicate has no portable
    escape, and a quote is never legitimate in a stream name or type here.
    """

    if "'" in stream_name or "'" in stream_type:
        raise ValueError("stream_name and stream_type must not contain a single quote")
    clauses: list[str] = []
    if stream_name:
        clauses.append(f"name='{stream_name}'")
    if stream_type:
        clauses.append(f"type='{stream_type}'")
    if not clauses:
        raise ValueError("set stream_name and/or stream_type to resolve an LSL stream")
    return " and ".join(clauses)


@dataclass
class LslClockAligner:
    """Map LSL sample-clock timestamps onto the ROS clock, re-anchoring on drift.

    LSL stamps each sample on its own sample clock, which is not the ROS clock.
    Anchoring ``(ros_time, lsl_time)`` at the first sample and deriving later stamps
    as ``ros_anchor + (lsl_ts - lsl_anchor)`` preserves the inter-sample spacing the
    stream reports, but a single fixed anchor lets the two clocks drift apart over a
    long session (and silently absorbs a backward LSL jump). This re-anchors whenever
    the *newest* sample's derived time leaves a ``resync_threshold_sec`` band around
    the real arrival (ROS) time, bounding that drift while staying on the sample clock
    within the band. Using the newest sample as the drift signal (not the oldest) is
    deliberate: it keeps a drained processing backlog, where the oldest sample is far
    behind arrival but the newest is recent, from being mistaken for drift.
    :meth:`reset` re-arms it after a stream drop.

    The emitted stamp is held monotonic non-decreasing: a re-anchor snaps the
    timebase by up to ``resync_threshold_sec``, which would otherwise step the
    published ``header.stamp`` backward when the LSL clock runs ahead of ROS. ROS
    sensor streams must not go backward, so a re-anchor that would do so plateaus at
    the last stamp until the clocks realign instead. (A smoothly slewed offset would
    avoid the plateau too, at more complexity than this path warrants.)

    Pure integer/float arithmetic (nanoseconds in, nanoseconds out) so the node's
    timing logic is unit-testable without a ROS clock.
    """

    resync_threshold_sec: float = DEFAULT_RESYNC_THRESHOLD_SEC
    _threshold_ns: int = field(default=0, init=False, repr=False)
    _ros_anchor_ns: int | None = field(default=None, init=False, repr=False)
    _lsl_anchor_sec: float = field(default=0.0, init=False, repr=False)
    _last_stamp_ns: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.resync_threshold_sec <= 0.0:
            raise ValueError("resync_threshold_sec must be greater than 0")
        # Fixed once; recomputing the int threshold on every chunk (~50 Hz) is waste.
        self._threshold_ns = round(self.resync_threshold_sec * NANOSECONDS_PER_SECOND)

    def reset(self) -> None:
        """Drop the anchor and clamp so the next sample starts a fresh monotonic run.

        Called on a stream drop/reconnect, a genuine discontinuity: clearing the
        last-stamp clamp too means an earlier-arriving reconnect re-anchors cleanly
        instead of freezing at the stale pre-drop stamp.
        """

        self._ros_anchor_ns = None
        self._last_stamp_ns = None

    def stamp_ns(
        self,
        ros_now_ns: int,
        first_lsl_ts_sec: float,
        last_lsl_ts_sec: float | None = None,
    ) -> int:
        """Return the ROS-clock nanosecond stamp for a chunk's first sample.

        ``last_lsl_ts_sec`` (the chunk's newest sample; defaults to the first) is the
        drift signal, so a processing backlog is not mistaken for clock drift: the
        newest sample is recent regardless of how many buffered samples drained, so
        only a genuine offset of the newest sample from arrival re-anchors. The
        emitted stamp is always derived for the first (oldest) sample, preserving its
        true acquisition time even when a backlog is being drained.
        """

        if last_lsl_ts_sec is None:
            last_lsl_ts_sec = first_lsl_ts_sec
        anchor_ns = self._ros_anchor_ns
        if (
            anchor_ns is None
            or abs(
                ros_now_ns
                - (
                    anchor_ns
                    + round((last_lsl_ts_sec - self._lsl_anchor_sec) * NANOSECONDS_PER_SECOND)
                )
            )
            > self._threshold_ns
        ):
            # First sample, or the newest sample is genuinely off from arrival (clock
            # drift, not a backlog): anchor the newest sample to the current arrival.
            anchor_ns = ros_now_ns
            self._ros_anchor_ns = ros_now_ns
            self._lsl_anchor_sec = last_lsl_ts_sec
        stamp_ns = anchor_ns + round(
            (first_lsl_ts_sec - self._lsl_anchor_sec) * NANOSECONDS_PER_SECOND
        )
        # Hold monotonic non-decreasing and non-negative (see the class docstring): a
        # re-anchor must never publish a stamp earlier than the last one.
        floor_ns = 0 if self._last_stamp_ns is None else self._last_stamp_ns
        if stamp_ns < floor_ns:
            stamp_ns = floor_ns
        self._last_stamp_ns = stamp_ns
        return stamp_ns


def import_pylsl() -> Any:
    """Import pylsl lazily, raising a clear setup error if it (or liblsl) is absent.

    Mirrors the lazy ``import mne`` pattern in ``data/gdf_recording.py``: the LSL
    nodes call this so the rest of the package stays importable without Lab
    Streaming Layer installed. Catches any import failure, including pylsl's
    RuntimeError when the native liblsl cannot be loaded.
    """

    try:
        import pylsl  # pyright: ignore[reportMissingImports]
    except Exception as error:  # pragma: no cover
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
    # Per-channel declared LSL type (e.g. "EEG"), document order, length
    # channel_count ("" where a channel declared none). Lets the node forward only
    # a subset (e.g. type="EEG") of a stream that also carries contact /
    # accelerometer / battery channels, as the BrainAccess MIDI does.
    channel_types: tuple[str, ...] = ()


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
    declared_labels, declared_types, declared_unit = _read_channel_desc(info.desc(), channel_count)
    channel_labels = resolve_channel_labels(declared_labels, channel_count)
    declared_present = any(label.strip() for label in declared_labels)
    labels_downgraded = declared_present and channel_labels == default_channel_labels(
        channel_count
    )
    # Pad to channel_count so the types align to sample columns even when the desc
    # declares fewer <channel> nodes than the stream advertises.
    channel_types = tuple(declared_types[:channel_count]) + ("",) * (
        channel_count - len(declared_types)
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
        channel_types=channel_types,
    )


def _read_channel_desc(
    desc: LslXmlElementLike,
    channel_count: int,
) -> tuple[list[str], list[str], str]:
    """Walk ``desc/channels/channel`` in document order for labels, types, and a unit.

    Single-unit assumption: the first non-empty channel unit is taken as the whole
    stream's unit. Heterogeneous per-channel units (rare for an EEG stream) are not
    modeled; every channel is scaled by that one unit. Per-channel ``type`` is kept
    individually so the node can select a subset (e.g. only ``type="EEG"``).
    """

    labels: list[str] = []
    types: list[str] = []
    declared_unit = ""
    node = desc.child("channels").child("channel")
    while not node.empty() and len(labels) < channel_count:
        labels.append(node.child_value("label"))
        types.append(node.child_value("type"))
        if not declared_unit:
            unit = node.child_value("unit").strip().lower()
            if unit:
                declared_unit = unit
        node = node.next_sibling()
    return labels, types, declared_unit


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


def select_channel_indices(channel_types: Sequence[str], select_type: str) -> tuple[int, ...]:
    """Return the indices of channels whose declared type equals ``select_type``.

    Case-insensitive, document order. Lets the node forward only e.g. the
    ``type="EEG"`` channels of a stream that also carries contact / accelerometer /
    battery channels (the BrainAccess MIDI publishes all of them in one outlet).
    Raises if ``select_type`` is empty or no channel matches, so a misconfigured
    selection fails fast at launch instead of yielding a zero-width frame.
    """

    target = select_type.strip().lower()
    if not target:
        raise ValueError("select_type must be non-empty")
    indices = tuple(
        index for index, ctype in enumerate(channel_types) if ctype.strip().lower() == target
    )
    if not indices:
        raise ValueError(f"no channels with declared type '{select_type}'")
    return indices


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
    keep_indices: Sequence[int] | None = None,
) -> list[float]:
    """Convert an LSL ``pull_chunk`` result to channel-major microvolt samples.

    ``sample_major_chunk`` is ``[n_samples][n_channels]`` (LSL's layout) where
    ``n_channels`` is ``channel_count`` (the full stream width, validated); the
    result is the flat channel-major ``[ch0_s0, ch0_s1, ..., ch1_s0, ...]`` list
    that ``EegFrame.samples`` requires, scaled by ``unit_scale``. When
    ``keep_indices`` is given, only those channel columns are forwarded (in the
    given order), so a stream carrying non-EEG channels can be subset down. An
    empty chunk yields ``[]``; a chunk whose per-sample width is not
    ``channel_count`` raises.
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
    if keep_indices is not None:
        data = data[:, list(keep_indices)]
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


@dataclass
class CausalDcBlocker:
    """Stateful one-pole high-pass (DC blocker) for channel-major microvolt frames.

    Dry-electrode EEG (e.g. the BrainAccess MIDI over LSL) rides on a large, slowly
    varying electrode DC offset, hundreds of millivolts, far above the EegFrame
    amplitude contract, even though the neural AC content is a healthy ~100 uV. This
    strips that pedestal causally (no look-ahead, so it is valid for real-time and
    shareable with the embedded front-end) with the classic one-pole DC blocker::

        y[n] = x[n] - x[n-1] + pole * y[n-1],   pole = exp(-2*pi*fc/fs)

    a transfer-function zero at DC and a pole near the unit circle: the response is
    ~0 at DC and ~unity through the mu/beta band, so a 0.5 Hz cutoff removes the
    offset without touching motor rhythms. State (previous input and output per
    channel) persists across chunks so the filter is continuous over the stream. On
    the first chunk it lazily anchors ``x[-1]`` to each channel's first sample
    (``y[-1] = 0``), so a constant pedestal yields ~0 output from the very first
    sample rather than a one-sample full-amplitude transient that would trip the
    contract at launch. :meth:`reset` re-arms it after a stream drop.

    Pure NumPy (channel-vectorized over a short per-chunk time loop) so the node's
    DSP is unit-testable without ROS or LSL.
    """

    cutoff_hz: float
    sampling_rate_hz: float
    channel_count: int
    _pole: float = field(default=0.0, init=False, repr=False)
    _x_prev: Any = field(default=None, init=False, repr=False)
    _y_prev: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be greater than 0")
        if self.sampling_rate_hz <= 0.0:
            raise ValueError("sampling_rate_hz must be greater than 0")
        if self.channel_count < 1:
            raise ValueError("channel_count must be at least 1")
        self._pole = math.exp(-2.0 * math.pi * self.cutoff_hz / self.sampling_rate_hz)

    def reset(self) -> None:
        """Drop the filter state so the next chunk re-anchors (after a stream drop)."""

        self._x_prev = None
        self._y_prev = None

    def process(self, channel_major_flat: Sequence[float]) -> list[float]:
        """High-pass a flat channel-major frame, carrying filter state across calls.

        Input/output layout matches :func:`chunk_to_channel_major_uv`
        (``[ch0_s0, ch0_s1, ..., ch1_s0, ...]``). An empty frame yields ``[]``; a
        length not divisible by ``channel_count`` raises.
        """

        length = len(channel_major_flat)
        if length == 0:
            return []
        if length % self.channel_count != 0:
            raise ValueError(
                f"channel-major length {length} is not divisible by "
                f"channel_count {self.channel_count}"
            )
        data = np.asarray(channel_major_flat, dtype=np.float64).reshape(self.channel_count, -1)
        if self._x_prev is None:
            self._x_prev = data[:, 0].copy()
            self._y_prev = np.zeros(self.channel_count, dtype=np.float64)
        x_prev = self._x_prev
        y_prev = self._y_prev
        pole = self._pole
        out = np.empty_like(data)
        for index in range(data.shape[1]):
            x_t = data[:, index]
            y_t = x_t - x_prev + pole * y_prev
            out[:, index] = y_t
            x_prev = x_t
            y_prev = y_t
        self._x_prev = np.asarray(x_prev).copy()
        self._y_prev = np.asarray(y_prev).copy()
        return out.reshape(-1).tolist()
