"""Node-level coverage for the `lsl_test_outlet` LSL test producer.

The pure helpers it leans on (mock_signal, gdf_recording, lsl_bridge transforms,
node_params) are covered in their own sibling tests; this drives the
`LslTestOutlet` node itself in-process. It never opens a real LSL outlet: a small
hand-written FakePylsl stands in for the pylsl module (capturing every push_chunk)
and is injected by monkeypatching `import_pylsl`, exactly the external boundary the
sibling `test_lsl_bridge.py` already exercises against real pylsl. GDF mode swaps
`read_gdf_recording`/`iter_replay_frames` for in-memory fakes carrying real
microvolt data, so the metadata, framing, and looping are asserted on real call
paths rather than mocks of the node's own logic.

Skipped when ROS (rclpy) is unavailable, as in a bare pytest run.
"""

from __future__ import annotations

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

import eeg_bci_pipeline.lsl_test_outlet as outlet_module  # noqa: E402
from eeg_bci_pipeline.data.gdf_recording import (  # noqa: E402
    EegRecording,
    EegReplayFrame,
)
from eeg_bci_pipeline.eeg_frame_contract import EEG_SAMPLE_UNIT  # noqa: E402
from eeg_bci_pipeline.lsl_test_outlet import LslTestOutlet  # noqa: E402

# --- FakePylsl: a capturing stand-in for the pylsl module --------------------


class FakeXmlChild:
    """A node in the StreamInfo desc tree; records appended children/values."""

    def __init__(self, name):
        self.name = name
        self.children = []
        self.values = {}

    def append_child(self, name):
        child = FakeXmlChild(name)
        self.children.append(child)
        return child

    def append_child_value(self, name, value):
        self.values[name] = value
        return self


class FakeStreamInfo:
    """Minimal stand-in for pylsl.StreamInfo that retains its desc tree."""

    def __init__(self, *, name, type, channel_count, nominal_srate, channel_format, source_id):
        self.name = name
        self.type = type
        self.channel_count = channel_count
        self.nominal_srate = nominal_srate
        self.channel_format = channel_format
        self.source_id = source_id
        self._desc = FakeXmlChild("desc")

    def desc(self):
        return self._desc


class FakeStreamOutlet:
    """Captures push_chunk payloads instead of pushing over the network."""

    def __init__(self, info, chunk_size):
        self.info = info
        self.chunk_size = chunk_size
        self.pushed = []

    def push_chunk(self, chunk):
        self.pushed.append(chunk)


class FakePylsl:
    """A drop-in for the pylsl module the node imports via import_pylsl()."""

    cf_float32 = 1

    def __init__(self):
        self.outlets = []

    def StreamInfo(self, **kwargs):
        return FakeStreamInfo(**kwargs)

    def StreamOutlet(self, info, chunk_size):
        outlet = FakeStreamOutlet(info, chunk_size)
        self.outlets.append(outlet)
        return outlet


def _channel_paths(channels_el):
    """Read back the (label, unit, type) tuples appended to a channels element."""

    triples = []
    for channel in channels_el.children:
        triples.append(
            (
                channel.values.get("label"),
                channel.values.get("unit"),
                channel.values.get("type"),
            )
        )
    return triples


# --- ROS lifecycle + injection ----------------------------------------------


def _init(monkeypatch, *param_overrides, fake_pylsl=None):
    """Init rclpy with the given `-p key:=value` overrides and inject FakePylsl.

    Returns the FakePylsl so a test can read back captured outlets/pushes.
    """

    args = ["--ros-args"]
    for override in param_overrides:
        args += ["-p", override]
    rclpy.init(args=args)
    fake = fake_pylsl if fake_pylsl is not None else FakePylsl()
    monkeypatch.setattr(outlet_module, "import_pylsl", lambda: fake)
    return fake


def _make_recording():
    # Three channels, eight samples each, distinct per channel so a channel mix-up
    # would be visible. Values are plain microvolts well within the contract band.
    labels = ("EEG-C3", "EEG-Cz", "EEG-C4")
    samples_uv = np.array(
        [[float(c * 100 + s) for s in range(8)] for c in range(len(labels))],
        dtype=np.float64,
    )
    return EegRecording(
        source_id="rec-src",
        sampling_rate_hz=250.0,
        channel_labels=labels,
        samples_uv=samples_uv,
    )


def _patch_gdf(monkeypatch, recording):
    monkeypatch.setattr(outlet_module, "read_gdf_recording", lambda *a, **k: recording)
    monkeypatch.setattr(
        outlet_module,
        "iter_replay_frames",
        lambda rec, samples_per_frame: _frames_for(rec, samples_per_frame),
    )


def _frames_for(recording, samples_per_frame):
    # Real channel-major framing over the fake recording, mirroring the production
    # iter_replay_frames layout so _next_gdf_frame / push ordering stay faithful.
    n = recording.samples_per_channel
    for start in range(0, n, samples_per_frame):
        stop = min(start + samples_per_frame, n)
        flat = recording.samples_uv[:, start:stop].reshape(-1).tolist()
        yield EegReplayFrame(start_sample_index=start, samples=flat)


# --- guard tests: these raise before _open_outlet, no pylsl needed -----------


def test_samples_per_frame_below_one_raises():
    rclpy.init(args=["--ros-args", "-p", "mode:=synthetic", "-p", "samples_per_frame:=0"])
    try:
        with pytest.raises(ValueError, match="samples_per_frame must be at least 1"):
            LslTestOutlet()
    finally:
        rclpy.shutdown()


def test_unknown_mode_raises():
    rclpy.init(args=["--ros-args", "-p", "mode:=bogus"])
    try:
        with pytest.raises(ValueError, match="mode must be 'gdf' or 'synthetic'"):
            LslTestOutlet()
    finally:
        rclpy.shutdown()


def test_gdf_mode_empty_path_raises():
    # gdf_path defaults to "" (declared empty), so gdf mode without a path is the
    # guard's trigger; an empty-string override is unparseable as a global arg.
    rclpy.init(args=["--ros-args", "-p", "mode:=gdf"])
    try:
        with pytest.raises(ValueError, match="gdf_path parameter must point to a GDF recording"):
            LslTestOutlet()
    finally:
        rclpy.shutdown()


def test_synthetic_non_positive_rate_raises():
    rclpy.init(args=["--ros-args", "-p", "mode:=synthetic", "-p", "sampling_rate_hz:=0.0"])
    try:
        with pytest.raises(ValueError, match="sampling_rate_hz must be greater than 0"):
            LslTestOutlet()
    finally:
        rclpy.shutdown()


# --- synthetic happy path ----------------------------------------------------


def test_synthetic_init_builds_indexed_labels_and_outlet(monkeypatch):
    fake = _init(
        monkeypatch,
        "mode:=synthetic",
        "channel_count:=3",
        "sampling_rate_hz:=100.0",
        "samples_per_frame:=4",
        "stream_name:=synthX",
    )
    node = LslTestOutlet()
    try:
        # ch_NN labels for synthetic; source_id falls back to the test-outlet default.
        assert node._channel_labels == ["ch_01", "ch_02", "ch_03"]
        assert node._channel_count == 3
        assert node._sampling_rate_hz == pytest.approx(100.0)
        assert node._source_id == "lsl-test-outlet"
        assert node._sample_index == 0
        assert node._frame_index == 0

        # _open_outlet built one outlet with the chunk size and per-channel desc.
        assert len(fake.outlets) == 1
        built = fake.outlets[0]
        assert built.chunk_size == 4
        assert built.info.name == "synthX"
        assert built.info.type == "EEG"
        assert built.info.channel_count == 3
        channels_el = built.info.desc().children[0]
        assert channels_el.name == "channels"
        assert _channel_paths(channels_el) == [
            ("ch_01", EEG_SAMPLE_UNIT, "EEG"),
            ("ch_02", EEG_SAMPLE_UNIT, "EEG"),
            ("ch_03", EEG_SAMPLE_UNIT, "EEG"),
        ]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_synthetic_next_frame_advances_indices_and_cycles_amplitude(monkeypatch):
    _init(
        monkeypatch,
        "mode:=synthetic",
        "channel_count:=2",
        "sampling_rate_hz:=100.0",
        "samples_per_frame:=5",
        "amplitude_cycle_uv:=[10.0,40.0]",
        "frames_per_intent:=1",
    )
    node = LslTestOutlet()
    try:
        first = node._next_synthetic_frame()
        # channel_count * samples_per_frame values, channel-major.
        assert len(first) == 2 * 5
        # Indices advanced by one frame's worth.
        assert node._sample_index == 5
        assert node._frame_index == 1

        second = node._next_synthetic_frame()
        assert len(second) == 2 * 5
        assert node._sample_index == 10
        assert node._frame_index == 2

        # frames_per_intent=1 over a 2-value cycle: frame 0 used amp 10, frame 1
        # used amp 40, so the waveforms differ (amplitude actually cycled).
        assert first != pytest.approx(second)
        # And the peak of the second frame is larger, matching the 10 -> 40 step.
        assert max(abs(v) for v in second) > max(abs(v) for v in first)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_pushes_sample_major_chunk(monkeypatch):
    fake = _init(
        monkeypatch,
        "mode:=synthetic",
        "channel_count:=2",
        "sampling_rate_hz:=100.0",
        "samples_per_frame:=3",
    )
    node = LslTestOutlet()
    try:
        outlet = fake.outlets[0]
        node._on_tick()

        assert len(outlet.pushed) == 1
        chunk = outlet.pushed[0]
        # Sample-major: samples_per_frame rows, channel_count columns each.
        assert len(chunk) == 3
        assert all(len(row) == 2 for row in chunk)

        # The pushed chunk is the channel-major frame transposed: rebuild the
        # channel-major form from the chunk and compare to a freshly generated frame.
        channel_major = [chunk[s][c] for c in range(2) for s in range(3)]
        node._sample_index = 0
        node._frame_index = 0
        expected = node._next_synthetic_frame()
        assert channel_major == pytest.approx(expected)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --- gdf mode ----------------------------------------------------------------


def test_gdf_mode_populates_frames_and_metadata(monkeypatch):
    recording = _make_recording()
    fake = _init(monkeypatch, "mode:=gdf", "gdf_path:=/fake/A01T.gdf", "samples_per_frame:=4")
    _patch_gdf(monkeypatch, recording)
    node = LslTestOutlet()
    try:
        # Recording's own labels/rate/source carried verbatim into the stream.
        assert node._channel_labels == list(recording.channel_labels)
        assert node._channel_count == 3
        assert node._sampling_rate_hz == pytest.approx(250.0)
        assert node._source_id == "rec-src"
        assert node._loop is True
        # 8 samples/channel at 4 per frame -> 2 frames, each channel-major flat.
        assert len(node._frames) == 2
        assert all(len(frame) == 3 * 4 for frame in node._frames)
        assert node._frame_index == 0

        # Outlet desc carries the recording's real labels with the microvolt unit.
        channels_el = fake.outlets[0].info.desc().children[0]
        assert _channel_paths(channels_el) == [
            (label, EEG_SAMPLE_UNIT, "EEG") for label in recording.channel_labels
        ]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_gdf_next_frame_loops_when_loop_true(monkeypatch):
    recording = _make_recording()
    _init(
        monkeypatch, "mode:=gdf", "gdf_path:=/fake/A01T.gdf", "samples_per_frame:=4", "loop:=true"
    )
    _patch_gdf(monkeypatch, recording)
    node = LslTestOutlet()
    try:
        first = node._next_gdf_frame()
        second = node._next_gdf_frame()
        assert node._frame_index == 2
        # Exhausted both frames; with loop on, the next call wraps to frame 0.
        wrapped = node._next_gdf_frame()
        assert wrapped is not None
        assert wrapped == first
        assert node._frame_index == 1
        assert second != first
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_gdf_next_frame_returns_none_at_end_when_loop_false(monkeypatch):
    recording = _make_recording()
    _init(
        monkeypatch, "mode:=gdf", "gdf_path:=/fake/A01T.gdf", "samples_per_frame:=4", "loop:=false"
    )
    _patch_gdf(monkeypatch, recording)
    node = LslTestOutlet()
    try:
        assert node._loop is False
        node._next_gdf_frame()
        node._next_gdf_frame()
        # Both frames consumed and looping off: the generator signals end-of-stream.
        assert node._next_gdf_frame() is None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_cancels_timer_at_end_of_non_looping_gdf(monkeypatch):
    recording = _make_recording()
    fake = _init(
        monkeypatch, "mode:=gdf", "gdf_path:=/fake/A01T.gdf", "samples_per_frame:=4", "loop:=false"
    )
    _patch_gdf(monkeypatch, recording)
    node = LslTestOutlet()
    try:
        # Two frames push, the third tick hits end-of-stream and cancels the timer.
        node._on_tick()
        node._on_tick()
        assert len(fake.outlets[0].pushed) == 2
        assert node._timer.is_canceled() is False

        node._on_tick()
        assert len(fake.outlets[0].pushed) == 2  # no further push after exhaustion
        assert node._timer.is_canceled() is True
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_gdf_empty_replay_frames_raises(monkeypatch):
    recording = _make_recording()
    _init(monkeypatch, "mode:=gdf", "gdf_path:=/fake/A01T.gdf", "samples_per_frame:=4")
    monkeypatch.setattr(outlet_module, "read_gdf_recording", lambda *a, **k: recording)
    monkeypatch.setattr(
        outlet_module, "iter_replay_frames", lambda rec, samples_per_frame: iter(())
    )
    try:
        with pytest.raises(ValueError, match="GDF recording contains no replay frames"):
            LslTestOutlet()
    finally:
        rclpy.shutdown()


def test_gdf_channel_labels_override_passed_through(monkeypatch):
    # The channel_labels parameter is forwarded to read_gdf_recording as an override;
    # capture it to pin the wiring (the recording itself supplies the final labels).
    recording = _make_recording()
    _init(
        monkeypatch,
        "mode:=gdf",
        "gdf_path:=/fake/A01T.gdf",
        "samples_per_frame:=4",
        "channel_labels:=[X1,X2,X3]",
    )
    captured = {}

    def fake_read(path, channel_labels=None):
        captured["channel_labels"] = channel_labels
        return recording

    monkeypatch.setattr(outlet_module, "read_gdf_recording", fake_read)
    monkeypatch.setattr(
        outlet_module,
        "iter_replay_frames",
        lambda rec, samples_per_frame: _frames_for(rec, samples_per_frame),
    )
    node = LslTestOutlet()
    try:
        assert captured["channel_labels"] == ["X1", "X2", "X3"]
    finally:
        node.destroy_node()
        rclpy.shutdown()
