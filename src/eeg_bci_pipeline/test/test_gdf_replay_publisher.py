"""Node-level coverage for the GDF -> EegFrame replay publisher.

The replay helpers (read_gdf_recording, iter_replay_frames, replay_elapsed_sec)
are covered in test_gdf_recording.py; this drives the GdfReplayPublisher node
itself. read_gdf_recording is the file-reading boundary, so it is monkeypatched
to hand back a small in-memory EegRecording instead of touching MNE/disk: a fake
that carries REAL channel-major microvolt samples, so the node walks its actual
flatten -> validate -> publish path on faithful data. Frames are captured by
replacing the publisher's publish with a list .append, and the timer is stepped
by calling _publish_frame directly (no wall-clock waits), so the test is
deterministic yet exercises the real frame assembly, loop wrap, and stamp logic.

Skipped when ROS (rclpy / the interfaces) is unavailable, as in a bare pytest run.
"""

from __future__ import annotations

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("eeg_bci_interfaces.msg")

import eeg_bci_pipeline.gdf_replay_publisher as module  # noqa: E402
from eeg_bci_pipeline.data.gdf_recording import (  # noqa: E402
    EegRecording,
    iter_replay_frames,
)
from eeg_bci_pipeline.eeg_frame_contract import EegFrameContractError  # noqa: E402
from eeg_bci_pipeline.gdf_replay_publisher import GdfReplayPublisher  # noqa: E402

CHANNELS = ("C3", "Cz", "C4")
RATE_HZ = 250.0
SAMPLES_PER_CHANNEL = 5


def _recording(
    *,
    channel_labels=CHANNELS,
    rate_hz=RATE_HZ,
    samples_per_channel=SAMPLES_PER_CHANNEL,
    source_id="A01T",
) -> EegRecording:
    # A per-channel, per-sample ramp in microvolts: distinct for every (channel,
    # sample) so a flatten/order bug would change the published payload detectably,
    # and small enough to stay under the contract's max_abs_sample_uv.
    channels = len(channel_labels)
    samples_uv = np.fromfunction(
        lambda c, s: c * 10.0 + s,
        (channels, samples_per_channel),
        dtype=np.float64,
    )
    return EegRecording(
        source_id=source_id,
        sampling_rate_hz=rate_hz,
        channel_labels=tuple(channel_labels),
        samples_uv=samples_uv,
    )


def _init_ros(**overrides) -> None:
    # GdfReplayPublisher reads its parameters during __init__, so overrides must be
    # present before the node is constructed; pass them through the global context
    # the same way a launch file's -p arguments would arrive.
    args = ["--ros-args"]
    for name, value in overrides.items():
        args += ["-p", f"{name}:={value}"]
    rclpy.init(args=args)


def _make_node(recording, monkeypatch, *, frames=None, **overrides) -> GdfReplayPublisher:
    # Boundary mock: stub the file-reading helper so no GDF/MNE is touched. The node
    # still runs its own iter_replay_frames flatten unless a frame list is injected.
    monkeypatch.setattr(module, "read_gdf_recording", lambda path, channel_labels=None: recording)
    if frames is not None:
        monkeypatch.setattr(module, "iter_replay_frames", lambda rec, samples_per_frame: frames)
    overrides.setdefault("gdf_path", "/data/A01T.gdf")
    # The first-frame contract check defaults to 22 channels (BCIC IV 2a); point it
    # at the fake recording's width so construction succeeds for the happy-path tests.
    overrides.setdefault("expected_channel_count", len(recording.channel_labels))
    _init_ros(**overrides)
    try:
        node = GdfReplayPublisher()
    except BaseException:
        rclpy.shutdown()  # don't leak the context into the next test
        raise
    node._published = []
    node._publisher.publish = node._published.append
    return node


def test_empty_gdf_path_raises_value_error():
    # The gdf_path parameter defaults to the empty string; left unset, the node must
    # refuse to start rather than try to read a nameless recording. (An empty value
    # cannot be passed as a ROS -p override, so this exercises the declared default.)
    _init_ros()
    try:
        with pytest.raises(ValueError, match="gdf_path"):
            GdfReplayPublisher()
    finally:
        rclpy.shutdown()


def test_empty_replay_frames_raises_value_error(monkeypatch):
    # A recording with zero samples flattens to no frames; the node must reject it
    # rather than create a publisher that never emits.
    monkeypatch.setattr(
        module, "read_gdf_recording", lambda path, channel_labels=None: _recording()
    )
    monkeypatch.setattr(module, "iter_replay_frames", lambda rec, samples_per_frame: [])
    _init_ros(gdf_path="/data/A01T.gdf")
    try:
        with pytest.raises(ValueError, match="no replay frames"):
            GdfReplayPublisher()
    finally:
        rclpy.shutdown()


def test_constructs_publisher_and_caches_frames(monkeypatch):
    recording = _recording()
    node = _make_node(recording, monkeypatch, samples_per_frame=2)
    try:
        # The frame cache is materialized (a list, not a one-shot iterator), so loop
        # mode can replay it without re-flattening.
        assert isinstance(node._frames, list)
        expected_frames = list(iter_replay_frames(recording, samples_per_frame=2))
        assert [f.samples for f in node._frames] == [f.samples for f in expected_frames]
        assert node._frame_index == 0
        assert node._loop_index == 0
        assert node._recording is recording
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_explicit_channel_labels_override_expected_count(monkeypatch):
    # An explicit channel_labels override pins _expected_channel_count to its length,
    # ignoring the declared expected_channel_count parameter (here a mismatched 22).
    recording = _recording()
    node = _make_node(
        recording,
        monkeypatch,
        channel_labels="[C3,Cz,C4]",
        expected_channel_count=22,
    )
    try:
        assert node._expected_channel_count == len(CHANNELS)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_expected_count_param_used_when_no_label_override(monkeypatch):
    # Without a channel_labels override, the declared expected_channel_count stands;
    # set it to match the fake recording's channel count so the first-frame check
    # passes (otherwise the count-mismatch branch would raise instead).
    recording = _recording()
    node = _make_node(recording, monkeypatch, expected_channel_count=len(CHANNELS))
    try:
        assert node._expected_channel_count == len(CHANNELS)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_first_frame_validation_propagates_contract_error(monkeypatch):
    # The first frame is validated at construction; a channel-count mismatch must
    # surface as the contract error, not be swallowed.
    recording = _recording()
    _init_ros(gdf_path="/data/A01T.gdf", expected_channel_count=22)
    monkeypatch.setattr(module, "read_gdf_recording", lambda path, channel_labels=None: recording)
    try:
        with pytest.raises(EegFrameContractError):
            GdfReplayPublisher()
    finally:
        rclpy.shutdown()


def test_timer_period_is_samples_per_frame_over_rate(monkeypatch):
    recording = _recording(rate_hz=100.0)
    node = _make_node(recording, monkeypatch, samples_per_frame=5, expected_sampling_rate_hz=100.0)
    try:
        # 5 samples / 100 Hz = 0.05 s per published frame. timer_period_ns is the
        # rclpy-visible record of the period the node asked for.
        assert node._timer.timer_period_ns == pytest.approx(0.05 * 1e9)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_publish_frame_emits_faithful_frames_and_advances_index(monkeypatch):
    recording = _recording()
    node = _make_node(recording, monkeypatch, samples_per_frame=2)
    try:
        node._publish_frame()
        assert node._frame_index == 1
        assert len(node._published) == 1

        frame = node._published[0]
        assert frame.source_id == recording.source_id
        assert frame.sampling_rate_hz == pytest.approx(recording.sampling_rate_hz)
        assert tuple(frame.channel_labels) == recording.channel_labels
        assert frame.header.frame_id == "eeg"
        # First frame is the first samples_per_frame columns, channel-major.
        expected = recording.samples_uv[:, 0:2].reshape(-1)
        assert list(frame.samples) == pytest.approx(expected.tolist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_publish_frame_cancels_timer_at_end_when_not_looping(monkeypatch):
    recording = _recording()
    # One sample per frame -> exactly SAMPLES_PER_CHANNEL frames, then the terminal
    # tick fires the no-more-frames, no-loop branch.
    node = _make_node(recording, monkeypatch, samples_per_frame=1, loop="false")
    try:
        for _ in range(SAMPLES_PER_CHANNEL):
            node._publish_frame()
        assert len(node._published) == SAMPLES_PER_CHANNEL
        assert not node._timer.is_canceled()

        node._publish_frame()  # index past end, loop off -> cancel, no new publish
        assert len(node._published) == SAMPLES_PER_CHANNEL
        assert node._timer.is_canceled()
        assert node._loop_index == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_publish_frame_wraps_and_increments_loop_index_when_looping(monkeypatch):
    recording = _recording()
    node = _make_node(recording, monkeypatch, samples_per_frame=1, loop="true")
    try:
        for _ in range(SAMPLES_PER_CHANNEL):
            node._publish_frame()
        assert node._frame_index == SAMPLES_PER_CHANNEL
        assert node._loop_index == 0

        # The wrap tick advances loop_index, resets frame_index, and republishes the
        # first frame instead of cancelling the timer.
        node._publish_frame()
        assert node._loop_index == 1
        assert node._frame_index == 1
        assert not node._timer.is_canceled()
        assert len(node._published) == SAMPLES_PER_CHANNEL + 1
        first = recording.samples_uv[:, 0:1].reshape(-1)
        assert list(node._published[-1].samples) == pytest.approx(first.tolist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_stamp_for_sample_is_monotonic_across_loop_cycles(monkeypatch):
    recording = _recording(rate_hz=10.0)
    node = _make_node(
        recording,
        monkeypatch,
        samples_per_frame=1,
        loop="true",
        expected_sampling_rate_hz=10.0,
    )
    try:
        anchor = node._replay_start_time
        # Sample 0 of the first cycle stamps at the replay anchor.
        first = node._stamp_for_sample(0)
        first_ns = first.sec * 1_000_000_000 + first.nanosec
        anchor_ns = anchor.nanoseconds
        assert first_ns == pytest.approx(anchor_ns, abs=1)

        # After one completed loop, sample 0 sits a full recording length later:
        # 5 samples / 10 Hz = 0.5 s past the anchor, so the stamp keeps climbing.
        node._loop_index = 1
        wrapped = node._stamp_for_sample(0)
        wrapped_ns = wrapped.sec * 1_000_000_000 + wrapped.nanosec
        assert wrapped_ns > first_ns
        assert wrapped_ns - anchor_ns == pytest.approx(0.5 * 1e9, abs=1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_optional_string_array_empty_is_none_and_nonempty_is_normalized(monkeypatch):
    recording = _recording()
    # No channel_labels override -> the parameter is the empty default -> None.
    node = _make_node(recording, monkeypatch)
    try:
        assert node._optional_string_array("channel_labels") is None
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # A populated, whitespace-padded override normalizes to a trimmed tuple.
    node2 = _make_node(recording, monkeypatch, channel_labels="[' C3 ',Cz,C4]")
    try:
        assert node2._optional_string_array("channel_labels") == ("C3", "Cz", "C4")
    finally:
        node2.destroy_node()
        rclpy.shutdown()
