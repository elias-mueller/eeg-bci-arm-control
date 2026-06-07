"""Node-level coverage for the synthetic `mock_eeg_publisher`.

The pure signal helpers (`select_cycle_value`, `generate_mock_eeg_samples`) are
covered in `test_mock_signal.py`; this drives the `MockEegPublisher` node itself,
which wires those helpers to an `EegFrame` publisher and a cycling amplitude
counter. Custom parameters are injected through real ROS global args at
`rclpy.init` (the node takes no constructor args), so the parameter-read path is
exercised end to end rather than poked. Published frames are captured by
replacing the publisher's `publish` with a list `.append`, avoiding any spin or
wall-clock wait, so the test is deterministic.

Skipped when ROS (rclpy / the interfaces) is unavailable, as in a bare pytest run.
"""

from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("eeg_bci_interfaces.msg")

from eeg_bci_pipeline.eeg_frame_contract import (  # noqa: E402
    DEFAULT_EEG_CHANNEL_COUNT,
    DEFAULT_EEG_SAMPLING_RATE_HZ,
    default_channel_labels,
)
from eeg_bci_pipeline.mock_eeg_publisher import MockEegPublisher  # noqa: E402
from eeg_bci_pipeline.mock_signal import DEFAULT_AMPLITUDE_CYCLE_UV  # noqa: E402


def _make_node(**ros_params):
    # The node declares its parameters and reads them in __init__, with no
    # constructor overrides, so custom values must arrive as real global ROS
    # args. init/return the node and let the caller own teardown via the fixture.
    args = ["--ros-args"]
    for name, value in ros_params.items():
        args += ["-p", f"{name}:={value}"]
    rclpy.init(args=args)
    return MockEegPublisher()


@pytest.fixture
def default_node():
    node = _make_node()
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.fixture
def custom_node():
    # Small, fast frames; a two-entry amplitude cycle with a large contrast so the
    # per-frame peak magnitude is unambiguous, and one frame per intent so the
    # amplitude advances on every published frame.
    node = _make_node(
        channel_count=4,
        samples_per_frame=8,
        amplitude_cycle_uv="[1.0,100.0]",
        frames_per_intent=1,
        sampling_rate_hz=200.0,
        source_id="custom-src",
    )
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _capture_published(node) -> list:
    published: list = []
    node._publisher.publish = published.append
    return published


def test_init_defaults_match_contract(default_node):
    assert default_node._channel_count == DEFAULT_EEG_CHANNEL_COUNT
    assert default_node._sampling_rate_hz == pytest.approx(DEFAULT_EEG_SAMPLING_RATE_HZ)
    assert default_node._samples_per_frame == 25
    assert default_node._amplitude_cycle_uv == DEFAULT_AMPLITUDE_CYCLE_UV
    assert default_node._frames_per_intent == 10
    assert default_node._source_id == "mock-eeg"
    # Labels are the stable fallback set, one per channel.
    assert default_node._channel_labels == list(default_channel_labels(DEFAULT_EEG_CHANNEL_COUNT))
    # Publisher and timer are constructed and the cursors start at zero.
    assert default_node._publisher is not None
    assert default_node._timer is not None
    assert default_node._sample_index == 0
    assert default_node._frame_index == 0


def test_init_custom_params_stored(custom_node):
    assert custom_node._channel_count == 4
    assert custom_node._samples_per_frame == 8
    assert custom_node._amplitude_cycle_uv == (1.0, 100.0)
    assert custom_node._frames_per_intent == 1
    assert custom_node._sampling_rate_hz == pytest.approx(200.0)
    assert custom_node._source_id == "custom-src"
    assert custom_node._channel_labels == list(default_channel_labels(4))


def test_publish_frame_builds_and_publishes_one_eeg_frame(custom_node):
    published = _capture_published(custom_node)

    custom_node._publish_frame()

    assert len(published) == 1
    frame = published[0]
    assert frame.header.frame_id == "eeg"
    assert frame.source_id == "custom-src"
    assert frame.sampling_rate_hz == pytest.approx(200.0)
    assert list(frame.channel_labels) == list(default_channel_labels(4))
    # Channel-major payload: one value per channel-sample, and non-empty.
    assert len(frame.samples) == 4 * 8
    assert any(value != 0.0 for value in frame.samples)


def test_generate_samples_shape_and_cursor_advance(custom_node):
    expected_len = custom_node._channel_count * custom_node._samples_per_frame

    for call_count in range(1, 4):
        samples = custom_node._generate_samples()
        assert len(samples) == expected_len
        # Each call advances the sample cursor by one frame and the frame counter
        # by one, so a later frame draws from a later point in the signal.
        assert custom_node._sample_index == call_count * custom_node._samples_per_frame
        assert custom_node._frame_index == call_count


def _record_cycle_amplitudes(node, monkeypatch, frame_count):
    # Capture the amplitude the node selects for each frame. A short sine window
    # does not reach its analytic peak, so comparing raw per-frame peaks is phase
    # dependent; recording the cycle value the node feeds the generator pins the
    # cycling logic directly and deterministically.
    chosen: list[float] = []
    import eeg_bci_pipeline.mock_eeg_publisher as module

    real_select = module.select_cycle_value

    def spy(values, frame_index, frames_per_value):
        amplitude = real_select(values, frame_index, frames_per_value)
        chosen.append(amplitude)
        return amplitude

    monkeypatch.setattr(module, "select_cycle_value", spy)
    for _ in range(frame_count):
        node._generate_samples()
    return chosen


def test_generate_samples_cycles_amplitude_across_intents(custom_node, monkeypatch):
    # frames_per_intent == 1 with a [1.0, 100.0] cycle means consecutive frames
    # alternate between the low and high amplitude, repeating every cycle length.
    amplitudes = _record_cycle_amplitudes(custom_node, monkeypatch, frame_count=4)

    assert amplitudes == [1.0, 100.0, 1.0, 100.0]


def test_generate_samples_holds_amplitude_within_one_intent():
    # frames_per_intent > 1 must keep the amplitude fixed for that many frames
    # before advancing; pins the cycle-index division, not just the modulo.
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "channel_count:=2",
            "-p",
            "samples_per_frame:=8",
            "-p",
            "amplitude_cycle_uv:=[1.0,100.0]",
            "-p",
            "frames_per_intent:=2",
        ]
    )
    node = MockEegPublisher()
    monkeypatch = pytest.MonkeyPatch()
    try:
        amplitudes = _record_cycle_amplitudes(node, monkeypatch, frame_count=4)
    finally:
        monkeypatch.undo()
        node.destroy_node()
        rclpy.shutdown()

    # Two frames at the low amplitude, then two at the high amplitude.
    assert amplitudes == [1.0, 1.0, 100.0, 100.0]
