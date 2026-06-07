"""Node-level coverage for the `mock_intent_decoder` frame-to-intent glue.

The pure decoder (`decode_mock_intent`) and the frame contract are covered in
`test_decoder.py` / `test_eeg_frame_contract.py`; this drives the
`MockIntentDecoder` node itself. It feeds real `EegFrame` messages through the
node's actual `_on_eeg_frame` path and captures what the node would publish by
replacing `_publisher.publish` with a list's `append`, so the assertions are on
the observable Intent output rather than poked-at private state. Frame payloads
are constructed with deterministic peaks/RMS so the contract's small-peak gate and
the decoder's class selection are predictable.

Skipped when ROS (rclpy / the interfaces) is unavailable, as in a bare pytest run.
"""

from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("eeg_bci_interfaces.msg")

from eeg_bci_pipeline.mock_intent_decoder import (  # noqa: E402
    CONTRACT_WARNING_INTERVAL_FRAMES,
    SMALL_PEAK_WARNING_INTERVAL_FRAMES,
    MockIntentDecoder,
)

from eeg_bci_interfaces.msg import EegFrame  # noqa: E402

CHANNELS = ("C5", "C3", "C1", "Cz", "C2", "C4", "C6", "Pz")
RATE_HZ = 250.0
SAMPLES_PER_CHANNEL = 4


@pytest.fixture
def ros_context():
    # Owns the rclpy lifecycle so a failure mid-test never leaks the context into
    # the next one.
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


@pytest.fixture
def decoder_node(ros_context):
    node = MockIntentDecoder()
    # Match the live channel count to our 8-channel test montage and let the
    # node learn labels from the first frame (the default expected labels unset).
    node._expected_channel_count = len(CHANNELS)
    node._expected_channel_labels = None
    published: list = []
    node._publisher.publish = published.append  # type: ignore[method-assign]
    node.published = published  # type: ignore[attr-defined]
    try:
        yield node
    finally:
        node.destroy_node()


def _frame(
    *,
    channels: tuple[str, ...] = CHANNELS,
    rate_hz: float = RATE_HZ,
    peak_uv: float = 20.0,
    samples_per_channel: int = SAMPLES_PER_CHANNEL,
    stamp_sec: int = 7,
) -> EegFrame:
    # A flat per-channel value of `peak_uv` gives an exactly-known peak amplitude
    # (so the suspiciously-small-peak gate is deterministic) and an RMS equal to
    # `peak_uv` (so the decoder's class selection is deterministic too).
    frame = EegFrame()
    frame.header.stamp.sec = stamp_sec
    frame.channel_labels = list(channels)
    frame.sampling_rate_hz = rate_hz
    frame.samples = [peak_uv] * (len(channels) * samples_per_channel)
    return frame


def test_init_sets_defaults_and_creates_pub_sub(ros_context):
    node = MockIntentDecoder()
    try:
        assert node._class_labels == ("rest", "left_hand", "right_hand")
        assert node._expected_channel_count == 16
        assert node._expected_channel_labels is None
        assert node._small_peak_frame_count == 0
        assert node._contract_error_count == 0
        assert node._last_contract_error == ""
        # The node wires up exactly one publisher and one subscription.
        assert node._publisher is not None
        assert node._subscription is not None
        assert node._publisher.topic_name.endswith("/bci/intent")
        assert node._subscription.topic_name.endswith("/bci/eeg")
    finally:
        node.destroy_node()


def test_valid_frame_publishes_one_intent_with_copied_header(decoder_node):
    frame = _frame(peak_uv=20.0, stamp_sec=42)

    decoder_node._on_eeg_frame(frame)

    assert len(decoder_node.published) == 1
    intent = decoder_node.published[0]
    # Header is copied through verbatim from the source frame.
    assert intent.header.stamp.sec == 42
    # The published intent mirrors the pure decoder's contract: a winning label
    # drawn from the class labels, a probability per class, and a confidence that
    # equals the winning class probability.
    assert tuple(intent.class_labels) == decoder_node._class_labels
    assert len(intent.probabilities) == len(intent.class_labels)
    assert intent.label in intent.class_labels
    winning_index = list(intent.class_labels).index(intent.label)
    assert intent.probabilities[winning_index] == pytest.approx(intent.confidence)
    assert sum(intent.probabilities) == pytest.approx(1.0)


def test_first_valid_frame_learns_and_normalizes_channel_labels(decoder_node):
    # Labels carry whitespace the node must trim when it adopts them from the
    # first valid frame (expected_channel_labels was unset).
    padded = tuple(f"  {label} " for label in CHANNELS)
    decoder_node._on_eeg_frame(_frame(channels=padded))

    assert decoder_node._expected_channel_labels == CHANNELS
    assert len(decoder_node.published) == 1


def test_invalid_sampling_rate_publishes_nothing(decoder_node):
    # Far outside the default 250 Hz +/- tolerance: the frame is rejected before
    # any intent is produced.
    decoder_node._on_eeg_frame(_frame(rate_hz=128.0))

    assert decoder_node.published == []
    assert decoder_node._contract_error_count == 1


def test_invalid_channel_count_publishes_nothing(decoder_node):
    # One channel short of the expected montage: rejected, nothing published.
    decoder_node._on_eeg_frame(_frame(channels=CHANNELS[:-1]))

    assert decoder_node.published == []
    assert decoder_node._contract_error_count == 1


def test_contract_error_then_valid_frame_resets_error_count(decoder_node):
    decoder_node._on_eeg_frame(_frame(rate_hz=128.0))  # contract error
    assert decoder_node._contract_error_count == 1

    decoder_node._on_eeg_frame(_frame())  # valid -> "recovered"

    assert decoder_node._contract_error_count == 0
    assert decoder_node._last_contract_error == ""
    assert len(decoder_node.published) == 1


def test_warn_about_contract_error_throttles_repeated_message(decoder_node):
    warnings: list[str] = []
    decoder_node.get_logger().warn = warnings.append  # type: ignore[method-assign]

    # Same message many times: warns on the 1st and again exactly on the interval
    # boundary, staying quiet in between.
    for _ in range(CONTRACT_WARNING_INTERVAL_FRAMES + 1):
        decoder_node._warn_about_contract_error("bad rate")

    assert decoder_node._contract_error_count == CONTRACT_WARNING_INTERVAL_FRAMES + 1
    assert len(warnings) == 2


def test_warn_about_contract_error_warns_on_changed_message(decoder_node):
    warnings: list[str] = []
    decoder_node.get_logger().warn = warnings.append  # type: ignore[method-assign]

    decoder_node._warn_about_contract_error("error A")  # count 1 -> warns
    decoder_node._warn_about_contract_error("error A")  # repeat -> silent
    decoder_node._warn_about_contract_error("error B")  # changed -> warns again

    assert len(warnings) == 2
    assert decoder_node._last_contract_error == "error B"


def test_warn_if_small_peak_throttles_then_recovers(decoder_node):
    warnings: list[str] = []
    infos: list[str] = []
    decoder_node.get_logger().warn = warnings.append  # type: ignore[method-assign]
    decoder_node.get_logger().info = infos.append  # type: ignore[method-assign]

    # First small-peak frame warns; subsequent ones within the interval stay quiet
    # until the interval boundary warns again.
    for _ in range(SMALL_PEAK_WARNING_INTERVAL_FRAMES):
        decoder_node._warn_if_suspiciously_small_peak(True)
    assert decoder_node._small_peak_frame_count == SMALL_PEAK_WARNING_INTERVAL_FRAMES
    assert len(warnings) == 2  # frame 1 and the interval boundary

    # A normal-peak frame resets the counter and logs the recovery exactly once.
    decoder_node._warn_if_suspiciously_small_peak(False)
    assert decoder_node._small_peak_frame_count == 0
    assert sum("returned to microvolt scale" in msg for msg in infos) == 1

    # A normal-peak frame while already at zero is a no-op (no extra recovery log).
    decoder_node._warn_if_suspiciously_small_peak(False)
    assert decoder_node._small_peak_frame_count == 0
    assert sum("returned to microvolt scale" in msg for msg in infos) == 1


def test_small_peak_frame_drives_warning_through_on_eeg_frame(decoder_node):
    warnings: list[str] = []
    decoder_node.get_logger().warn = warnings.append  # type: ignore[method-assign]

    # A peak strictly between 0 and the 0.01 uV threshold trips the small-peak
    # gate end to end, yet the frame is still valid and an intent is published.
    decoder_node._on_eeg_frame(_frame(peak_uv=0.001))

    assert decoder_node._small_peak_frame_count == 1
    assert len(decoder_node.published) == 1
    assert any("very small" in msg for msg in warnings)


def test_read_optional_labels_empty_becomes_none(decoder_node):
    # The default unset parameter is an empty list, which the optional read
    # collapses to None (so the node falls back to learning labels).
    decoder_node.set_parameters([rclpy.parameter.Parameter("expected_channel_labels", value=[])])
    assert decoder_node._read_optional_labels("expected_channel_labels") is None


def test_read_optional_labels_trims_populated_list(decoder_node):
    # A populated list is trimmed entry-wise and kept.
    decoder_node.set_parameters(
        [rclpy.parameter.Parameter("expected_channel_labels", value=[" Cz ", "Fz"])]
    )
    assert decoder_node._read_optional_labels("expected_channel_labels") == ("Cz", "Fz")


def test_normalized_labels_strips_each_entry(decoder_node):
    assert decoder_node._normalized_labels([" a ", "b\t", "\nc"]) == ("a", "b", "c")


def test_init_with_expected_labels_derives_channel_count_and_skips_learning():
    # When expected_channel_labels is supplied at construction (via ROS parameter
    # overrides), __init__ derives the channel count from them (line 58) and a
    # later valid frame keeps those labels rather than learning new ones
    # (the 96->103 branch). Overriding the count to a mismatched value proves the
    # label-derived count wins.
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "expected_channel_labels:=[" + ",".join(CHANNELS) + "]",
            "-p",
            "expected_channel_count:=3",
        ]
    )
    try:
        node = MockIntentDecoder()
        try:
            assert node._expected_channel_labels == CHANNELS
            # Label-derived count overrides the explicit (mismatched) count param.
            assert node._expected_channel_count == len(CHANNELS)

            published: list = []
            node._publisher.publish = published.append  # type: ignore[method-assign]
            node._on_eeg_frame(_frame())

            assert node._expected_channel_labels == CHANNELS
            assert len(published) == 1
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
