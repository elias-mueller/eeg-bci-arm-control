"""Node-level coverage for the `calibrate_capture` cue/record glue.

The pure calibration helpers (schedule, slice, assemble) are covered in
`test_calibration.py`; this drives the `CalibrateCapture` node itself, which was
otherwise untested. It steps the real cue state machine through `_on_tick` by
expiring each state's deadline (no wall-clock waits), so the test is deterministic
yet exercises the actual REST -> CUE -> IMAGERY transitions rather than poking
private state. It then asserts the saved epochs are aligned to the cue schedule
and faithful to the streamed content, the cue-to-content correspondence the
hardware-free GDF-over-LSL demo cannot establish.

Skipped when ROS (rclpy / the interfaces) is unavailable, as in a bare pytest run.
"""

from __future__ import annotations

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("eeg_bci_interfaces.msg")

from math import cos, sin  # noqa: E402

from eeg_bci_pipeline.calibrate_capture import CalibrateCapture  # noqa: E402
from eeg_bci_pipeline.data.bciciv2a_dataset import load_labeled_epochs  # noqa: E402
from eeg_bci_pipeline.intent_marker_mapping import style_for_intent  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from eeg_bci_interfaces.msg import EegFrame  # noqa: E402
from eeg_bci_pipeline import calibrate_capture as calibrate_capture_module  # noqa: E402

CHANNELS = ("C5", "C3", "C1", "Cz", "C2", "C4", "C6", "Pz")
RATE_HZ = 250.0
SETTLE_OFFSET = 125  # round(0.5 s * 250 Hz), the node's default settle window
SAMPLES_PER_EPOCH = 750  # round(3.0 s * 250 Hz)
BLOCK_SAMPLES = SETTLE_OFFSET + SAMPLES_PER_EPOCH


@pytest.fixture
def ros_context():
    # Owns the rclpy lifecycle so a failure mid-test never leaks the context into
    # the next one (a bare init/finally in each test would skip shutdown if the
    # node constructor raised).
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


@pytest.fixture
def capture_node(ros_context):
    node = CalibrateCapture()
    try:
        yield node
    finally:
        node.destroy_node()


def _trial_block(trial_index: int) -> np.ndarray:
    # A per-trial, per-channel, per-sample ramp: distinct for every trial so a
    # cross-trial mix-up would change the recorded content detectably. Rounded to
    # float32 to match the EegFrame.samples wire type.
    channels = len(CHANNELS)
    samples = np.fromfunction(
        lambda c, s: trial_index * 1000.0 + c + s * 0.01,
        (channels, BLOCK_SAMPLES),
        dtype=np.float64,
    )
    return samples.astype(np.float32).astype(np.float64)


def _eeg_frame(block: np.ndarray) -> EegFrame:
    frame = EegFrame()
    frame.channel_labels = list(CHANNELS)
    frame.sampling_rate_hz = RATE_HZ
    frame.samples = block.reshape(-1).astype(np.float32).tolist()
    return frame


def _tick(node: CalibrateCapture) -> None:
    # Expire the current state's deadline, then let the node's own timer callback
    # advance the state machine, exactly as it would on a real clock tick.
    node._state_deadline = node.get_clock().now() - Duration(seconds=1)
    node._on_tick()


def _run_trial(node: CalibrateCapture, block: np.ndarray) -> None:
    _tick(node)  # REST -> CUE
    _tick(node)  # CUE -> IMAGERY (clears the buffer)
    node._on_eeg_frame(_eeg_frame(block))
    _tick(node)  # IMAGERY -> record this trial (and finish the session on the last)


def _prime(node: CalibrateCapture, output_path, schedule) -> None:
    # The first frame fixes the stream contract (channels, rate, samples/epoch) and
    # enters REST; then pin a short known schedule and output for the run.
    node._on_eeg_frame(_eeg_frame(_trial_block(0)))
    node._output_path = str(output_path)
    node._schedule = schedule


def test_calibrate_capture_records_cue_aligned_epochs(capture_node, tmp_path):
    schedule = ("left_hand", "right_hand", "right_hand", "left_hand")
    output_path = tmp_path / "capture.joblib"
    _prime(capture_node, output_path, schedule)
    assert capture_node._samples_per_epoch == SAMPLES_PER_EPOCH
    assert capture_node._settle_offset == SETTLE_OFFSET

    for trial_index in range(len(schedule)):
        _run_trial(capture_node, _trial_block(trial_index))

    saved = load_labeled_epochs(output_path)
    # Each epoch carries the cue that was active when it was recorded.
    assert saved.labels == schedule
    assert saved.channel_labels == CHANNELS
    assert saved.epochs_uv.shape == (len(schedule), len(CHANNELS), SAMPLES_PER_EPOCH)
    # And it holds exactly the settle-offset slice of that trial's streamed block,
    # proving the right data was recorded for the right cue (no cross-trial mix-up).
    for trial_index in range(len(schedule)):
        expected = _trial_block(trial_index)[:, SETTLE_OFFSET:BLOCK_SAMPLES]
        np.testing.assert_array_equal(saved.epochs_uv[trial_index], expected)


def test_calibrate_capture_accumulates_multi_frame_imagery(capture_node, tmp_path):
    # Imagery usually arrives as several frames per trial; the node stitches them
    # with np.concatenate (calibrate_capture.py), a path single-frame trials never
    # exercise. Split one trial's block across two frames and assert the recorded
    # epoch still equals the contiguous settle slice.
    output_path = tmp_path / "multi.joblib"
    _prime(capture_node, output_path, ("left_hand",))
    block = _trial_block(0)
    half = block.shape[1] // 2

    _tick(capture_node)  # REST -> CUE
    _tick(capture_node)  # CUE -> IMAGERY
    capture_node._on_eeg_frame(_eeg_frame(block[:, :half]))
    capture_node._on_eeg_frame(_eeg_frame(block[:, half:]))
    _tick(capture_node)  # IMAGERY -> record + finish

    saved = load_labeled_epochs(output_path)
    expected = block[:, SETTLE_OFFSET:BLOCK_SAMPLES]
    np.testing.assert_array_equal(saved.epochs_uv[0], expected)


def test_calibrate_capture_tick_waits_for_deadline(capture_node, tmp_path):
    # A tick before the state deadline must not advance the machine; pins the
    # remaining-time comparison direction (a sign flip would advance immediately).
    _prime(capture_node, tmp_path / "wait.joblib", ("left_hand", "right_hand"))
    capture_node._state_deadline = capture_node.get_clock().now() + Duration(seconds=5)
    state_before = capture_node._state

    capture_node._on_tick()

    assert capture_node._state == state_before


def test_calibrate_capture_skips_short_trials(capture_node, tmp_path):
    # A trial whose imagery buffer is too short to fill one epoch is dropped and
    # counted, not silently padded or crashed.
    schedule = ("left_hand", "right_hand")
    output_path = tmp_path / "short.joblib"
    _prime(capture_node, output_path, schedule)

    _run_trial(capture_node, _trial_block(0))  # full block -> recorded
    _run_trial(capture_node, _trial_block(1)[:, : SAMPLES_PER_EPOCH - 1])  # short -> skipped

    saved = load_labeled_epochs(output_path)
    assert saved.labels == ("left_hand",)
    assert saved.skipped_epoch_count == 1


def _raw_frame(labels, rate_hz, samples) -> EegFrame:
    # Build an EegFrame from explicit fields so edge cases (empty labels, bad
    # rate, ragged sample counts) can be constructed without a valid block.
    frame = EegFrame()
    frame.channel_labels = list(labels)
    frame.sampling_rate_hz = float(rate_hz)
    frame.samples = [float(value) for value in samples]
    return frame


@pytest.mark.parametrize(
    "labels, rate_hz",
    [
        ((), RATE_HZ),  # empty channel labels
        (CHANNELS, 0.0),  # non-positive sampling rate
        (CHANNELS, float("nan")),  # non-finite sampling rate
    ],
)
def test_calibrate_capture_rejects_unusable_first_frame(capture_node, labels, rate_hz):
    # The very first frame fixes the stream contract; a frame with no channels or a
    # non-positive / non-finite rate is ignored and the node stays uninitialized
    # (no timer, no state machine) so a later good frame can still start the session.
    capture_node._on_eeg_frame(_raw_frame(labels, rate_hz, [0.0] * len(CHANNELS)))

    assert capture_node._channel_labels is None
    assert capture_node._timer is None
    assert capture_node._state == "WAIT"

    # A subsequent well-formed frame still initializes normally.
    capture_node._on_eeg_frame(_eeg_frame(_trial_block(0)))
    assert capture_node._channel_labels == list(CHANNELS)
    assert capture_node._state == "REST"


def test_calibrate_capture_drops_mismatched_labels_and_warns_once(capture_node, tmp_path):
    # If the stream's channel labels change mid-session the frames are dropped (they
    # can't be aligned to the established montage) and the warning is emitted only
    # once, not per frame, so a wrong cable doesn't flood the log.
    _prime(capture_node, tmp_path / "mismatch.joblib", ("left_hand", "right_hand"))
    warnings = []
    capture_node.get_logger().warn = warnings.append

    mismatched = _eeg_frame(_trial_block(0))
    mismatched.channel_labels = list(CHANNELS[:-1]) + ["X9"]
    capture_node._on_eeg_frame(mismatched)
    capture_node._on_eeg_frame(mismatched)

    assert len(warnings) == 1
    assert capture_node._labels_warned is True

    # A matching-label frame outside IMAGERY is accepted but not buffered.
    capture_node._on_eeg_frame(_eeg_frame(_trial_block(0)))
    assert capture_node._imagery_buffer is None


def test_calibrate_capture_drops_malformed_imagery_frame(capture_node, tmp_path):
    # During IMAGERY a frame whose flat length is not a multiple of channel_count is
    # malformed; it is dropped rather than reshaped into scrambled channels, and it
    # does not disturb already-buffered samples.
    _prime(capture_node, tmp_path / "malformed.joblib", ("left_hand", "right_hand"))
    _tick(capture_node)  # REST -> CUE
    _tick(capture_node)  # CUE -> IMAGERY (clears buffer)

    good = _trial_block(0)
    capture_node._on_eeg_frame(_eeg_frame(good))
    buffered = capture_node._imagery_buffer.copy()

    ragged = _raw_frame(CHANNELS, RATE_HZ, [1.0] * (len(CHANNELS) + 1))  # not divisible
    capture_node._on_eeg_frame(ragged)

    np.testing.assert_array_equal(capture_node._imagery_buffer, buffered)


def test_calibrate_capture_tick_ignores_uninitialized_and_done(capture_node, tmp_path):
    # _on_tick is created only after init, but the guard is explicit: before any
    # frame (channel_labels None) and after the session is DONE it must early-return
    # without advancing the machine or touching the deadline.
    before_init = capture_node._state
    capture_node._on_tick()  # channel_labels is None
    assert capture_node._state == before_init

    _prime(capture_node, tmp_path / "done.joblib", ("left_hand", "right_hand"))
    capture_node._state = calibrate_capture_module.DONE_STATE
    capture_node._state_deadline = capture_node.get_clock().now() - Duration(seconds=1)
    capture_node._on_tick()
    assert capture_node._state == calibrate_capture_module.DONE_STATE


def test_calibrate_capture_advance_state_ignores_unknown_state(capture_node, tmp_path):
    # _advance_state is a no-op for any state outside the REST/CUE/IMAGERY chain, so
    # a stray transition cannot enter a phantom phase or write a record.
    _prime(capture_node, tmp_path / "unknown.joblib", ("left_hand", "right_hand"))
    capture_node._state = "WAIT"
    records_before = list(capture_node._records)

    capture_node._advance_state()

    assert capture_node._state == "WAIT"
    assert capture_node._records == records_before


def test_calibrate_capture_zero_epochs_saves_nothing(capture_node, tmp_path):
    # A trial that receives no imagery frames leaves the buffer None and is skipped;
    # when every trial is dropped the session ends with no records, logs an error,
    # and writes no file. Also covers the timer-already-cancelled finish path.
    output_path = tmp_path / "empty.joblib"
    _prime(capture_node, output_path, ("left_hand",))
    errors = []
    capture_node.get_logger().error = errors.append
    capture_node._timer = None  # finish must tolerate a missing timer

    _tick(capture_node)  # REST -> CUE
    _tick(capture_node)  # CUE -> IMAGERY (no frames arrive)
    _tick(capture_node)  # IMAGERY -> skip + finish (zero records)

    assert capture_node._state == calibrate_capture_module.DONE_STATE
    assert capture_node._skipped == 1
    assert capture_node._records == []
    assert len(errors) == 1
    assert not output_path.exists()


def test_calibrate_capture_save_oserror_is_caught(capture_node, tmp_path, monkeypatch):
    # A filesystem failure during the final save is reported, not raised, so the
    # captured session does not crash the node on an unwritable path.
    schedule = ("left_hand", "right_hand")
    output_path = tmp_path / "unwritable.joblib"
    _prime(capture_node, output_path, schedule)

    def raise_oserror(epochs, path):
        raise OSError("disk full")

    monkeypatch.setattr(calibrate_capture_module, "save_labeled_epochs", raise_oserror)
    errors = []
    capture_node.get_logger().error = errors.append

    for trial_index in range(len(schedule)):
        _run_trial(capture_node, _trial_block(trial_index))

    assert capture_node._state == calibrate_capture_module.DONE_STATE
    assert len(errors) == 1
    assert not output_path.exists()


def test_calibrate_capture_state_metadata(capture_node):
    # The per-state durations and cue labels back the schedule the test harness
    # relies on; pin them, including the rest/0.0 fallbacks for off-chain states.
    node = capture_node
    assert node._state_duration(calibrate_capture_module.REST_STATE) == pytest.approx(
        node._rest_sec
    )
    assert node._state_duration(calibrate_capture_module.CUE_STATE) == pytest.approx(node._cue_sec)
    expected_imagery = (
        node._settle_sec + node._epoch_sec + calibrate_capture_module.IMAGERY_MARGIN_SEC
    )
    assert node._state_duration(calibrate_capture_module.IMAGERY_STATE) == pytest.approx(
        expected_imagery
    )
    assert node._state_duration(calibrate_capture_module.DONE_STATE) == pytest.approx(0.0)

    node._schedule = ("left_hand",)
    node._trial_index = 0
    assert node._cue_label_for_state(calibrate_capture_module.CUE_STATE) == "left_hand"
    assert node._cue_label_for_state(calibrate_capture_module.IMAGERY_STATE) == "left_hand"
    assert node._cue_label_for_state(calibrate_capture_module.REST_STATE) == "rest"
    assert node._cue_label_for_state(calibrate_capture_module.DONE_STATE) == "rest"


def test_calibrate_capture_cue_markers_encode_direction_and_clear(capture_node):
    # The RViz cue markers are the operator's only feedback; a swapped arrow or a
    # dropped clear would silently mislead. Assert the published marker fields, the
    # per-cue arrow direction, and the DELETEALL clear (reached via _on_tick but
    # otherwise never inspected).
    node = capture_node
    markers = []
    node._marker_publisher.publish = markers.append
    node._schedule = ("left_hand", "right_hand")
    node._trial_index = 0
    node._state = calibrate_capture_module.CUE_STATE

    node._publish_cue_marker(2.5)
    arrow, text = markers
    assert arrow.ns == "bci_cue" and arrow.id == 0 and arrow.type == Marker.ARROW
    assert text.ns == "bci_cue" and text.id == 1 and text.type == Marker.TEXT_VIEW_FACING
    assert text.text == "CUE: left_hand  2.5s"
    left = style_for_intent("left_hand", 1.0)
    assert arrow.pose.orientation.z == pytest.approx(sin(left.yaw_rad / 2.0))
    assert arrow.pose.orientation.w == pytest.approx(cos(left.yaw_rad / 2.0))

    # The right-hand cue points the opposite way: pin that the arrow direction
    # tracks the cue label, not a constant.
    markers.clear()
    node._trial_index = 1
    node._publish_cue_marker(1.0)
    assert markers[0].pose.orientation.z != pytest.approx(arrow.pose.orientation.z)

    # The clear emits a single DELETEALL in the cue namespace.
    markers.clear()
    node._clear_cue_marker()
    assert len(markers) == 1
    assert markers[0].action == Marker.DELETEALL
    assert markers[0].ns == "bci_cue"
