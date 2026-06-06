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

from eeg_bci_pipeline.calibrate_capture import CalibrateCapture  # noqa: E402
from eeg_bci_pipeline.data.bciciv2a_dataset import load_labeled_epochs  # noqa: E402
from rclpy.duration import Duration  # noqa: E402

from eeg_bci_interfaces.msg import EegFrame  # noqa: E402

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
