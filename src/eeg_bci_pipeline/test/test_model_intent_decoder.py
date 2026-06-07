"""Node-level coverage for the `model_intent_decoder` ROS shell.

The pure sliding-window, gating, and decode helpers are covered in
`test_model_decode.py`; this drives the `ModelIntentDecoder` node itself. It
loads a REAL CSP+LDA `HandClassifierArtifact` (trained on small seeded synthetic
epochs and saved to a tmp `.joblib`), points the node's `model_path` parameter at
it, and feeds real `EegFrame` messages through the actual `_on_eeg_frame`
callback. Published intents are captured by replacing the publisher's `publish`
with a list append, so the genuine publish path runs without spinning up a
subscriber. No wall-clock waits.

Skipped when ROS (rclpy / the interfaces) is unavailable, as in a bare pytest run.
"""

from __future__ import annotations

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("eeg_bci_interfaces.msg")

import eeg_bci_pipeline.model_intent_decoder as decoder_module  # noqa: E402
from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs  # noqa: E402
from eeg_bci_pipeline.model_intent_decoder import (  # noqa: E402
    CONTRACT_WARNING_INTERVAL_FRAMES,
    ModelIntentDecoder,
)
from eeg_bci_pipeline.training.hand_classifier import (  # noqa: E402
    save_hand_classifier_artifact,
    train_hand_classifier,
)

from eeg_bci_interfaces.msg import EegFrame  # noqa: E402

CHANNELS = ("C3", "Cz", "C4", "Pz")
RATE_HZ = 100.0
SAMPLES_PER_EPOCH = 100


def _labeled_epochs(labels: tuple[str, ...]) -> LabeledEpochs:
    # Seeded synthetic epochs with a 10 Hz tone injected into C3 for left_hand and
    # C4 for right_hand, the separable fixture the model_decode/hand_classifier
    # tests use so a real CSP+LDA can be fit and produce confident predictions.
    rng = np.random.default_rng(3)
    epochs = rng.normal(scale=0.05, size=(len(labels), len(CHANNELS), SAMPLES_PER_EPOCH))
    time = np.arange(SAMPLES_PER_EPOCH, dtype=float) / float(SAMPLES_PER_EPOCH)
    signal = np.sin(2.0 * np.pi * 10.0 * time)
    for index, label in enumerate(labels):
        if label == "left_hand":
            epochs[index, 0, :] += signal
        elif label == "right_hand":
            epochs[index, 1, :] += signal
    return LabeledEpochs(
        source_id="synthetic",
        sampling_rate_hz=RATE_HZ,
        channel_labels=CHANNELS,
        class_labels=("left_hand", "right_hand", "feet", "tongue"),
        labels=labels,
        start_sample_indices=tuple(index * SAMPLES_PER_EPOCH for index, _ in enumerate(labels)),
        epochs_uv=epochs,
    )


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory):
    # One real artifact for the whole module: train + save once, reuse the file.
    labels = ("left_hand", "right_hand") * 8
    artifact = train_hand_classifier(
        _labeled_epochs(labels),
        csp_components=2,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
    )
    path = tmp_path_factory.mktemp("model") / "hand-csp-lda.joblib"
    save_hand_classifier_artifact(artifact, path)
    return path


@pytest.fixture(scope="module")
def left_hand_window() -> np.ndarray:
    # A clean left_hand epoch (channels, samples) the trained model classifies
    # confidently; reused to drive the happy-path decode.
    return _labeled_epochs(("left_hand",)).epochs_uv[0]


def _make_node(overrides: dict[str, object]) -> ModelIntentDecoder:
    # Inject parameters through the real ROS arg path so the node reads them exactly
    # as it would at launch; rclpy.init is global, so each test owns its context.
    args = ["--ros-args"]
    for name, value in overrides.items():
        args += ["-p", f"{name}:={value}"]
    rclpy.init(args=args)
    return ModelIntentDecoder()


def _captured_intents(node: ModelIntentDecoder) -> list:
    published: list = []
    node._publisher.publish = published.append
    return published


def _frame(window: np.ndarray, *, frame_id: str = "bci") -> EegFrame:
    frame = EegFrame()
    frame.header.frame_id = frame_id
    frame.channel_labels = list(CHANNELS)
    frame.sampling_rate_hz = RATE_HZ
    frame.samples = np.asarray(window).reshape(-1).astype(np.float32).tolist()
    return frame


def test_init_wires_pub_sub_buffer_and_contract(artifact_path):
    node = _make_node({"model_path": str(artifact_path)})
    try:
        # The node derives its frame contract from the loaded artifact.
        assert node._artifact.channel_labels == CHANNELS
        assert node._buffer._channel_count == len(CHANNELS)
        assert node._buffer._samples_per_epoch == SAMPLES_PER_EPOCH
        # Rest is synthesized for a hand model, so it heads the runtime vocabulary.
        assert node._runtime_class_labels == ("rest", "left_hand", "right_hand")
        assert not node._window_filled
        assert node._contract_error_count == 0
        assert node._decode_error_count == 0
        from eeg_bci_interfaces.msg import Intent

        assert node._publisher.msg_type is Intent
        assert node._subscription.msg_type is EegFrame
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_init_rejects_missing_model_path():
    rclpy.init()
    try:
        with pytest.raises(ValueError, match="model_path"):
            ModelIntentDecoder()
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_init_rejects_rest_threshold_outside_unit_interval(artifact_path, threshold):
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"model_path:={artifact_path}",
            "-p",
            f"rest_confidence_threshold:={threshold}",
        ]
    )
    try:
        with pytest.raises(ValueError, match="between 0 and 1"):
            ModelIntentDecoder()
    finally:
        rclpy.shutdown()


def test_warmup_frame_publishes_rest_before_window_fills(artifact_path, left_hand_window):
    node = _make_node({"model_path": str(artifact_path)})
    try:
        published = _captured_intents(node)
        # Half an epoch per channel: the sliding buffer is still warming up.
        half = SAMPLES_PER_EPOCH // 2
        node._on_eeg_frame(_frame(left_hand_window[:, :half], frame_id="warm"))

        assert len(published) == 1
        intent = published[0]
        assert intent.label == "rest"
        assert intent.header.frame_id == "warm"
        assert list(intent.class_labels) == ["rest", "left_hand", "right_hand"]
        assert not node._window_filled
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_full_window_decodes_hand_and_sets_flag_once(artifact_path, left_hand_window):
    node = _make_node({"model_path": str(artifact_path)})
    try:
        published = _captured_intents(node)
        # A full epoch fills the window in one push and yields a real model intent.
        node._on_eeg_frame(_frame(left_hand_window, frame_id="full"))

        assert node._window_filled
        assert len(published) == 1
        intent = published[0]
        # The fixture is separable enough that the trained model decodes left_hand.
        assert intent.label == "left_hand"
        assert intent.confidence >= node._rest_confidence_threshold
        assert intent.header.frame_id == "full"
        assert list(intent.class_labels) == ["rest", "left_hand", "right_hand"]
        # Probabilities are a valid distribution whose argmax agrees with the label.
        assert sum(intent.probabilities) == pytest.approx(1.0)
        assert intent.class_labels[int(np.argmax(intent.probabilities))] == intent.label

        # A second full window must not re-log "window filled" but keeps decoding.
        node._on_eeg_frame(_frame(left_hand_window))
        assert node._window_filled
        assert len(published) == 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_contract_violation_resets_buffer_and_drops_frame(artifact_path, left_hand_window):
    node = _make_node({"model_path": str(artifact_path)})
    try:
        published = _captured_intents(node)
        # Fill the window first so the reset has visible state to clear.
        node._on_eeg_frame(_frame(left_hand_window))
        assert node._window_filled
        published.clear()

        # A frame whose sampling rate is far off the artifact's violates the contract.
        bad = _frame(left_hand_window)
        bad.sampling_rate_hz = RATE_HZ * 2.0
        node._on_eeg_frame(bad)

        # No intent is published, the buffer is dropped, and the filled flag clears
        # so windows can't stitch across the gap on recovery.
        assert published == []
        assert not node._buffer.is_full
        assert not node._window_filled
        assert node._contract_error_count == 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_contract_recovery_logged_once_and_resets_counter(artifact_path, left_hand_window):
    node = _make_node({"model_path": str(artifact_path)})
    try:
        published = _captured_intents(node)
        bad = _frame(left_hand_window)
        bad.sampling_rate_hz = RATE_HZ * 2.0
        node._on_eeg_frame(bad)
        assert node._contract_error_count == 1
        assert published == []

        # A valid frame after a violation recovers: the error counter resets and a
        # warm-up rest intent is published again (buffer was cleared by the reset).
        node._on_eeg_frame(_frame(left_hand_window[:, : SAMPLES_PER_EPOCH // 2]))
        assert node._contract_error_count == 0
        assert node._last_contract_error == ""
        assert len(published) == 1
        assert published[0].label == "rest"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_decode_failure_holds_rest_and_warns(artifact_path, left_hand_window, monkeypatch):
    node = _make_node({"model_path": str(artifact_path)})
    try:
        published = _captured_intents(node)

        def _boom(*args, **kwargs):
            raise RuntimeError("decode exploded")

        # The node imported decode_window into its own namespace, so patch it there.
        monkeypatch.setattr(decoder_module, "decode_window", _boom)
        node._on_eeg_frame(_frame(left_hand_window, frame_id="boom"))

        # A decode failure must not kill the node; it holds at rest and counts it.
        assert len(published) == 1
        assert published[0].label == "rest"
        assert published[0].header.frame_id == "boom"
        assert node._decode_error_count == 1
        assert node._window_filled
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_contract_warning_throttles_by_interval_and_change():
    # _warn_about_contract_error is throttling logic: the count IS the behavior, so
    # assert which calls warn. It warns on the first, on every CONTRACT_WARNING_-
    # INTERVAL_FRAMES-th, and whenever the message text changes.
    warned: list[str] = []
    node = ModelIntentDecoder.__new__(ModelIntentDecoder)
    node._contract_error_count = 0
    node._last_contract_error = ""
    node.get_logger = lambda: type("L", (), {"warn": lambda _self, msg: warned.append(msg)})()

    node._warn_about_contract_error("same")  # count 1 -> warns
    for _ in range(CONTRACT_WARNING_INTERVAL_FRAMES - 2):
        node._warn_about_contract_error("same")  # suppressed
    assert len(warned) == 1
    node._warn_about_contract_error("same")  # count == interval -> warns
    assert len(warned) == 2
    assert node._contract_error_count == CONTRACT_WARNING_INTERVAL_FRAMES

    node._warn_about_contract_error("different")  # changed text -> warns mid-interval
    assert len(warned) == 3


def test_decode_warning_throttles_by_interval_and_change():
    # Mirror of the contract throttle for _warn_about_decode_error, which logs at
    # error level. The first, the interval-th, and any text change emit.
    logged: list[str] = []
    node = ModelIntentDecoder.__new__(ModelIntentDecoder)
    node._decode_error_count = 0
    node._last_decode_error = ""
    node.get_logger = lambda: type("L", (), {"error": lambda _self, msg: logged.append(msg)})()

    node._warn_about_decode_error("a")  # count 1 -> logs
    for _ in range(CONTRACT_WARNING_INTERVAL_FRAMES - 2):
        node._warn_about_decode_error("a")  # suppressed
    assert len(logged) == 1
    node._warn_about_decode_error("a")  # interval-th -> logs
    assert len(logged) == 2
    node._warn_about_decode_error("b")  # changed text -> logs
    assert len(logged) == 3
