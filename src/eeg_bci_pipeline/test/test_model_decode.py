import numpy as np
import pytest
from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs
from eeg_bci_pipeline.model_decode import (
    SlidingEpochBuffer,
    decode_window,
    gate_intent,
    rest_intent,
    runtime_labels_for_artifact,
)
from eeg_bci_pipeline.training.hand_classifier import train_hand_classifier


def make_labeled_epochs(labels: tuple[str, ...]) -> LabeledEpochs:
    rng = np.random.default_rng(3)
    epochs = rng.normal(scale=0.05, size=(len(labels), 4, 100))
    time = np.arange(100, dtype=float) / 100.0
    signal = np.sin(2.0 * np.pi * 10.0 * time)
    for index, label in enumerate(labels):
        if label == "left_hand":
            epochs[index, 0, :] += signal
        elif label == "right_hand":
            epochs[index, 1, :] += signal

    return LabeledEpochs(
        source_id="synthetic",
        sampling_rate_hz=100.0,
        channel_labels=("C3", "Cz", "C4", "Pz"),
        class_labels=("left_hand", "right_hand", "feet", "tongue"),
        labels=labels,
        start_sample_indices=tuple(index * 100 for index, _ in enumerate(labels)),
        epochs_uv=epochs,
    )


def test_sliding_buffer_rejects_nonpositive_channel_count():
    with pytest.raises(ValueError, match="channel_count must be at least 1"):
        SlidingEpochBuffer(channel_count=0, samples_per_epoch=4)
    with pytest.raises(ValueError, match="channel_count must be at least 1"):
        SlidingEpochBuffer(channel_count=-1, samples_per_epoch=4)


def test_sliding_buffer_rejects_nonpositive_samples_per_epoch():
    with pytest.raises(ValueError, match="samples_per_epoch must be at least 1"):
        SlidingEpochBuffer(channel_count=2, samples_per_epoch=0)
    with pytest.raises(ValueError, match="samples_per_epoch must be at least 1"):
        SlidingEpochBuffer(channel_count=2, samples_per_epoch=-1)


def test_sliding_buffer_rejects_multidimensional_frame():
    buffer = SlidingEpochBuffer(channel_count=2, samples_per_epoch=4)
    with pytest.raises(ValueError, match="1-D channel-major"):
        buffer.push([[10.0, 11.0], [20.0, 21.0]])  # 2-D, not flat channel-major


def test_sliding_buffer_warms_up_then_fills_in_order():
    buffer = SlidingEpochBuffer(channel_count=2, samples_per_epoch=4)

    assert buffer.push([10.0, 11.0, 20.0, 21.0]) is None  # channel-major: ch0=[10,11], ch1=[20,21]
    assert not buffer.is_full

    window = buffer.push([12.0, 13.0, 22.0, 23.0])
    assert buffer.is_full
    assert window is not None
    np.testing.assert_array_equal(window, [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]])


def test_sliding_buffer_slides_dropping_oldest_samples():
    buffer = SlidingEpochBuffer(channel_count=2, samples_per_epoch=4)
    buffer.push([10.0, 11.0, 20.0, 21.0])
    buffer.push([12.0, 13.0, 22.0, 23.0])

    window = buffer.push([14.0, 24.0])  # one sample per channel
    assert window is not None
    np.testing.assert_array_equal(window, [[11.0, 12.0, 13.0, 14.0], [21.0, 22.0, 23.0, 24.0]])


def test_sliding_buffer_handles_frame_larger_than_window():
    buffer = SlidingEpochBuffer(channel_count=2, samples_per_epoch=3)

    window = buffer.push([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])  # 4 samples/channel
    assert window is not None
    np.testing.assert_array_equal(window, [[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]])


def test_sliding_buffer_rejects_frame_not_divisible_by_channel_count():
    buffer = SlidingEpochBuffer(channel_count=2, samples_per_epoch=4)
    with pytest.raises(ValueError, match="divisible by channel count"):
        buffer.push([1.0, 2.0, 3.0])


def test_sliding_buffer_reset_returns_to_warmup():
    buffer = SlidingEpochBuffer(channel_count=2, samples_per_epoch=2)
    assert buffer.push([1.0, 2.0, 3.0, 4.0]) is not None  # 2 samples/channel fills it
    assert buffer.is_full
    buffer.reset()
    assert not buffer.is_full
    assert buffer.push([5.0, 6.0]) is None  # 1 sample/channel → still warming up


def test_runtime_labels_prepend_rest_for_hand_model():
    assert runtime_labels_for_artifact(("left_hand", "right_hand")) == (
        "rest",
        "left_hand",
        "right_hand",
    )


def test_runtime_labels_passthrough_when_model_already_has_rest():
    assert runtime_labels_for_artifact(("rest", "left_hand", "right_hand")) == (
        "rest",
        "left_hand",
        "right_hand",
    )


def test_runtime_labels_accept_nonstandard_classes_without_gate_crash():
    # Regression: model_intent_decoder used to hardcode the runtime set to
    # rest/left_hand/right_hand, so an artifact trained on other classes made
    # gate_intent raise on its own winning label. Deriving the set keeps the winner
    # in range.
    labels = runtime_labels_for_artifact(("feet", "tongue"))
    assert labels == ("rest", "feet", "tongue")
    intent = gate_intent([0.9, 0.1], ("feet", "tongue"), runtime_class_labels=labels)
    assert intent.label == "feet"


def test_runtime_labels_reject_empty_and_duplicates():
    with pytest.raises(ValueError, match="non-empty"):
        runtime_labels_for_artifact(())
    with pytest.raises(ValueError, match="unique"):
        runtime_labels_for_artifact(("left_hand", "left_hand"))


def test_rest_intent_rejects_runtime_labels_without_rest():
    with pytest.raises(ValueError, match="rest label"):
        rest_intent(("left_hand", "right_hand"))


def test_gate_intent_reports_confident_winner():
    intent = gate_intent([0.9, 0.1], ("left_hand", "right_hand"), rest_threshold=0.6)

    assert intent.label == "left_hand"
    assert intent.confidence == pytest.approx(0.9)
    assert intent.class_labels == ("rest", "left_hand", "right_hand")
    assert len(intent.probabilities) == 3
    assert sum(intent.probabilities) == pytest.approx(1.0)
    assert intent.class_labels[int(np.argmax(intent.probabilities))] == "left_hand"


def test_gate_intent_falls_back_to_rest_when_unsure():
    intent = gate_intent([0.52, 0.48], ("left_hand", "right_hand"), rest_threshold=0.6)

    assert intent.label == "rest"
    assert intent.confidence == pytest.approx(0.48)  # 1 - winning probability
    assert intent.class_labels[int(np.argmax(intent.probabilities))] == "rest"


def test_gate_intent_threshold_boundary_acts():
    intent = gate_intent([0.6, 0.4], ("left_hand", "right_hand"), rest_threshold=0.6)

    assert intent.label == "left_hand"  # winner_prob == threshold acts (rest uses strict <)


def test_gate_intent_validates_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        gate_intent([], ("left_hand", "right_hand"))
    with pytest.raises(ValueError, match="equal length"):
        gate_intent([0.6], ("left_hand", "right_hand"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gate_intent([1.5, -0.5], ("left_hand", "right_hand"))
    with pytest.raises(ValueError, match="finite"):
        gate_intent([float("nan"), 0.4], ("left_hand", "right_hand"))
    with pytest.raises(ValueError, match="between 0 and 1"):
        gate_intent([0.6, 0.4], ("left_hand", "right_hand"), rest_threshold=1.5)
    with pytest.raises(ValueError, match="rest label"):
        gate_intent(
            [0.6, 0.4],
            ("left_hand", "right_hand"),
            runtime_class_labels=("left_hand", "right_hand"),
        )


def test_gate_intent_rejects_confident_winner_absent_from_runtime_labels():
    # A confident winner (>= threshold, so it does not gate to rest) whose model
    # label is missing from a runtime set that still includes rest must raise.
    with pytest.raises(ValueError, match="not in runtime_class_labels"):
        gate_intent(
            [0.9, 0.1],
            ("feet", "tongue"),
            runtime_class_labels=("rest", "left_hand", "right_hand"),
            rest_threshold=0.6,
        )


def test_gate_intent_rest_probabilities_stay_consistent_at_high_threshold():
    # A confident-but-below-threshold winner gates to rest; the probability vector's
    # argmax must still agree with label == "rest" (the soft-spread bug regression).
    intent = gate_intent([0.8, 0.2], ("left_hand", "right_hand"), rest_threshold=0.9)

    assert intent.label == "rest"
    assert sum(intent.probabilities) == pytest.approx(1.0)
    assert intent.class_labels[int(np.argmax(intent.probabilities))] == "rest"


def test_decode_window_predicts_hand_for_clear_window():
    labels = ("left_hand", "right_hand") * 8
    epochs = make_labeled_epochs(labels)
    artifact = train_hand_classifier(
        epochs, csp_components=2, bandpass_low_hz=None, bandpass_high_hz=None
    )
    left_window = epochs.epochs_uv[0]  # labels[0] == "left_hand"

    intent = decode_window(artifact, left_window, rest_threshold=0.6)

    assert intent.label == "left_hand"
    assert intent.confidence >= 0.6
    # The rest-gating branch of decode_window is exercised deterministically by the
    # gate_intent tests above; the synthetic fixture is too separable to drive a real
    # CSP+LDA posterior below threshold.


def test_decode_window_rejects_non_2d_window():
    labels = ("left_hand", "right_hand") * 4
    epochs = make_labeled_epochs(labels)
    artifact = train_hand_classifier(
        epochs, csp_components=2, bandpass_low_hz=None, bandpass_high_hz=None
    )
    with pytest.raises(ValueError, match="channels, samples"):
        decode_window(artifact, epochs.epochs_uv)  # 3-D, not a single window


def test_decode_window_holds_rest_for_degenerate_window():
    labels = ("left_hand", "right_hand") * 8
    epochs = make_labeled_epochs(labels)
    artifact = train_hand_classifier(
        epochs, csp_components=2, bandpass_low_hz=None, bandpass_high_hz=None
    )
    n_channels = epochs.epochs_uv.shape[1]
    n_samples = epochs.epochs_uv.shape[2]

    # A flatlined (all-zero) window would make CSP's log-power -inf and crash the
    # classifier; the decoder must hold at rest instead of raising.
    intent = decode_window(artifact, np.zeros((n_channels, n_samples)))

    assert intent.label == "rest"
    assert intent.confidence == pytest.approx(1.0)
