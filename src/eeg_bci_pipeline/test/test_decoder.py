import pytest

from eeg_bci_pipeline.decoder import DEFAULT_CLASS_LABELS, decode_mock_intent
from eeg_bci_pipeline.mock_signal import (
    DEFAULT_AMPLITUDE_CYCLE_UV,
    generate_mock_eeg_samples,
    select_cycle_value,
)


def test_empty_samples_publish_rest_with_full_confidence():
    prediction = decode_mock_intent([])

    assert prediction.label == DEFAULT_CLASS_LABELS[0]
    assert prediction.confidence == pytest.approx(1.0)
    assert prediction.probabilities == pytest.approx((1.0, 0.0, 0.0))


def test_low_energy_samples_select_first_class():
    prediction = decode_mock_intent([1.0, -1.0, 2.0, -2.0])

    assert prediction.label == "rest"
    assert 0.5 <= prediction.confidence <= 0.95
    assert sum(prediction.probabilities) == pytest.approx(1.0)


def test_higher_energy_samples_move_to_later_class():
    prediction = decode_mock_intent([30.0, -30.0, 30.0, -30.0])

    assert prediction.label == "right_hand"
    assert prediction.class_labels == DEFAULT_CLASS_LABELS
    assert prediction.probabilities[2] == pytest.approx(prediction.confidence)


@pytest.mark.parametrize(
    ("sample", "expected_label"),
    [
        (14.99, "rest"),
        (15.0, "left_hand"),
        (29.99, "left_hand"),
        (30.0, "right_hand"),
    ],
)
def test_decoder_threshold_boundaries(sample, expected_label):
    prediction = decode_mock_intent([sample])

    assert prediction.label == expected_label


def test_custom_labels_are_supported():
    prediction = decode_mock_intent([20.0, 20.0], class_labels=("idle", "move"))

    assert prediction.label == "move"
    assert prediction.class_labels == ("idle", "move")


def test_empty_class_labels_are_rejected():
    with pytest.raises(ValueError, match="class_labels"):
        decode_mock_intent([1.0], class_labels=())


def test_default_mock_signal_cycle_reaches_all_intent_buckets():
    labels = []
    for frame_index in (0, 10, 20):
        amplitude = select_cycle_value(DEFAULT_AMPLITUDE_CYCLE_UV, frame_index, frames_per_value=10)
        samples = generate_mock_eeg_samples(
            start_sample_index=frame_index * 25,
            samples_per_frame=25,
            channel_count=16,
            sampling_rate_hz=250.0,
            amplitude_uv=amplitude,
        )
        labels.append(decode_mock_intent(samples).label)

    assert labels == ["rest", "left_hand", "right_hand"]
