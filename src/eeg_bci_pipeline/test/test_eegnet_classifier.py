import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs  # noqa: E402
from eeg_bci_pipeline.training.eegnet_classifier import (  # noqa: E402
    _epochs_to_tensor,
    evaluate_eegnet_classifier,
)


def make_labeled_epochs(labels: tuple[str, ...]) -> LabeledEpochs:
    rng = np.random.default_rng(3)
    epochs = rng.normal(scale=0.05, size=(len(labels), 4, 128))
    time = np.arange(128, dtype=float) / 128.0
    signal = np.sin(2.0 * np.pi * 10.0 * time)
    for index, label in enumerate(labels):
        if label == "left_hand":
            epochs[index, 0, :] += signal
        elif label == "right_hand":
            epochs[index, 1, :] += signal

    return LabeledEpochs(
        source_id="synthetic",
        sampling_rate_hz=128.0,
        channel_labels=("C3", "Cz", "C4", "Pz"),
        class_labels=("left_hand", "right_hand", "feet", "tongue"),
        labels=labels,
        start_sample_indices=tuple(index * 128 for index, _ in enumerate(labels)),
        epochs_uv=epochs,
    )


def test_epochs_to_tensor_adds_channel_dimension():
    arr = np.random.randn(10, 4, 64).astype(np.float64)

    tensor = _epochs_to_tensor(arr)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (10, 1, 4, 64)
    assert tensor.dtype == torch.float32
    assert np.allclose(tensor.numpy()[:, 0, :, :], arr, atol=1e-5)


def test_evaluate_eegnet_classifier_reports_cross_validation_metadata():
    labels = ("left_hand", "right_hand") * 8
    epochs = make_labeled_epochs(labels)

    evaluation = evaluate_eegnet_classifier(
        epochs,
        cv_splits=2,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
        n_epochs=5,
        batch_size=4,
        patience=3,
        kernel_length=32,
    )

    assert evaluation.source_id == "synthetic"
    assert evaluation.classifier_name == "eegnet"
    assert evaluation.class_counts == (8, 8)
    assert evaluation.cv_splits == 2
    assert len(evaluation.fold_scores) == 2


def test_evaluate_eegnet_classifier_learns_above_chance_on_separable_classes():
    labels = ("left_hand", "right_hand") * 10
    epochs = make_labeled_epochs(labels)

    evaluation = evaluate_eegnet_classifier(
        epochs,
        cv_splits=2,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
        n_epochs=60,
        batch_size=4,
        patience=15,
        kernel_length=32,
    )

    assert evaluation.mean_accuracy > 0.6


def test_evaluate_eegnet_classifier_caps_cv_splits_to_min_class_count():
    labels = ("left_hand", "right_hand") * 3
    epochs = make_labeled_epochs(labels)

    evaluation = evaluate_eegnet_classifier(
        epochs,
        cv_splits=10,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
        n_epochs=3,
        batch_size=2,
        patience=2,
        kernel_length=32,
    )

    assert evaluation.cv_splits <= 3
