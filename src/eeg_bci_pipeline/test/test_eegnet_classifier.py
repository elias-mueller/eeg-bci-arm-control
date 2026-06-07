import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs  # noqa: E402
from eeg_bci_pipeline.training.eegnet import EEGNet  # noqa: E402
from eeg_bci_pipeline.training.eegnet_classifier import (  # noqa: E402
    _apply_channel_stats,
    _epochs_to_tensor,
    _fit_channel_stats,
    _stratified_holdout,
    evaluate_eegnet_classifier,
    train_eegnet_fold,
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


def test_channel_stats_standardize_per_channel():
    rng = np.random.default_rng(0)
    epochs = rng.normal(loc=5.0, scale=3.0, size=(20, 4, 64))

    mean, std = _fit_channel_stats(epochs)
    standardized = _apply_channel_stats(epochs, mean, std)

    assert mean.shape == (1, 4, 1)
    assert np.allclose(standardized.mean(axis=(0, 2)), 0.0, atol=1e-6)
    assert np.allclose(standardized.std(axis=(0, 2)), 1.0, atol=1e-6)


def test_fit_channel_stats_guards_zero_variance_channel():
    epochs = np.ones((5, 3, 16))

    _, std = _fit_channel_stats(epochs)

    assert np.all(std == 1.0)


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
        kernel_length=32,
    )

    assert evaluation.cv_splits == 3


def test_stratified_holdout_returns_whole_fold_when_too_small_to_stratify():
    # 3 epochs across 3 distinct classes: n_total (3) < 2 * n_classes (6), so the
    # split cannot place one sample per class on both sides and falls back to
    # using the entire fold for both inner train and inner validation.
    epochs = np.random.default_rng(7).normal(size=(3, 4, 64))
    labels = np.array([0, 1, 2], dtype=np.int64)

    train_uv, train_labels, val_uv, val_labels = _stratified_holdout(
        epochs,
        labels,
        val_fraction=0.2,
        random_state=0,
    )

    assert train_uv is epochs
    assert val_uv is epochs
    assert np.array_equal(train_labels, labels)
    assert np.array_equal(val_labels, labels)


def test_train_eegnet_fold_with_no_epochs_leaves_model_unchanged():
    # n_epochs=0: the training loop never runs, so best_state stays None and the
    # best-checkpoint restore is skipped (the model keeps its initial weights).
    torch.manual_seed(0)
    model = EEGNet(n_channels=4, n_samples=128, n_classes=2, kernel_length=32)
    before = copy.deepcopy(model.state_dict())

    rng = np.random.default_rng(0)
    train_uv = rng.normal(size=(8, 4, 128))
    train_labels = np.array([0, 1] * 4, dtype=np.int64)

    train_eegnet_fold(
        model,
        train_uv,
        train_labels,
        train_uv,
        train_labels,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        weight_decay=1e-2,
        n_epochs=0,
        batch_size=4,
    )

    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])


def test_train_eegnet_fold_restores_best_checkpoint_over_many_epochs():
    # Over several epochs the inner-val loss improves on some epochs and worsens
    # on others; the best (lowest-loss) checkpoint is restored at the end. This
    # exercises both the improve and the no-improve branches of the loop.
    torch.manual_seed(0)
    model = EEGNet(n_channels=4, n_samples=128, n_classes=2, kernel_length=32)

    rng = np.random.default_rng(1)
    train_uv = rng.normal(size=(12, 4, 128))
    train_labels = np.array([0, 1] * 6, dtype=np.int64)
    val_uv = rng.normal(size=(4, 4, 128))
    val_labels = np.array([0, 1, 0, 1], dtype=np.int64)

    criterion = torch.nn.CrossEntropyLoss()
    val_x = _epochs_to_tensor(val_uv)
    val_y = torch.from_numpy(val_labels)

    val_losses = train_eegnet_fold(
        model,
        train_uv,
        train_labels,
        val_uv,
        val_labels,
        device=torch.device("cpu"),
        learning_rate=5e-2,
        weight_decay=1e-2,
        n_epochs=20,
        batch_size=4,
    )

    # The inner-val loss both improved and regressed across epochs, and its lowest
    # point was not the final epoch, so restoring the best checkpoint is observable
    # (a no-op restore would leave the higher last-epoch loss).
    assert len(val_losses) == 20
    assert min(val_losses) < val_losses[-1]

    # The restored model reproduces exactly that lowest inner-val loss, proving the
    # best checkpoint (not the final or an intermediate one) was loaded back.
    model.eval()
    with torch.no_grad():
        restored_loss = float(criterion(model(val_x), val_y))
    assert restored_loss == pytest.approx(min(val_losses))
