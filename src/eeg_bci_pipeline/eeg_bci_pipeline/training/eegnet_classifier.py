"""EEGNet training and cross-validated evaluation for hand motor-imagery."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np

from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs
from eeg_bci_pipeline.data.gdf_recording import FloatArray
from eeg_bci_pipeline.training.hand_classifier import (
    DEFAULT_BANDPASS_HIGH_HZ,
    DEFAULT_BANDPASS_LOW_HZ,
    DEFAULT_CV_RANDOM_STATE,
    DEFAULT_CV_SPLITS,
    DEFAULT_HAND_CLASS_LABELS,
    HandClassifierEvaluation,
    IntArray,
    cross_validate_folds,
    select_hand_epochs,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as _err:
    raise RuntimeError("EEGNet classifier requires PyTorch. Install: pip install torch") from _err

from eeg_bci_pipeline.training.eegnet import EEGNet

DEFAULT_F1 = 8
DEFAULT_DEPTH_MULTIPLIER = 2
DEFAULT_KERNEL_LENGTH = 125
DEFAULT_DROPOUT_RATE = 0.5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_TRAINING_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32

# Fraction of each training fold held out (stratified) as an inner validation
# set for early stopping, so the outer CV fold stays untouched until scoring.
INNER_VAL_FRACTION = 0.2


def evaluate_eegnet_classifier(
    labeled_epochs: LabeledEpochs,
    *,
    class_labels: Sequence[str] = DEFAULT_HAND_CLASS_LABELS,
    cv_splits: int = DEFAULT_CV_SPLITS,
    cv_random_state: int = DEFAULT_CV_RANDOM_STATE,
    bandpass_low_hz: float | None = DEFAULT_BANDPASS_LOW_HZ,
    bandpass_high_hz: float | None = DEFAULT_BANDPASS_HIGH_HZ,
    f1: int = DEFAULT_F1,
    d: int = DEFAULT_DEPTH_MULTIPLIER,
    kernel_length: int = DEFAULT_KERNEL_LENGTH,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    n_epochs: int = DEFAULT_TRAINING_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> HandClassifierEvaluation:
    """Evaluate an EEGNet classifier with stratified cross-validation."""

    training_data = select_hand_epochs(labeled_epochs, class_labels=class_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def score_fold(
        fold_index: int,
        train_epochs_uv: FloatArray,
        train_labels: IntArray,
        val_epochs_uv: FloatArray,
        val_labels: IntArray,
    ) -> float:
        torch.manual_seed(cv_random_state + fold_index)  # pyright: ignore[reportUnknownMemberType]

        # Standardize per channel using training-fold statistics only, then
        # apply the same shift/scale to the held-out fold (no leakage).
        channel_mean, channel_std = _fit_channel_stats(train_epochs_uv)
        train_epochs_uv = _apply_channel_stats(train_epochs_uv, channel_mean, channel_std)
        val_epochs_uv = _apply_channel_stats(val_epochs_uv, channel_mean, channel_std)

        (
            inner_train_uv,
            inner_train_labels,
            inner_val_uv,
            inner_val_labels,
        ) = _stratified_holdout(
            train_epochs_uv,
            train_labels,
            val_fraction=INNER_VAL_FRACTION,
            random_state=cv_random_state + fold_index,
        )

        model = EEGNet(
            n_channels=training_data.channel_count,
            n_samples=training_data.samples_per_epoch,
            n_classes=len(training_data.class_labels),
            f1=f1,
            d=d,
            kernel_length=kernel_length,
            dropout_rate=dropout_rate,
        ).to(device)

        train_eegnet_fold(
            model,
            inner_train_uv,
            inner_train_labels,
            inner_val_uv,
            inner_val_labels,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )

        # Score the untouched outer fold exactly once.
        return _accuracy(
            model,
            _epochs_to_tensor(val_epochs_uv).to(device),
            _labels_to_tensor(val_labels).to(device),
        )

    return cross_validate_folds(
        training_data,
        classifier_name="eegnet",
        cv_splits=cv_splits,
        cv_random_state=cv_random_state,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        score_fold=score_fold,
    )


def train_eegnet_fold(
    model: nn.Module,
    train_epochs_uv: FloatArray,
    train_labels: IntArray,
    val_epochs_uv: FloatArray,
    val_labels: IntArray,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    n_epochs: int,
    batch_size: int,
) -> None:
    """Train ``model`` for ``n_epochs`` and restore its best checkpoint.

    The lowest-validation-loss checkpoint (on an inner split of the training
    fold, never the outer CV fold) is restored before returning, so the reported
    accuracy stays unbiased. Training runs the full schedule with no early stop:
    on small folds the inner-val loss often worsens for tens of epochs before
    recovering, so stopping early restored a near-random checkpoint.
    """

    train_x = _epochs_to_tensor(train_epochs_uv).to(device)
    train_y = _labels_to_tensor(train_labels).to(device)
    val_x = _epochs_to_tensor(val_epochs_uv).to(device)
    val_y = _labels_to_tensor(val_labels).to(device)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None

    for _epoch in range(n_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)


def _fit_channel_stats(epochs_uv: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-channel mean and std over the training epochs (epoch and time axes)."""
    mean = epochs_uv.mean(axis=(0, 2), keepdims=True)
    std = epochs_uv.std(axis=(0, 2), keepdims=True)
    std[std == 0.0] = 1.0
    return mean, std


def _apply_channel_stats(epochs_uv: FloatArray, mean: FloatArray, std: FloatArray) -> FloatArray:
    """Z-score epochs with precomputed per-channel statistics."""
    return (epochs_uv - mean) / std


def _stratified_holdout(
    epochs_uv: FloatArray,
    labels: IntArray,
    *,
    val_fraction: float,
    random_state: int,
) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
    """Carve a stratified inner train/validation split from a training fold.

    Falls back to using the whole fold for both sides when it is too small to
    split with at least one sample per class on each side; the outer scoring
    fold is unaffected either way.
    """

    n_classes = int(np.unique(labels).size)
    n_total = int(labels.shape[0])
    if n_total < 2 * n_classes:
        return epochs_uv, labels, epochs_uv, labels

    train_test_split = cast(
        Callable[..., list[Any]],
        getattr(importlib.import_module("sklearn.model_selection"), "train_test_split"),
    )
    val_count = min(max(int(round(val_fraction * n_total)), n_classes), n_total - n_classes)
    inner_train_uv, inner_val_uv, inner_train_labels, inner_val_labels = train_test_split(
        epochs_uv,
        labels,
        test_size=val_count,
        stratify=labels,
        random_state=random_state,
        shuffle=True,
    )
    return inner_train_uv, inner_train_labels, inner_val_uv, inner_val_labels


def _accuracy(model: nn.Module, epochs: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the model's classification accuracy on the given tensors."""
    model.eval()
    with torch.no_grad():
        predictions = model(epochs).argmax(dim=1)
        return float((predictions == labels).float().mean())


def _epochs_to_tensor(epochs_uv: FloatArray) -> torch.Tensor:
    """Convert (n_epochs, n_channels, n_samples) to (n_epochs, 1, n_channels, n_samples)."""
    arr = np.asarray(epochs_uv, dtype=np.float32)
    return torch.from_numpy(arr[:, np.newaxis, :, :])  # pyright: ignore[reportUnknownMemberType]


def _labels_to_tensor(labels: IntArray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(labels, dtype=np.int64))  # pyright: ignore[reportUnknownMemberType]
