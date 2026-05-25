"""Offline CSP + LDA baseline for left/right motor-imagery epochs."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
import numpy.typing as npt

from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs
from eeg_bci_pipeline.data.gdf_recording import FloatArray

DEFAULT_HAND_CLASS_LABELS = ("left_hand", "right_hand")
DEFAULT_BANDPASS_LOW_HZ = 8.0
DEFAULT_BANDPASS_HIGH_HZ = 30.0
DEFAULT_CSP_COMPONENTS = 4
DEFAULT_CV_SPLITS = 5
DEFAULT_CV_RANDOM_STATE = 7

IntArray = npt.NDArray[np.int_]
CrossValScore = Callable[..., FloatArray]
PipelineFactory = Callable[..., object]
FilterData = Callable[..., FloatArray]


@dataclass(frozen=True)
class HandTrainingData:
    source_id: str
    sampling_rate_hz: float
    channel_labels: tuple[str, ...]
    class_labels: tuple[str, ...]
    class_counts: tuple[int, ...]
    epochs_uv: FloatArray
    encoded_labels: IntArray

    @property
    def epoch_count(self) -> int:
        return int(self.epochs_uv.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.epochs_uv.shape[1])

    @property
    def samples_per_epoch(self) -> int:
        return int(self.epochs_uv.shape[2])


@dataclass(frozen=True)
class HandClassifierEvaluation:
    source_id: str
    sampling_rate_hz: float
    channel_count: int
    samples_per_epoch: int
    class_labels: tuple[str, ...]
    class_counts: tuple[int, ...]
    csp_components: int
    cv_splits: int
    fold_scores: tuple[float, ...]
    mean_accuracy: float
    std_accuracy: float
    bandpass_low_hz: float | None = DEFAULT_BANDPASS_LOW_HZ
    bandpass_high_hz: float | None = DEFAULT_BANDPASS_HIGH_HZ


def select_hand_epochs(
    labeled_epochs: LabeledEpochs,
    *,
    class_labels: Sequence[str] = DEFAULT_HAND_CLASS_LABELS,
) -> HandTrainingData:
    """Select and encode epochs for the requested hand classes."""

    labels = tuple(str(label) for label in class_labels)
    if len(labels) < 2:
        raise ValueError("class_labels must contain at least two classes")
    if len(set(labels)) != len(labels):
        raise ValueError("class_labels must be unique")

    label_to_index = {label: index for index, label in enumerate(labels)}
    selected_indices: list[int] = []
    encoded_labels: list[int] = []
    for epoch_index, label in enumerate(labeled_epochs.labels):
        class_index = label_to_index.get(label)
        if class_index is None:
            continue
        selected_indices.append(epoch_index)
        encoded_labels.append(class_index)

    if not selected_indices:
        raise ValueError("no epochs found for requested class labels")

    class_counts = tuple(encoded_labels.count(index) for index in range(len(labels)))
    missing_labels = [label for label, count in zip(labels, class_counts) if count == 0]
    if missing_labels:
        raise ValueError(f"missing epochs for class labels: {missing_labels}")

    return HandTrainingData(
        source_id=labeled_epochs.source_id,
        sampling_rate_hz=labeled_epochs.sampling_rate_hz,
        channel_labels=labeled_epochs.channel_labels,
        class_labels=labels,
        class_counts=class_counts,
        epochs_uv=np.asarray(labeled_epochs.epochs_uv[selected_indices], dtype=np.float64),
        encoded_labels=np.asarray(encoded_labels, dtype=np.int_),
    )


def evaluate_hand_classifier(
    labeled_epochs: LabeledEpochs,
    *,
    class_labels: Sequence[str] = DEFAULT_HAND_CLASS_LABELS,
    csp_components: int = DEFAULT_CSP_COMPONENTS,
    cv_splits: int = DEFAULT_CV_SPLITS,
    cv_random_state: int = DEFAULT_CV_RANDOM_STATE,
    bandpass_low_hz: float | None = DEFAULT_BANDPASS_LOW_HZ,
    bandpass_high_hz: float | None = DEFAULT_BANDPASS_HIGH_HZ,
) -> HandClassifierEvaluation:
    """Evaluate a CSP + LDA classifier with stratified cross-validation."""

    training_data = select_hand_epochs(labeled_epochs, class_labels=class_labels)
    _validate_csp_components(csp_components, channel_count=training_data.channel_count)
    resolved_cv_splits = _resolve_cv_splits(cv_splits, training_data.class_counts)
    epochs_uv = _maybe_bandpass_epochs(
        training_data.epochs_uv,
        sampling_rate_hz=training_data.sampling_rate_hz,
        low_hz=bandpass_low_hz,
        high_hz=bandpass_high_hz,
    )

    try:
        from mne import use_log_level
        from sklearn.model_selection import StratifiedKFold
    except ImportError as error:
        raise RuntimeError(
            "Hand classifier evaluation requires MNE and scikit-learn. "
            "Install: python3-mne python3-sklearn"
        ) from error

    pipeline = build_csp_lda_pipeline(csp_components=csp_components)
    cv = StratifiedKFold(
        n_splits=resolved_cv_splits,
        shuffle=True,
        random_state=cv_random_state,
    )
    cross_val_score_typed = _cross_val_score()
    with use_log_level("ERROR"):
        scores = cross_val_score_typed(
            pipeline,
            epochs_uv,
            training_data.encoded_labels,
            cv=cv,
            scoring="accuracy",
            error_score="raise",
        )
    fold_scores = tuple(float(score) for score in scores)
    return HandClassifierEvaluation(
        source_id=training_data.source_id,
        sampling_rate_hz=training_data.sampling_rate_hz,
        channel_count=training_data.channel_count,
        samples_per_epoch=training_data.samples_per_epoch,
        class_labels=training_data.class_labels,
        class_counts=training_data.class_counts,
        csp_components=csp_components,
        cv_splits=resolved_cv_splits,
        fold_scores=fold_scores,
        mean_accuracy=float(np.mean(scores)),
        std_accuracy=float(np.std(scores)),
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
    )


def build_csp_lda_pipeline(*, csp_components: int = DEFAULT_CSP_COMPONENTS) -> object:
    """Build the baseline classifier pipeline."""

    try:
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except ImportError as error:
        raise RuntimeError(
            "Hand classifier evaluation requires MNE and scikit-learn. "
            "Install: python3-mne python3-sklearn"
        ) from error

    csp = CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    return _make_pipeline()(csp, lda)


def format_evaluation_report(evaluation: HandClassifierEvaluation) -> str:
    """Format a compact human-readable evaluation report."""

    class_lines = [
        f"  {label}: {count}"
        for label, count in zip(evaluation.class_labels, evaluation.class_counts)
    ]
    bandpass = "disabled"
    if evaluation.bandpass_low_hz is not None and evaluation.bandpass_high_hz is not None:
        bandpass = f"{evaluation.bandpass_low_hz:g}-{evaluation.bandpass_high_hz:g} Hz"

    fold_scores = ", ".join(f"{score:.3f}" for score in evaluation.fold_scores)
    lines = [
        "Hand classifier evaluation",
        f"source: {evaluation.source_id}",
        f"classes: {', '.join(evaluation.class_labels)}",
        "class counts:",
        *class_lines,
        f"epochs: {sum(evaluation.class_counts)}",
        f"shape: {evaluation.channel_count} channels x {evaluation.samples_per_epoch} samples",
        f"sampling rate: {evaluation.sampling_rate_hz:g} Hz",
        f"bandpass: {bandpass}",
        f"csp components: {evaluation.csp_components}",
        f"cv splits: {evaluation.cv_splits}",
        f"fold accuracies: {fold_scores}",
        f"mean accuracy: {evaluation.mean_accuracy:.3f} +/- {evaluation.std_accuracy:.3f}",
    ]
    return "\n".join(lines)


def _maybe_bandpass_epochs(
    epochs_uv: FloatArray,
    *,
    sampling_rate_hz: float,
    low_hz: float | None,
    high_hz: float | None,
) -> FloatArray:
    if low_hz is None and high_hz is None:
        return np.asarray(epochs_uv, dtype=np.float64)
    if low_hz is None or high_hz is None:
        raise ValueError("bandpass low and high frequencies must both be set or both disabled")
    if low_hz <= 0.0:
        raise ValueError("bandpass low frequency must be greater than 0")
    if high_hz <= low_hz:
        raise ValueError("bandpass high frequency must be greater than low frequency")
    nyquist_hz = sampling_rate_hz / 2.0
    if high_hz >= nyquist_hz:
        raise ValueError("bandpass high frequency must be below Nyquist")

    return _filter_data()(
        np.asarray(epochs_uv, dtype=np.float64),
        sfreq=sampling_rate_hz,
        l_freq=low_hz,
        h_freq=high_hz,
        verbose="ERROR",
    )


def _resolve_cv_splits(requested_splits: int, class_counts: Sequence[int]) -> int:
    if requested_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    min_class_count = min(class_counts)
    if min_class_count < 2:
        raise ValueError("each class must contain at least two epochs for cross-validation")
    return min(requested_splits, min_class_count)


def _validate_csp_components(csp_components: int, *, channel_count: int) -> None:
    if csp_components < 1:
        raise ValueError("csp_components must be at least 1")
    if csp_components > channel_count:
        raise ValueError("csp_components must not exceed channel count")


def _cross_val_score() -> CrossValScore:
    return cast(
        CrossValScore,
        getattr(importlib.import_module("sklearn.model_selection"), "cross_val_score"),
    )


def _make_pipeline() -> PipelineFactory:
    return cast(
        PipelineFactory,
        getattr(importlib.import_module("sklearn.pipeline"), "make_pipeline"),
    )


def _filter_data() -> FilterData:
    try:
        filter_module = importlib.import_module("mne.filter")
    except ImportError as error:
        raise RuntimeError("Bandpass filtering requires MNE. Install: python3-mne") from error

    return cast(FilterData, getattr(filter_module, "filter_data"))
