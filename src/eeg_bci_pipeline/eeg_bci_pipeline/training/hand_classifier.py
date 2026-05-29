"""Offline CSP + LDA baseline for left/right motor-imagery epochs."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, cast

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
HAND_CLASSIFIER_ARTIFACT_VERSION = 1

IntArray = npt.NDArray[np.int_]
FoldScorer = Callable[[int, FloatArray, IntArray, FloatArray, IntArray], float]
PipelineFactory = Callable[..., object]
FilterData = Callable[..., FloatArray]
FitPipeline = Callable[[FloatArray, IntArray], object]
PredictPipeline = Callable[[FloatArray], IntArray]


class JoblibLike(Protocol):
    def dump(self, value: object, filename: str | Path) -> object: ...

    def load(self, filename: str | Path) -> object: ...


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
    cv_splits: int
    fold_scores: tuple[float, ...]
    mean_accuracy: float
    std_accuracy: float
    classifier_name: str = "csp-lda"
    csp_components: int = 0
    bandpass_low_hz: float | None = DEFAULT_BANDPASS_LOW_HZ
    bandpass_high_hz: float | None = DEFAULT_BANDPASS_HIGH_HZ


@dataclass(frozen=True)
class HandClassifierArtifact:
    """A final-fit classifier and the runtime contract needed to use it."""

    source_id: str
    sampling_rate_hz: float
    channel_labels: tuple[str, ...]
    class_labels: tuple[str, ...]
    class_counts: tuple[int, ...]
    samples_per_epoch: int
    pipeline: object
    epoch_tmin_sec: float | None = None
    epoch_tmax_sec: float | None = None
    classifier_name: str = "csp-lda"
    csp_components: int = DEFAULT_CSP_COMPONENTS
    bandpass_low_hz: float | None = DEFAULT_BANDPASS_LOW_HZ
    bandpass_high_hz: float | None = DEFAULT_BANDPASS_HIGH_HZ
    artifact_version: int = HAND_CLASSIFIER_ARTIFACT_VERSION

    @property
    def channel_count(self) -> int:
        return len(self.channel_labels)


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


def train_hand_classifier(
    labeled_epochs: LabeledEpochs,
    *,
    class_labels: Sequence[str] = DEFAULT_HAND_CLASS_LABELS,
    csp_components: int = DEFAULT_CSP_COMPONENTS,
    bandpass_low_hz: float | None = DEFAULT_BANDPASS_LOW_HZ,
    bandpass_high_hz: float | None = DEFAULT_BANDPASS_HIGH_HZ,
    epoch_tmin_sec: float | None = None,
    epoch_tmax_sec: float | None = None,
) -> HandClassifierArtifact:
    """Fit one deployable CSP + LDA classifier on all calibration epochs."""

    training_data = select_hand_epochs(labeled_epochs, class_labels=class_labels)
    _validate_csp_components(csp_components, channel_count=training_data.channel_count)

    try:
        from mne import use_log_level
    except ImportError as error:
        raise RuntimeError(
            "Hand classifier training requires MNE and scikit-learn. "
            "Install: python3-mne python3-sklearn"
        ) from error

    epochs_uv = maybe_bandpass_epochs(
        training_data.epochs_uv,
        sampling_rate_hz=training_data.sampling_rate_hz,
        low_hz=bandpass_low_hz,
        high_hz=bandpass_high_hz,
    )
    pipeline = build_csp_lda_pipeline(csp_components=csp_components)
    fit = cast(FitPipeline, getattr(pipeline, "fit"))
    with use_log_level("ERROR"):
        fit(epochs_uv, training_data.encoded_labels)

    return HandClassifierArtifact(
        source_id=training_data.source_id,
        sampling_rate_hz=training_data.sampling_rate_hz,
        channel_labels=training_data.channel_labels,
        class_labels=training_data.class_labels,
        class_counts=training_data.class_counts,
        samples_per_epoch=training_data.samples_per_epoch,
        pipeline=pipeline,
        epoch_tmin_sec=epoch_tmin_sec,
        epoch_tmax_sec=epoch_tmax_sec,
        csp_components=csp_components,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
    )


def save_hand_classifier_artifact(
    artifact: HandClassifierArtifact,
    output_path: str | Path,
) -> None:
    """Persist a trained hand classifier artifact."""

    _joblib().dump(artifact, Path(output_path))


def load_hand_classifier_artifact(path: str | Path) -> HandClassifierArtifact:
    """Load and validate a persisted hand classifier artifact."""

    artifact = _joblib().load(Path(path))
    if not isinstance(artifact, HandClassifierArtifact):
        raise ValueError("artifact does not contain a hand classifier")
    if artifact.artifact_version != HAND_CLASSIFIER_ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported hand classifier artifact version: {artifact.artifact_version}"
        )
    return artifact


def predict_hand_classifier(
    artifact: HandClassifierArtifact,
    epochs_uv: FloatArray,
) -> tuple[str, ...]:
    """Predict class labels for epoch-shaped EEG windows with a saved artifact."""

    epochs = _validate_artifact_epoch_shape(artifact, epochs_uv)
    prepared_epochs = maybe_bandpass_epochs(
        epochs,
        sampling_rate_hz=artifact.sampling_rate_hz,
        low_hz=artifact.bandpass_low_hz,
        high_hz=artifact.bandpass_high_hz,
    )
    predict = cast(PredictPipeline, getattr(artifact.pipeline, "predict"))
    predicted_indices = np.asarray(predict(prepared_epochs), dtype=np.int_)
    labels: list[str] = []
    for class_index in predicted_indices:
        index = int(class_index)
        if index < 0 or index >= len(artifact.class_labels):
            raise ValueError(f"classifier predicted unknown class index: {index}")
        labels.append(artifact.class_labels[index])
    return tuple(labels)


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

    try:
        from mne import use_log_level
    except ImportError as error:
        raise RuntimeError(
            "Hand classifier evaluation requires MNE and scikit-learn. "
            "Install: python3-mne python3-sklearn"
        ) from error

    def score_fold(
        _fold_index: int,
        train_epochs_uv: FloatArray,
        train_labels: IntArray,
        val_epochs_uv: FloatArray,
        val_labels: IntArray,
    ) -> float:
        pipeline = cast(Any, build_csp_lda_pipeline(csp_components=csp_components))
        with use_log_level("ERROR"):
            pipeline.fit(train_epochs_uv, train_labels)
            return float(pipeline.score(val_epochs_uv, val_labels))

    return cross_validate_folds(
        training_data,
        classifier_name="csp-lda",
        csp_components=csp_components,
        cv_splits=cv_splits,
        cv_random_state=cv_random_state,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        score_fold=score_fold,
    )


def cross_validate_folds(
    training_data: HandTrainingData,
    *,
    classifier_name: str,
    cv_splits: int,
    cv_random_state: int,
    bandpass_low_hz: float | None,
    bandpass_high_hz: float | None,
    score_fold: FoldScorer,
    csp_components: int = 0,
) -> HandClassifierEvaluation:
    """Run stratified k-fold CV, scoring each fold with ``score_fold``.

    ``score_fold`` receives ``(fold_index, train_epochs_uv, train_labels,
    val_epochs_uv, val_labels)`` and returns that fold's accuracy. Shared by the
    CSP+LDA and EEGNet evaluators so the bandpass, fold splitting, and report
    assembly live in one place.
    """

    epochs_uv = maybe_bandpass_epochs(
        training_data.epochs_uv,
        sampling_rate_hz=training_data.sampling_rate_hz,
        low_hz=bandpass_low_hz,
        high_hz=bandpass_high_hz,
    )
    resolved_cv_splits = resolve_cv_splits(cv_splits, training_data.class_counts)

    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as error:
        raise RuntimeError(
            "Cross-validation requires scikit-learn. Install: python3-sklearn"
        ) from error

    cv = StratifiedKFold(
        n_splits=resolved_cv_splits,
        shuffle=True,
        random_state=cv_random_state,
    )
    labels = training_data.encoded_labels
    fold_scores: list[float] = []
    for fold_index, (train_idx, val_idx) in enumerate(
        cv.split(epochs_uv, labels)  # pyright: ignore[reportUnknownMemberType]
    ):
        fold_scores.append(
            score_fold(
                fold_index,
                epochs_uv[train_idx],
                labels[train_idx],
                epochs_uv[val_idx],
                labels[val_idx],
            )
        )

    scores = np.array(fold_scores)
    return HandClassifierEvaluation(
        source_id=training_data.source_id,
        sampling_rate_hz=training_data.sampling_rate_hz,
        channel_count=training_data.channel_count,
        samples_per_epoch=training_data.samples_per_epoch,
        class_labels=training_data.class_labels,
        class_counts=training_data.class_counts,
        cv_splits=resolved_cv_splits,
        fold_scores=tuple(fold_scores),
        mean_accuracy=float(np.mean(scores)),
        std_accuracy=float(np.std(scores)),
        classifier_name=classifier_name,
        csp_components=csp_components,
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
        f"Hand classifier evaluation ({evaluation.classifier_name})",
        f"source: {evaluation.source_id}",
        f"classes: {', '.join(evaluation.class_labels)}",
        "class counts:",
        *class_lines,
        f"epochs: {sum(evaluation.class_counts)}",
        f"shape: {evaluation.channel_count} channels x {evaluation.samples_per_epoch} samples",
        f"sampling rate: {evaluation.sampling_rate_hz:g} Hz",
        f"bandpass: {bandpass}",
    ]
    if evaluation.classifier_name == "csp-lda":
        lines.append(f"csp components: {evaluation.csp_components}")
    lines += [
        f"cv splits: {evaluation.cv_splits}",
        f"fold accuracies: {fold_scores}",
        f"mean accuracy: {evaluation.mean_accuracy:.3f} +/- {evaluation.std_accuracy:.3f}",
    ]
    return "\n".join(lines)


def maybe_bandpass_epochs(
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


def resolve_cv_splits(requested_splits: int, class_counts: Sequence[int]) -> int:
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


def _validate_artifact_epoch_shape(
    artifact: HandClassifierArtifact,
    epochs_uv: FloatArray,
) -> FloatArray:
    epochs = np.asarray(epochs_uv, dtype=np.float64)
    if epochs.ndim != 3:
        raise ValueError("epochs must have shape (n_epochs, n_channels, n_samples)")
    if epochs.shape[1] != artifact.channel_count:
        raise ValueError(
            "epoch channel count does not match artifact: "
            f"{epochs.shape[1]} != {artifact.channel_count}"
        )
    if epochs.shape[2] != artifact.samples_per_epoch:
        raise ValueError(
            "epoch sample count does not match artifact: "
            f"{epochs.shape[2]} != {artifact.samples_per_epoch}"
        )
    return epochs


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


def _joblib() -> JoblibLike:
    try:
        return cast(JoblibLike, importlib.import_module("joblib"))
    except ImportError as error:
        raise RuntimeError("Model persistence requires joblib. Install: python3-joblib") from error
