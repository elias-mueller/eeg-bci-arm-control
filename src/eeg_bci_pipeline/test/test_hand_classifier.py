import importlib

import numpy as np
import pytest
from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs
from eeg_bci_pipeline.training.hand_classifier import (
    HAND_CLASSIFIER_ARTIFACT_VERSION,
    HandClassifierArtifact,
    HandClassifierEvaluation,
    HandTrainingData,
    _validate_artifact_epoch_shape,
    _validate_csp_components,
    evaluate_hand_classifier,
    format_evaluation_report,
    load_hand_classifier_artifact,
    maybe_bandpass_epochs,
    predict_hand_classifier,
    predict_hand_proba,
    resolve_cv_splits,
    save_hand_classifier_artifact,
    select_hand_epochs,
    train_hand_classifier,
)


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


def test_select_hand_epochs_filters_and_encodes_requested_classes():
    epochs = make_labeled_epochs(
        ("feet", "left_hand", "right_hand", "left_hand", "tongue", "right_hand")
    )

    training_data = select_hand_epochs(epochs)

    assert training_data.source_id == "synthetic"
    assert training_data.class_labels == ("left_hand", "right_hand")
    assert training_data.class_counts == (2, 2)
    assert training_data.epochs_uv.shape == (4, 4, 100)
    assert training_data.encoded_labels.tolist() == [0, 1, 0, 1]


def test_select_hand_epochs_rejects_missing_requested_class():
    epochs = make_labeled_epochs(("left_hand", "left_hand", "feet"))

    with pytest.raises(ValueError, match="missing"):
        select_hand_epochs(epochs)


def test_evaluate_hand_classifier_reports_cross_validation_metadata():
    labels = ("left_hand", "right_hand") * 8
    epochs = make_labeled_epochs(labels)

    evaluation = evaluate_hand_classifier(
        epochs,
        csp_components=2,
        cv_splits=4,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
    )

    assert evaluation.source_id == "synthetic"
    assert evaluation.class_counts == (8, 8)
    assert evaluation.csp_components == 2
    assert evaluation.cv_splits == 4
    assert len(evaluation.fold_scores) == 4
    assert 0.0 <= evaluation.mean_accuracy <= 1.0


def test_train_hand_classifier_saves_loads_and_predicts(tmp_path):
    labels = ("left_hand", "right_hand") * 8
    epochs = make_labeled_epochs(labels)
    artifact = train_hand_classifier(
        epochs,
        csp_components=2,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
        epoch_tmin_sec=0.5,
        epoch_tmax_sec=3.5,
    )

    output_path = tmp_path / "hand-csp-lda.joblib"
    save_hand_classifier_artifact(artifact, output_path)
    loaded = load_hand_classifier_artifact(output_path)
    training_data = select_hand_epochs(epochs)

    assert loaded.source_id == "synthetic"
    assert loaded.channel_labels == ("C3", "Cz", "C4", "Pz")
    assert loaded.class_labels == ("left_hand", "right_hand")
    assert loaded.samples_per_epoch == 100
    assert loaded.epoch_tmin_sec == 0.5
    assert loaded.epoch_tmax_sec == 3.5
    assert predict_hand_classifier(loaded, training_data.epochs_uv) == predict_hand_classifier(
        artifact,
        training_data.epochs_uv,
    )


def test_predict_hand_classifier_rejects_epoch_shape_mismatch():
    labels = ("left_hand", "right_hand") * 4
    epochs = make_labeled_epochs(labels)
    artifact = train_hand_classifier(
        epochs,
        csp_components=2,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
    )

    with pytest.raises(ValueError, match="sample count"):
        predict_hand_classifier(artifact, epochs.epochs_uv[:, :, :-1])


def test_predict_hand_proba_normalizes_and_aligns_with_labels():
    labels = ("left_hand", "right_hand") * 8
    epochs = make_labeled_epochs(labels)
    artifact = train_hand_classifier(
        epochs,
        csp_components=2,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
    )
    training_data = select_hand_epochs(epochs)

    proba_labels, probabilities = predict_hand_proba(artifact, training_data.epochs_uv)

    assert probabilities.shape == (len(labels), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    # Columns are ordered to match artifact.class_labels, so the proba argmax label
    # equals the hard prediction from predict_hand_classifier.
    predicted = predict_hand_classifier(artifact, training_data.epochs_uv)
    assert proba_labels == predicted
    argmax_labels = tuple(
        artifact.class_labels[int(index)] for index in probabilities.argmax(axis=1)
    )
    assert argmax_labels == predicted


def test_predict_hand_proba_remaps_columns_by_estimator_class_order():
    # Estimator whose internal class order is REVERSED relative to class_labels, so an
    # identity (non-remapping) implementation would transpose the probabilities.
    class _FakePipeline:
        classes_ = np.array([1, 0])

        def predict_proba(self, epochs):
            n = np.asarray(epochs).shape[0]
            # raw column 0 -> encoded class 1, column 1 -> encoded class 0
            return np.tile([0.3, 0.7], (n, 1))

    artifact = HandClassifierArtifact(
        source_id="fake",
        sampling_rate_hz=100.0,
        channel_labels=("a", "b", "c", "d"),
        class_labels=("left_hand", "right_hand"),
        class_counts=(1, 1),
        samples_per_epoch=8,
        pipeline=_FakePipeline(),
        bandpass_low_hz=None,
        bandpass_high_hz=None,
    )

    labels, probabilities = predict_hand_proba(artifact, np.zeros((1, 4, 8)))

    # P(class 1)=right_hand=0.3 must land at index 1, P(class 0)=left_hand=0.7 at index 0.
    np.testing.assert_allclose(probabilities, [[0.7, 0.3]])
    assert labels == ("left_hand",)


def test_format_evaluation_report_includes_classifier_summary():
    labels = ("left_hand", "right_hand") * 4
    epochs = make_labeled_epochs(labels)
    evaluation = evaluate_hand_classifier(
        epochs,
        csp_components=2,
        cv_splits=2,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
    )

    report = format_evaluation_report(evaluation)

    assert "Hand classifier evaluation" in report
    assert "left_hand" in report
    assert "right_hand" in report
    assert "mean accuracy" in report


def test_format_evaluation_report_eegnet_omits_csp_components():
    evaluation = HandClassifierEvaluation(
        source_id="synthetic",
        sampling_rate_hz=128.0,
        channel_count=4,
        samples_per_epoch=128,
        class_labels=("left_hand", "right_hand"),
        class_counts=(8, 8),
        cv_splits=2,
        fold_scores=(0.5, 0.6),
        mean_accuracy=0.55,
        std_accuracy=0.05,
        classifier_name="eegnet",
    )

    report = format_evaluation_report(evaluation)

    assert "Hand classifier evaluation (eegnet)" in report
    assert "csp components" not in report


def make_artifact(pipeline: object, **overrides: object) -> HandClassifierArtifact:
    defaults: dict[str, object] = dict(
        source_id="fake",
        sampling_rate_hz=100.0,
        channel_labels=("a", "b", "c", "d"),
        class_labels=("left_hand", "right_hand"),
        class_counts=(1, 1),
        samples_per_epoch=8,
        pipeline=pipeline,
        bandpass_low_hz=None,
        bandpass_high_hz=None,
    )
    defaults.update(overrides)
    return HandClassifierArtifact(**defaults)  # type: ignore[arg-type]


def test_hand_training_data_epoch_count_reports_leading_dimension():
    training_data = HandTrainingData(
        source_id="synthetic",
        sampling_rate_hz=100.0,
        channel_labels=("C3", "Cz", "C4", "Pz"),
        class_labels=("left_hand", "right_hand"),
        class_counts=(2, 1),
        epochs_uv=np.zeros((3, 4, 8), dtype=np.float64),
        encoded_labels=np.array([0, 0, 1], dtype=np.int_),
    )

    assert training_data.epoch_count == 3
    assert training_data.channel_count == 4
    assert training_data.samples_per_epoch == 8


def test_select_hand_epochs_rejects_fewer_than_two_classes():
    epochs = make_labeled_epochs(("left_hand", "right_hand"))

    with pytest.raises(ValueError, match="at least two classes"):
        select_hand_epochs(epochs, class_labels=("left_hand",))


def test_select_hand_epochs_rejects_duplicate_classes():
    epochs = make_labeled_epochs(("left_hand", "right_hand"))

    with pytest.raises(ValueError, match="unique"):
        select_hand_epochs(epochs, class_labels=("left_hand", "left_hand"))


def test_select_hand_epochs_rejects_no_matching_epochs():
    epochs = make_labeled_epochs(("feet", "tongue", "feet"))

    with pytest.raises(ValueError, match="no epochs found"):
        select_hand_epochs(epochs, class_labels=("left_hand", "right_hand"))


def test_load_hand_classifier_artifact_rejects_wrong_type(tmp_path):
    joblib = importlib.import_module("joblib")
    output_path = tmp_path / "not-an-artifact.joblib"
    joblib.dump({"not": "an artifact"}, output_path)

    with pytest.raises(ValueError, match="does not contain a hand classifier"):
        load_hand_classifier_artifact(output_path)


def test_load_hand_classifier_artifact_rejects_version_mismatch(tmp_path):
    artifact = make_artifact(
        object(),
        artifact_version=HAND_CLASSIFIER_ARTIFACT_VERSION + 1,
    )
    output_path = tmp_path / "stale-version.joblib"
    save_hand_classifier_artifact(artifact, output_path)

    with pytest.raises(ValueError, match="unsupported hand classifier artifact version"):
        load_hand_classifier_artifact(output_path)


def test_predict_hand_classifier_rejects_out_of_range_index():
    class _OutOfRangePipeline:
        def predict(self, epochs):
            return np.full(np.asarray(epochs).shape[0], 5, dtype=np.int_)

    artifact = make_artifact(_OutOfRangePipeline())

    with pytest.raises(ValueError, match="unknown class index"):
        predict_hand_classifier(artifact, np.zeros((2, 4, 8)))


def test_predict_hand_proba_requires_predict_proba_support():
    class _HardOnlyPipeline:
        def predict(self, epochs):
            return np.zeros(np.asarray(epochs).shape[0], dtype=np.int_)

    artifact = make_artifact(_HardOnlyPipeline())

    with pytest.raises(RuntimeError, match="does not support probability output"):
        predict_hand_proba(artifact, np.zeros((1, 4, 8)))


def test_predict_hand_proba_rejects_out_of_range_class_order():
    class _BadClassOrderPipeline:
        classes_ = np.array([0, 9])

        def predict_proba(self, epochs):
            n = np.asarray(epochs).shape[0]
            return np.tile([0.6, 0.4], (n, 1))

    artifact = make_artifact(_BadClassOrderPipeline())

    with pytest.raises(ValueError, match="unknown class index"):
        predict_hand_proba(artifact, np.zeros((1, 4, 8)))


def test_maybe_bandpass_epochs_rejects_partial_band_specification():
    epochs = np.zeros((2, 4, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="both be set or both disabled"):
        maybe_bandpass_epochs(epochs, sampling_rate_hz=100.0, low_hz=8.0, high_hz=None)


def test_maybe_bandpass_epochs_rejects_non_positive_low():
    epochs = np.zeros((2, 4, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="low frequency must be greater than 0"):
        maybe_bandpass_epochs(epochs, sampling_rate_hz=100.0, low_hz=0.0, high_hz=30.0)


def test_maybe_bandpass_epochs_rejects_high_not_above_low():
    epochs = np.zeros((2, 4, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="high frequency must be greater than low"):
        maybe_bandpass_epochs(epochs, sampling_rate_hz=100.0, low_hz=30.0, high_hz=30.0)


def test_maybe_bandpass_epochs_rejects_high_at_or_above_nyquist():
    epochs = np.zeros((2, 4, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="below Nyquist"):
        maybe_bandpass_epochs(epochs, sampling_rate_hz=100.0, low_hz=8.0, high_hz=50.0)


def test_maybe_bandpass_epochs_attenuates_out_of_band_content():
    pytest.importorskip("mne")
    time = np.arange(256, dtype=float) / 100.0
    in_band = np.sin(2.0 * np.pi * 15.0 * time)
    out_of_band = np.sin(2.0 * np.pi * 1.0 * time)
    epochs = (in_band + out_of_band)[None, None, :].repeat(3, axis=1)

    filtered = maybe_bandpass_epochs(epochs, sampling_rate_hz=100.0, low_hz=8.0, high_hz=30.0)

    assert filtered.shape == epochs.shape
    # The 1 Hz drift sits well below the 8 Hz lower edge, so the band-passed signal
    # tracks the 15 Hz component far more closely than the unfiltered mixture does.
    filtered_error = float(np.std(filtered[0, 0] - in_band))
    unfiltered_error = float(np.std(epochs[0, 0] - in_band))
    assert filtered_error < unfiltered_error


def test_resolve_cv_splits_rejects_fewer_than_two_splits():
    with pytest.raises(ValueError, match="cv_splits must be at least 2"):
        resolve_cv_splits(1, (8, 8))


def test_resolve_cv_splits_rejects_class_with_single_epoch():
    with pytest.raises(ValueError, match="at least two epochs"):
        resolve_cv_splits(5, (8, 1))


def test_validate_csp_components_rejects_fewer_than_one():
    with pytest.raises(ValueError, match="csp_components must be at least 1"):
        _validate_csp_components(0, channel_count=4)


def test_validate_csp_components_rejects_more_than_channels():
    with pytest.raises(ValueError, match="must not exceed channel count"):
        _validate_csp_components(5, channel_count=4)


def test_validate_artifact_epoch_shape_rejects_wrong_dimensionality():
    artifact = make_artifact(object())

    with pytest.raises(ValueError, match=r"shape \(n_epochs, n_channels, n_samples\)"):
        _validate_artifact_epoch_shape(artifact, np.zeros((4, 8)))


def test_validate_artifact_epoch_shape_rejects_channel_mismatch():
    artifact = make_artifact(object())

    with pytest.raises(ValueError, match="channel count does not match artifact"):
        _validate_artifact_epoch_shape(artifact, np.zeros((1, 3, 8)))
