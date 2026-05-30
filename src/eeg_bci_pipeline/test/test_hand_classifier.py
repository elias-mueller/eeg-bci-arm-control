import numpy as np
import pytest
from eeg_bci_pipeline.data.bciciv2a_dataset import LabeledEpochs
from eeg_bci_pipeline.training.hand_classifier import (
    HandClassifierArtifact,
    HandClassifierEvaluation,
    evaluate_hand_classifier,
    format_evaluation_report,
    load_hand_classifier_artifact,
    predict_hand_classifier,
    predict_hand_proba,
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
