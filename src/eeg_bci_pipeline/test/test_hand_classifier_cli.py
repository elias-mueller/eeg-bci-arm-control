import sys
import types

import numpy as np
import pytest
from eeg_bci_pipeline.calibration import assemble_labeled_epochs
from eeg_bci_pipeline.data.bciciv2a_dataset import (
    LabeledEpochs,
    save_labeled_epochs,
)
from eeg_bci_pipeline.training import hand_classifier_cli
from eeg_bci_pipeline.training.hand_classifier import (
    HandClassifierEvaluation,
    load_hand_classifier_artifact,
)
from eeg_bci_pipeline.training.hand_classifier_cli import main


def _make_epochs(source_id, *, per_class, seed, channels=8, samples=750):
    """Build a real LabeledEpochs with `per_class` epochs in each of two classes."""
    rng = np.random.default_rng(seed)
    records = []
    for index in range(per_class * 2):
        label = "left_hand" if index % 2 == 0 else "right_hand"
        records.append((label, rng.standard_normal((channels, samples)) * 20.0))
    return assemble_labeled_epochs(
        source_id=source_id,
        sampling_rate_hz=250.0,
        channel_labels=tuple(f"EEG-{i:02d}" for i in range(channels)),
        class_labels=("left_hand", "right_hand"),
        records=records,
    )


def _save_epochs(tmp_path, name, epochs):
    path = tmp_path / name
    save_labeled_epochs(epochs, path)
    return path


def test_save_model_rejects_non_csp_classifier(tmp_path, capsys):
    # --save-model is only meaningful for csp-lda; pairing it with eegnet must
    # fail fast before any (expensive) training happens.
    epochs_path = _save_epochs(tmp_path, "cap.joblib", _make_epochs("a", per_class=4, seed=1))
    model_path = tmp_path / "model.joblib"

    exit_code = main([str(epochs_path), "--classifier", "eegnet", "--save-model", str(model_path)])

    assert exit_code == 1
    assert "save-model currently supports only --classifier csp-lda" in capsys.readouterr().err
    assert not model_path.exists()


def test_gdf_input_stamps_window_provenance(tmp_path, monkeypatch, capsys):
    # A .gdf path is epoched on read; the requested --tmin/--tmax window must be
    # recorded on the saved artifact as the epoch provenance.
    captured = {}

    def fake_read(gdf_path, *, tmin_sec, tmax_sec, class_labels):
        captured["gdf_path"] = gdf_path
        captured["tmin_sec"] = tmin_sec
        captured["tmax_sec"] = tmax_sec
        captured["class_labels"] = class_labels
        return _make_epochs("gdf", per_class=4, seed=2)

    monkeypatch.setattr(hand_classifier_cli, "read_bciciv2a_epochs", fake_read)
    gdf_path = tmp_path / "A01T.gdf"
    model_path = tmp_path / "model.joblib"

    exit_code = main(
        [
            str(gdf_path),
            "--tmin-sec",
            "0.5",
            "--tmax-sec",
            "3.5",
            "--save-model",
            str(model_path),
            "--cv-splits",
            "2",
        ]
    )

    assert exit_code == 0
    assert captured["tmin_sec"] == pytest.approx(0.5)
    assert captured["tmax_sec"] == pytest.approx(3.5)
    artifact = load_hand_classifier_artifact(model_path)
    assert artifact.epoch_tmin_sec == pytest.approx(0.5)
    assert artifact.epoch_tmax_sec == pytest.approx(3.5)
    assert "saved model" in capsys.readouterr().out


def test_eegnet_import_failure_returns_one(tmp_path, monkeypatch, capsys):
    # If importing the EEGNet evaluator raises RuntimeError (e.g. torch missing on
    # the live rig), the CLI must surface the message and exit 1, not crash.
    epochs_path = _save_epochs(tmp_path, "cap.joblib", _make_epochs("a", per_class=4, seed=3))

    class _Raiser(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("EEGNet classifier requires PyTorch")

    monkeypatch.setitem(
        sys.modules,
        "eeg_bci_pipeline.training.eegnet_classifier",
        _Raiser("eeg_bci_pipeline.training.eegnet_classifier"),
    )

    exit_code = main([str(epochs_path), "--classifier", "eegnet"])

    assert exit_code == 1
    assert "EEGNet classifier requires PyTorch" in capsys.readouterr().err


def test_eegnet_evaluation_path_reports_and_exits_zero(tmp_path, monkeypatch, capsys):
    # When the EEGNet evaluator is available, the CLI runs it and prints its report.
    # Patch the evaluator so the test stays fast and deterministic while still
    # exercising the real eegnet call site (and the no-save-model exit path).
    epochs = _make_epochs("a", per_class=4, seed=4)
    epochs_path = _save_epochs(tmp_path, "cap.joblib", epochs)

    seen = {}

    def fake_evaluate(labeled_epochs, *, class_labels, **kwargs):
        seen["epoch_count"] = labeled_epochs.epoch_count
        seen["class_labels"] = class_labels
        return HandClassifierEvaluation(
            source_id="a",
            sampling_rate_hz=250.0,
            channel_count=8,
            samples_per_epoch=750,
            class_labels=("left_hand", "right_hand"),
            class_counts=(4, 4),
            cv_splits=2,
            fold_scores=(0.5, 0.5),
            mean_accuracy=0.5,
            std_accuracy=0.0,
            classifier_name="eegnet",
        )

    pytest.importorskip("torch")
    import eeg_bci_pipeline.training.eegnet_classifier as eegnet_module

    monkeypatch.setattr(eegnet_module, "evaluate_eegnet_classifier", fake_evaluate)

    exit_code = main([str(epochs_path), "--classifier", "eegnet", "--cv-splits", "2"])

    assert exit_code == 0
    assert seen["epoch_count"] == 8
    assert seen["class_labels"] == ("left_hand", "right_hand")
    out = capsys.readouterr().out
    assert "eegnet" in out


def test_cv_failure_without_save_model_returns_one(tmp_path, capsys):
    # One epoch per class cannot be cross-validated. Without --save-model there is
    # nothing to fall back to, so the CLI reports the failure and exits 1.
    epochs = _make_epochs("tiny", per_class=1, seed=5)
    assert epochs.epoch_count == 2
    epochs_path = _save_epochs(tmp_path, "tiny.joblib", epochs)

    exit_code = main([str(epochs_path), "--cv-splits", "10"])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "evaluation failed" in err


def test_cv_failure_with_save_model_skips_cv_and_saves(tmp_path, capsys):
    # An imbalanced capture (2 left, 1 right) cannot be cross-validated because one
    # class has a single epoch, but the final LDA fit still has enough samples.
    # With --save-model the CLI must skip CV and still write the artifact, exiting 0.
    rng = np.random.default_rng(6)
    records = [
        ("left_hand", rng.standard_normal((8, 750)) * 20.0),
        ("right_hand", rng.standard_normal((8, 750)) * 20.0),
        ("left_hand", rng.standard_normal((8, 750)) * 20.0),
    ]
    epochs = assemble_labeled_epochs(
        source_id="tiny",
        sampling_rate_hz=250.0,
        channel_labels=tuple(f"EEG-{i:02d}" for i in range(8)),
        class_labels=("left_hand", "right_hand"),
        records=records,
    )
    epochs_path = _save_epochs(tmp_path, "tiny.joblib", epochs)
    model_path = tmp_path / "model.joblib"

    exit_code = main([str(epochs_path), "--cv-splits", "10", "--save-model", str(model_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "cross-validation skipped" in captured.err
    assert "saved model" in captured.out
    artifact = load_hand_classifier_artifact(model_path)
    assert artifact.channel_count == 8
    # Captured (non-.gdf) epochs carry no GDF window, so provenance is None.
    assert artifact.epoch_tmin_sec is None
    assert artifact.epoch_tmax_sec is None


def test_cli_trains_from_captured_epochs(tmp_path):
    # The central seam: calibration saves a LabeledEpochs joblib; the CLI loads it
    # (non-.gdf path), evaluates with CV, and trains a saveable artifact.
    epochs = _make_epochs("cli-test", per_class=4, seed=7)
    epochs_path = _save_epochs(tmp_path, "cap.joblib", epochs)
    model_path = tmp_path / "model.joblib"

    exit_code = main([str(epochs_path), "--save-model", str(model_path), "--cv-splits", "2"])

    assert exit_code == 0
    artifact = load_hand_classifier_artifact(model_path)
    assert isinstance(artifact.class_labels, tuple)
    assert artifact.channel_count == 8
    assert artifact.samples_per_epoch == 750
    assert artifact.epoch_tmin_sec is None
    assert artifact.epoch_tmax_sec is None


def test_cli_evaluate_only_loads_captured_epochs(tmp_path, capsys):
    # Without --save-model and with a CV-sized capture, the CLI just prints the
    # evaluation report and exits 0 (the loaded-epochs branch is a real LabeledEpochs).
    epochs = _make_epochs("eval-only", per_class=4, seed=8)
    assert isinstance(epochs, LabeledEpochs)
    epochs_path = _save_epochs(tmp_path, "cap.joblib", epochs)

    exit_code = main([str(epochs_path), "--cv-splits", "2"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "csp-lda" in out
