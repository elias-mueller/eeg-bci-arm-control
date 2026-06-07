import numpy as np
import pytest
from eeg_bci_pipeline.data.bciciv2a_dataset import (
    extract_bciciv2a_epochs,
    load_labeled_epochs,
)


class FakeAnnotations:
    # Instance-level so each FakeRaw gets its own annotations: several tests mutate
    # raw.annotations.description/onset, and a shared class-level instance would
    # leak those edits across tests (passing only by collection order).
    def __init__(self):
        self.onset = [1.0, 2.0, 3.0, 4.0]
        self.description = ["769", "770", "771", "999"]


class FakeRaw:
    ch_names = ["C3", "Cz", "EOG-left"]
    info = {"sfreq": 10.0}

    def __init__(self):
        self.annotations = FakeAnnotations()
        self._data = {
            "C3": np.arange(100, dtype=float) * 1e-6,
            "Cz": (np.arange(100, dtype=float) + 100.0) * 1e-6,
            "EOG-left": np.arange(100, dtype=float) * 1e-3,
        }

    def get_channel_types(self):
        return ["eeg", "eeg", "eog"]

    def get_data(self, picks):
        return np.vstack([self._data[pick] for pick in picks])


def test_extract_bciciv2a_epochs_uses_cue_labels_and_microvolt_epochs():
    epochs = extract_bciciv2a_epochs(
        FakeRaw(),
        source_id="A01T",
        tmin_sec=0.0,
        tmax_sec=1.0,
    )

    assert epochs.source_id == "A01T"
    assert epochs.class_labels == ("left_hand", "right_hand", "feet", "tongue")
    assert epochs.labels == ("left_hand", "right_hand", "feet")
    assert epochs.channel_labels == ("C3", "Cz")
    assert epochs.epochs_uv.shape == (3, 2, 10)
    assert epochs.start_sample_indices == (10, 20, 30)
    assert epochs.epochs_uv[0, 0, 0] == pytest.approx(10.0)
    assert epochs.epochs_uv[0, 1, 0] == pytest.approx(110.0)


def test_extract_bciciv2a_epochs_can_filter_left_right():
    epochs = extract_bciciv2a_epochs(
        FakeRaw(),
        source_id="A01T",
        tmin_sec=0.0,
        tmax_sec=1.0,
        class_labels=("left_hand", "right_hand"),
    )

    assert epochs.class_labels == ("left_hand", "right_hand")
    assert epochs.labels == ("left_hand", "right_hand")


def test_extract_bciciv2a_epochs_accepts_bytes_annotation_descriptions():
    raw = FakeRaw()
    raw.annotations.description = [b"769", b"770", b"999"]
    raw.annotations.onset = [1.0, 2.0, 3.0]

    epochs = extract_bciciv2a_epochs(
        raw,
        source_id="A01T",
        tmin_sec=0.0,
        tmax_sec=1.0,
    )

    assert epochs.labels == ("left_hand", "right_hand")


def test_extract_bciciv2a_epochs_counts_skipped_boundary_epochs():
    raw = FakeRaw()
    raw.annotations.onset = [0.0, 1.0, 9.5]
    raw.annotations.description = ["769", "770", "771"]

    epochs = extract_bciciv2a_epochs(
        raw,
        source_id="A01T",
        tmin_sec=-0.1,
        tmax_sec=1.0,
    )

    assert epochs.labels == ("right_hand",)
    assert epochs.skipped_epoch_count == 2


def test_extract_bciciv2a_epochs_rejects_invalid_windows_and_labels():
    with pytest.raises(ValueError, match="tmax_sec"):
        extract_bciciv2a_epochs(FakeRaw(), source_id="A01T", tmin_sec=1.0, tmax_sec=1.0)

    with pytest.raises(ValueError, match="unknown"):
        extract_bciciv2a_epochs(
            FakeRaw(),
            source_id="A01T",
            class_labels=("left_hand", "blink"),
        )

    with pytest.raises(ValueError, match="unique"):
        extract_bciciv2a_epochs(
            FakeRaw(),
            source_id="A01T",
            class_labels=("left_hand", "left_hand"),
        )


def test_extract_bciciv2a_epochs_rejects_sub_sample_window():
    # sfreq=10 Hz: a 0.04 s window rounds to 0 samples, leaving no room for data.
    with pytest.raises(ValueError, match="at least one sample"):
        extract_bciciv2a_epochs(
            FakeRaw(),
            source_id="A01T",
            tmin_sec=0.0,
            tmax_sec=0.04,
        )


def test_extract_bciciv2a_epochs_raises_when_no_cue_epochs_found():
    raw = FakeRaw()
    # No description maps to a BCIC IV 2a cue label, so nothing is kept.
    raw.annotations.onset = [1.0, 2.0, 3.0]
    raw.annotations.description = ["999", "1023", "768"]

    with pytest.raises(ValueError, match="no BCIC IV 2a cue epochs found"):
        extract_bciciv2a_epochs(
            raw,
            source_id="A01T",
            tmin_sec=0.0,
            tmax_sec=1.0,
        )


def test_extract_bciciv2a_epochs_skips_cues_outside_selected_classes():
    raw = FakeRaw()
    # Real cues exist, but the class filter excludes both, leaving zero epochs.
    raw.annotations.onset = [1.0, 2.0]
    raw.annotations.description = ["771", "772"]

    with pytest.raises(ValueError, match="no BCIC IV 2a cue epochs found"):
        extract_bciciv2a_epochs(
            raw,
            source_id="A01T",
            tmin_sec=0.0,
            tmax_sec=1.0,
            class_labels=("left_hand", "right_hand"),
        )


def test_load_labeled_epochs_rejects_corrupted_joblib_file(tmp_path):
    corrupted = tmp_path / "x.joblib"
    corrupted.write_bytes(b"not a real joblib payload \x00\x01\x02")

    with pytest.raises(ValueError, match="could not read"):
        load_labeled_epochs(corrupted)
