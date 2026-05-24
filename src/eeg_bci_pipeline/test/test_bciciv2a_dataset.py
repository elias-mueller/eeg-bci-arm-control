import numpy as np
import pytest

from eeg_bci_pipeline.bciciv2a_dataset import extract_bciciv2a_epochs


class FakeAnnotations:
    onset = [1.0, 2.0, 3.0, 4.0]
    description = ["769", "770", "771", "999"]


class FakeRaw:
    ch_names = ["C3", "Cz", "EOG-left"]
    info = {"sfreq": 10.0}
    annotations = FakeAnnotations()

    def __init__(self):
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


def test_extract_bciciv2a_epochs_rejects_invalid_windows_and_labels():
    with pytest.raises(ValueError, match="tmax_sec"):
        extract_bciciv2a_epochs(FakeRaw(), source_id="A01T", tmin_sec=1.0, tmax_sec=1.0)

    with pytest.raises(ValueError, match="unknown"):
        extract_bciciv2a_epochs(
            FakeRaw(),
            source_id="A01T",
            class_labels=("left_hand", "blink"),
        )
