import numpy as np
import pytest
from eeg_bci_pipeline.data.gdf_recording import (
    VOLTS_TO_MICROVOLTS,
    iter_replay_frames,
    normalize_channel_labels,
    recording_from_mne_raw,
    replay_elapsed_sec,
)


class FakeRaw:
    ch_names = ["C3", "Cz", "EOG-left"]
    info = {"sfreq": 250.0}

    def __init__(self):
        self._data = {
            "C3": np.array([1e-6, 2e-6, 3e-6]),
            "Cz": np.array([4e-6, 5e-6, 6e-6]),
            "EOG-left": np.array([100e-6, 200e-6, 300e-6]),
        }

    def get_channel_types(self):
        return ["eeg", "eeg", "eog"]

    def get_data(self, picks):
        return np.vstack([self._data[pick] for pick in picks])


def test_recording_from_mne_raw_selects_eeg_channels_and_converts_units():
    recording = recording_from_mne_raw(FakeRaw(), source_id="A01T")

    assert recording.source_id == "A01T"
    assert recording.sampling_rate_hz == pytest.approx(250.0)
    assert recording.channel_labels == ("C3", "Cz")
    assert recording.samples_uv[0, 0] == pytest.approx(1e-6 * VOLTS_TO_MICROVOLTS)
    assert recording.samples_uv[1, 2] == pytest.approx(6e-6 * VOLTS_TO_MICROVOLTS)


def test_recording_from_mne_raw_excludes_eog_named_channels_even_if_typed_eeg():
    raw = FakeRaw()
    raw.get_channel_types = lambda: ["eeg", "eeg", "eeg"]

    recording = recording_from_mne_raw(raw, source_id="A01T")

    assert recording.channel_labels == ("C3", "Cz")


def test_recording_from_mne_raw_honors_explicit_channel_order():
    recording = recording_from_mne_raw(
        FakeRaw(),
        source_id="A01T",
        channel_labels=("Cz", "C3"),
    )

    assert recording.channel_labels == ("Cz", "C3")
    assert recording.samples_uv[0, 0] == pytest.approx(4.0)
    assert recording.samples_uv[1, 0] == pytest.approx(1.0)


def test_recording_from_mne_raw_requires_eeg_typed_channels_without_explicit_labels():
    raw = FakeRaw()
    raw.get_channel_types = lambda: ["misc", "stim", "eog"]

    with pytest.raises(ValueError, match="no channels typed as EEG"):
        recording_from_mne_raw(raw, source_id="A01T")


def test_normalize_channel_labels_rejects_scalar_strings_and_empty_labels():
    with pytest.raises(ValueError, match="single string"):
        normalize_channel_labels("C3")

    with pytest.raises(ValueError, match="empty"):
        normalize_channel_labels(("C3", " "))


def test_iter_replay_frames_flattens_channel_major_chunks():
    recording = recording_from_mne_raw(FakeRaw(), source_id="A01T")

    frames = list(iter_replay_frames(recording, samples_per_frame=2))

    assert frames[0].start_sample_index == 0
    assert frames[0].samples == pytest.approx([1.0, 2.0, 4.0, 5.0])
    assert frames[1].start_sample_index == 2
    assert frames[1].samples == pytest.approx([3.0, 6.0])


def test_replay_elapsed_sec_keeps_looped_replay_time_monotonic():
    assert replay_elapsed_sec(
        0,
        loop_index=1,
        samples_per_channel=100,
        sampling_rate_hz=10.0,
    ) == pytest.approx(10.0)
    assert replay_elapsed_sec(
        20,
        loop_index=2,
        samples_per_channel=100,
        sampling_rate_hz=10.0,
    ) == pytest.approx(22.0)
