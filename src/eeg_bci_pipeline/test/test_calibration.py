import numpy as np
import pytest
from eeg_bci_pipeline.calibration import (
    assemble_labeled_epochs,
    build_trial_schedule,
    extract_epoch,
    reshape_frame_to_channel_major,
)
from eeg_bci_pipeline.data.bciciv2a_dataset import (
    LabeledEpochs,
    load_labeled_epochs,
    save_labeled_epochs,
)


def test_build_trial_schedule_is_balanced_and_deterministic():
    schedule = build_trial_schedule(5, ["left_hand", "right_hand"], seed=0)

    assert len(schedule) == 10
    assert schedule.count("left_hand") == 5
    assert schedule.count("right_hand") == 5
    assert build_trial_schedule(5, ["left_hand", "right_hand"], seed=0) == schedule


def test_build_trial_schedule_varies_with_seed_but_stays_balanced():
    a = build_trial_schedule(8, ["left_hand", "right_hand"], seed=0)
    b = build_trial_schedule(8, ["left_hand", "right_hand"], seed=1)

    assert a != b
    assert a.count("left_hand") == b.count("left_hand") == 8


def test_build_trial_schedule_validates_inputs():
    with pytest.raises(ValueError, match="at least 1"):
        build_trial_schedule(0, ["left_hand", "right_hand"])
    with pytest.raises(ValueError, match="at least two"):
        build_trial_schedule(5, ["left_hand"])
    with pytest.raises(ValueError, match="unique"):
        build_trial_schedule(5, ["left_hand", "left_hand"])


def test_extract_epoch_slices_after_settle_offset():
    # 2 channels x 10 samples; channel c, sample s -> c*100 + s.
    buffer = np.array([[float(c * 100 + s) for s in range(10)] for c in range(2)])

    window = extract_epoch(buffer, settle_offset=2, samples_per_epoch=5)

    assert window is not None
    assert window.shape == (2, 5)
    np.testing.assert_array_equal(window[0], [2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(window[1], [102.0, 103.0, 104.0, 105.0, 106.0])


def test_extract_epoch_returns_none_on_dropout():
    assert extract_epoch(np.zeros((2, 4)), settle_offset=1, samples_per_epoch=5) is None


def test_extract_epoch_validates_inputs():
    buffer = np.zeros((2, 10))
    with pytest.raises(ValueError, match="non-negative"):
        extract_epoch(buffer, settle_offset=-1, samples_per_epoch=5)
    with pytest.raises(ValueError, match="at least 1"):
        extract_epoch(buffer, settle_offset=0, samples_per_epoch=0)
    with pytest.raises(ValueError, match="2-D"):
        extract_epoch(np.zeros(10), settle_offset=0, samples_per_epoch=5)


def test_assemble_labeled_epochs_stacks_and_labels():
    channels = ("C3", "Cz", "C4")
    records = [
        ("left_hand", np.ones((3, 4))),
        ("right_hand", np.full((3, 4), 2.0)),
        ("left_hand", np.full((3, 4), 3.0)),
    ]

    epochs = assemble_labeled_epochs(
        source_id="cal",
        sampling_rate_hz=250.0,
        channel_labels=channels,
        class_labels=("left_hand", "right_hand"),
        records=records,
        skipped_epoch_count=1,
    )

    assert epochs.epochs_uv.shape == (3, 3, 4)
    assert epochs.epochs_uv.dtype == np.float64
    assert epochs.labels == ("left_hand", "right_hand", "left_hand")
    assert epochs.channel_labels == channels
    assert epochs.class_labels == ("left_hand", "right_hand")
    assert epochs.sampling_rate_hz == pytest.approx(250.0)
    assert epochs.skipped_epoch_count == 1
    assert epochs.epoch_count == 3


def test_assemble_labeled_epochs_validates():
    with pytest.raises(ValueError, match="no epochs"):
        assemble_labeled_epochs(
            source_id="cal",
            sampling_rate_hz=250.0,
            channel_labels=("C3",),
            class_labels=("left_hand", "right_hand"),
            records=[],
        )
    with pytest.raises(ValueError, match="share"):
        assemble_labeled_epochs(
            source_id="cal",
            sampling_rate_hz=250.0,
            channel_labels=("C3", "C4"),
            class_labels=("left_hand", "right_hand"),
            records=[("left_hand", np.zeros((2, 4))), ("right_hand", np.zeros((2, 5)))],
        )
    with pytest.raises(ValueError, match="channels"):
        assemble_labeled_epochs(
            source_id="cal",
            sampling_rate_hz=250.0,
            channel_labels=("C3", "C4", "Cz"),
            class_labels=("left_hand", "right_hand"),
            records=[("left_hand", np.zeros((2, 4)))],
        )


def test_labeled_epochs_save_load_round_trip(tmp_path):
    epochs = assemble_labeled_epochs(
        source_id="cal",
        sampling_rate_hz=250.0,
        channel_labels=("C3", "Cz", "C4"),
        class_labels=("left_hand", "right_hand"),
        records=[("left_hand", np.ones((3, 4))), ("right_hand", np.full((3, 4), 2.0))],
    )
    path = tmp_path / "epochs.joblib"

    save_labeled_epochs(epochs, path)
    loaded = load_labeled_epochs(path)

    assert isinstance(loaded, LabeledEpochs)
    assert loaded.labels == epochs.labels
    assert loaded.channel_labels == epochs.channel_labels
    np.testing.assert_array_equal(loaded.epochs_uv, epochs.epochs_uv)


def test_load_labeled_epochs_rejects_wrong_type(tmp_path):
    import joblib

    path = tmp_path / "not-epochs.joblib"
    joblib.dump({"not": "epochs"}, path)

    with pytest.raises(ValueError, match="does not contain labeled epochs"):
        load_labeled_epochs(path)


def test_load_labeled_epochs_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_labeled_epochs(tmp_path / "nope.joblib")


def test_reshape_frame_to_channel_major():
    block = reshape_frame_to_channel_major([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)

    assert block is not None
    np.testing.assert_array_equal(block, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert reshape_frame_to_channel_major([], 2) is None
    assert reshape_frame_to_channel_major([1.0, 2.0, 3.0], 2) is None  # not divisible


def test_extract_epoch_exact_boundary():
    # settle_offset(2) + samples_per_epoch(5) == buffer width(7): just enough.
    window = extract_epoch(np.zeros((2, 7)), 2, 5)
    assert window is not None and window.shape == (2, 5)
    assert extract_epoch(np.zeros((2, 6)), 2, 5) is None  # one short


def test_build_trial_schedule_supports_more_than_two_classes():
    schedule = build_trial_schedule(3, ["left_hand", "right_hand", "rest"], seed=2)

    assert len(schedule) == 9
    for label in ("left_hand", "right_hand", "rest"):
        assert schedule.count(label) == 3


def test_reshape_frame_to_channel_major_rejects_nonpositive_channel_count():
    with pytest.raises(ValueError, match="channel_count must be at least 1"):
        reshape_frame_to_channel_major([1.0, 2.0], 0)
    with pytest.raises(ValueError, match="channel_count must be at least 1"):
        reshape_frame_to_channel_major([1.0, 2.0], -1)


def test_assemble_labeled_epochs_rejects_unknown_label():
    with pytest.raises(ValueError, match="not in class_labels"):
        assemble_labeled_epochs(
            source_id="cal",
            sampling_rate_hz=250.0,
            channel_labels=("C3", "C4"),
            class_labels=("left_hand", "right_hand"),
            records=[("feet", np.zeros((2, 4)))],
        )
