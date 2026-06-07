import pytest
from eeg_bci_pipeline.mock_signal import select_cycle_value


def test_empty_values_are_rejected():
    with pytest.raises(ValueError, match="at least one item"):
        select_cycle_value(values=[], frame_index=0, frames_per_value=1)


@pytest.mark.parametrize("frames_per_value", [0, -1])
def test_non_positive_frames_per_value_are_rejected(frames_per_value):
    with pytest.raises(ValueError, match="at least 1"):
        select_cycle_value(values=(1.0,), frame_index=0, frames_per_value=frames_per_value)
