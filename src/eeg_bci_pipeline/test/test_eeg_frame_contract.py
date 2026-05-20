import pytest

from eeg_bci_pipeline.eeg_frame_contract import (
    DEFAULT_EEG_CHANNEL_COUNT,
    DEFAULT_EEG_SAMPLING_RATE_HZ,
    DEFAULT_MAX_ABS_SAMPLE_UV,
    EegFrameContractError,
    default_channel_labels,
    validate_eeg_frame_payload,
)
from eeg_bci_pipeline.mock_signal import generate_mock_eeg_samples


def test_valid_default_brainaccess_shape_is_accepted():
    labels = default_channel_labels()
    samples = [0.0] * (DEFAULT_EEG_CHANNEL_COUNT * 25)

    shape = validate_eeg_frame_payload(
        sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
        channel_labels=labels,
        samples=samples,
    )

    assert shape.channel_count == DEFAULT_EEG_CHANNEL_COUNT
    assert shape.samples_per_channel == 25
    assert shape.duration_sec == pytest.approx(0.1)


def test_mock_publisher_payload_shape_satisfies_contract():
    samples_per_frame = 25
    samples = generate_mock_eeg_samples(
        start_sample_index=0,
        samples_per_frame=samples_per_frame,
        channel_count=DEFAULT_EEG_CHANNEL_COUNT,
        sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
        amplitude_uv=25.0,
    )

    shape = validate_eeg_frame_payload(
        sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
        channel_labels=default_channel_labels(),
        samples=samples,
    )

    assert shape.samples_per_channel == samples_per_frame


def test_channel_count_is_configurable_for_smoke_rigs():
    shape = validate_eeg_frame_payload(
        sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
        channel_labels=("bench_1", "bench_2", "bench_3", "bench_4"),
        samples=[1.0] * 12,
        expected_channel_count=4,
    )

    assert shape.channel_count == 4
    assert shape.samples_per_channel == 3


def test_expected_labels_define_count_and_order():
    labels = ("c3", "cz", "c4")

    shape = validate_eeg_frame_payload(
        sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
        channel_labels=labels,
        samples=[1.0] * 6,
        expected_channel_labels=labels,
    )

    assert shape.channel_count == 3

    with pytest.raises(EegFrameContractError, match="stable and ordered"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=tuple(reversed(labels)),
            samples=[1.0] * 6,
            expected_channel_labels=labels,
        )

    with pytest.raises(EegFrameContractError, match="exactly 4"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=labels,
            samples=[1.0] * 6,
            expected_channel_labels=(*labels, "p4"),
        )


def test_sampling_rate_must_match_configured_rate():
    validate_eeg_frame_payload(
        sampling_rate_hz=249.75,
        channel_labels=default_channel_labels(),
        samples=[1.0] * DEFAULT_EEG_CHANNEL_COUNT,
    )

    with pytest.raises(EegFrameContractError, match="250 Hz"):
        validate_eeg_frame_payload(
            sampling_rate_hz=249.0,
            channel_labels=default_channel_labels(),
            samples=[1.0] * DEFAULT_EEG_CHANNEL_COUNT,
        )


def test_sampling_rate_must_be_positive():
    with pytest.raises(EegFrameContractError, match="greater than 0"):
        validate_eeg_frame_payload(
            sampling_rate_hz=0.0,
            channel_labels=default_channel_labels(),
            samples=[1.0] * DEFAULT_EEG_CHANNEL_COUNT,
        )


def test_channel_labels_must_be_complete_non_empty_and_unique():
    duplicate_labels = list(default_channel_labels())
    duplicate_labels[-1] = duplicate_labels[0]

    with pytest.raises(EegFrameContractError, match="unique"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=duplicate_labels,
            samples=[1.0] * DEFAULT_EEG_CHANNEL_COUNT,
        )

    with pytest.raises(EegFrameContractError, match="exactly 16"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=default_channel_labels(15),
            samples=[1.0] * 15,
        )


def test_samples_must_be_non_empty_divisible_and_finite():
    with pytest.raises(EegFrameContractError, match="non-empty"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=default_channel_labels(),
            samples=[],
        )

    with pytest.raises(EegFrameContractError, match="divisible"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=default_channel_labels(),
            samples=[1.0] * (DEFAULT_EEG_CHANNEL_COUNT + 1),
        )

    with pytest.raises(EegFrameContractError, match="samples\\[0\\]"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=default_channel_labels(),
            samples=[float("nan")] + [1.0] * (DEFAULT_EEG_CHANNEL_COUNT - 1),
        )

    with pytest.raises(EegFrameContractError, match="numeric and finite"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=default_channel_labels(),
            samples=["bad"] + [1.0] * (DEFAULT_EEG_CHANNEL_COUNT - 1),
        )


def test_samples_must_stay_inside_microvolt_sanity_bounds():
    validate_eeg_frame_payload(
        sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
        channel_labels=default_channel_labels(),
        samples=[DEFAULT_MAX_ABS_SAMPLE_UV] + [1.0] * (DEFAULT_EEG_CHANNEL_COUNT - 1),
    )

    with pytest.raises(EegFrameContractError, match="microvolts"):
        validate_eeg_frame_payload(
            sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
            channel_labels=default_channel_labels(),
            samples=[
                DEFAULT_MAX_ABS_SAMPLE_UV + 1.0,
                *([1.0] * (DEFAULT_EEG_CHANNEL_COUNT - 1)),
            ],
        )


def test_suspiciously_small_peak_flags_possible_volt_units():
    shape = validate_eeg_frame_payload(
        sampling_rate_hz=DEFAULT_EEG_SAMPLING_RATE_HZ,
        channel_labels=default_channel_labels(),
        samples=[50e-6] * DEFAULT_EEG_CHANNEL_COUNT,
    )

    assert shape.suspiciously_small_peak
