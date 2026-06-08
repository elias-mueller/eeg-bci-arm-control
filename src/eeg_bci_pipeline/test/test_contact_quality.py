"""TDD spec for the contact-quality helpers (tests written before the impl).

Pure label/value logic, so no ROS/pylsl: these run under plain pytest. The
numeric threshold and polarity in ``poor_contact_channels`` are deliberately
parameters, not constants, because the BrainAccess contact-value semantics still
need a seated-vs-lifted hardware spike to pin down; these tests fix the *logic*,
not those hardware constants.
"""

import pytest
from eeg_bci_pipeline.contact_quality import pair_contact_to_eeg, poor_contact_channels

# --- pair_contact_to_eeg -----------------------------------------------------


def test_pairs_eeg_channels_to_their_contact_channels():
    labels = ["C3", "Cz", "contact_C3", "contact_Cz", "Accel_x"]
    types = ["EEG", "EEG", "contact", "contact", "Accel"]

    assert pair_contact_to_eeg(labels, types) == {"C3": "contact_C3", "Cz": "contact_Cz"}


def test_pairing_preserves_eeg_order():
    labels = ["C4", "C3", "contact_C3", "contact_C4"]
    types = ["EEG", "EEG", "contact", "contact"]

    assert list(pair_contact_to_eeg(labels, types)) == ["C4", "C3"]


def test_eeg_channel_without_a_contact_partner_is_omitted():
    labels = ["C3", "C4", "contact_C3"]
    types = ["EEG", "EEG", "contact"]

    assert pair_contact_to_eeg(labels, types) == {"C3": "contact_C3"}


def test_contact_channel_without_an_eeg_partner_is_ignored():
    labels = ["C3", "contact_C3", "contact_Zz"]
    types = ["EEG", "contact", "contact"]

    assert pair_contact_to_eeg(labels, types) == {"C3": "contact_C3"}


def test_pairing_matches_types_case_insensitively():
    labels = ["C3", "contact_C3"]
    types = ["eeg", "CONTACT"]

    assert pair_contact_to_eeg(labels, types) == {"C3": "contact_C3"}


def test_pairing_rejects_mismatched_label_and_type_lengths():
    with pytest.raises(ValueError, match="same length"):
        pair_contact_to_eeg(["C3", "contact_C3"], ["EEG"])


def test_pairing_of_empty_stream_is_empty():
    assert pair_contact_to_eeg([], []) == {}


# --- poor_contact_channels ---------------------------------------------------


def test_flags_channels_below_threshold_when_higher_is_better():
    eeg_to_contact = {"C3": "contact_C3", "Cz": "contact_Cz"}
    contact_values = {"contact_C3": 1.0, "contact_Cz": 0.2}

    assert poor_contact_channels(eeg_to_contact, contact_values, good_threshold=0.5) == ["Cz"]


def test_flags_channels_above_threshold_when_lower_is_better():
    # e.g. an impedance-like value where a larger number is worse contact.
    eeg_to_contact = {"C3": "contact_C3", "Cz": "contact_Cz"}
    contact_values = {"contact_C3": 5.0, "contact_Cz": 50.0}

    poor = poor_contact_channels(
        eeg_to_contact, contact_values, good_threshold=10.0, higher_is_better=False
    )
    assert poor == ["Cz"]


def test_missing_contact_value_counts_as_poor():
    # Absence of a reading is not evidence of good contact, so gate it as poor.
    eeg_to_contact = {"C3": "contact_C3"}

    assert poor_contact_channels(eeg_to_contact, {}, good_threshold=0.5) == ["C3"]


def test_value_exactly_at_threshold_is_good():
    # Strict inequality: a value sitting exactly on the threshold is acceptable.
    eeg_to_contact = {"C3": "contact_C3"}
    contact_values = {"contact_C3": 0.5}

    assert poor_contact_channels(eeg_to_contact, contact_values, good_threshold=0.5) == []


def test_all_good_contacts_yields_no_poor_channels():
    eeg_to_contact = {"C3": "contact_C3", "Cz": "contact_Cz"}
    contact_values = {"contact_C3": 1.0, "contact_Cz": 0.9}

    assert poor_contact_channels(eeg_to_contact, contact_values, good_threshold=0.5) == []
