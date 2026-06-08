"""Map and classify BrainAccess electrode-contact-quality channels.

A BrainAccess LSL outlet carries, alongside each EEG channel, a paired
``type="contact"`` channel named ``contact_<eeg_label>`` reporting that
electrode's contact quality. The LSL bridge forwards only the EEG channels, so
this module turns the contact channels into a per-EEG-channel quality read the
calibration flow can warn on, catching a poorly-seated electrode (e.g. F3)
*before* it is baked into a trained model rather than after.

Pure label/value logic: no pylsl, ROS, or NumPy, so it is unit-testable in
isolation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

CONTACT_CHANNEL_TYPE = "contact"
CONTACT_LABEL_PREFIX = "contact_"


def pair_contact_to_eeg(
    channel_labels: Sequence[str],
    channel_types: Sequence[str],
    *,
    eeg_type: str = "EEG",
    contact_type: str = CONTACT_CHANNEL_TYPE,
    contact_prefix: str = CONTACT_LABEL_PREFIX,
) -> dict[str, str]:
    """Map each EEG channel label to its contact-quality channel label.

    Pairs a ``type=eeg_type`` channel ``"C3"`` to a ``type=contact_type`` channel
    ``"contact_C3"`` by the BrainAccess ``contact_<label>`` convention. Only EEG
    channels that have a matching contact channel present are included (in EEG
    order); an EEG channel with no contact partner is omitted, and a contact
    channel with no EEG partner is ignored. ``channel_labels`` and ``channel_types``
    are positionally aligned and must be the same length. Types are matched
    case-insensitively (stripped), consistent with the rest of the bridge.
    """

    if len(channel_labels) != len(channel_types):
        raise ValueError("channel_labels and channel_types must be the same length")
    eeg_target = eeg_type.strip().lower()
    contact_target = contact_type.strip().lower()
    contact_labels = {
        label
        for label, ctype in zip(channel_labels, channel_types)
        if ctype.strip().lower() == contact_target
    }
    pairing: dict[str, str] = {}
    for label, ctype in zip(channel_labels, channel_types):
        if ctype.strip().lower() != eeg_target:
            continue
        contact_label = f"{contact_prefix}{label}"
        if contact_label in contact_labels:
            pairing[label] = contact_label
    return pairing


def poor_contact_channels(
    eeg_to_contact: Mapping[str, str],
    contact_values: Mapping[str, float],
    *,
    good_threshold: float,
    higher_is_better: bool = True,
) -> list[str]:
    """Return the EEG labels whose paired contact value signals poor contact.

    ``contact_values`` maps a contact channel label to its current value. A channel
    is poor when its value is on the wrong side of ``good_threshold``: strictly below
    it when ``higher_is_better`` (the default: a larger value means firmer contact),
    or strictly above it otherwise. An EEG channel whose contact value is missing
    from ``contact_values`` is treated as poor (absence is not evidence of good
    contact). Order follows ``eeg_to_contact`` iteration order.

    ``good_threshold`` and ``higher_is_better`` are hardware constants: they depend
    on what the BrainAccess contact channel actually reports, which must be confirmed
    against a seated-vs-lifted electrode spike before the gate is trusted. They are
    parameters here rather than baked-in guesses.
    """

    poor: list[str] = []
    for eeg_label, contact_label in eeg_to_contact.items():
        if contact_label not in contact_values:
            poor.append(eeg_label)
            continue
        value = contact_values[contact_label]
        is_poor = value < good_threshold if higher_is_better else value > good_threshold
        if is_poor:
            poor.append(eeg_label)
    return poor
