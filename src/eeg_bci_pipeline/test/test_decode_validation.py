"""Hardware-free validation that the calibration-capture format decodes above chance.

The calibration node (`calibrate_capture`) records cue-labeled epochs into a
`LabeledEpochs` via `assemble_labeled_epochs`, and `evaluate-hand-classifier`
trains/cross-validates a CSP+LDA model on it. The only previously available
hardware-free check (replaying a looped GDF over LSL while the cue schedule runs
on its own clock) is at-chance *by construction*: the assigned cue label is not
time-aligned to the streamed brain activity, so it validates plumbing only.

These tests close that gap without a headset by feeding the capture format data
whose label genuinely matches its content:

* a deterministic, class-separable synthetic signal that a correct CSP+LDA
  pipeline must score well above chance (and a broken one, e.g. mislabeled or
  mis-sliced, would not), and
* real BCIC IV 2a motor imagery re-assembled through the same capture container,
  skipped when the gitignored dataset is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from eeg_bci_pipeline.calibration import assemble_labeled_epochs

SEPARABLE_RATE_HZ = 250.0
# Eight sensorimotor-style channels; C3 (index 1) and C4 (index 5) carry the
# class-discriminative rhythm, the rest are common background.
SEPARABLE_CHANNELS = ("C5", "C3", "C1", "Cz", "C2", "C4", "C6", "Pz")
SEPARABLE_SAMPLES = 750  # 3.0 s at 250 Hz: the calibration epoch length
LEFT_MOTOR_INDEX = 1
RIGHT_MOTOR_INDEX = 5


def _make_separable_records(n_per_class: int, *, seed: int):
    """Build ``(label, (channels, samples))`` records CSP+LDA can separate by design.

    Two motor channels carry an in-band (10-22 Hz) rhythm whose amplitude swaps
    between the classes over a common broadband background. That spatial-variance
    contrast is exactly what CSP extracts and LDA classifies, so a correct
    pipeline scores well above chance while a mislabeled or mis-shaped one cannot.
    Deterministic given ``seed``.
    """

    rng = np.random.default_rng(seed)
    times = np.arange(SEPARABLE_SAMPLES) / SEPARABLE_RATE_HZ
    freqs = (10.0, 15.0, 22.0)
    channels = len(SEPARABLE_CHANNELS)
    records: list[tuple[str, np.ndarray]] = []
    for class_index, label in enumerate(("left_hand", "right_hand")):
        strong, weak = (6.0, 1.0) if class_index == 0 else (1.0, 6.0)
        for _ in range(n_per_class):
            epoch = rng.standard_normal((channels, SEPARABLE_SAMPLES))
            for freq in freqs:
                wave = np.sin(2.0 * np.pi * freq * times + rng.uniform(0.0, 2.0 * np.pi))
                epoch[LEFT_MOTOR_INDEX] += strong * wave
                epoch[RIGHT_MOTOR_INDEX] += weak * wave
            records.append((label, epoch * 5.0))
    return records


def test_separable_capture_format_trains_above_chance():
    # The capture container (assemble_labeled_epochs) feeding CSP+LDA must decode
    # cue-aligned, class-separable data well above the 0.5 chance line. This is the
    # hardware-free, above-chance-by-construction counterpart to the at-chance
    # GDF-over-LSL demo.
    from eeg_bci_pipeline.training.hand_classifier import evaluate_hand_classifier

    records = _make_separable_records(24, seed=11)
    epochs = assemble_labeled_epochs(
        source_id="separable",
        sampling_rate_hz=SEPARABLE_RATE_HZ,
        channel_labels=SEPARABLE_CHANNELS,
        class_labels=("left_hand", "right_hand"),
        records=records,
    )

    evaluation = evaluate_hand_classifier(epochs, cv_splits=4)

    assert evaluation.mean_accuracy > 0.85


def test_shuffled_labels_collapse_to_chance():
    # Control: same signals, labels decorrelated from content. Confirms the
    # above-chance result comes from the label-content alignment, not a leak in the
    # pipeline. The scramble is a seeded permutation of the *actual* labels (still
    # balanced), so it holds regardless of the order records are emitted in.
    from eeg_bci_pipeline.training.hand_classifier import evaluate_hand_classifier

    records = _make_separable_records(24, seed=11)
    windows = [window for _, window in records]
    true_labels = [label for label, _ in records]
    permutation = np.random.default_rng(99).permutation(len(true_labels))
    shuffled_labels = [true_labels[index] for index in permutation]
    shuffled = list(zip(shuffled_labels, windows))
    epochs = assemble_labeled_epochs(
        source_id="scrambled",
        sampling_rate_hz=SEPARABLE_RATE_HZ,
        channel_labels=SEPARABLE_CHANNELS,
        class_labels=("left_hand", "right_hand"),
        records=shuffled,
    )

    evaluation = evaluate_hand_classifier(epochs, cv_splits=4)

    assert evaluation.mean_accuracy < 0.7


def _find_bcic_gdf(name: str) -> Path | None:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "data" / "raw" / "bciciv2a" / name
        if candidate.is_file():
            return candidate
    return None


def test_real_bcic_capture_format_decodes_above_chance():
    # Real left/right motor imagery re-routed through the calibration capture
    # container, proving the *deployable* path decodes real EEG above chance
    # without a headset. Skipped when the gitignored dataset (or MNE) is absent.
    gdf_path = _find_bcic_gdf("A01T.gdf")
    if gdf_path is None:
        pytest.skip("BCIC IV 2a A01T.gdf not present (gitignored dataset)")
    pytest.importorskip("mne")

    from eeg_bci_pipeline.data.bciciv2a_dataset import read_bciciv2a_epochs
    from eeg_bci_pipeline.training.hand_classifier import evaluate_hand_classifier

    bcic = read_bciciv2a_epochs(
        gdf_path,
        tmin_sec=0.5,
        tmax_sec=3.5,
        class_labels=("left_hand", "right_hand"),
    )
    # Rebuild through assemble_labeled_epochs so the data takes the same shape and
    # path a live capture session produces, then train/evaluate exactly as the CLI
    # would on a captured file.
    records = [
        (label, np.asarray(epoch, dtype=np.float64))
        for label, epoch in zip(bcic.labels, bcic.epochs_uv)
    ]
    captured = assemble_labeled_epochs(
        source_id="A01T-as-capture",
        sampling_rate_hz=bcic.sampling_rate_hz,
        channel_labels=bcic.channel_labels,
        class_labels=("left_hand", "right_hand"),
        records=records,
    )

    evaluation = evaluate_hand_classifier(captured, cv_splits=3)

    # A01 is a strong subject (offline benchmark mean ~0.75-0.85). A conservative
    # floor keeps this from flaking while still proving real MI decodes through the
    # capture format, not just synthetic separability.
    assert evaluation.mean_accuracy > 0.6
