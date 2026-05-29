"""Command-line entry point for offline hand-classifier evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from eeg_bci_pipeline.data.bciciv2a_dataset import read_bciciv2a_epochs
from eeg_bci_pipeline.training.hand_classifier import (
    DEFAULT_BANDPASS_HIGH_HZ,
    DEFAULT_BANDPASS_LOW_HZ,
    DEFAULT_CSP_COMPONENTS,
    DEFAULT_CV_RANDOM_STATE,
    DEFAULT_CV_SPLITS,
    DEFAULT_HAND_CLASS_LABELS,
    evaluate_hand_classifier,
    format_evaluation_report,
)

DEFAULT_EPOCH_TMIN_SEC = 0.5
DEFAULT_EPOCH_TMAX_SEC = 3.5


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    bandpass_low_hz = None if args.no_bandpass else args.bandpass_low_hz
    bandpass_high_hz = None if args.no_bandpass else args.bandpass_high_hz
    class_labels = tuple(args.class_labels)
    epochs = read_bciciv2a_epochs(
        args.gdf_path,
        tmin_sec=args.tmin_sec,
        tmax_sec=args.tmax_sec,
        class_labels=class_labels,
    )

    if args.classifier == "eegnet":
        try:
            from eeg_bci_pipeline.training.eegnet_classifier import evaluate_eegnet_classifier
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1

        evaluation = evaluate_eegnet_classifier(
            epochs,
            class_labels=class_labels,
            cv_splits=args.cv_splits,
            cv_random_state=args.cv_random_state,
            bandpass_low_hz=bandpass_low_hz,
            bandpass_high_hz=bandpass_high_hz,
            f1=args.f1,
            d=args.depth_multiplier,
            kernel_length=args.kernel_length,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            n_epochs=args.training_epochs,
            batch_size=args.batch_size,
        )
    else:
        evaluation = evaluate_hand_classifier(
            epochs,
            class_labels=class_labels,
            csp_components=args.csp_components,
            cv_splits=args.cv_splits,
            cv_random_state=args.cv_random_state,
            bandpass_low_hz=bandpass_low_hz,
            bandpass_high_hz=bandpass_high_hz,
        )

    print(format_evaluation_report(evaluation))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an offline hand classifier on BCIC IV 2a data."
    )
    parser.add_argument(
        "gdf_path",
        type=Path,
        help="Path to a BCIC IV 2a GDF recording, for example data/raw/bciciv2a/A01T.gdf.",
    )
    parser.add_argument(
        "--classifier",
        choices=["csp-lda", "eegnet"],
        default="csp-lda",
        help="Classifier to evaluate. Default: csp-lda.",
    )
    parser.add_argument("--tmin-sec", type=float, default=DEFAULT_EPOCH_TMIN_SEC)
    parser.add_argument("--tmax-sec", type=float, default=DEFAULT_EPOCH_TMAX_SEC)
    parser.add_argument(
        "--class-labels",
        nargs="+",
        default=list(DEFAULT_HAND_CLASS_LABELS),
        help="Class labels to train/evaluate. Defaults to left_hand right_hand.",
    )
    parser.add_argument("--csp-components", type=int, default=DEFAULT_CSP_COMPONENTS)
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    parser.add_argument("--cv-random-state", type=int, default=DEFAULT_CV_RANDOM_STATE)
    parser.add_argument("--bandpass-low-hz", type=float, default=DEFAULT_BANDPASS_LOW_HZ)
    parser.add_argument("--bandpass-high-hz", type=float, default=DEFAULT_BANDPASS_HIGH_HZ)
    parser.add_argument(
        "--no-bandpass",
        action="store_true",
        help="Disable the default bandpass filter.",
    )

    eegnet = parser.add_argument_group("eegnet", "EEGNet-specific options")
    eegnet.add_argument("--f1", type=int, default=8)
    eegnet.add_argument("--depth-multiplier", type=int, default=2)
    eegnet.add_argument("--kernel-length", type=int, default=125)
    eegnet.add_argument("--dropout-rate", type=float, default=0.5)
    eegnet.add_argument("--learning-rate", type=float, default=1e-3)
    eegnet.add_argument("--weight-decay", type=float, default=1e-2)
    eegnet.add_argument("--training-epochs", type=int, default=200)
    eegnet.add_argument("--batch-size", type=int, default=32)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
