# EEG-BCI Manipulator Control

Real-time motor-imagery EEG decoding driving a simulated Franka Panda manipulator over ROS 2, a closed loop from EEG frame to joint motion. A CSP + LDA classifier turns EEG windows into movement intents, and an EEGNet (PyTorch) decoder is benchmarked offline against the same baseline. The system is a C++/Python ROS 2 graph: a Python (`rclpy`) EEG and decoding pipeline feeding C++ (`rclcpp`) control nodes through shared `EegFrame`/`Intent` messages.

## Quick Start

Requires ROS 2 Jazzy with RViz: [install it for Ubuntu](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) first. `scripts/setup-dev-tools` installs the Python lint and type-check tooling (ruff, basedpyright).

```bash
scripts/build           # colcon build --symlink-install
scripts/run-robot-rviz  # Panda in RViz: real CSP+LDA decode of BCIC replay if present, else mock
scripts/test            # colcon test + verbose results
```

`run-robot-rviz` prefers real decoding: when the BCIC IV 2a dataset is present it
trains the CSP+LDA artifact if needed and drives the Panda from held-out replay,
and falls back to the synthetic mock decoder only when the dataset is absent.

## Layout

Three ROS 2 packages under `src/`:

- `eeg_bci_interfaces`: shared `EegFrame` and `Intent` messages
- `eeg_bci_pipeline`: Python (`rclpy`) mock EEG publisher, GDF replay, mock and model-backed intent decoders, RViz intent marker, and launch files, plus offline CSP+LDA / EEGNet training under `training/`
- `manipulator_control`: C++ (`rclcpp`) intent logger and intent-driven joint-state driver, plus the Panda visualization URDF

## Scripts

`scripts/` wraps the common workflows: building, testing, running each launch, and linting. Each is a short, self-documenting wrapper, so read its header for what it does; launch and test scripts need a prior `scripts/build`.

The pure decoder tests can also run without ROS: `PYTHONPATH=src/eeg_bci_pipeline pytest src/eeg_bci_pipeline/test`.

## Replaying with a trained model

`run-bciciv2a-model` swaps the mock decoder for a real CSP + LDA classifier (this
is the path `run-robot-rviz` auto-selects when the dataset is present):

```bash
# Train CSP+LDA on the A01T session and save the artifact.
scripts/evaluate-hand-classifier data/raw/bciciv2a/A01T.gdf --save-model tmp/hand-csp-lda-A01T.joblib

# Replay the held-out A01E session, not the A01T session the model was trained on.
scripts/run-bciciv2a-model gdf_path:=data/raw/bciciv2a/A01E.gdf model_path:=tmp/hand-csp-lda-A01T.joblib

# Same decode path, but drives the Panda in RViz. Defaults to the A01E recording
# and the artifact above; override with gdf_path:= / model_path:=.
scripts/run-bciciv2a-model-rviz
```

The `model_intent_decoder` node takes its frame contract (channels, sampling rate, window length) from the artifact and rejects non-matching frames. It publishes `rest` until one ~3 s window has buffered, then gates any window below `rest_confidence_threshold` (default `0.6`) to `rest`.

The replay drives the arm but does not score predictions against ground truth; for cross-validated accuracy, run `scripts/evaluate-hand-classifier`.

## Mock robot behavior

The mock robot launch maps decoded intents to `panda_joint2` (`rest` holds, `left_hand` moves negative, `right_hand` moves positive) while the other joints publish zero so `robot_state_publisher` keeps the full TF tree. `intent_joint_state_driver` ignores intents below `confidence_threshold` (default `0.55`), holds after `intent_timeout_sec` (default `0.3`) without messages, and drives `driven_joint_name` (default `panda_joint2`).
