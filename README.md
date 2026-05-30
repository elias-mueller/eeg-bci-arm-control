# EEG-BCI Manipulator Control

Real-time motor-imagery EEG decoding driving a simulated Franka Panda manipulator over ROS 2. It runs as a no-device ROS graph: synthetic EEG frames are decoded into mock intents and consumed by a robot-control node.

## Quick Start

Requires ROS 2 Jazzy. The devcontainer installs it along with RViz and the ROS dev tools; otherwise [install it for Ubuntu](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) first.

```bash
scripts/build           # colcon build --symlink-install
scripts/run-robot-rviz  # mock pipeline + Panda model in RViz
scripts/test            # colcon test + verbose results
```

## Layout

Three ROS 2 packages under `src/`:

- `eeg_bci_interfaces` — shared `EegFrame` and `Intent` messages
- `eeg_bci_pipeline` — Python (`rclpy`) mock EEG publisher, baseline decoder, RViz intent marker, and launch files
- `manipulator_control` — C++ (`rclcpp`) intent logger and intent-driven joint-state driver, plus the Panda visualization URDF

## Scripts

`scripts/` wraps the common workflows — building, testing, running each launch, and linting. Each is a short, self-documenting wrapper, so read its header for what it does; launch and test scripts need a prior `scripts/build`.

The pure decoder tests can also run without ROS: `PYTHONPATH=src/eeg_bci_pipeline pytest src/eeg_bci_pipeline/test`.

## Replaying with a trained model

`run-bciciv2a-model` swaps the mock decoder for a real CSP + LDA classifier:

```bash
scripts/evaluate-hand-classifier data/raw/bciciv2a/A01T.gdf --save-model tmp/hand-csp-lda-A01T.joblib
scripts/run-bciciv2a-model gdf_path:=data/raw/bciciv2a/A01T.gdf model_path:=tmp/hand-csp-lda-A01T.joblib
```

The `model_intent_decoder` node takes its frame contract (channels, sampling rate, window length) from the artifact and rejects non-matching frames. It publishes `rest` until one ~3 s window has buffered, then gates any window below `rest_confidence_threshold` (default `0.6`) to `rest`.

Caveats: per-frame decoding with no temporal smoothing; `loop:=true` contaminates ~one window per wrap at the stitched file boundary; and the 22-channel 2a artifact is a benchmark, not the 16-channel BrainAccess rig. Train and replay different sessions (`A01T` then `A01E`) for an honest check, and keep the driver's `confidence_threshold` ≤ `rest_confidence_threshold`.

## Mock robot behavior

The mock robot launch maps decoded intents to `panda_joint2` — `rest` holds, `left_hand` moves negative, `right_hand` moves positive — while the other joints publish zero so `robot_state_publisher` keeps the full TF tree. `intent_joint_state_driver` ignores intents below `confidence_threshold` (default `0.55`), holds after `intent_timeout_sec` (default `0.3`) without messages, and drives `driven_joint_name` (default `panda_joint2`).
