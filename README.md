# EEG-BCI Manipulator Control

Real-time motor-imagery EEG decoding driving a simulated Franka Panda manipulator over ROS 2.

The workspace can run a no-device ROS graph where synthetic EEG frames are decoded into mock intents and consumed by a robot-control boundary node.

## Quick Start

The devcontainer installs ROS 2 Jazzy, RViz, and the ROS development tools. Outside the devcontainer, install ROS 2 Jazzy for Ubuntu first: <https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>.

Build the workspace:

```bash
scripts/build
```

Run the mock EEG pipeline with the Panda-style robot model in RViz:

```bash
scripts/run-robot-rviz
```

Run the full test suite:

```bash
scripts/test
```

## Components

- **ROS 2 target**: Jazzy on Ubuntu 24.04
- **EEG pipeline**: Python (`rclpy`) mock EEG publisher and deterministic baseline decoder
- **Interfaces**: custom EEG frame and intent messages
- **Robot control**: C++ (`rclcpp`) intent subscriber/logger and intent-driven joint-state driver
- **Visualization**: RViz marker for the current decoded intent and a lightweight Panda-style robot model

## Packages

- `eeg_bci_interfaces`: shared ROS messages
- `eeg_bci_pipeline`: mock EEG publisher, baseline decoder, RViz intent marker, and launch files
- `manipulator_control`: robot-control boundary nodes and Panda visualization URDF

## Available Scripts

- `scripts/build`: build the workspace with `colcon build --symlink-install`
- `scripts/test`: run `colcon test` and print verbose test results
- `scripts/run-mock`: launch the mock EEG publisher, decoder, and intent logger
- `scripts/run-rviz`: launch the mock pipeline with the intent marker in RViz
- `scripts/run-robot`: launch the mock pipeline with the Panda-style robot state publisher
- `scripts/run-robot-rviz`: launch the mock robot pipeline and RViz robot visualization
- `scripts/run-bciciv2a`: replay a BCI Competition IV 2a GDF recording into the pipeline
- `scripts/evaluate-hand-classifier`: evaluate an offline BCIC IV 2a left/right CSP + LDA
  baseline; pass `--save-model path/to/model.joblib` to also fit and save one final CSP+LDA
  calibration artifact
- `scripts/run-bciciv2a-model`: replay a BCIC IV 2a recording through a saved CSP + LDA model
  (from `--save-model`) instead of the mock decoder, gating low-confidence windows to `rest`
- `scripts/lint`: check Python formatting/imports and Ruff lint rules
- `scripts/format`: format Python files and sort imports with Ruff
- `scripts/typecheck`: run basedpyright on the pure Python pipeline helpers
- `scripts/setup-dev-tools`: install Ruff and basedpyright with pipx

The ROS build, test, and launch scripts source `/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash`.
ROS launch and test scripts also source `install/setup.bash` and require a prior `scripts/build`.
Extra arguments are passed through to `colcon`, `ros2 launch`, Ruff, or basedpyright.
Run `scripts/setup-dev-tools` once to install optional Python developer tools.

## Replaying with a Trained Model

`scripts/run-bciciv2a-model` swaps the mock decoder for a real CSP + LDA classifier on the
replay path:

```
scripts/evaluate-hand-classifier data/raw/bciciv2a/A01T.gdf --save-model tmp/hand-csp-lda-A01T.joblib
scripts/run-bciciv2a-model gdf_path:=data/raw/bciciv2a/A01T.gdf model_path:=tmp/hand-csp-lda-A01T.joblib
```

The `model_intent_decoder` node takes its frame contract (channels, sampling rate, window
length) from the artifact and rejects non-matching frames. It publishes `rest` until one epoch
window (~3 s) has buffered, then gates any window below `rest_confidence_threshold` (default
`0.6`) to `rest`.

Limitations: per-frame decoding, no temporal smoothing; `loop:=true` contaminates ~one window
per wrap (stitched file boundary); and the 22-channel BCIC artifact is a benchmark, not for the
16-channel BrainAccess rig. Train and replay different sessions (`A01T` then `A01E`) for an
honest check; in a robot launch keep the driver's `confidence_threshold` ≤
`rest_confidence_threshold`.

## Mock Robot Behavior

The mock robot launch maps decoded intents to `panda_joint2`:

```text
rest       -> hold position
left_hand  -> move negative
right_hand -> move positive
```

`intent_joint_state_driver` ignores movement intents below its `confidence_threshold` parameter, which defaults to `0.55`. The driven joint can be changed with `driven_joint_name`; it defaults to `panda_joint2`. If intent messages stop arriving, `intent_timeout_sec` defaults to `0.3` seconds before motion is held.

The remaining Panda joints are published at zero so `robot_state_publisher` can maintain the complete TF tree.

## Manual ROS Commands

The scripts are thin wrappers around the standard ROS commands:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

```bash
ros2 launch eeg_bci_pipeline mock_pipeline.launch.py
ros2 launch eeg_bci_pipeline mock_rviz.launch.py
ros2 launch eeg_bci_pipeline mock_robot.launch.py
ros2 launch eeg_bci_pipeline mock_robot_rviz.launch.py
ros2 launch eeg_bci_pipeline bciciv2a_replay.launch.py gdf_path:=data/raw/bciciv2a/A01T.gdf
```

Pick one launch file for the view you want to run.

```bash
colcon test
colcon test-result --verbose
```

The pure decoder tests can also run without ROS by setting `PYTHONPATH`:

```bash
PYTHONPATH=src/eeg_bci_pipeline pytest src/eeg_bci_pipeline/test
```

This shortcut works because those tests import only pure Python helpers.
