# EEG-BCI Manipulator Control

Real-time motor-imagery EEG decoding driving a simulated Franka Panda manipulator over ROS 2.

The workspace can run a no-device ROS graph where synthetic EEG frames are decoded into mock intents and consumed by a robot-control boundary node.

## Components

- **ROS 2 target**: Jazzy on Ubuntu 24.04
- **EEG pipeline**: Python (`rclpy`) mock EEG publisher and deterministic baseline decoder
- **Interfaces**: custom EEG frame and intent messages
- **Robot control**: C++ (`rclcpp`) intent subscriber/logger

## Packages

- `eeg_bci_interfaces`: shared ROS messages
- `eeg_bci_pipeline`: mock EEG publisher, baseline decoder, and launch file
- `manipulator_control`: robot-control boundary node

## Build

The devcontainer installs `ros-jazzy-ros-base` and `ros-dev-tools`. If you are not using the devcontainer, install ROS 2 Jazzy first using the official Ubuntu deb package instructions: <https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>.

Source ROS before building:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run Without BCI Hardware

Start the full mock pipeline:

```bash
ros2 launch eeg_bci_pipeline mock_pipeline.launch.py
```

In another shell:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 topic echo /bci/intent
```

## Test

```bash
source /opt/ros/jazzy/setup.bash
colcon test
colcon test-result --verbose
```

The pure decoder tests can also run without ROS by setting `PYTHONPATH`:

```bash
PYTHONPATH=src/eeg_bci_pipeline pytest src/eeg_bci_pipeline/test
```

This shortcut works because those tests import only pure Python helpers, not `rclpy` or generated ROS message modules.
