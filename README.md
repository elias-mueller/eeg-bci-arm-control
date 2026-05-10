# EEG-BCI Manipulator Control

Personal bridge project (in progress): real-time motor-imagery EEG decoding driving a simulated Franka Panda manipulator over ROS 2.

## Stack

- **EEG pipeline** — Python (`rclpy`), MNE, scikit-learn (CSP + LDA baseline), PyTorch (EEGNet)
- **Hardware** — Unicorn Hybrid Black recordings (may swap MNE-LSL for BrainFlow)
- **Robot control** — C++ (`rclcpp`) nodes
- **Simulation** — Gazebo (Franka Panda)

## Status

Early. Scope and dependencies still in flux.
