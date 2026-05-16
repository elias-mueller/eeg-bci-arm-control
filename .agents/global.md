# Shared Instructions

## Agent instruction rendering

`.agents/global.md` is the source of truth for durable repo-local agent memory. `CLAUDE.md` and `AGENTS.md` are rendered outputs.

- Do not edit `CLAUDE.md` or `AGENTS.md` directly for durable conventions.
- Put shared repo rules and preferences in `.agents/global.md`.
- Put tool-specific wording only in `.agents/claude-extra.md` or `.agents/codex-extra.md`.
- After changing any `.agents/*.md` source, run `.agents/render.sh` and include the rendered files in the diff.

## Project

EEG-BCI Manipulator Control: real-time motor-imagery EEG pipeline driving a simulated Franka Panda manipulator over ROS 2.

- **EEG pipeline.** Python (`rclpy`), MNE, scikit-learn. 16-channel BrainAccess MIDI recordings at 250 Hz. CSP + LDA baseline; EEGNet (PyTorch) experiments.
- **Robot control.** C++ (`rclcpp`) ROS 2 nodes.
- **Simulation.** Gazebo with a Franka Panda model.
