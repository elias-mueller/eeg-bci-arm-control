# EEG-BCI Manipulator Control

Real-time motor-imagery EEG decoding driving a simulated Franka Panda manipulator over ROS 2, a closed loop from EEG frame to joint motion. A CSP + LDA classifier turns EEG windows into movement intents, and an EEGNet (PyTorch) decoder is benchmarked offline against the same baseline. The system is a C++/Python ROS 2 graph: a Python (`rclpy`) EEG and decoding pipeline feeding C++ (`rclcpp`) control nodes through shared `EegFrame`/`Intent` messages.

## Setup

One-time setup, in order; each item is only needed for the parts you use.

- **ROS 2 Jazzy + RViz** (required for building and any launch). [Install for Ubuntu](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
- **Dev tooling** (optional, for `scripts/lint` and `scripts/typecheck`): `scripts/setup-dev-tools` installs `ruff` and `basedpyright` via `pipx`.
- **Live LSL** (optional, for `scripts/run-lsl`): `scripts/setup-python-env` creates a repo `.venv` and installs `pylsl`. Its wheel bundles the native `liblsl`, so on x86_64 nothing else is needed; the script self-checks and only prints a one-time `liblsl` `.deb` fallback if the bundled library fails to load. The repo scripts and `.envrc` put the `.venv` on `PYTHONPATH`, so the system Python stays untouched.
- **BCIC IV 2a dataset** (optional, for real decode and the replay demos): put the `.gdf` files under `data/raw/bciciv2a/` (gitignored). Without it, `run-robot-rviz` uses the synthetic mock.

The pipeline's other Python dependencies (mne, numpy, scikit-learn) come from apt via `package.xml`; only LSL uses the `.venv`.

## Quick Start

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

## Live LSL (Lab Streaming Layer)

`run-lsl` drives the Panda from a live LSL EEG stream instead of in-process GDF
replay. The bridge (`lsl_eeg_bridge`) is vendor-agnostic: it resolves any
`type="EEG"` LSL stream (a BrainAccess MIDI, a replay outlet, or a synthetic test
stream) and republishes it as `EegFrame` on `/bci/eeg`, so the rest of the graph
is unchanged. Set up `pylsl` once with `scripts/setup-python-env` (see [Setup](#setup)).

Hardware-free demo (default): a test outlet replays the held-out A01E session over
LSL through the same 22-channel CSP+LDA model, so the arm moves from real decoded
EEG with no headset:

```bash
scripts/run-lsl
```

`mode:=synthetic` streams the deterministic mock waveform instead; that is a
bridge transport smoke test only (its `ch_NN` labels do not match the model, so it
does not move the arm). Inspect the bridged stream with `ros2 topic echo /bci/eeg`.

A live BrainAccess MIDI is the same launch with the test outlet off and a
same-montage model (a 16-channel artifact from a calibration session, not the
22-channel BCIC one):

```bash
scripts/run-lsl start_test_outlet:=false stream_name:=<headset stream> \
    expected_channel_count:=16 model_path:=tmp/hand-16ch.joblib
```

The bridge auto-detects the stream's declared unit (microvolts or volts) and
scales to microvolts accordingly. `scale_to_microvolts` is only the fallback for a
stream that declares no recognized unit; set it then (e.g.
`scale_to_microvolts:=1000000.0` for an unlabeled volts stream).

## Calibration (record a same-montage model)

A model only drives the arm from EEG whose channel labels and rate match the
trained artifact, so a real BrainAccess MIDI session needs a model trained on
*your* montage. `run-calibrate` records that calibration set: it shows left/right
cues in RViz and saves one labeled motor-imagery epoch per trial from `/bci/eeg`.

Stream the MIDI to LSL (a BrainAccess SDK/Board step) and place the dry electrodes
over motor cortex (C3 / Cz / C4 matter most for left/right hand), then:

```bash
# Cued recording -> tmp/calibration-epochs.joblib (watch the RViz arrows).
scripts/run-calibrate stream_name:=<headset stream>
# Train a same-montage CSP+LDA model on the captured epochs.
scripts/evaluate-hand-classifier tmp/calibration-epochs.joblib --save-model tmp/hand-16ch.joblib
# Drive the arm from your live motor imagery (start_test_outlet:=false so the
# bridge binds your headset, not the GDF test outlet).
scripts/run-lsl start_test_outlet:=false model_path:=tmp/hand-16ch.joblib \
    stream_name:=<headset stream> expected_channel_count:=16
```

Tune the protocol with `trials_per_class:=` (default 20), `epoch_sec:=`,
`rest_sec:=`, `cue_sec:=`, `settle_sec:=`. Rest is the inter-trial baseline, not a
trained class: training stays 2-class (left/right) and the runtime synthesizes
rest below `rest_confidence_threshold`. Dry-electrode signal quality and
motor-imagery difficulty dominate accuracy, so expect to iterate over a few
sessions.

Hardware-free plumbing test (no headset), replaying BCIC over LSL:

```bash
scripts/run-calibrate start_test_outlet:=true mode:=gdf trials_per_class:=2
```

The cues won't match the replayed brain activity, so the resulting model is at
chance, but the capture -> train -> load chain is exercised end to end. Cross-
validation is skipped automatically when a capture has too few epochs per class,
and the model still saves: `scripts/evaluate-hand-classifier
tmp/calibration-epochs.joblib --save-model tmp/test.joblib`.

## Mock robot behavior

The mock robot launch maps decoded intents to `panda_joint2` (`rest` holds, `left_hand` moves negative, `right_hand` moves positive) while the other joints publish zero so `robot_state_publisher` keeps the full TF tree. `intent_joint_state_driver` ignores intents below `confidence_threshold` (default `0.55`), holds after `intent_timeout_sec` (default `0.3`) without messages, and drives `driven_joint_name` (default `panda_joint2`).
