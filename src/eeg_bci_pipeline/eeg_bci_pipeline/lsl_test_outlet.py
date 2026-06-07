# pyright: basic
# rclpy and the generated message classes are only partially typed; strict mode
# floods this shell with reportUnknown* noise. Basic still catches real mistakes.
"""Stream EEG to a Lab Streaming Layer outlet so the bridge runs without hardware.

A test producer for `lsl_eeg_bridge`. It owns a pylsl `StreamOutlet` (it publishes
no ROS topic) in one of two modes:

* ``gdf``: replay a BCIC IV 2a recording over LSL via the existing
  `read_gdf_recording` / `iter_replay_frames` helpers, carrying the recording's
  own channel labels verbatim in the stream metadata. This is the path that
  drives the arm: the labels match the trained 22-ch artifact exactly, so
  `model_intent_decoder` accepts the bridged frames and decodes real EEG.
* ``synthetic``: stream the deterministic `mock_signal` waveform with ``ch_NN``
  labels. A transport smoke test for the bridge only; those labels do not match
  the 22-ch model, so synthetic mode does not move the arm.

The outlet declares ``microvolts`` and pushes microvolt samples, so the bridge
applies no unit scaling for this source.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

from eeg_bci_pipeline.data.gdf_recording import (
    DEFAULT_REPLAY_SAMPLES_PER_FRAME,
    iter_replay_frames,
    read_gdf_recording,
)
from eeg_bci_pipeline.eeg_frame_contract import (
    DEFAULT_EEG_CHANNEL_COUNT,
    DEFAULT_EEG_SAMPLING_RATE_HZ,
    EEG_SAMPLE_UNIT,
    default_channel_labels,
    validate_eeg_frame_payload,
)
from eeg_bci_pipeline.lsl_bridge import channel_major_to_sample_major, import_pylsl
from eeg_bci_pipeline.mock_signal import (
    DEFAULT_AMPLITUDE_CYCLE_UV,
    generate_mock_eeg_samples,
    select_cycle_value,
)
from eeg_bci_pipeline.node_params import (
    bool_param,
    float_list_param,
    float_param,
    int_param,
    str_list_param,
    str_param,
)


class LslTestOutlet(Node):
    """Pushes synthetic or GDF-replayed EEG to a pylsl StreamOutlet."""

    def __init__(self) -> None:
        super().__init__("lsl_test_outlet")
        self.declare_parameter("mode", "gdf")
        self.declare_parameter("stream_name", "eeg-bci-test")
        self.declare_parameter("source_id", "")
        self.declare_parameter("samples_per_frame", DEFAULT_REPLAY_SAMPLES_PER_FRAME)
        # gdf mode
        self.declare_parameter("gdf_path", "")
        self.declare_parameter("channel_labels", [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("loop", True)
        # synthetic mode
        self.declare_parameter("sampling_rate_hz", DEFAULT_EEG_SAMPLING_RATE_HZ)
        self.declare_parameter("channel_count", DEFAULT_EEG_CHANNEL_COUNT)
        self.declare_parameter("amplitude_cycle_uv", list(DEFAULT_AMPLITUDE_CYCLE_UV))
        self.declare_parameter("frames_per_intent", 10)

        self._mode = str_param(self, "mode")
        self._stream_name = str_param(self, "stream_name")
        self._source_id_param = str_param(self, "source_id")
        self._samples_per_frame = int_param(self, "samples_per_frame")
        if self._samples_per_frame < 1:
            raise ValueError("samples_per_frame must be at least 1")

        if self._mode == "gdf":
            self._setup_gdf_mode()
        elif self._mode == "synthetic":
            self._setup_synthetic_mode()
        else:
            raise ValueError(f"mode must be 'gdf' or 'synthetic', got '{self._mode}'")

        self._outlet = self._open_outlet()
        publish_period_sec = self._samples_per_frame / self._sampling_rate_hz
        self._timer = self.create_timer(publish_period_sec, self._on_tick)
        self.get_logger().info(
            f"Streaming {self._channel_count} ch at {self._sampling_rate_hz:g} Hz to LSL outlet "
            f"'{self._stream_name}' (type=EEG, mode={self._mode}, source_id='{self._source_id}')"
        )

    def _setup_gdf_mode(self) -> None:
        gdf_path_value = str_param(self, "gdf_path")
        if not gdf_path_value:
            raise ValueError("gdf_path parameter must point to a GDF recording")
        labels_override = str_list_param(self, "channel_labels")
        recording = read_gdf_recording(
            Path(gdf_path_value).expanduser(),
            channel_labels=labels_override or None,
        )
        frames = list(iter_replay_frames(recording, samples_per_frame=self._samples_per_frame))
        if not frames:
            raise ValueError("GDF recording contains no replay frames")
        # Fail fast on a malformed recording (finiteness, magnitude, divisibility,
        # label uniqueness). Channel count and rate are checked against the recording's
        # own values here (self-consistency); the bridge re-validates against the
        # launch's expected_* downstream.
        validate_eeg_frame_payload(
            sampling_rate_hz=recording.sampling_rate_hz,
            channel_labels=recording.channel_labels,
            samples=frames[0].samples,
            expected_channel_count=len(recording.channel_labels),
            expected_sampling_rate_hz=recording.sampling_rate_hz,
        )

        self._channel_labels = list(recording.channel_labels)
        self._channel_count = len(self._channel_labels)
        self._sampling_rate_hz = recording.sampling_rate_hz
        self._source_id = self._source_id_param or recording.source_id
        self._loop = bool_param(self, "loop")
        self._frames = [frame.samples for frame in frames]
        self._frame_index = 0

    def _setup_synthetic_mode(self) -> None:
        self._channel_count = int_param(self, "channel_count")
        self._channel_labels = list(default_channel_labels(self._channel_count))
        self._sampling_rate_hz = float_param(self, "sampling_rate_hz")
        if self._sampling_rate_hz <= 0.0:
            raise ValueError("sampling_rate_hz must be greater than 0")
        self._source_id = self._source_id_param or "lsl-test-outlet"
        self._amplitude_cycle_uv = tuple(float_list_param(self, "amplitude_cycle_uv"))
        self._frames_per_intent = int_param(self, "frames_per_intent")
        self._sample_index = 0
        self._frame_index = 0

    def _open_outlet(self):
        pylsl = import_pylsl()
        info = pylsl.StreamInfo(
            name=self._stream_name,
            type="EEG",
            channel_count=self._channel_count,
            nominal_srate=self._sampling_rate_hz,
            channel_format=pylsl.cf_float32,
            source_id=self._source_id,
        )
        channels = info.desc().append_child("channels")
        for label in self._channel_labels:
            channel = channels.append_child("channel")
            channel.append_child_value("label", label)
            channel.append_child_value("unit", EEG_SAMPLE_UNIT)
            channel.append_child_value("type", "EEG")
        return pylsl.StreamOutlet(info, chunk_size=self._samples_per_frame)

    def _on_tick(self) -> None:
        samples = self._next_channel_major()
        if samples is None:
            self.get_logger().info("Finished LSL replay")
            self._timer.cancel()
            return
        self._outlet.push_chunk(channel_major_to_sample_major(samples, self._channel_count))

    def _next_channel_major(self):
        if self._mode == "gdf":
            return self._next_gdf_frame()
        return self._next_synthetic_frame()

    def _next_gdf_frame(self):
        if self._frame_index >= len(self._frames):
            if not self._loop:
                return None
            self._frame_index = 0
        samples = self._frames[self._frame_index]
        self._frame_index += 1
        return samples

    def _next_synthetic_frame(self):
        amplitude_uv = select_cycle_value(
            self._amplitude_cycle_uv, self._frame_index, self._frames_per_intent
        )
        samples = generate_mock_eeg_samples(
            self._sample_index,
            self._samples_per_frame,
            self._channel_count,
            self._sampling_rate_hz,
            amplitude_uv,
        )
        self._sample_index += self._samples_per_frame
        self._frame_index += 1
        return samples


def main(args: list[str] | None = None) -> None:  # pragma: no cover
    rclpy.init(args=args)
    node = None
    try:
        node = LslTestOutlet()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
