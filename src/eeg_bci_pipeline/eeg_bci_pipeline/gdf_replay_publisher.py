"""Replay GDF EEG recordings as EegFrame messages."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from eeg_bci_interfaces.msg import EegFrame
from eeg_bci_pipeline.eeg_frame_contract import (
    DEFAULT_EEG_SAMPLING_RATE_HZ,
    DEFAULT_MAX_ABS_SAMPLE_UV,
    DEFAULT_SAMPLING_RATE_TOLERANCE_HZ,
    validate_eeg_frame_payload,
)
from eeg_bci_pipeline.gdf_recording import (
    BCICIV2A_EEG_CHANNEL_COUNT,
    DEFAULT_REPLAY_SAMPLES_PER_FRAME,
    EegRecording,
    iter_replay_frames,
    normalize_channel_labels,
    read_gdf_recording,
    replay_elapsed_sec,
)


class GdfReplayPublisher(Node):
    """Publishes EEG frames from a GDF recording."""

    def __init__(self) -> None:
        super().__init__("gdf_replay_publisher")
        self.declare_parameter("gdf_path", "")
        self.declare_parameter("topic", "/bci/eeg")
        self.declare_parameter("samples_per_frame", DEFAULT_REPLAY_SAMPLES_PER_FRAME)
        self.declare_parameter("loop", False)
        self.declare_parameter("channel_labels", [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("expected_channel_count", BCICIV2A_EEG_CHANNEL_COUNT)
        self.declare_parameter("expected_sampling_rate_hz", DEFAULT_EEG_SAMPLING_RATE_HZ)
        self.declare_parameter("sampling_rate_tolerance_hz", DEFAULT_SAMPLING_RATE_TOLERANCE_HZ)
        self.declare_parameter("max_abs_sample_uv", DEFAULT_MAX_ABS_SAMPLE_UV)

        gdf_path_value = str(self.get_parameter("gdf_path").value)
        if not gdf_path_value:
            raise ValueError("gdf_path parameter must point to a GDF recording")
        gdf_path = Path(gdf_path_value).expanduser()

        self._samples_per_frame = int(self.get_parameter("samples_per_frame").value)
        self._loop = bool(self.get_parameter("loop").value)
        self._expected_channel_count = int(self.get_parameter("expected_channel_count").value)
        self._expected_sampling_rate_hz = float(
            self.get_parameter("expected_sampling_rate_hz").value
        )
        self._sampling_rate_tolerance_hz = float(
            self.get_parameter("sampling_rate_tolerance_hz").value
        )
        self._max_abs_sample_uv = float(self.get_parameter("max_abs_sample_uv").value)
        channel_labels = self._optional_string_array("channel_labels")

        self._recording = read_gdf_recording(gdf_path, channel_labels=channel_labels)
        if channel_labels:
            self._expected_channel_count = len(channel_labels)
        # Cache frame payloads so loop mode can replay without re-flattening the recording.
        self._frames = list(
            iter_replay_frames(self._recording, samples_per_frame=self._samples_per_frame)
        )
        if not self._frames:
            raise ValueError("GDF recording contains no replay frames")

        self._validate_first_frame(self._recording)
        topic = self.get_parameter("topic").value
        self._publisher = self.create_publisher(EegFrame, topic, qos_profile_sensor_data)
        self._frame_index = 0
        self._loop_index = 0
        self._replay_start_time = self.get_clock().now()
        publish_period_sec = self._samples_per_frame / self._recording.sampling_rate_hz
        self._timer = self.create_timer(publish_period_sec, self._publish_frame)
        self.get_logger().info(
            f"Replaying {gdf_path} to {topic}: {len(self._recording.channel_labels)} "
            f"channels at {self._recording.sampling_rate_hz:g} Hz"
        )

    def _publish_frame(self) -> None:
        if self._frame_index >= len(self._frames):
            if not self._loop:
                self.get_logger().info("Finished GDF replay")
                self._timer.cancel()
                return
            self._loop_index += 1
            self._frame_index = 0

        replay_frame = self._frames[self._frame_index]
        frame = EegFrame()
        frame.header.stamp = self._stamp_for_sample(replay_frame.start_sample_index)
        frame.header.frame_id = "eeg"
        frame.source_id = self._recording.source_id
        frame.sampling_rate_hz = self._recording.sampling_rate_hz
        frame.channel_labels = list(self._recording.channel_labels)
        frame.samples = replay_frame.samples
        self._publisher.publish(frame)
        self._frame_index += 1

    def _stamp_for_sample(self, sample_index: int):
        elapsed_sec = replay_elapsed_sec(
            sample_index,
            loop_index=self._loop_index,
            samples_per_channel=self._recording.samples_per_channel,
            sampling_rate_hz=self._recording.sampling_rate_hz,
        )
        return (self._replay_start_time + Duration(seconds=elapsed_sec)).to_msg()

    def _validate_first_frame(self, recording: EegRecording) -> None:
        first_frame = self._frames[0]
        validate_eeg_frame_payload(
            sampling_rate_hz=recording.sampling_rate_hz,
            channel_labels=recording.channel_labels,
            samples=first_frame.samples,
            expected_channel_count=self._expected_channel_count,
            expected_sampling_rate_hz=self._expected_sampling_rate_hz,
            sampling_rate_tolerance_hz=self._sampling_rate_tolerance_hz,
            max_abs_sample_uv=self._max_abs_sample_uv,
        )

    def _optional_string_array(self, parameter_name: str) -> tuple[str, ...] | None:
        labels = normalize_channel_labels(self.get_parameter(parameter_name).value)
        return labels or None


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GdfReplayPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
