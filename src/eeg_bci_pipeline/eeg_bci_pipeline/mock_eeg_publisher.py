# pyright: basic
# rclpy and the generated message classes are only partially typed; strict mode
# floods this shell with reportUnknown* noise. Basic still catches real mistakes.
"""Publish synthetic EEG frames so the pipeline can run without hardware."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from eeg_bci_interfaces.msg import EegFrame
from eeg_bci_pipeline.eeg_frame_contract import (
    DEFAULT_EEG_CHANNEL_COUNT,
    DEFAULT_EEG_SAMPLING_RATE_HZ,
    default_channel_labels,
)
from eeg_bci_pipeline.mock_signal import (
    DEFAULT_AMPLITUDE_CYCLE_UV,
    generate_mock_eeg_samples,
    select_cycle_value,
)
from eeg_bci_pipeline.node_params import (
    float_list_param,
    float_param,
    int_param,
    str_param,
)


class MockEegPublisher(Node):
    """Publishes deterministic synthetic EEG frames that cycle signal energy."""

    def __init__(self) -> None:
        super().__init__("mock_eeg_publisher")
        self.declare_parameter("source_id", "mock-eeg")
        self.declare_parameter("sampling_rate_hz", DEFAULT_EEG_SAMPLING_RATE_HZ)
        self.declare_parameter("channel_count", DEFAULT_EEG_CHANNEL_COUNT)
        self.declare_parameter("samples_per_frame", 25)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("amplitude_cycle_uv", list(DEFAULT_AMPLITUDE_CYCLE_UV))
        self.declare_parameter("frames_per_intent", 10)
        self.declare_parameter("topic", "/bci/eeg")

        self._source_id = str_param(self, "source_id")
        self._sampling_rate_hz = float_param(self, "sampling_rate_hz")
        self._channel_count = int_param(self, "channel_count")
        self._samples_per_frame = int_param(self, "samples_per_frame")
        self._amplitude_cycle_uv = tuple(float_list_param(self, "amplitude_cycle_uv"))
        self._frames_per_intent = int_param(self, "frames_per_intent")
        publish_rate_hz = float_param(self, "publish_rate_hz")
        topic = str_param(self, "topic")

        self._channel_labels = list(default_channel_labels(self._channel_count))
        self._sample_index = 0
        self._frame_index = 0
        self._publisher = self.create_publisher(EegFrame, topic, qos_profile_sensor_data)
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish_frame)
        self.get_logger().info(
            f"Publishing mock EEG frames to {topic} "
            f"({self._channel_count} channels at {self._sampling_rate_hz:.1f} Hz; "
            f"amplitude cycle {self._amplitude_cycle_uv} uV)"
        )

    def _publish_frame(self) -> None:
        frame = EegFrame()
        frame.header.stamp = self.get_clock().now().to_msg()
        frame.header.frame_id = "eeg"
        frame.source_id = self._source_id
        frame.sampling_rate_hz = self._sampling_rate_hz
        frame.channel_labels = self._channel_labels
        frame.samples = self._generate_samples()
        self._publisher.publish(frame)

    def _generate_samples(self) -> list[float]:
        amplitude_uv = select_cycle_value(
            self._amplitude_cycle_uv,
            self._frame_index,
            self._frames_per_intent,
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
    node = MockEegPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
