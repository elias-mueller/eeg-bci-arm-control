"""Decode mock EEG frames into intent messages."""

from __future__ import annotations

import rclpy
from eeg_bci_interfaces.msg import EegFrame, Intent
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from eeg_bci_pipeline.decoder import DEFAULT_CLASS_LABELS, decode_mock_intent


class BaselineIntentDecoder(Node):
    """Subscribes to EEG frames and publishes deterministic mock intents."""

    def __init__(self) -> None:
        super().__init__("baseline_intent_decoder")
        self.declare_parameter("input_topic", "/bci/eeg")
        self.declare_parameter("output_topic", "/bci/intent")
        self.declare_parameter("class_labels", list(DEFAULT_CLASS_LABELS))

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._class_labels = tuple(self.get_parameter("class_labels").value)

        self._publisher = self.create_publisher(Intent, output_topic, 10)
        self._subscription = self.create_subscription(
            EegFrame,
            input_topic,
            self._on_eeg_frame,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Decoding EEG frames from {input_topic} to intents on {output_topic}")

    def _on_eeg_frame(self, frame: EegFrame) -> None:
        prediction = decode_mock_intent(frame.samples, self._class_labels)
        intent = Intent()
        intent.header = frame.header
        intent.label = prediction.label
        intent.confidence = prediction.confidence
        intent.class_labels = list(prediction.class_labels)
        intent.probabilities = list(prediction.probabilities)
        self._publisher.publish(intent)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = BaselineIntentDecoder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
