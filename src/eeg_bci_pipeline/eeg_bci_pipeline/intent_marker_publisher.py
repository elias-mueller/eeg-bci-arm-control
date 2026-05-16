"""Publish RViz markers for decoded BCI intents."""

from __future__ import annotations

from math import cos, sin

import rclpy
from eeg_bci_interfaces.msg import Intent
from rclpy.node import Node
from visualization_msgs.msg import Marker

from eeg_bci_pipeline.intent_marker_mapping import IntentMarkerStyle, style_for_intent


class IntentMarkerPublisher(Node):
    """Converts decoded intents into a simple directional marker."""

    def __init__(self) -> None:
        super().__init__("intent_marker_publisher")
        self.declare_parameter("input_topic", "/bci/intent")
        self.declare_parameter("marker_topic", "/bci/intent_marker")
        self.declare_parameter("frame_id", "bci_world")

        input_topic = self.get_parameter("input_topic").value
        marker_topic = self.get_parameter("marker_topic").value
        self._frame_id = self.get_parameter("frame_id").value

        self._publisher = self.create_publisher(Marker, marker_topic, 10)
        self._subscription = self.create_subscription(
            Intent,
            input_topic,
            self._on_intent,
            10,
        )
        self.get_logger().info(
            f"Publishing intent markers from {input_topic} to {marker_topic}"
        )

    def _on_intent(self, intent: Intent) -> None:
        style = style_for_intent(intent.label, intent.confidence)
        stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._arrow_marker(style, stamp))
        self._publisher.publish(self._text_marker(style, intent.confidence, stamp))

    def _arrow_marker(self, style: IntentMarkerStyle, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = stamp
        marker.ns = "bci_intent"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.z = sin(style.yaw_rad / 2.0)
        marker.pose.orientation.w = cos(style.yaw_rad / 2.0)
        marker.scale.x = style.length
        marker.scale.y = 0.08
        marker.scale.z = 0.12
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = style.color_rgba
        return marker

    def _text_marker(self, style: IntentMarkerStyle, confidence: float, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = stamp
        marker.ns = "bci_intent"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 0.45
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.16
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = style.color_rgba
        marker.text = f"{style.label} {confidence:.2f}"
        return marker


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = IntentMarkerPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
