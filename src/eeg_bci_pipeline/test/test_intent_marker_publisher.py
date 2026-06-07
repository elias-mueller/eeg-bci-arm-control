"""Node-level coverage for the `intent_marker_publisher` RViz glue.

The pure label -> style mapping is covered in `test_intent_marker_mapping.py`;
this drives the `IntentMarkerPublisher` node itself. It feeds real `Intent`
messages through the actual `_on_intent` callback (capturing what would be
published by replacing the publisher's `publish` with a list append), and checks
the two emitted markers against the geometry the node promises: an ARROW whose
orientation quaternion is the half-yaw (sin, cos) encoding of the style yaw, and
a TEXT_VIEW_FACING marker whose text is the label plus the formatted confidence.
No wall-clock waits: the clock is read once per call and never compared against.

Skipped when ROS (rclpy / the interfaces) is unavailable, as in a bare pytest run.
"""

from __future__ import annotations

from math import cos, pi, sin

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("eeg_bci_interfaces.msg")

from eeg_bci_pipeline.intent_marker_mapping import (  # noqa: E402
    IntentMarkerStyle,
    style_for_intent,
)
from eeg_bci_pipeline.intent_marker_publisher import IntentMarkerPublisher  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from eeg_bci_interfaces.msg import Intent  # noqa: E402


@pytest.fixture
def ros_context():
    # Owns the rclpy lifecycle so a failure mid-test never leaks the context into
    # the next one (a bare init/finally in each test would skip shutdown if the
    # node constructor raised).
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


@pytest.fixture
def marker_node(ros_context):
    node = IntentMarkerPublisher()
    try:
        yield node
    finally:
        node.destroy_node()


def _captured_markers(node) -> list:
    # Redirect the real publisher to a list so _on_intent's actual publish path
    # is exercised without spinning up a subscriber.
    published: list = []
    node._publisher.publish = published.append
    return published


def _intent(label: str, confidence: float) -> Intent:
    intent = Intent()
    intent.label = label
    intent.confidence = confidence
    return intent


def test_init_creates_pub_sub_and_reads_frame_id(marker_node):
    # The constructor wires up a Marker publisher, an Intent subscription, and
    # caches the frame_id parameter that every emitted marker stamps its header
    # with.
    assert marker_node._frame_id == "bci_world"
    assert marker_node._publisher.msg_type is Marker
    assert marker_node._subscription.msg_type is Intent


def test_on_intent_publishes_arrow_then_text(marker_node):
    # One intent emits exactly two markers sharing a namespace but split by id:
    # the ARROW (id 0) followed by the TEXT_VIEW_FACING label (id 1).
    published = _captured_markers(marker_node)

    marker_node._on_intent(_intent("right_hand", 0.9))

    assert len(published) == 2
    arrow, text = published
    assert arrow.type == Marker.ARROW
    assert arrow.id == 0
    assert arrow.ns == "bci_intent"
    assert text.type == Marker.TEXT_VIEW_FACING
    assert text.id == 1
    assert text.ns == "bci_intent"
    # The markers reflect THIS intent, not hardcoded defaults: the text renders the
    # label and confidence, and the arrow points along the label's style yaw.
    # right_hand maps to yaw 0; a swapped left/right (yaw pi) or wrong confidence
    # would flip these. (Without this, _on_intent could call style_for_intent with
    # the wrong args and still pass.)
    assert text.text == "right_hand 0.90"
    assert arrow.pose.orientation.z == pytest.approx(0.0)
    assert arrow.pose.orientation.w == pytest.approx(1.0)


def test_arrow_marker_encodes_style(marker_node):
    # The arrow carries the style's color, length, the fixed cross-section scale,
    # and the node's frame_id, with action ADD.
    style = style_for_intent("right_hand", 0.9)
    stamp = marker_node.get_clock().now().to_msg()

    arrow = marker_node._arrow_marker(style, stamp)

    assert arrow.header.frame_id == "bci_world"
    assert arrow.header.stamp == stamp
    assert arrow.action == Marker.ADD
    assert arrow.scale.x == pytest.approx(style.length)
    assert arrow.scale.y == pytest.approx(0.08)
    assert arrow.scale.z == pytest.approx(0.12)
    assert (arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a) == pytest.approx(
        style.color_rgba
    )


@pytest.mark.parametrize("yaw", [0.0, pi, pi / 2.0])
def test_arrow_quaternion_is_half_yaw(marker_node, yaw):
    # The arrow's yaw is encoded as a 2D quaternion about z: (z, w) = (sin, cos)
    # of half the yaw. Checks all three task yaws, including the pi case where
    # sin(pi/2)=1 distinguishes a half-yaw from a full-yaw bug.
    style = IntentMarkerStyle("custom", yaw, 1.0, (0.1, 0.2, 0.3, 1.0))
    stamp = marker_node.get_clock().now().to_msg()

    arrow = marker_node._arrow_marker(style, stamp)

    assert arrow.pose.orientation.z == pytest.approx(sin(yaw / 2.0))
    assert arrow.pose.orientation.w == pytest.approx(cos(yaw / 2.0))


def test_text_marker_renders_label_and_confidence(marker_node):
    # The text marker shows the style label plus the confidence to two decimals,
    # raised above the arrow and billboarded with a neutral orientation.
    style = style_for_intent("left_hand", 0.875)
    stamp = marker_node.get_clock().now().to_msg()

    text = marker_node._text_marker(style, 0.875, stamp)

    assert text.text == "left_hand 0.88"
    assert text.header.frame_id == "bci_world"
    assert text.header.stamp == stamp
    assert text.action == Marker.ADD
    assert text.pose.position.z == pytest.approx(0.45)
    assert text.pose.orientation.w == pytest.approx(1.0)
    assert text.scale.z == pytest.approx(0.16)
    assert (text.color.r, text.color.g, text.color.b, text.color.a) == pytest.approx(
        style.color_rgba
    )
