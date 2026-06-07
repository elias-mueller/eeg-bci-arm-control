import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from sensor_msgs.msg import JointState

from eeg_bci_interfaces.msg import Intent

INTENT_TOPIC = "/test/bci/intent"
JOINT_STATE_TOPIC = "/test/joint_states"
DRIVEN_JOINT_NAME = "panda_joint2"
JOINT_LIMIT_RAD = 0.4
INTENT_TIMEOUT_SEC = 0.15


@pytest.mark.launch_test
def generate_test_description():
    driver = launch_ros.actions.Node(
        package="manipulator_control",
        executable="intent_joint_state_driver",
        name="intent_joint_state_driver_test",
        parameters=[
            {
                "intent_topic": INTENT_TOPIC,
                "joint_state_topic": JOINT_STATE_TOPIC,
                "confidence_threshold": 0.55,
                "joint_velocity_rad_s": 1.0,
                "joint_limit_rad": JOINT_LIMIT_RAD,
                "publish_rate_hz": 20.0,
                "intent_timeout_sec": INTENT_TIMEOUT_SEC,
            }
        ],
        output="screen",
    )

    return (
        launch.LaunchDescription(
            [
                driver,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"driver": driver},
    )


class TestIntentJointStateDriver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("intent_joint_state_driver_test_client")
        self.publisher = self.node.create_publisher(Intent, INTENT_TOPIC, 10)
        self.joint_states = []
        self.subscription = self.node.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self.joint_states.append,
            10,
        )

    def tearDown(self):
        self.node.destroy_node()

    def test_intents_drive_joint_in_expected_direction(self):
        self.assertTrue(
            self._spin_until(
                lambda: self.publisher.get_subscription_count() > 0 and self.joint_states
            )
        )

        initial_position = self._latest_driven_joint_position()
        self._publish_intent_for("right_hand", 0.1, 0.4)
        low_confidence_position = self._latest_driven_joint_position()
        self.assertLess(abs(low_confidence_position - initial_position), 0.05)

        self._publish_intent_for("right_hand", 0.9, 0.12)
        moving_position = self._latest_driven_joint_position()
        self.assertGreater(moving_position, low_confidence_position + 0.05)

        self._spin_for(INTENT_TIMEOUT_SEC + 0.2)
        stale_timeout_position = self._latest_driven_joint_position()
        self.assertLess(stale_timeout_position, JOINT_LIMIT_RAD - 0.05)

        self._spin_for(0.3)
        stale_hold_position = self._latest_driven_joint_position()
        self.assertLess(abs(stale_hold_position - stale_timeout_position), 0.05)

        self._publish_intent_for("right_hand", 0.9, 0.8)
        right_position = self._latest_driven_joint_position()
        self.assertGreater(right_position, stale_hold_position + 0.1)
        self.assertLessEqual(right_position, JOINT_LIMIT_RAD + 0.01)
        self.assertGreater(right_position, JOINT_LIMIT_RAD - 0.1)

        self._publish_intent_for("rest", 1.0, 0.4)
        rest_position = self._latest_driven_joint_position()
        self.assertLess(abs(rest_position - right_position), 0.05)

        self._publish_intent_for("left_hand", 0.9, 1.0)
        left_position = self._latest_driven_joint_position()
        self.assertLess(left_position, rest_position - 0.1)
        self.assertGreaterEqual(left_position, -JOINT_LIMIT_RAD - 0.01)
        self.assertLess(left_position, -JOINT_LIMIT_RAD + 0.1)

    def test_unknown_intent_label_is_ignored(self):
        self.assertTrue(
            self._spin_until(
                lambda: self.publisher.get_subscription_count() > 0 and self.joint_states
            )
        )

        # Nudge to a strictly-interior position so "held" is distinguishable from
        # "pinned at a clamp" (a non-zero unknown velocity of either sign would move
        # the joint off this point). A short drive never reaches the 0.4 clamp.
        self._publish_intent_for("right_hand", 0.95, 0.15)
        self.assertLess(abs(self._latest_driven_joint_position()), JOINT_LIMIT_RAD - 0.05)

        # Switch to the unknown label and let the previous command expire and the
        # zero-velocity command settle (one handoff tick can still apply the old
        # velocity), then confirm the joint holds across a further window. The
        # baseline is captured after the settle, not before, so the handoff tick
        # cannot be mistaken for motion under the unknown label.
        self._publish_intent_for("blink", 0.95, INTENT_TIMEOUT_SEC + 0.3)
        settled = self._latest_driven_joint_position()
        self._publish_intent_for("blink", 0.95, 0.3)
        self.assertAlmostEqual(self._latest_driven_joint_position(), settled, delta=0.02)

    def _publish_intent_for(self, label, confidence, duration_sec):
        intent = Intent()
        intent.label = label
        intent.confidence = confidence

        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            self.publisher.publish(intent)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _spin_for(self, duration_sec):
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _latest_driven_joint_position(self):
        latest = self.joint_states[-1]
        joint_index = latest.name.index(DRIVEN_JOINT_NAME)
        return latest.position[joint_index]

    def _spin_until(self, predicate, timeout_sec=5.0):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if predicate():
                return True
        return predicate()
