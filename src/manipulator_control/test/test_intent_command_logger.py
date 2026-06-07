"""Launch test for the intent_command_logger C++ node.

The logger publishes no topic; its only observable behavior is the log lines it
emits, so the assertions are on the node's process output (the node's contract).
A client publishes Intent messages while the test polls proc_output, which
absorbs the subscription-discovery race without a fixed sleep.
"""

import threading
import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy

from eeg_bci_interfaces.msg import Intent

INTENT_TOPIC = "/test/bci/intent"


@pytest.mark.launch_test
def generate_test_description():
    logger = launch_ros.actions.Node(
        package="manipulator_control",
        executable="intent_command_logger",
        name="intent_command_logger_test",
        parameters=[{"intent_topic": INTENT_TOPIC}],
        output="screen",
    )

    return (
        launch.LaunchDescription(
            [
                logger,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"logger": logger},
    )


class TestIntentCommandLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node("intent_command_logger_test_client")
        self.publisher = self.node.create_publisher(Intent, INTENT_TOPIC, 10)

    def tearDown(self):
        self.node.destroy_node()

    def test_logs_the_configured_topic_on_startup(self, proc_output, logger):
        proc_output.assertWaitFor(
            f"Listening for BCI intents on {INTENT_TOPIC}",
            process=logger,
            timeout=10,
        )

    def test_logs_each_received_intent(self, proc_output, logger):
        # Keep publishing in the background so the match does not depend on
        # pub/sub discovery completing before a single publish.
        stop = threading.Event()

        def pump():
            intent = Intent()
            intent.label = "left_hand"
            intent.confidence = 0.87
            while not stop.is_set():
                self.publisher.publish(intent)
                rclpy.spin_once(self.node, timeout_sec=0.05)
                time.sleep(0.02)

        pump_thread = threading.Thread(target=pump, daemon=True)
        pump_thread.start()
        try:
            proc_output.assertWaitFor(
                "Received intent 'left_hand' with confidence 0.870",
                process=logger,
                timeout=15,
            )
        finally:
            stop.set()
            pump_thread.join()
