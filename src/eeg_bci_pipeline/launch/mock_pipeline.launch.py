from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="eeg_bci_pipeline",
                executable="mock_eeg_publisher",
                name="mock_eeg_publisher",
                parameters=[
                    {
                        "amplitude_cycle_uv": [10.0, 25.0, 45.0],
                        "frames_per_intent": 10,
                    }
                ],
                output="screen",
            ),
            Node(
                package="eeg_bci_pipeline",
                executable="mock_intent_decoder",
                name="mock_intent_decoder",
                parameters=[
                    {
                        "class_labels": ["rest", "left_hand", "right_hand"],
                    }
                ],
                output="screen",
            ),
            Node(
                package="manipulator_control",
                executable="intent_command_logger",
                name="intent_command_logger",
                output="screen",
            ),
        ]
    )
