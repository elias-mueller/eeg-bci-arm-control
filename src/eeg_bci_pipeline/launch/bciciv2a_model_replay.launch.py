from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    gdf_path = LaunchConfiguration("gdf_path")
    model_path = LaunchConfiguration("model_path")
    samples_per_frame = LaunchConfiguration("samples_per_frame")
    loop = LaunchConfiguration("loop")
    rest_confidence_threshold = LaunchConfiguration("rest_confidence_threshold")

    return LaunchDescription(
        [
            DeclareLaunchArgument("gdf_path"),
            DeclareLaunchArgument("model_path"),
            DeclareLaunchArgument("samples_per_frame", default_value="25"),
            DeclareLaunchArgument("loop", default_value="false"),
            DeclareLaunchArgument("rest_confidence_threshold", default_value="0.6"),
            Node(
                package="eeg_bci_pipeline",
                executable="gdf_replay_publisher",
                name="gdf_replay_publisher",
                parameters=[
                    {
                        "gdf_path": gdf_path,
                        "samples_per_frame": ParameterValue(samples_per_frame, value_type=int),
                        "loop": ParameterValue(loop, value_type=bool),
                        "expected_channel_count": 22,
                    }
                ],
                output="screen",
            ),
            Node(
                package="eeg_bci_pipeline",
                executable="model_intent_decoder",
                name="model_intent_decoder",
                parameters=[
                    {
                        "model_path": model_path,
                        "rest_confidence_threshold": ParameterValue(
                            rest_confidence_threshold, value_type=float
                        ),
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
