from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("eeg_bci_pipeline")
    mock_robot_launch = PathJoinSubstitution([package_share, "launch", "mock_robot.launch.py"])
    rviz_config = PathJoinSubstitution([package_share, "rviz", "mock_robot.rviz"])

    return LaunchDescription(
        [
            IncludeLaunchDescription(PythonLaunchDescriptionSource(mock_robot_launch)),
            Node(
                package="eeg_bci_pipeline",
                executable="intent_marker_publisher",
                name="intent_marker_publisher",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                output="screen",
            ),
        ]
    )
