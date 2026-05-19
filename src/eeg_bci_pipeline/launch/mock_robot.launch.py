from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pipeline_launch = PathJoinSubstitution(
        [FindPackageShare("eeg_bci_pipeline"), "launch", "mock_pipeline.launch.py"]
    )
    robot_description_path = PathJoinSubstitution(
        [FindPackageShare("manipulator_control"), "urdf", "panda_visual.urdf"]
    )
    robot_description = ParameterValue(
        Command(["cat ", robot_description_path]),
        value_type=str,
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(PythonLaunchDescriptionSource(pipeline_launch)),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                output="screen",
            ),
            Node(
                package="manipulator_control",
                executable="intent_joint_state_driver",
                name="intent_joint_state_driver",
                output="screen",
            ),
        ]
    )
