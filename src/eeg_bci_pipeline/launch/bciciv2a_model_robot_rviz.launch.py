"""Replay a BCIC IV 2a recording through a trained model and drive the Panda in RViz.

This is the model counterpart to ``mock_robot_rviz.launch.py``: it composes the
logger-only ``bciciv2a_model_replay.launch.py`` (gdf_replay_publisher ->
model_intent_decoder -> /bci/intent) with the visualization half of
``mock_robot.launch.py`` (robot_state_publisher + intent_joint_state_driver) plus
the intent marker and RViz, so decoded EEG actually moves the arm instead of only
printing intents.

The path defaults assume launch from the repo root (``scripts/run-bciciv2a-model-rviz``
cd's there). Train on a subject's T session and replay the held-out E session for an
honest check, e.g.::

    scripts/evaluate-hand-classifier data/raw/bciciv2a/A01T.gdf \
        --save-model tmp/hand-csp-lda-A01T.joblib
    scripts/run-bciciv2a-model-rviz
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("eeg_bci_pipeline")
    model_replay_launch = PathJoinSubstitution(
        [package_share, "launch", "bciciv2a_model_replay.launch.py"]
    )
    rviz_config = PathJoinSubstitution([package_share, "rviz", "mock_robot.rviz"])
    robot_description_path = PathJoinSubstitution(
        [FindPackageShare("manipulator_control"), "urdf", "panda_visual.urdf"]
    )
    robot_description = ParameterValue(
        Command(["cat ", robot_description_path]),
        value_type=str,
    )

    gdf_path = LaunchConfiguration("gdf_path")
    model_path = LaunchConfiguration("model_path")
    loop = LaunchConfiguration("loop")
    samples_per_frame = LaunchConfiguration("samples_per_frame")
    rest_confidence_threshold = LaunchConfiguration("rest_confidence_threshold")
    confidence_threshold = LaunchConfiguration("confidence_threshold")

    return LaunchDescription(
        [
            # Relative defaults resolve against the working directory; the run script
            # cd's to the repo root before launching.
            DeclareLaunchArgument("gdf_path", default_value="data/raw/bciciv2a/A01E.gdf"),
            DeclareLaunchArgument("model_path", default_value="tmp/hand-csp-lda-A01T.joblib"),
            DeclareLaunchArgument("loop", default_value="true"),
            DeclareLaunchArgument("samples_per_frame", default_value="25"),
            DeclareLaunchArgument("rest_confidence_threshold", default_value="0.6"),
            # Keep the driver gate <= the decoder's rest gate so confident hand intents
            # are not dropped before they reach the joint driver.
            DeclareLaunchArgument("confidence_threshold", default_value="0.55"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(model_replay_launch),
                launch_arguments={
                    "gdf_path": gdf_path,
                    "model_path": model_path,
                    "loop": loop,
                    "samples_per_frame": samples_per_frame,
                    "rest_confidence_threshold": rest_confidence_threshold,
                }.items(),
            ),
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
                parameters=[
                    {
                        "confidence_threshold": ParameterValue(
                            confidence_threshold, value_type=float
                        )
                    }
                ],
                output="screen",
            ),
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
