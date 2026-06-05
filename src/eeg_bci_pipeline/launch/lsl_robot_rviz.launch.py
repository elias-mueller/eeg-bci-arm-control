"""Drive the Panda from a live LSL EEG stream through a trained model.

The live-LSL counterpart to ``bciciv2a_model_robot_rviz.launch.py``: it swaps the
in-process ``gdf_replay_publisher`` EEG source for an LSL pair, ``lsl_test_outlet``
(synthetic or GDF-over-LSL) feeding ``lsl_eeg_bridge``, and keeps the same
``model_intent_decoder`` -> ``/bci/intent`` -> robot / driver / marker / RViz half.

Hardware-free demo (default): replay A01E.gdf over LSL through the 22-ch model, so
the arm moves from real decoded EEG with no headset::

    scripts/run-lsl

Real headset later: skip the test outlet and load a same-montage 16-ch model::

    scripts/run-lsl start_test_outlet:=false stream_name:=<headset stream> \
        expected_channel_count:=16 model_path:=tmp/hand-16ch.joblib

Relative defaults resolve against the working directory; the run script cd's to
the repo root before launching.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("eeg_bci_pipeline")
    rviz_config = PathJoinSubstitution([package_share, "rviz", "mock_robot.rviz"])
    robot_description_path = PathJoinSubstitution(
        [FindPackageShare("manipulator_control"), "urdf", "panda_visual.urdf"]
    )
    robot_description = ParameterValue(
        Command(["cat ", robot_description_path]),
        value_type=str,
    )

    mode = LaunchConfiguration("mode")
    gdf_path = LaunchConfiguration("gdf_path")
    model_path = LaunchConfiguration("model_path")
    loop = LaunchConfiguration("loop")
    samples_per_frame = LaunchConfiguration("samples_per_frame")
    stream_name = LaunchConfiguration("stream_name")
    stream_type = LaunchConfiguration("stream_type")
    start_test_outlet = LaunchConfiguration("start_test_outlet")
    expected_channel_count = LaunchConfiguration("expected_channel_count")
    expected_sampling_rate_hz = LaunchConfiguration("expected_sampling_rate_hz")
    scale_to_microvolts = LaunchConfiguration("scale_to_microvolts")
    rest_confidence_threshold = LaunchConfiguration("rest_confidence_threshold")
    confidence_threshold = LaunchConfiguration("confidence_threshold")

    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="gdf"),
            DeclareLaunchArgument("gdf_path", default_value="data/raw/bciciv2a/A01E.gdf"),
            DeclareLaunchArgument("model_path", default_value="tmp/hand-csp-lda-A01T.joblib"),
            DeclareLaunchArgument("loop", default_value="true"),
            DeclareLaunchArgument("samples_per_frame", default_value="25"),
            DeclareLaunchArgument("stream_name", default_value="eeg-bci-test"),
            # Clear (stream_type:="") to resolve a headset that advertises a non-EEG type.
            DeclareLaunchArgument("stream_type", default_value="EEG"),
            DeclareLaunchArgument("start_test_outlet", default_value="true"),
            DeclareLaunchArgument("expected_channel_count", default_value="22"),
            DeclareLaunchArgument("expected_sampling_rate_hz", default_value="250.0"),
            # 1.0: the test outlet emits microvolts. A volts headset sets 1000000.0.
            DeclareLaunchArgument("scale_to_microvolts", default_value="1.0"),
            DeclareLaunchArgument("rest_confidence_threshold", default_value="0.6"),
            # Keep the driver gate <= the decoder's rest gate so confident hand
            # intents are not dropped before they reach the joint driver.
            DeclareLaunchArgument("confidence_threshold", default_value="0.55"),
            # EEG source (hardware-free): synthetic / GDF -> LSL. Skipped for a real headset.
            Node(
                package="eeg_bci_pipeline",
                executable="lsl_test_outlet",
                name="lsl_test_outlet",
                condition=IfCondition(start_test_outlet),
                parameters=[
                    {
                        "mode": mode,
                        "gdf_path": gdf_path,
                        "stream_name": stream_name,
                        "loop": ParameterValue(loop, value_type=bool),
                        "samples_per_frame": ParameterValue(samples_per_frame, value_type=int),
                        # Synthetic mode uses these; gdf mode takes them from the recording.
                        # Tying them to the bridge's expected_* keeps the two consistent.
                        "channel_count": ParameterValue(expected_channel_count, value_type=int),
                        "sampling_rate_hz": ParameterValue(
                            expected_sampling_rate_hz, value_type=float
                        ),
                    }
                ],
                output="screen",
            ),
            # Bridge: resolve the LSL stream and republish as EegFrame on /bci/eeg.
            Node(
                package="eeg_bci_pipeline",
                executable="lsl_eeg_bridge",
                name="lsl_eeg_bridge",
                parameters=[
                    {
                        "stream_name": stream_name,
                        "stream_type": stream_type,
                        "expected_channel_count": ParameterValue(
                            expected_channel_count, value_type=int
                        ),
                        "expected_sampling_rate_hz": ParameterValue(
                            expected_sampling_rate_hz, value_type=float
                        ),
                        "scale_to_microvolts": ParameterValue(
                            scale_to_microvolts, value_type=float
                        ),
                    }
                ],
                output="screen",
            ),
            # Decoder + visualization: identical to bciciv2a_model_robot_rviz.launch.py.
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
