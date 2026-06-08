"""Record a calibration session from a live (or test) LSL EEG stream.

Composes the LSL source (an optional test outlet + the lsl_eeg_bridge) with the
calibrate_capture node and an RViz cue display. For a real BrainAccess session,
stream the headset to LSL and run with the test outlet off (the default),
pointing stream_name at the headset's stream::

    scripts/run-calibrate stream_name:=<headset stream>

Hardware-free plumbing test (no headset): drive /bci/eeg from GDF-over-LSL::

    scripts/run-calibrate start_test_outlet:=true mode:=gdf trials_per_class:=2

Relative defaults resolve against the working directory; the run script cd's to
the repo root before launching.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("eeg_bci_pipeline"), "rviz", "calibrate.rviz"]
    )

    mode = LaunchConfiguration("mode")
    gdf_path = LaunchConfiguration("gdf_path")
    stream_name = LaunchConfiguration("stream_name")
    stream_type = LaunchConfiguration("stream_type")
    scale_to_microvolts = LaunchConfiguration("scale_to_microvolts")
    select_channel_type = LaunchConfiguration("select_channel_type")
    highpass_hz = LaunchConfiguration("highpass_hz")
    expected_channel_count = LaunchConfiguration("expected_channel_count")
    source_id = LaunchConfiguration("source_id")
    start_test_outlet = LaunchConfiguration("start_test_outlet")
    trials_per_class = LaunchConfiguration("trials_per_class")
    output_path = LaunchConfiguration("output_path")
    rest_sec = LaunchConfiguration("rest_sec")
    cue_sec = LaunchConfiguration("cue_sec")
    settle_sec = LaunchConfiguration("settle_sec")
    epoch_sec = LaunchConfiguration("epoch_sec")
    seed = LaunchConfiguration("seed")

    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="gdf"),
            DeclareLaunchArgument("gdf_path", default_value="data/raw/bciciv2a/A01E.gdf"),
            DeclareLaunchArgument("stream_name", default_value="eeg-bci-test"),
            DeclareLaunchArgument("stream_type", default_value="EEG"),
            # 1.0: the test outlet (and a unit-declaring headset) emit microvolts. Set
            # 1000000.0 for a headset that streams volts with no declared LSL unit.
            DeclareLaunchArgument("scale_to_microvolts", default_value="1.0"),
            # Empty keeps every channel. A BrainAccess headset sets "EEG" to drop its
            # contact / accelerometer / battery channels down to the EEG montage.
            DeclareLaunchArgument("select_channel_type", default_value=""),
            # 0 disables the DC blocker. A dry headset sets ~0.5 to strip the
            # electrode offset before the amplitude contract / epoch capture.
            DeclareLaunchArgument("highpass_hz", default_value="0.0"),
            # 0 accepts any channel count; with select_channel_type:=EEG set e.g. 16
            # to guard the *selected* EEG count (the headset's motor montage).
            DeclareLaunchArgument("expected_channel_count", default_value="0"),
            DeclareLaunchArgument("source_id", default_value="calibration"),
            # Default off: a real session streams the headset. Set true for a no-headset test.
            DeclareLaunchArgument("start_test_outlet", default_value="false"),
            DeclareLaunchArgument("trials_per_class", default_value="20"),
            DeclareLaunchArgument("output_path", default_value="tmp/calibration-epochs.joblib"),
            DeclareLaunchArgument("rest_sec", default_value="2.0"),
            DeclareLaunchArgument("cue_sec", default_value="1.0"),
            DeclareLaunchArgument("settle_sec", default_value="0.5"),
            DeclareLaunchArgument("epoch_sec", default_value="3.0"),
            DeclareLaunchArgument("seed", default_value="0"),
            # Optional EEG source for a no-headset run (synthetic / GDF over LSL).
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
                    }
                ],
                output="screen",
            ),
            # Bridge: resolve the headset/outlet stream onto /bci/eeg (any channel count).
            Node(
                package="eeg_bci_pipeline",
                executable="lsl_eeg_bridge",
                name="lsl_eeg_bridge",
                parameters=[
                    {
                        "stream_name": stream_name,
                        "stream_type": stream_type,
                        "scale_to_microvolts": ParameterValue(
                            scale_to_microvolts, value_type=float
                        ),
                        "select_channel_type": select_channel_type,
                        "highpass_hz": ParameterValue(highpass_hz, value_type=float),
                        "expected_channel_count": ParameterValue(
                            expected_channel_count, value_type=int
                        ),
                    }
                ],
                output="screen",
            ),
            # Cue protocol + epoch recorder. class_labels / eeg_topic / marker_topic are
            # intentionally not launch args (the wired flow is fixed to left/right);
            # override with `--ros-args -p` if needed.
            Node(
                package="eeg_bci_pipeline",
                executable="calibrate_capture",
                name="calibrate_capture",
                parameters=[
                    {
                        "trials_per_class": ParameterValue(trials_per_class, value_type=int),
                        "output_path": output_path,
                        "source_id": source_id,
                        "rest_sec": ParameterValue(rest_sec, value_type=float),
                        "cue_sec": ParameterValue(cue_sec, value_type=float),
                        "settle_sec": ParameterValue(settle_sec, value_type=float),
                        "epoch_sec": ParameterValue(epoch_sec, value_type=float),
                        "seed": ParameterValue(seed, value_type=int),
                    }
                ],
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
