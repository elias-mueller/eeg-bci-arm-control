# pyright: basic
# rclpy and the generated message classes are only partially typed; strict mode
# floods this shell with reportUnknown* noise. Basic still catches real mistakes.
"""Record cue-labeled motor-imagery epochs from the live EEG stream.

Thin ROS shell over the pure `calibration` helpers. It subscribes to `/bci/eeg`,
derives the channel labels / sampling rate from the first frame, then runs a
cued protocol per trial -- REST -> CUE (show a left/right arrow in RViz) ->
IMAGERY (accumulate frames) -- collecting one epoch per trial. On completion it
assembles a `LabeledEpochs` and saves it, ready for
`scripts/evaluate-hand-classifier <file> --save-model ...` to train a model on
this montage. Rest is the inter-trial baseline (not a trained class); training
stays 2-class and the runtime synthesizes rest via its confidence threshold.
"""

from __future__ import annotations

from math import cos, isfinite, sin
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from visualization_msgs.msg import Marker

from eeg_bci_interfaces.msg import EegFrame
from eeg_bci_pipeline.calibration import (
    assemble_labeled_epochs,
    build_trial_schedule,
    extract_epoch,
    reshape_frame_to_channel_major,
)
from eeg_bci_pipeline.data.bciciv2a_dataset import save_labeled_epochs
from eeg_bci_pipeline.data.gdf_recording import FloatArray
from eeg_bci_pipeline.intent_marker_mapping import IntentMarkerStyle, style_for_intent
from eeg_bci_pipeline.node_params import (
    float_param,
    int_param,
    str_list_param,
    str_param,
)

REST_STATE = "REST"
CUE_STATE = "CUE"
IMAGERY_STATE = "IMAGERY"
DONE_STATE = "DONE"
TICK_PERIOD_SEC = 0.05
# Collect a little past settle+epoch so timer granularity / frame jitter never
# leaves the imagery buffer short of one full epoch; the extra tail is unused.
IMAGERY_MARGIN_SEC = 0.5


class CalibrateCapture(Node):
    """Runs the cue protocol and records labeled epochs from /bci/eeg."""

    def __init__(self) -> None:
        super().__init__("calibrate_capture")
        self.declare_parameter("eeg_topic", "/bci/eeg")
        self.declare_parameter("marker_topic", "/bci/cue_marker")
        self.declare_parameter("frame_id", "bci_world")
        self.declare_parameter("class_labels", [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("trials_per_class", 20)
        self.declare_parameter("rest_sec", 2.0)
        self.declare_parameter("cue_sec", 1.0)
        self.declare_parameter("settle_sec", 0.5)
        self.declare_parameter("epoch_sec", 3.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("source_id", "calibration")
        self.declare_parameter("output_path", "tmp/calibration-epochs.joblib")

        self._frame_id = str_param(self, "frame_id")
        labels = str_list_param(self, "class_labels")
        self._class_labels = labels or ["left_hand", "right_hand"]
        self._rest_sec = float_param(self, "rest_sec")
        self._cue_sec = float_param(self, "cue_sec")
        self._settle_sec = float_param(self, "settle_sec")
        self._epoch_sec = float_param(self, "epoch_sec")
        self._source_id = str_param(self, "source_id")
        self._output_path = str_param(self, "output_path")
        trials_per_class = int_param(self, "trials_per_class")
        seed = int_param(self, "seed")
        eeg_topic = str_param(self, "eeg_topic")
        marker_topic = str_param(self, "marker_topic")

        self._schedule = build_trial_schedule(trials_per_class, self._class_labels, seed=seed)
        self._records: list[tuple[str, FloatArray]] = []
        self._skipped = 0
        self._trial_index = 0
        self._labels_warned = False
        self._state = "WAIT"
        self._channel_labels: list[str] | None = None
        self._sampling_rate_hz = 0.0
        self._channel_count = 0
        self._samples_per_epoch = 0
        self._settle_offset = 0
        self._imagery_buffer = None
        self._state_deadline = self.get_clock().now()
        self._timer = None

        self._marker_publisher = self.create_publisher(Marker, marker_topic, 10)
        self._subscription = self.create_subscription(
            EegFrame, eeg_topic, self._on_eeg_frame, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Calibration ready: {len(self._schedule)} trials "
            f"({self._class_labels}); waiting for EEG on {eeg_topic}"
        )

    # --- EEG ingestion -----------------------------------------------------

    def _on_eeg_frame(self, frame: EegFrame) -> None:
        if self._channel_labels is None:
            self._initialize_from_frame(frame)
            return
        if list(cast(Sequence[str], frame.channel_labels)) != self._channel_labels:
            if not self._labels_warned:
                self._labels_warned = True
                self.get_logger().warn(
                    "EEG channel labels changed mid-session; dropping mismatched frames"
                )
            return
        if self._state == IMAGERY_STATE:
            self._append_imagery(cast(Sequence[float], frame.samples))

    def _initialize_from_frame(self, frame: EegFrame) -> None:
        channel_labels = list(cast(Sequence[str], frame.channel_labels))
        sampling_rate_hz = float(frame.sampling_rate_hz)
        if not channel_labels or not isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
            self.get_logger().warn(
                "Ignoring EEG frame with empty channel labels or non-positive rate; waiting"
            )
            return
        self._channel_labels = channel_labels
        self._channel_count = len(channel_labels)
        self._sampling_rate_hz = sampling_rate_hz
        self._samples_per_epoch = max(1, round(self._epoch_sec * sampling_rate_hz))
        self._settle_offset = max(0, round(self._settle_sec * sampling_rate_hz))
        self.get_logger().info(
            f"Stream: {self._channel_count} ch at {self._sampling_rate_hz:g} Hz; "
            f"{self._samples_per_epoch} samples/epoch (settle {self._settle_offset}). Starting."
        )
        self._enter_state(REST_STATE)
        self._timer = self.create_timer(TICK_PERIOD_SEC, self._on_tick)

    def _append_imagery(self, samples: Sequence[float]) -> None:
        block = reshape_frame_to_channel_major(samples, self._channel_count)
        if block is None:
            return
        if self._imagery_buffer is None:
            self._imagery_buffer = block
        else:
            self._imagery_buffer = np.concatenate([self._imagery_buffer, block], axis=1)

    # --- Cue state machine -------------------------------------------------

    def _on_tick(self) -> None:
        # The timer is created only after init, so channel_labels is set here; the
        # guard keeps that invariant explicit.
        if self._channel_labels is None or self._state == DONE_STATE:
            return
        remaining = (self._state_deadline - self.get_clock().now()).nanoseconds / 1e9
        self._publish_cue_marker(remaining)
        if remaining <= 0.0:
            self._advance_state()

    def _advance_state(self) -> None:
        if self._state == REST_STATE:
            self._enter_state(CUE_STATE)
        elif self._state == CUE_STATE:
            self._imagery_buffer = None
            self._enter_state(IMAGERY_STATE)
        elif self._state == IMAGERY_STATE:
            self._finish_imagery()

    def _enter_state(self, state: str) -> None:
        self._state = state
        duration = self._state_duration(state)
        self._state_deadline = self.get_clock().now() + Duration(seconds=duration)
        self.get_logger().info(
            f"Trial {self._trial_index + 1}/{len(self._schedule)}: "
            f"{state} ({self._cue_label_for_state(state)}) for {duration:g}s"
        )

    def _finish_imagery(self) -> None:
        label = self._schedule[self._trial_index]
        window = None
        if self._imagery_buffer is not None:
            window = extract_epoch(
                self._imagery_buffer, self._settle_offset, self._samples_per_epoch
            )
        if window is None:
            self._skipped += 1
            self.get_logger().warn(f"Trial {self._trial_index + 1}: too few samples, skipped")
        else:
            self._records.append((label, window))
        self._trial_index += 1
        if self._trial_index >= len(self._schedule):
            self._finish_session()
        else:
            self._enter_state(REST_STATE)

    def _finish_session(self) -> None:
        self._state = DONE_STATE
        if self._timer is not None:
            self._timer.cancel()
        self._clear_cue_marker()
        if not self._records:
            self.get_logger().error(
                "No epochs captured (every trial dropped); nothing saved. Ctrl-C to stop."
            )
            return
        epochs = assemble_labeled_epochs(
            source_id=self._source_id,
            sampling_rate_hz=self._sampling_rate_hz,
            channel_labels=self._channel_labels or [],
            class_labels=self._class_labels,
            records=self._records,
            skipped_epoch_count=self._skipped,
        )
        counts = {label: self._records_count(label) for label in self._class_labels}
        low = [label for label, count in counts.items() if count < 2]
        if low:
            self.get_logger().warn(
                f"Low epoch count for {low} ({counts}); training or CV may fail or be biased. "
                "Re-run with more trials or fewer dropouts."
            )
        output = Path(self._output_path).expanduser()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            save_labeled_epochs(epochs, output)
        except OSError as error:
            self.get_logger().error(f"Failed to save epochs to {output}: {error}")
            return
        self.get_logger().info(
            f"Saved {epochs.epoch_count} epochs {counts} ({self._skipped} skipped) to {output}"
        )
        self.get_logger().info(
            f"Train: scripts/evaluate-hand-classifier {output} --save-model tmp/hand-16ch.joblib"
        )
        self.get_logger().info(
            "Then drive the arm: scripts/run-lsl start_test_outlet:=false "
            "model_path:=tmp/hand-16ch.joblib stream_name:=<headset stream> "
            f"expected_channel_count:={self._channel_count}"
        )
        self.get_logger().info("Calibration complete. Ctrl-C to stop.")

    def _records_count(self, label: str) -> int:
        return sum(1 for record_label, _ in self._records if record_label == label)

    def _state_duration(self, state: str) -> float:
        if state == REST_STATE:
            return self._rest_sec
        if state == CUE_STATE:
            return self._cue_sec
        if state == IMAGERY_STATE:
            return self._settle_sec + self._epoch_sec + IMAGERY_MARGIN_SEC
        return 0.0

    def _cue_label_for_state(self, state: str) -> str:
        if state in (CUE_STATE, IMAGERY_STATE):
            return self._schedule[self._trial_index]
        return "rest"

    # --- RViz cue marker ---------------------------------------------------

    def _publish_cue_marker(self, remaining: float) -> None:
        label = self._cue_label_for_state(self._state)
        style = style_for_intent(label, 1.0)
        stamp = self.get_clock().now().to_msg()
        self._marker_publisher.publish(self._arrow_marker(style, stamp))
        self._marker_publisher.publish(self._text_marker(style, remaining, stamp))

    def _clear_cue_marker(self) -> None:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bci_cue"
        marker.action = Marker.DELETEALL
        self._marker_publisher.publish(marker)

    def _arrow_marker(self, style: IntentMarkerStyle, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = stamp
        marker.ns = "bci_cue"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.z = sin(style.yaw_rad / 2.0)
        marker.pose.orientation.w = cos(style.yaw_rad / 2.0)
        marker.scale.x = style.length
        marker.scale.y = 0.12
        marker.scale.z = 0.18
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = style.color_rgba
        return marker

    def _text_marker(self, style: IntentMarkerStyle, remaining: float, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = stamp
        marker.ns = "bci_cue"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 0.6
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.22
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = style.color_rgba
        marker.text = f"{self._state}: {style.label}  {max(0.0, remaining):.1f}s"
        return marker


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CalibrateCapture()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
