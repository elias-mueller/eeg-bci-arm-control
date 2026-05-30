"""Decode EEG frames into intents with a trained hand-classifier model.

Thin ROS shell over the pure `model_decode` helpers: it loads a saved
`HandClassifierArtifact`, derives its frame contract (channel labels, sampling
rate, window length) from the artifact, buffers streaming frames into a sliding
epoch window, and publishes a gated intent per frame. Until the window fills it
publishes `rest` so the robot holds position.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from eeg_bci_interfaces.msg import EegFrame, Intent
from eeg_bci_pipeline.decoder import DEFAULT_CLASS_LABELS, IntentPrediction
from eeg_bci_pipeline.eeg_frame_contract import (
    DEFAULT_MAX_ABS_SAMPLE_UV,
    DEFAULT_SAMPLING_RATE_TOLERANCE_HZ,
    EegFrameContractError,
    validate_eeg_frame_payload,
)
from eeg_bci_pipeline.model_decode import (
    DEFAULT_REST_CONFIDENCE_THRESHOLD,
    SlidingEpochBuffer,
    decode_window,
    rest_intent,
)
from eeg_bci_pipeline.training.hand_classifier import load_hand_classifier_artifact

CONTRACT_WARNING_INTERVAL_FRAMES = 100


class ModelIntentDecoder(Node):
    """Subscribes to EEG frames and publishes intents from a trained CSP+LDA model."""

    def __init__(self) -> None:
        super().__init__("model_intent_decoder")
        self.declare_parameter("input_topic", "/bci/eeg")
        self.declare_parameter("output_topic", "/bci/intent")
        self.declare_parameter("model_path", "")
        self.declare_parameter("rest_confidence_threshold", DEFAULT_REST_CONFIDENCE_THRESHOLD)
        self.declare_parameter("sampling_rate_tolerance_hz", DEFAULT_SAMPLING_RATE_TOLERANCE_HZ)
        self.declare_parameter("max_abs_sample_uv", DEFAULT_MAX_ABS_SAMPLE_UV)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        model_path = str(self.get_parameter("model_path").value)
        if not model_path:
            raise ValueError("model_path parameter must point to a saved classifier artifact")
        model_path = str(Path(model_path).expanduser())
        self._rest_confidence_threshold = float(
            self.get_parameter("rest_confidence_threshold").value
        )
        if not 0.0 <= self._rest_confidence_threshold <= 1.0:
            raise ValueError("rest_confidence_threshold must be between 0 and 1")
        self._sampling_rate_tolerance_hz = float(
            self.get_parameter("sampling_rate_tolerance_hz").value
        )
        self._max_abs_sample_uv = float(self.get_parameter("max_abs_sample_uv").value)

        # Loading fails loudly (missing/corrupt/unsupported artifact) — this node is
        # only launched when real model decoding is intended.
        self._artifact = load_hand_classifier_artifact(model_path)
        self._runtime_class_labels = DEFAULT_CLASS_LABELS
        self._buffer = SlidingEpochBuffer(
            self._artifact.channel_count,
            self._artifact.samples_per_epoch,
        )
        self._contract_error_count = 0
        self._last_contract_error = ""
        self._decode_error_count = 0
        self._last_decode_error = ""
        self._window_filled = False

        self._publisher = self.create_publisher(Intent, output_topic, 10)
        self._subscription = self.create_subscription(
            EegFrame,
            input_topic,
            self._on_eeg_frame,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Loaded hand classifier '{self._artifact.source_id}' "
            f"({self._artifact.classifier_name}): {self._artifact.channel_count} channels "
            f"x {self._artifact.samples_per_epoch} samples at "
            f"{self._artifact.sampling_rate_hz:g} Hz"
        )
        self.get_logger().info(
            f"Decoding EEG frames from {input_topic} to intents on {output_topic} "
            f"(rest threshold {self._rest_confidence_threshold:g})"
        )

    def _on_eeg_frame(self, frame: EegFrame) -> None:
        try:
            validate_eeg_frame_payload(
                sampling_rate_hz=frame.sampling_rate_hz,
                channel_labels=frame.channel_labels,
                samples=frame.samples,
                expected_channel_labels=self._artifact.channel_labels,
                expected_sampling_rate_hz=self._artifact.sampling_rate_hz,
                sampling_rate_tolerance_hz=self._sampling_rate_tolerance_hz,
                max_abs_sample_uv=self._max_abs_sample_uv,
            )
        except EegFrameContractError as error:
            # Drop buffered samples so windows don't stitch across the gap on recovery.
            self._buffer.reset()
            self._window_filled = False
            self._warn_about_contract_error(str(error))
            return

        if self._contract_error_count > 0:
            self.get_logger().info("EEG frame contract recovered")
            self._contract_error_count = 0
            self._last_contract_error = ""

        window = self._buffer.push(frame.samples)
        if window is None:
            self._publish(frame, rest_intent(self._runtime_class_labels))
            return
        if not self._window_filled:
            self.get_logger().info("Sliding window filled; emitting model intents")
            self._window_filled = True

        try:
            prediction = decode_window(
                self._artifact,
                window,
                runtime_class_labels=self._runtime_class_labels,
                rest_threshold=self._rest_confidence_threshold,
            )
        except Exception as error:  # keep the decoder alive; hold at rest on any decode failure
            self._warn_about_decode_error(str(error))
            prediction = rest_intent(self._runtime_class_labels)
        self._publish(frame, prediction)

    def _publish(self, frame: EegFrame, prediction: IntentPrediction) -> None:
        intent = Intent()
        intent.header = frame.header
        intent.label = prediction.label
        intent.confidence = prediction.confidence
        intent.class_labels = list(prediction.class_labels)
        intent.probabilities = list(prediction.probabilities)
        self._publisher.publish(intent)

    def _warn_about_contract_error(self, error_message: str) -> None:
        self._contract_error_count += 1
        changed_error = error_message != self._last_contract_error
        self._last_contract_error = error_message
        if (
            changed_error
            or self._contract_error_count == 1
            or self._contract_error_count % CONTRACT_WARNING_INTERVAL_FRAMES == 0
        ):
            self.get_logger().warn(
                "Ignoring EEG frame that violates contract "
                f"({self._contract_error_count}): {error_message}"
            )

    def _warn_about_decode_error(self, error_message: str) -> None:
        self._decode_error_count += 1
        changed_error = error_message != self._last_decode_error
        self._last_decode_error = error_message
        if (
            changed_error
            or self._decode_error_count == 1
            or self._decode_error_count % CONTRACT_WARNING_INTERVAL_FRAMES == 0
        ):
            self.get_logger().error(
                f"Decode failed; holding at rest ({self._decode_error_count}): {error_message}"
            )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ModelIntentDecoder()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
