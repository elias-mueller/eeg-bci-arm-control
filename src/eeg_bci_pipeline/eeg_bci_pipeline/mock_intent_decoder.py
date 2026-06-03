"""Decode mock EEG frames into intent messages."""

from __future__ import annotations

from typing import Sequence

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from eeg_bci_interfaces.msg import EegFrame, Intent
from eeg_bci_pipeline.decoder import DEFAULT_CLASS_LABELS, decode_mock_intent
from eeg_bci_pipeline.eeg_frame_contract import (
    DEFAULT_EEG_CHANNEL_COUNT,
    DEFAULT_EEG_SAMPLING_RATE_HZ,
    DEFAULT_MAX_ABS_SAMPLE_UV,
    DEFAULT_SAMPLING_RATE_TOLERANCE_HZ,
    EegFrameContractError,
    validate_eeg_frame_payload,
)

SMALL_PEAK_WARNING_INTERVAL_FRAMES = 100
CONTRACT_WARNING_INTERVAL_FRAMES = 100


class MockIntentDecoder(Node):
    """Subscribes to EEG frames and publishes deterministic mock intents."""

    def __init__(self) -> None:
        super().__init__("mock_intent_decoder")
        self.declare_parameter("input_topic", "/bci/eeg")
        self.declare_parameter("output_topic", "/bci/intent")
        self.declare_parameter("class_labels", list(DEFAULT_CLASS_LABELS))
        self.declare_parameter("expected_channel_count", DEFAULT_EEG_CHANNEL_COUNT)
        self.declare_parameter(
            "expected_channel_labels",
            [],
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("expected_sampling_rate_hz", DEFAULT_EEG_SAMPLING_RATE_HZ)
        self.declare_parameter(
            "sampling_rate_tolerance_hz",
            DEFAULT_SAMPLING_RATE_TOLERANCE_HZ,
        )
        self.declare_parameter("max_abs_sample_uv", DEFAULT_MAX_ABS_SAMPLE_UV)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._class_labels = tuple(self.get_parameter("class_labels").value)
        self._expected_channel_count = int(self.get_parameter("expected_channel_count").value)
        self._expected_channel_labels = self._read_optional_labels("expected_channel_labels")
        if self._expected_channel_labels is not None:
            self._expected_channel_count = len(self._expected_channel_labels)
        self._expected_sampling_rate_hz = float(
            self.get_parameter("expected_sampling_rate_hz").value
        )
        self._sampling_rate_tolerance_hz = float(
            self.get_parameter("sampling_rate_tolerance_hz").value
        )
        self._max_abs_sample_uv = float(self.get_parameter("max_abs_sample_uv").value)
        self._small_peak_frame_count = 0
        self._contract_error_count = 0
        self._last_contract_error = ""

        self._publisher = self.create_publisher(Intent, output_topic, 10)
        self._subscription = self.create_subscription(
            EegFrame,
            input_topic,
            self._on_eeg_frame,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Decoding EEG frames from {input_topic} to intents on {output_topic}"
        )

    def _on_eeg_frame(self, frame: EegFrame) -> None:
        try:
            shape = validate_eeg_frame_payload(
                sampling_rate_hz=frame.sampling_rate_hz,
                channel_labels=frame.channel_labels,
                samples=frame.samples,
                expected_channel_count=self._expected_channel_count,
                expected_channel_labels=self._expected_channel_labels,
                expected_sampling_rate_hz=self._expected_sampling_rate_hz,
                sampling_rate_tolerance_hz=self._sampling_rate_tolerance_hz,
                max_abs_sample_uv=self._max_abs_sample_uv,
            )
        except EegFrameContractError as error:
            self._warn_about_contract_error(str(error))
            return

        if self._expected_channel_labels is None:
            # Dev/mock convenience: production model launch should set expected_channel_labels.
            self._expected_channel_labels = self._normalized_labels(frame.channel_labels)
            self.get_logger().info(
                "Learned EEG channel order from first valid frame: "
                f"{list(self._expected_channel_labels)}"
            )
        if self._contract_error_count > 0:
            self.get_logger().info("EEG frame contract recovered")
            self._contract_error_count = 0
            self._last_contract_error = ""
        self._warn_if_suspiciously_small_peak(shape.suspiciously_small_peak)

        prediction = decode_mock_intent(frame.samples, self._class_labels)
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

    def _warn_if_suspiciously_small_peak(self, suspiciously_small_peak: bool) -> None:
        if suspiciously_small_peak:
            self._small_peak_frame_count += 1
        elif self._small_peak_frame_count > 0:
            self.get_logger().info("EEG frame peak amplitude returned to microvolt scale")
            self._small_peak_frame_count = 0
            return

        if self._small_peak_frame_count == 0:
            return
        if self._small_peak_frame_count == 1 or (
            self._small_peak_frame_count % SMALL_PEAK_WARNING_INTERVAL_FRAMES == 0
        ):
            self.get_logger().warn(
                "EEG frame peak amplitude is very small; check whether source data "
                "is in volts instead of microvolts"
            )

    def _read_optional_labels(self, parameter_name: str) -> tuple[str, ...] | None:
        labels = self._normalized_labels(self.get_parameter(parameter_name).value)
        return labels or None

    def _normalized_labels(self, labels: Sequence[str]) -> tuple[str, ...]:
        return tuple(str(label).strip() for label in labels)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MockIntentDecoder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
