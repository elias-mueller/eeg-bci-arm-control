# pyright: basic
# rclpy and the generated message classes are only partially typed; strict mode
# floods this shell with reportUnknown* noise. Basic still catches real mistakes.
"""Bridge a live LSL EEG stream onto the pipeline's EegFrame topic.

Thin ROS shell over the pure `lsl_bridge` helpers: it resolves an LSL inlet
(BrainAccess MIDI, a replay outlet, or any `type="EEG"` source), reads the
stream's channel labels / rate / unit from its metadata, and republishes each
pulled chunk as a channel-major microvolt `EegFrame` on `/bci/eeg`. The node is
montage-agnostic: it forwards the stream's own channel labels when they are
complete and unique (else it falls back to ch_NN and warns), so the same bridge
serves the 22-ch BCIC replay demo and a future 16-ch headset by changing only
parameters, not code.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from eeg_bci_interfaces.msg import EegFrame
from eeg_bci_pipeline.eeg_frame_contract import (
    DEFAULT_MAX_ABS_SAMPLE_UV,
    DEFAULT_SAMPLING_RATE_TOLERANCE_HZ,
    EegFrameContractError,
    validate_eeg_frame_payload,
)
from eeg_bci_pipeline.lsl_bridge import (
    DEFAULT_RESYNC_THRESHOLD_SEC,
    LslClockAligner,
    build_resolve_predicate,
    chunk_to_channel_major_uv,
    extract_lsl_metadata,
    import_pylsl,
    unit_scale_for,
)
from eeg_bci_pipeline.node_params import bool_param, float_param, int_param, str_param

# pylsl pull_chunk needs a positive max_samples; this is the cap when the
# max_chunk_samples parameter is left at its 0 ("unset") default.
DEFAULT_LSL_MAX_SAMPLES = 1024

# A small cap for the one-off priming pull so startup stays fast and the discarded
# prime is tiny (~0.1 s at 250 Hz), regardless of pull_chunk's fill-vs-drain timing.
PRIME_MAX_SAMPLES = 32


class LslEegBridge(Node):
    """Republishes a resolved LSL EEG stream as EegFrame messages."""

    def __init__(self) -> None:
        super().__init__("lsl_eeg_bridge")
        self.declare_parameter("topic", "/bci/eeg")
        self.declare_parameter("source_id", "")
        self.declare_parameter("stream_type", "EEG")
        self.declare_parameter("stream_name", "")
        self.declare_parameter("resolve_timeout_sec", 5.0)
        self.declare_parameter("pull_period_sec", 0.02)
        self.declare_parameter("pull_timeout_sec", 0.0)
        self.declare_parameter("max_chunk_samples", 0)
        self.declare_parameter("scale_to_microvolts", 1.0)
        self.declare_parameter("validate_frames", True)
        self.declare_parameter("expected_channel_count", 0)
        self.declare_parameter("expected_sampling_rate_hz", 0.0)
        self.declare_parameter("sampling_rate_tolerance_hz", DEFAULT_SAMPLING_RATE_TOLERANCE_HZ)
        self.declare_parameter("max_abs_sample_uv", DEFAULT_MAX_ABS_SAMPLE_UV)
        self.declare_parameter("use_lsl_timestamps", True)
        self.declare_parameter("resync_threshold_sec", DEFAULT_RESYNC_THRESHOLD_SEC)
        self.declare_parameter("reconnect", True)

        self._source_id_param = str_param(self, "source_id")
        self._stream_type = str_param(self, "stream_type")
        self._stream_name = str_param(self, "stream_name")
        self._resolve_timeout_sec = float_param(self, "resolve_timeout_sec")
        self._pull_timeout_sec = float_param(self, "pull_timeout_sec")
        self._max_chunk_samples = int_param(self, "max_chunk_samples")
        self._scale_to_microvolts = float_param(self, "scale_to_microvolts")
        self._validate_frames = bool_param(self, "validate_frames")
        self._expected_channel_count = int_param(self, "expected_channel_count")
        self._expected_sampling_rate_hz = float_param(self, "expected_sampling_rate_hz")
        self._sampling_rate_tolerance_hz = float_param(self, "sampling_rate_tolerance_hz")
        self._max_abs_sample_uv = float_param(self, "max_abs_sample_uv")
        self._use_lsl_timestamps = bool_param(self, "use_lsl_timestamps")
        self._reconnect = bool_param(self, "reconnect")
        topic = str_param(self, "topic")
        pull_period_sec = float_param(self, "pull_period_sec")

        self._dropped_frames = 0
        self._pull_errors = 0
        self._clock_aligner = LslClockAligner(
            resync_threshold_sec=float_param(self, "resync_threshold_sec")
        )

        pylsl = import_pylsl()
        info = self._resolve_stream(pylsl)
        self._apply_metadata(extract_lsl_metadata(info))

        self._publisher = self.create_publisher(EegFrame, topic, qos_profile_sensor_data)
        self._validate_first_frame()
        self._timer = self.create_timer(pull_period_sec, self._on_tick)

    def _resolve_stream(self, pylsl):
        predicate = self._resolve_predicate()
        streams = pylsl.resolve_bypred(predicate, 1, self._resolve_timeout_sec)
        if not streams:
            raise RuntimeError(
                f"No LSL stream matched [{predicate}] within {self._resolve_timeout_sec:g} s"
            )
        # recover=True lets pylsl transparently reconnect transient drops.
        self._inlet = pylsl.StreamInlet(streams[0], recover=self._reconnect)
        # The resolve result may omit the desc; inlet.info() returns the full
        # StreamInfo including the channel labels the decoder contract needs.
        info = self._inlet.info(self._resolve_timeout_sec)
        self.get_logger().info(
            f"Resolved LSL stream '{info.name()}' "
            f"(type={info.type()}, {info.channel_count()} ch at {info.nominal_srate():g} Hz)"
        )
        return info

    def _resolve_predicate(self) -> str:
        # Default stream_type="EEG"; clear it (stream_type:="") to resolve by name
        # only for a headset that advertises a non-standard type. The predicate
        # construction (conjunction, quote rejection) is a pure, tested helper.
        return build_resolve_predicate(self._stream_name, self._stream_type)

    def _apply_metadata(self, meta) -> None:
        if self._expected_channel_count > 0 and meta.channel_count != self._expected_channel_count:
            raise RuntimeError(
                f"LSL stream has {meta.channel_count} channels, "
                f"expected {self._expected_channel_count}"
            )
        rate = self._resolve_rate(meta.nominal_srate_hz)

        self._channel_labels = list(meta.channel_labels)
        self._sampling_rate_hz = rate
        self._unit_scale = unit_scale_for(meta.declared_unit, self._scale_to_microvolts)
        self._source_id = self._source_id_param or meta.source_id
        if meta.labels_downgraded:
            self.get_logger().warn(
                "LSL stream declared channel labels that were incomplete, duplicated, or the "
                f"wrong count; using generic {self._channel_labels[0]}..{self._channel_labels[-1]} "
                "labels instead. A label-strict decoder will reject these frames."
            )
        self.get_logger().info(
            f"Bridging {meta.channel_count} ch at {rate:g} Hz to EegFrame; "
            f"unit_scale={self._unit_scale:g} (declared unit "
            f"'{meta.declared_unit or 'unspecified'}'), source_id='{self._source_id}'"
        )

    def _resolve_rate(self, nominal_srate_hz: float) -> float:
        # With an expected rate set, validate the stream's declared rate against it
        # (an independent check) and adopt it. Otherwise trust the stream's rate;
        # the decoder still enforces its own rate contract downstream.
        if self._expected_sampling_rate_hz > 0.0:
            if (
                abs(nominal_srate_hz - self._expected_sampling_rate_hz)
                > self._sampling_rate_tolerance_hz
            ):
                raise RuntimeError(
                    f"LSL stream rate {nominal_srate_hz:g} Hz differs from expected "
                    f"{self._expected_sampling_rate_hz:g} Hz by more than "
                    f"{self._sampling_rate_tolerance_hz:g} Hz"
                )
            return self._expected_sampling_rate_hz
        if nominal_srate_hz <= 0.0:
            raise RuntimeError(
                "LSL stream rate is irregular or unknown; set expected_sampling_rate_hz"
            )
        return nominal_srate_hz

    def _max_samples(self) -> int:
        return self._max_chunk_samples if self._max_chunk_samples > 0 else DEFAULT_LSL_MAX_SAMPLES

    def _validate_first_frame(self) -> None:
        # Intentional fail-fast: wait up to the resolve timeout for the first samples
        # and validate the contract shape/units, so a misconfigured stream errors at
        # launch instead of dropping every frame. PRIME_MAX_SAMPLES caps this read so
        # startup is fast and the discarded prime is tiny (~0.1 s), regardless of
        # pull_chunk's fill-vs-drain timing. Steady-state frames flow from the timer.
        chunk, _timestamps = self._inlet.pull_chunk(
            timeout=self._resolve_timeout_sec, max_samples=PRIME_MAX_SAMPLES
        )
        if not chunk:
            self.get_logger().warn(
                "No samples received while priming; validating on the first live frame instead"
            )
            return
        samples = chunk_to_channel_major_uv(
            chunk, len(self._channel_labels), unit_scale=self._unit_scale
        )
        self._validate_payload(samples)
        peak = max((abs(value) for value in samples), default=0.0)
        self.get_logger().info(
            f"First frame validated: {len(samples) // len(self._channel_labels)} samples/ch, "
            f"peak {peak:.3g} uV"
        )

    def _on_tick(self) -> None:
        try:
            chunk, timestamps = self._inlet.pull_chunk(
                timeout=self._pull_timeout_sec, max_samples=self._max_samples()
            )
        except Exception as error:  # pylsl.LostError and friends
            self._handle_pull_error(error)
            return
        if not chunk:
            return
        # Guard the conversion too: a wrong-width chunk (e.g. after a transparent
        # reconnect to a different stream) raises ValueError, which would otherwise
        # escape the timer callback and tear the node down.
        try:
            samples = chunk_to_channel_major_uv(
                chunk, len(self._channel_labels), unit_scale=self._unit_scale
            )
        except ValueError as error:
            self._note_dropped_frame(f"malformed chunk: {error}")
            return
        if self._validate_frames and not self._frame_is_valid(samples):
            return
        stamp = self._stamp_for_chunk(timestamps)
        self._publish(samples, stamp)

    def _frame_is_valid(self, samples) -> bool:
        try:
            self._validate_payload(samples)
        except EegFrameContractError as error:
            self._note_dropped_frame(f"fails the EEG contract: {error}")
            return False
        return True

    def _note_dropped_frame(self, reason: str) -> None:
        self._dropped_frames += 1
        if self._dropped_frames == 1 or self._dropped_frames % 50 == 0:
            self.get_logger().warn(
                f"Dropping frame ({self._dropped_frames} dropped so far): {reason}"
            )

    def _validate_payload(self, samples) -> None:
        validate_eeg_frame_payload(
            sampling_rate_hz=self._sampling_rate_hz,
            channel_labels=self._channel_labels,
            samples=samples,
            expected_channel_count=len(self._channel_labels),
            expected_sampling_rate_hz=self._sampling_rate_hz,
            sampling_rate_tolerance_hz=self._sampling_rate_tolerance_hz,
            max_abs_sample_uv=self._max_abs_sample_uv,
        )

    def _publish(self, samples, stamp) -> None:
        frame = EegFrame()
        frame.header.stamp = stamp
        frame.header.frame_id = "eeg"
        frame.source_id = self._source_id
        frame.sampling_rate_hz = self._sampling_rate_hz
        frame.channel_labels = list(self._channel_labels)
        frame.samples = samples
        self._publisher.publish(frame)

    def _stamp_for_chunk(self, timestamps):
        # Map the stream's LSL sample clock onto the ROS clock via the aligner. It
        # uses the chunk's oldest (timestamps[0]) and newest (timestamps[-1]) sample
        # times so a drained backlog is not mistaken for clock drift, and re-anchors
        # on genuine drift so a long session does not accumulate offset. Arrival time
        # is the explicit fallback (timestamps disabled, or a chunk with no LSL stamp)
        # per the EegFrame contract.
        if not self._use_lsl_timestamps or not timestamps or timestamps[0] == 0.0:
            return self.get_clock().now().to_msg()
        stamp_ns = self._clock_aligner.stamp_ns(
            self.get_clock().now().nanoseconds, timestamps[0], timestamps[-1]
        )
        return Time(nanoseconds=stamp_ns).to_msg()

    def _handle_pull_error(self, error) -> None:
        self._clock_aligner.reset()
        if not self._reconnect:
            self._timer.cancel()
            self.get_logger().error(
                f"LSL stream lost and reconnect disabled; bridge stopped: {error}"
            )
            return
        self._pull_errors += 1
        if self._pull_errors == 1 or self._pull_errors % 50 == 0:
            self.get_logger().warn(
                f"LSL pull failed ({self._pull_errors} so far), recovering: {error}"
            )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = LslEegBridge()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
