"""Node-level coverage for the `lsl_eeg_bridge` ROS shell.

The pure LSL helpers (predicate, metadata walk, chunk transpose, clock aligner)
are covered in `test_lsl_bridge.py`; this drives the `LslEegBridge` node itself.
It substitutes a hand-written FakePylsl that carries real data: `resolve_bypred`
hands back a `FakeStreamInfo` (the same desc idiom as the helper tests), and the
fake inlet's `pull_chunk` returns canned sample-major chunks plus LSL timestamps,
so the bridge runs its real resolve -> metadata -> publish path against an
inlet's actual API rather than a mocked-out internal.

Construction is in-process: `rclpy.init(args=[...])` carries the per-test
parameter overrides, the node is built with `import_pylsl` monkeypatched to the
fake, private callbacks (`_on_tick`) are called directly with no spin/sleep, and
published frames are captured by replacing `_publisher.publish` with a list
append. Skipped when ROS (rclpy / the interfaces) is unavailable.
"""

from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("eeg_bci_interfaces.msg")

import eeg_bci_pipeline.lsl_eeg_bridge as bridge_mod  # noqa: E402
from eeg_bci_pipeline.lsl_eeg_bridge import LslEegBridge  # noqa: E402

# A small, exact montage so a transpose/scale bug is detectable per channel.
LABELS = ("C3", "Cz", "C4")
RATE_HZ = 250.0


class FakeXml:
    """Minimal stand-in for pylsl's XMLElement (a libxml node wrapper)."""

    def __init__(self, *, values=None, children=None, nxt=None, empty=False):
        self._values = values or {}
        self._children = children or {}
        self._next = nxt
        self._empty = empty

    def child(self, name):
        return self._children.get(name, _EMPTY_XML)

    def child_value(self, name):
        return self._values.get(name, "")

    def next_sibling(self):
        return self._next if self._next is not None else _EMPTY_XML

    def empty(self):
        return self._empty


_EMPTY_XML = FakeXml(empty=True)


def _make_desc(channels):
    """Build a desc tree from [(label, unit[, type]), ...], linking channel siblings."""

    node = None
    for entry in reversed(channels):
        label, unit = entry[0], entry[1]
        values = {"label": label, "unit": unit}
        if len(entry) > 2 and entry[2]:
            values["type"] = entry[2]
        node = FakeXml(values=values, nxt=node)
    channels_el = FakeXml(children={"channel": node} if node is not None else {})
    return FakeXml(children={"channels": channels_el})


class FakeStreamInfo:
    """Minimal stand-in for pylsl's StreamInfo."""

    def __init__(
        self,
        *,
        channel_count,
        nominal_srate=RATE_HZ,
        name="eeg-bci-test",
        stype="EEG",
        source_id="lsl-test-outlet",
        channels=None,
    ):
        self._channel_count = channel_count
        self._nominal_srate = nominal_srate
        self._name = name
        self._type = stype
        self._source_id = source_id
        self._desc = _make_desc(channels or [])

    def channel_count(self):
        return self._channel_count

    def nominal_srate(self):
        return self._nominal_srate

    def name(self):
        return self._name

    def type(self):
        return self._type

    def source_id(self):
        return self._source_id

    def desc(self):
        return self._desc


class FakeInlet:
    """Stand-in for pylsl.StreamInlet: serves canned chunks and records the open."""

    def __init__(self, info, *, chunks, recover):
        self._info = info
        self._recover = recover
        # Each entry is (chunk, timestamps) or an Exception to raise on pull.
        self._chunks = list(chunks)

    def info(self, timeout=None):
        return self._info

    def pull_chunk(self, timeout, max_samples):
        # Record the most recent caller-supplied max_samples so a test can pin the
        # default-vs-param cap that the node hands pylsl.
        self.last_max_samples = max_samples
        self.last_timeout = timeout
        if not self._chunks:
            return [], []
        item = self._chunks.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakePylsl:
    """Stand-in for the pylsl module the node pulls in via import_pylsl()."""

    def __init__(self, info, *, chunks, streams=None):
        self._info = info
        self._chunks = chunks
        self._streams = [info] if streams is None else streams
        self.resolve_calls = []

    def resolve_bypred(self, predicate, minimum, timeout):
        self.resolve_calls.append((predicate, minimum, timeout))
        return list(self._streams)

    def StreamInlet(self, stream, recover):
        # The resolve result is the same FakeStreamInfo; inlet.info() returns it.
        self.last_inlet = FakeInlet(self._info, chunks=self._chunks, recover=recover)
        return self.last_inlet


def _info(channel_count=len(LABELS), nominal_srate=RATE_HZ, unit="microvolts", **kw):
    channels = [(label, unit) for label in LABELS[:channel_count]]
    # Pad with extra distinct labels if asked for more channels than LABELS holds.
    while len(channels) < channel_count:
        channels.append((f"X{len(channels)}", unit))
    return FakeStreamInfo(
        channel_count=channel_count, nominal_srate=nominal_srate, channels=channels, **kw
    )


def _build(monkeypatch, *, info=None, chunks=(), overrides=None, streams=None):
    """Init rclpy with the param overrides, monkeypatch import_pylsl, build node.

    Returns (node, fake_pylsl). The caller must destroy the node and shutdown;
    `_make` wraps that in a try/finally for the common case.
    """

    info = info if info is not None else _info()
    fake = FakePylsl(info, chunks=list(chunks), streams=streams)
    monkeypatch.setattr(bridge_mod, "import_pylsl", lambda: fake)
    args = ["--ros-args"]
    for name, value in (overrides or {}).items():
        args += ["-p", f"{name}:={value}"]
    rclpy.init(args=args)
    node = LslEegBridge()
    return node, fake


def _capture(node):
    """Replace the publisher's publish with a list-appending capture."""

    published = []
    node._publisher.publish = published.append
    return published


# --- init / resolve / metadata ----------------------------------------------


def test_init_resolves_stream_and_applies_metadata(monkeypatch):
    # A clean stream: resolve hands back the FakeStreamInfo, the node adopts its
    # labels/rate/unit and stands up a publisher. The prime pull validates a first
    # frame, so feed one in-range chunk for it to consume.
    chunk = [[1.0, 2.0, 3.0]]  # 1 sample x 3 channels, microvolts
    node, fake = _build(monkeypatch, chunks=[(chunk, [50.0])])
    try:
        assert node._channel_labels == list(LABELS)
        assert node._sampling_rate_hz == pytest.approx(RATE_HZ)
        assert node._unit_scale == pytest.approx(1.0)  # declared microvolts -> 1x
        assert node._source_id == "lsl-test-outlet"
        assert node._publisher is not None
        # The resolve used the default EEG-type predicate, once.
        assert len(fake.resolve_calls) == 1
        assert fake.resolve_calls[0][0] == "type='EEG'"
        # recover follows the reconnect param (default True).
        assert fake.last_inlet._recover is True
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_init_scales_volts_to_microvolts(monkeypatch):
    # A volts-declared stream gets a 1e6 unit scale so the prime/first-frame
    # validation sees microvolt-range values, not 1e-6 ones.
    chunk = [[1e-6, 2e-6, 3e-6]]
    node, _ = _build(monkeypatch, info=_info(unit="volts"), chunks=[(chunk, [50.0])])
    try:
        assert node._unit_scale == pytest.approx(1e6)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_init_source_id_param_overrides_stream(monkeypatch):
    chunk = [[1.0, 2.0, 3.0]]
    node, _ = _build(monkeypatch, chunks=[(chunk, [50.0])], overrides={"source_id": "headset-9"})
    try:
        assert node._source_id == "headset-9"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_init_warns_and_keeps_generic_labels_when_downgraded(monkeypatch):
    # A stream that declares duplicate labels is downgraded to ch_NN; the node must
    # adopt the generic labels (and warn). Pins the labels_downgraded branch.
    info = FakeStreamInfo(channel_count=3, channels=[("C3", "uV"), ("C3", "uV"), ("Cz", "uV")])
    chunk = [[1.0, 2.0, 3.0]]
    node, _ = _build(monkeypatch, info=info, chunks=[(chunk, [50.0])])
    try:
        assert node._channel_labels == ["ch_01", "ch_02", "ch_03"]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_init_wrong_channel_count_raises(monkeypatch):
    # expected_channel_count set, stream has fewer -> fail fast at launch.
    rclpy_inited = False
    fake = FakePylsl(_info(channel_count=3), chunks=[])
    monkeypatch.setattr(bridge_mod, "import_pylsl", lambda: fake)
    rclpy.init(args=["--ros-args", "-p", "expected_channel_count:=8"])
    rclpy_inited = True
    try:
        with pytest.raises(RuntimeError, match="3 channels, expected 8"):
            LslEegBridge()
    finally:
        if rclpy_inited:
            rclpy.shutdown()


def test_init_rate_mismatch_beyond_tolerance_raises(monkeypatch):
    # Stream declares 200 Hz but we expect 250 +/- 0.5 -> reject.
    fake = FakePylsl(_info(nominal_srate=200.0), chunks=[])
    monkeypatch.setattr(bridge_mod, "import_pylsl", lambda: fake)
    rclpy.init(args=["--ros-args", "-p", "expected_sampling_rate_hz:=250.0"])
    try:
        with pytest.raises(RuntimeError, match="differs from expected"):
            LslEegBridge()
    finally:
        rclpy.shutdown()


def test_init_rate_within_tolerance_adopts_expected(monkeypatch):
    # 249.6 Hz declared, expected 250 +/- 0.5 -> in band, adopt the expected rate.
    chunk = [[1.0, 2.0, 3.0]]
    fake = FakePylsl(_info(nominal_srate=249.6), chunks=[(chunk, [50.0])])
    monkeypatch.setattr(bridge_mod, "import_pylsl", lambda: fake)
    rclpy.init(args=["--ros-args", "-p", "expected_sampling_rate_hz:=250.0"])
    try:
        node = LslEegBridge()
        try:
            assert node._sampling_rate_hz == pytest.approx(250.0)
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()


def test_init_irregular_rate_without_expected_raises(monkeypatch):
    # No expected rate and the stream declares an irregular/unknown rate (0) -> the
    # node cannot stamp frames, so it fails fast pointing at expected_sampling_rate_hz.
    fake = FakePylsl(_info(nominal_srate=0.0), chunks=[])
    monkeypatch.setattr(bridge_mod, "import_pylsl", lambda: fake)
    rclpy.init()
    try:
        with pytest.raises(RuntimeError, match="irregular or unknown"):
            LslEegBridge()
    finally:
        rclpy.shutdown()


def test_init_no_stream_resolved_raises(monkeypatch):
    fake = FakePylsl(_info(), chunks=[], streams=[])
    monkeypatch.setattr(bridge_mod, "import_pylsl", lambda: fake)
    rclpy.init()
    try:
        with pytest.raises(RuntimeError, match="No LSL stream matched"):
            LslEegBridge()
    finally:
        rclpy.shutdown()


# --- priming / first-frame validation ---------------------------------------


def test_prime_empty_response_warns_and_defers(monkeypatch):
    # No samples available while priming: the node must not error; it defers the
    # contract check to the first live frame (the empty-chunk prime branch).
    node, _ = _build(monkeypatch, chunks=[([], [])])
    try:
        # The node still came up with a publisher and timer.
        assert node._timer is not None
        assert node._dropped_frames == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_prime_validates_first_chunk_and_rejects_bad_units(monkeypatch):
    # The prime pull returns out-of-range values (way above max_abs_sample_uv), so
    # the first-frame contract check fails and the node refuses to launch.
    chunk = [[1e9, 2e9, 3e9]]
    fake = FakePylsl(_info(), chunks=[(chunk, [50.0])])
    monkeypatch.setattr(bridge_mod, "import_pylsl", lambda: fake)
    rclpy.init()
    try:
        with pytest.raises(Exception) as excinfo:
            LslEegBridge()
        # Match the discriminating amplitude-bound phrase, not a generic "sample"
        # substring (which also appears in unrelated contract errors), so a
        # shape/emptiness regression in the prime path can't satisfy this test.
        assert "less than or equal to" in str(excinfo.value)
    finally:
        rclpy.shutdown()


def test_prime_caps_read_at_prime_max_samples(monkeypatch):
    # The one-off prime pull uses PRIME_MAX_SAMPLES, not the steady-state cap.
    chunk = [[1.0, 2.0, 3.0]]
    node, fake = _build(monkeypatch, chunks=[(chunk, [50.0])])
    try:
        # After init the inlet's last pull was the prime (no _on_tick yet).
        assert fake.last_inlet.last_max_samples == bridge_mod.PRIME_MAX_SAMPLES
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --- _on_tick publish path ---------------------------------------------------


def test_on_tick_publishes_transposed_scaled_eeg_frame(monkeypatch):
    # 2 samples x 3 channels, sample-major. Channel-major microvolt output must be
    # [ch0_s0, ch0_s1, ch1_s0, ch1_s1, ch2_s0, ch2_s1].
    prime = ([[0.0, 0.0, 0.0]], [50.0])
    tick = ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [51.0, 51.004])
    node, _ = _build(monkeypatch, chunks=[prime, tick])
    published = _capture(node)
    try:
        node._on_tick()
        assert len(published) == 1
        frame = published[0]
        assert frame.header.frame_id == "eeg"
        assert frame.channel_labels == list(LABELS)
        assert frame.sampling_rate_hz == pytest.approx(RATE_HZ)
        assert list(frame.samples) == pytest.approx([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        assert frame.source_id == "lsl-test-outlet"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_empty_chunk_does_not_publish(monkeypatch):
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    node, _ = _build(monkeypatch, chunks=[prime, ([], [])])
    published = _capture(node)
    try:
        node._on_tick()
        assert published == []
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_malformed_chunk_width_dropped_and_counted(monkeypatch):
    # A chunk with the wrong per-sample width (4 cols for a 3-channel stream) raises
    # ValueError inside the transpose; the node drops + counts it, no publish.
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    bad = ([[1.0, 2.0, 3.0, 4.0]], [51.0])
    node, _ = _build(monkeypatch, chunks=[prime, bad])
    published = _capture(node)
    try:
        node._on_tick()
        assert published == []
        assert node._dropped_frames == 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_contract_failing_frame_dropped_and_counted(monkeypatch):
    # A well-shaped chunk whose values exceed max_abs_sample_uv fails the contract;
    # validate_frames is on by default, so the frame is dropped + counted, no publish.
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    huge = ([[1e9, 2e9, 3e9]], [51.0])
    node, _ = _build(monkeypatch, chunks=[prime, huge])
    published = _capture(node)
    try:
        node._on_tick()
        assert published == []
        assert node._dropped_frames == 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_validation_disabled_publishes_unchecked(monkeypatch):
    # With validate_frames:=false the contract check is skipped, so even an
    # out-of-range frame is published (the not-_validate_frames branch).
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    huge = ([[1e9, 2e9, 3e9]], [51.0])
    node, _ = _build(monkeypatch, chunks=[prime, huge], overrides={"validate_frames": "false"})
    published = _capture(node)
    try:
        node._on_tick()
        assert len(published) == 1
        assert node._dropped_frames == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --- timestamp mapping -------------------------------------------------------


def test_on_tick_uses_lsl_timestamps_via_aligner(monkeypatch):
    # use_lsl_timestamps (default True): the first stamped chunk anchors the LSL
    # clock to ROS arrival, so the published stamp equals the ROS arrival time the
    # aligner saw, not the raw LSL value. Pin that the aligner path ran by checking
    # the aligner's internal anchor was set.
    prime = ([[0.0, 0.0, 0.0]], [50.0])
    tick = ([[1.0, 2.0, 3.0]], [51.0])
    node, _ = _build(monkeypatch, chunks=[prime, tick])
    published = _capture(node)
    try:
        assert node._clock_aligner._ros_anchor_ns is None
        node._on_tick()
        assert len(published) == 1
        # The aligner anchored on this chunk's LSL timestamp.
        assert node._clock_aligner._ros_anchor_ns is not None
        assert node._clock_aligner._lsl_anchor_sec == pytest.approx(51.0)
        # The published frame carries the aligner-mapped ROS stamp (the anchor),
        # not the raw LSL value, confirming the timestamp path drove the output.
        stamp = published[0].header.stamp
        assert stamp.sec * 1_000_000_000 + stamp.nanosec == node._clock_aligner._ros_anchor_ns
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_zero_lsl_timestamp_falls_back_to_arrival(monkeypatch):
    # A chunk whose first LSL stamp is 0.0 (no LSL stamp) must use arrival time, not
    # the aligner: the aligner anchor stays unset.
    prime = ([[0.0, 0.0, 0.0]], [50.0])
    tick = ([[1.0, 2.0, 3.0]], [0.0])
    node, _ = _build(monkeypatch, chunks=[prime, tick])
    published = _capture(node)
    try:
        node._on_tick()
        assert len(published) == 1
        assert node._clock_aligner._ros_anchor_ns is None
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_timestamps_disabled_uses_arrival_time(monkeypatch):
    # use_lsl_timestamps:=false routes every chunk through arrival time; the aligner
    # is never consulted even with a non-zero LSL stamp.
    prime = ([[0.0, 0.0, 0.0]], [50.0])
    tick = ([[1.0, 2.0, 3.0]], [51.0])
    node, _ = _build(monkeypatch, chunks=[prime, tick], overrides={"use_lsl_timestamps": "false"})
    published = _capture(node)
    try:
        node._on_tick()
        assert len(published) == 1
        assert node._clock_aligner._ros_anchor_ns is None
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --- pull error handling -----------------------------------------------------


def test_on_tick_pull_error_with_reconnect_resets_and_counts(monkeypatch):
    # A pull that raises (e.g. LostError) with reconnect=True resets the aligner,
    # counts the error, warns, and leaves the timer running so recovery can happen.
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    node, _ = _build(monkeypatch, chunks=[prime, RuntimeError("lost")])
    published = _capture(node)
    try:
        # Seed the aligner so we can observe the reset clearing its anchor.
        node._clock_aligner.stamp_ns(1_000_000_000, 50.0)
        assert node._clock_aligner._ros_anchor_ns is not None
        node._on_tick()
        assert published == []
        assert node._pull_errors == 1
        assert node._clock_aligner._ros_anchor_ns is None  # reset() ran
        assert not node._timer.is_canceled()  # timer kept running
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_pull_error_with_reconnect_disabled_cancels_timer(monkeypatch):
    # reconnect:=false: a pull error is terminal, the node cancels its timer and
    # logs an error rather than counting a recoverable pull failure.
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    node, fake = _build(
        monkeypatch, chunks=[prime, RuntimeError("lost")], overrides={"reconnect": "false"}
    )
    published = _capture(node)
    try:
        # recover was passed through as False to the inlet.
        assert fake.last_inlet._recover is False
        node._on_tick()
        assert published == []
        assert node._pull_errors == 0  # not counted as recoverable
        assert node._timer.is_canceled()  # bridge stopped
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --- dropped-frame log throttling (count IS the behavior) --------------------


def test_dropped_frame_warning_is_throttled(monkeypatch):
    # _note_dropped_frame warns on the 1st drop and then every 50th. Capture the
    # logger's warn calls and feed malformed chunks: warnings fire at drop 1 and 50
    # but not on the in-between drops. The count itself is the throttle behavior.
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    node, _ = _build(monkeypatch, chunks=[prime])
    warns = []
    node.get_logger().warn = lambda msg, *a, **k: warns.append(msg)
    try:
        for _ in range(50):
            node._note_dropped_frame("malformed chunk: boom")
        assert node._dropped_frames == 50
        # Exactly two warnings: the first drop and the 50th.
        assert len(warns) == 2
        assert "1 dropped" in warns[0]
        assert "50 dropped" in warns[1]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_pull_error_warning_is_throttled(monkeypatch):
    # _handle_pull_error warns on the 1st failure and then every 50th, same throttle.
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    node, _ = _build(monkeypatch, chunks=[prime])
    warns = []
    node.get_logger().warn = lambda msg, *a, **k: warns.append(msg)
    try:
        for _ in range(50):
            node._handle_pull_error(RuntimeError("lost"))
        assert node._pull_errors == 50
        assert len(warns) == 2
        assert "1 so far" in warns[0]
        assert "50 so far" in warns[1]
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --- max_samples cap ---------------------------------------------------------


def test_max_samples_uses_default_when_param_unset(monkeypatch):
    # max_chunk_samples defaults to 0 ("unset") -> the steady-state pull uses the
    # DEFAULT_LSL_MAX_SAMPLES cap.
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    tick = ([[1.0, 2.0, 3.0]], [51.0])
    node, fake = _build(monkeypatch, chunks=[prime, tick])
    _capture(node)
    try:
        node._on_tick()
        assert fake.last_inlet.last_max_samples == bridge_mod.DEFAULT_LSL_MAX_SAMPLES
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_max_samples_respects_param_when_set(monkeypatch):
    prime = ([[1.0, 2.0, 3.0]], [50.0])
    tick = ([[1.0, 2.0, 3.0]], [51.0])
    node, fake = _build(monkeypatch, chunks=[prime, tick], overrides={"max_chunk_samples": "64"})
    _capture(node)
    try:
        assert node._max_samples() == 64
        node._on_tick()
        assert fake.last_inlet.last_max_samples == 64
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --- channel selection by type -----------------------------------------------


def _mixed_info(channel_count=5):
    # A BrainAccess-like outlet: EEG channels plus aux (accel / battery) channels.
    channels = [
        ("C3", "microvolts", "EEG"),
        ("Cz", "microvolts", "EEG"),
        ("C4", "microvolts", "EEG"),
        ("Accel_x", "g", "Accel"),
        ("Battery", "pct", "Battery"),
    ][:channel_count]
    return FakeStreamInfo(channel_count=channel_count, channels=channels)


def test_init_selects_channels_by_type(monkeypatch):
    # select_channel_type:=EEG keeps only the EEG columns (with their labels) and
    # records the keep indices + the full source width for the chunk subset.
    prime = ([[1.0, 2.0, 3.0, 9.0, 99.0]], [50.0])
    node, _ = _build(
        monkeypatch, info=_mixed_info(), chunks=[prime], overrides={"select_channel_type": "EEG"}
    )
    try:
        assert node._channel_labels == ["C3", "Cz", "C4"]
        assert node._keep_indices == (0, 1, 2)
        assert node._source_channel_count == 5
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_on_tick_publishes_only_selected_channels(monkeypatch):
    # The published frame carries only the EEG columns, transposed to channel-major.
    prime = ([[0.0, 0.0, 0.0, 0.0, 0.0]], [50.0])
    tick = ([[1.0, 2.0, 3.0, 7.0, 88.0], [4.0, 5.0, 6.0, 8.0, 89.0]], [51.0, 51.004])
    node, _ = _build(
        monkeypatch,
        info=_mixed_info(),
        chunks=[prime, tick],
        overrides={"select_channel_type": "EEG"},
    )
    published = _capture(node)
    try:
        node._on_tick()
        assert len(published) == 1
        frame = published[0]
        assert frame.channel_labels == ["C3", "Cz", "C4"]
        # EEG columns 0,1,2 only: channel-major [C3_s0, C3_s1, Cz_s0, Cz_s1, C4_s0, C4_s1].
        assert list(frame.samples) == pytest.approx([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_init_selected_channel_count_mismatch_raises(monkeypatch):
    # expected_channel_count validates the *selected* count: 3 EEG channels but the
    # user expects 4 -> fail fast, naming the selection so the cause is obvious.
    fake = FakePylsl(_mixed_info(), chunks=[])
    monkeypatch.setattr(bridge_mod, "import_pylsl", lambda: fake)
    rclpy.init(
        args=["--ros-args", "-p", "select_channel_type:=EEG", "-p", "expected_channel_count:=4"]
    )
    try:
        with pytest.raises(RuntimeError, match="after selecting 'EEG'"):
            LslEegBridge()
    finally:
        rclpy.shutdown()


# --- causal high-pass (DC blocker) -------------------------------------------


def test_highpass_removes_dc_offset_before_contract(monkeypatch):
    # A stream on a 200 mV DC pedestal (far over the 10000 uV ceiling) with small AC:
    # highpass_hz:=0.5 centers it so the contract passes and the published frame is
    # the AC, not the pedestal. Without the high-pass this stream cannot launch.
    prime = ([[200000.0, 200000.0, 200000.0]], [50.0])
    tick = ([[200000.0, 200000.0, 200000.0], [200010.0, 200000.0, 199990.0]], [51.0, 51.004])
    node, _ = _build(monkeypatch, chunks=[prime, tick], overrides={"highpass_hz": "0.5"})
    published = _capture(node)
    try:
        assert node._highpass is not None
        node._on_tick()
        assert len(published) == 1
        assert max(abs(value) for value in published[0].samples) < 1000.0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_pull_error_resets_highpass(monkeypatch):
    # A pull error is a discontinuity: the DC blocker must re-arm so the next chunk
    # re-anchors its offset instead of carrying stale filter state across the gap.
    prime = ([[100000.0, 100000.0, 100000.0]], [50.0])
    node, _ = _build(
        monkeypatch, chunks=[prime, RuntimeError("lost")], overrides={"highpass_hz": "0.5"}
    )
    _capture(node)
    try:
        assert node._highpass._x_prev is not None  # primed the filter state
        node._on_tick()  # pulls the RuntimeError -> _handle_pull_error
        assert node._highpass._x_prev is None  # reset() re-armed it
        assert node._pull_errors == 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
