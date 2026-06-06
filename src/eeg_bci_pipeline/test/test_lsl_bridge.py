import pytest
from eeg_bci_pipeline.data.gdf_recording import VOLTS_TO_MICROVOLTS
from eeg_bci_pipeline.lsl_bridge import (
    NANOSECONDS_PER_SECOND,
    LslClockAligner,
    build_resolve_predicate,
    channel_major_to_sample_major,
    chunk_to_channel_major_uv,
    extract_lsl_metadata,
    resolve_channel_labels,
    unit_scale_for,
)

# The exact, ordered channel labels stored in the BCIC IV 2a CSP+LDA artifact
# (tmp/hand-csp-lda-A01T.joblib). model_intent_decoder enforces byte-for-byte
# equality, so the bridge must reproduce these verbatim from the LSL desc.
ARTIFACT_LABELS = (
    "EEG-Fz",
    "EEG-0",
    "EEG-1",
    "EEG-2",
    "EEG-3",
    "EEG-4",
    "EEG-5",
    "EEG-C3",
    "EEG-6",
    "EEG-Cz",
    "EEG-7",
    "EEG-C4",
    "EEG-8",
    "EEG-9",
    "EEG-10",
    "EEG-11",
    "EEG-12",
    "EEG-13",
    "EEG-14",
    "EEG-Pz",
    "EEG-15",
    "EEG-16",
)


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
    """Build a desc tree from [(label, unit), ...], linking channel siblings."""

    node = None
    for label, unit in reversed(channels):
        node = FakeXml(values={"label": label, "unit": unit}, nxt=node)
    channels_el = FakeXml(children={"channel": node} if node is not None else {})
    return FakeXml(children={"channels": channels_el})


class FakeStreamInfo:
    """Minimal stand-in for pylsl's StreamInfo."""

    def __init__(
        self,
        *,
        channel_count,
        nominal_srate=250.0,
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


def test_extract_lsl_metadata_preserves_exact_channel_labels_in_order():
    info = FakeStreamInfo(
        channel_count=len(ARTIFACT_LABELS),
        nominal_srate=250.0,
        channels=[(label, "microvolts") for label in ARTIFACT_LABELS],
    )

    meta = extract_lsl_metadata(info)

    assert meta.channel_labels == ARTIFACT_LABELS
    assert meta.channel_count == len(ARTIFACT_LABELS)
    assert meta.nominal_srate_hz == pytest.approx(250.0)
    assert meta.declared_unit == "microvolts"
    assert meta.stream_type == "EEG"
    assert meta.labels_downgraded is False


def test_extract_lsl_metadata_source_id_fallback_chain():
    channels = [("C3", "uV"), ("C4", "uV")]

    explicit = FakeStreamInfo(channel_count=2, source_id="dev-1", channels=channels)
    assert extract_lsl_metadata(explicit).source_id == "dev-1"

    named = FakeStreamInfo(channel_count=2, source_id="", name="streamX", channels=channels)
    assert extract_lsl_metadata(named).source_id == "streamX"

    anonymous = FakeStreamInfo(channel_count=2, source_id="", name="", channels=channels)
    assert extract_lsl_metadata(anonymous).source_id == "lsl-eeg"


def test_extract_lsl_metadata_falls_back_to_default_labels_when_desc_empty():
    meta = extract_lsl_metadata(FakeStreamInfo(channel_count=3, channels=[]))

    assert meta.channel_labels == ("ch_01", "ch_02", "ch_03")
    assert meta.declared_unit == ""


def test_extract_lsl_metadata_rejects_non_positive_channel_count():
    with pytest.raises(ValueError, match="at least one channel"):
        extract_lsl_metadata(FakeStreamInfo(channel_count=0, channels=[]))


def test_extract_lsl_metadata_flags_downgrade_when_declared_labels_rejected():
    # Duplicate declared labels -> ch_NN fallback, flagged so the node can warn.
    info = FakeStreamInfo(channel_count=3, channels=[("C3", "uV"), ("C3", "uV"), ("Cz", "uV")])

    meta = extract_lsl_metadata(info)

    assert meta.channel_labels == ("ch_01", "ch_02", "ch_03")
    assert meta.labels_downgraded is True


def test_extract_lsl_metadata_partial_label_desc_is_a_downgrade():
    # Fewer <channel> nodes than channel_count -> partial -> fallback + flag.
    info = FakeStreamInfo(channel_count=3, channels=[("C3", "uV")])

    meta = extract_lsl_metadata(info)

    assert meta.channel_labels == ("ch_01", "ch_02", "ch_03")
    assert meta.labels_downgraded is True


def test_extract_lsl_metadata_no_declared_labels_is_not_a_downgrade():
    # A stream that simply declares no labels is normal, not a rejected downgrade.
    meta = extract_lsl_metadata(FakeStreamInfo(channel_count=3, channels=[]))

    assert meta.channel_labels == ("ch_01", "ch_02", "ch_03")
    assert meta.labels_downgraded is False


def test_extract_lsl_metadata_passes_through_declared_rate():
    info = FakeStreamInfo(
        channel_count=2, nominal_srate=128.0, channels=[("C3", "uV"), ("Cz", "uV")]
    )

    assert extract_lsl_metadata(info).nominal_srate_hz == pytest.approx(128.0)


def test_resolve_channel_labels_uses_exact_unique_labels_verbatim():
    assert resolve_channel_labels(["C3", "Cz", "C4"], 3) == ("C3", "Cz", "C4")
    assert resolve_channel_labels([" C3 ", "Cz"], 2) == ("C3", "Cz")


def test_resolve_channel_labels_falls_back_on_partial_duplicate_or_empty():
    assert resolve_channel_labels(["C3", "Cz"], 3) == ("ch_01", "ch_02", "ch_03")
    assert resolve_channel_labels(["C3", "C3"], 2) == ("ch_01", "ch_02")
    assert resolve_channel_labels(["C3", ""], 2) == ("ch_01", "ch_02")
    assert resolve_channel_labels([], 2) == ("ch_01", "ch_02")


def test_unit_scale_for_known_and_unknown_units():
    assert unit_scale_for("microvolts") == pytest.approx(1.0)
    assert unit_scale_for("uV") == pytest.approx(1.0)
    assert unit_scale_for("volts") == pytest.approx(VOLTS_TO_MICROVOLTS)
    assert unit_scale_for("V") == pytest.approx(VOLTS_TO_MICROVOLTS)
    assert unit_scale_for("") == pytest.approx(1.0)
    assert unit_scale_for("", default_scale=VOLTS_TO_MICROVOLTS) == pytest.approx(
        VOLTS_TO_MICROVOLTS
    )
    assert unit_scale_for("bogus", default_scale=2.0) == pytest.approx(2.0)


def test_unit_scale_for_normalizes_case_whitespace_and_micro_signs():
    assert unit_scale_for("  MicroVolts ") == pytest.approx(1.0)
    assert unit_scale_for("µV") == pytest.approx(1.0)  # U+00B5 micro sign
    assert unit_scale_for("μV") == pytest.approx(1.0)  # U+03BC greek small mu
    assert unit_scale_for(" VOLTS ") == pytest.approx(VOLTS_TO_MICROVOLTS)


def test_chunk_to_channel_major_uv_transposes_sample_major_input():
    # 3 samples x 2 channels, sample-major -> channel-major [ch0..., ch1...].
    chunk = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    assert chunk_to_channel_major_uv(chunk, 2) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_chunk_to_channel_major_uv_empty_chunk_returns_empty():
    assert chunk_to_channel_major_uv([], 4) == []


def test_chunk_to_channel_major_uv_applies_unit_scale():
    chunk = [[1e-6, 2e-6]]

    assert chunk_to_channel_major_uv(chunk, 2, unit_scale=VOLTS_TO_MICROVOLTS) == pytest.approx(
        [1.0, 2.0]
    )


def test_chunk_to_channel_major_uv_rejects_wrong_channel_width():
    with pytest.raises(ValueError, match="channels per sample"):
        chunk_to_channel_major_uv([[1.0, 2.0, 3.0]], 2)


def test_chunk_to_channel_major_uv_rejects_ragged_chunk():
    # A ragged chunk (unequal per-sample widths) must raise, not silently mangle;
    # _on_tick relies on this ValueError to drop the frame instead of crashing.
    with pytest.raises(ValueError):
        chunk_to_channel_major_uv([[1.0, 2.0], [3.0]], 2)


def test_channel_major_round_trips_through_sample_major():
    flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2 channels x 3 samples, channel-major

    sample_major = channel_major_to_sample_major(flat, 2)

    assert sample_major == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert chunk_to_channel_major_uv(sample_major, 2) == pytest.approx(flat)


def test_channel_major_to_sample_major_validates_divisibility():
    assert channel_major_to_sample_major([], 4) == []
    with pytest.raises(ValueError, match="divisible"):
        channel_major_to_sample_major([1.0, 2.0, 3.0], 2)


def test_real_pylsl_streaminfo_desc_round_trips_through_extract():
    # Validates the metadata walk against the real pylsl XML API, not just the
    # fakes. Skipped when pylsl or its native liblsl is unavailable.
    try:
        import pylsl
    except Exception as error:  # ImportError, or RuntimeError if liblsl is missing
        pytest.skip(f"pylsl/liblsl unavailable: {error}")

    labels = ("C3", "Cz", "C4")
    info = pylsl.StreamInfo("rt-test", "EEG", len(labels), 250.0, pylsl.cf_float32, "rt-src")
    channels = info.desc().append_child("channels")
    for label in labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "microvolts")

    meta = extract_lsl_metadata(info)

    assert meta.channel_labels == labels
    assert meta.channel_count == 3
    assert meta.nominal_srate_hz == pytest.approx(250.0)
    assert meta.declared_unit == "microvolts"
    assert meta.source_id == "rt-src"


# --- build_resolve_predicate -------------------------------------------------


def test_build_resolve_predicate_conjoins_name_and_type():
    assert build_resolve_predicate("rt-test", "EEG") == "name='rt-test' and type='EEG'"


def test_build_resolve_predicate_single_clause():
    assert build_resolve_predicate("", "EEG") == "type='EEG'"
    assert build_resolve_predicate("headset-7", "") == "name='headset-7'"


def test_build_resolve_predicate_requires_a_clause():
    with pytest.raises(ValueError, match="stream_name and/or stream_type"):
        build_resolve_predicate("", "")


def test_build_resolve_predicate_rejects_single_quote():
    # A quote would break out of the predicate's quoted literal; reject, not escape.
    with pytest.raises(ValueError, match="single quote"):
        build_resolve_predicate("rogue' or '1'='1", "EEG")
    with pytest.raises(ValueError, match="single quote"):
        build_resolve_predicate("", "EE'G")


# --- LslClockAligner ---------------------------------------------------------


def test_clock_aligner_anchors_to_ros_time_on_first_sample():
    aligner = LslClockAligner(resync_threshold_sec=0.25)
    # The first sample defines the anchor, so its stamp is the ROS arrival time
    # regardless of the LSL clock's absolute value.
    assert aligner.stamp_ns(1_000_000_000, first_lsl_ts_sec=50.0) == 1_000_000_000


def test_clock_aligner_tracks_lsl_spacing_within_band():
    aligner = LslClockAligner(resync_threshold_sec=0.25)
    aligner.stamp_ns(1_000_000_000, first_lsl_ts_sec=50.0)  # anchor ros=1.0s, lsl=50.0s
    # LSL advanced 0.1 s; ROS arrival is a hair early (within band), so the stamp
    # follows the sample clock (anchor + 0.1 s), not the jittery arrival time.
    stamp = aligner.stamp_ns(1_080_000_000, first_lsl_ts_sec=50.1)
    assert stamp == 1_100_000_000


def test_clock_aligner_reanchors_when_drift_exceeds_threshold():
    aligner = LslClockAligner(resync_threshold_sec=0.25)
    aligner.stamp_ns(1_000_000_000, first_lsl_ts_sec=50.0)
    # LSL says +1.0 s but ROS arrival is +1.4 s: 0.4 s drift > 0.25 s, so the
    # aligner re-anchors on arrival instead of emitting the stale derived stamp.
    stamp = aligner.stamp_ns(2_400_000_000, first_lsl_ts_sec=51.0)
    assert stamp == 2_400_000_000
    # Subsequent in-band samples now derive from the new anchor.
    assert aligner.stamp_ns(2_700_000_000, first_lsl_ts_sec=51.3) == 2_700_000_000


def test_clock_aligner_reset_reanchors_after_a_drop():
    aligner = LslClockAligner()
    aligner.stamp_ns(1_000_000_000, first_lsl_ts_sec=50.0)
    aligner.reset()
    # After a stream drop the LSL clock may be discontinuous; the next sample
    # re-anchors rather than projecting across the gap.
    assert aligner.stamp_ns(9_000_000_000, first_lsl_ts_sec=99.0) == 9_000_000_000


def test_clock_aligner_stays_monotonic_when_lsl_outruns_ros():
    # When the LSL clock runs ahead of ROS, the derived stamp climbs past arrival
    # until the band trips; the re-anchor then snaps to arrival, which is *earlier*
    # than the last emitted stamp. The aligner must hold the stamp, not step it
    # backward (a backward header.stamp violates the ROS sensor-stream convention).
    aligner = LslClockAligner(resync_threshold_sec=0.25)
    aligner.stamp_ns(1_000_000_000, first_lsl_ts_sec=50.0)  # anchor
    in_band = aligner.stamp_ns(1_100_000_000, first_lsl_ts_sec=50.3)  # derived 1.3, drift 0.2
    assert in_band == 1_300_000_000
    # LSL now 0.45 ahead while ROS advanced 0.15: derived 1.45 vs arrival 1.15 is
    # 0.3 > 0.25, so it re-anchors to 1.15 but must not regress below 1.3.
    after_resync = aligner.stamp_ns(1_150_000_000, first_lsl_ts_sec=50.45)
    assert after_resync >= in_band
    assert after_resync == 1_300_000_000


def test_clock_aligner_does_not_resync_on_drained_backlog():
    # An executor stall, then pull_chunk drains a 2 s backlog: the oldest sample is
    # 2 s behind arrival but the newest is current. That must NOT be read as drift,
    # and the frame must be stamped at the oldest sample's true time, not arrival.
    aligner = LslClockAligner(resync_threshold_sec=0.25)
    # First chunk anchors the newest sample (lsl 50.0) to arrival (ros 10.0 s).
    aligner.stamp_ns(10_000_000_000, first_lsl_ts_sec=49.96, last_lsl_ts_sec=50.0)
    # Backlog drains: oldest lsl 50.0, newest lsl 52.0; arrival 12.0 s -> the newest
    # derives to exactly 12.0 s (in band), so no re-anchor; oldest stays at 10.0 s.
    stamp = aligner.stamp_ns(12_000_000_000, first_lsl_ts_sec=50.0, last_lsl_ts_sec=52.0)
    assert stamp == 10_000_000_000  # the oldest sample's true time, not arrival (12.0 s)


def test_clock_aligner_exact_threshold_stays_in_band():
    # The drift check is a strict >, so drift exactly equal to the threshold is in
    # band and does not re-anchor. Pins the comparison against a >= regression.
    aligner = LslClockAligner(resync_threshold_sec=0.25)
    aligner.stamp_ns(1_000_000_000, first_lsl_ts_sec=50.0)  # anchor ros 1.0 / lsl 50.0
    stamp = aligner.stamp_ns(1_250_000_000, first_lsl_ts_sec=50.5)  # derived 1.5, drift == 0.25
    assert stamp == 1_500_000_000


def test_clock_aligner_holds_on_backward_lsl_jump_within_band():
    # The docstring claims the clamp absorbs a backward LSL jump; pin it. A within-
    # band backward step in the LSL clock would regress the derived stamp, so the
    # clamp holds it at the last emitted value instead.
    aligner = LslClockAligner(resync_threshold_sec=0.25)
    aligner.stamp_ns(1_000_000_000, first_lsl_ts_sec=50.0)
    forward = aligner.stamp_ns(1_100_000_000, first_lsl_ts_sec=50.1)
    assert forward == 1_100_000_000
    held = aligner.stamp_ns(1_150_000_000, first_lsl_ts_sec=50.05)  # lsl steps back 0.05 s
    assert held == 1_100_000_000


def test_clock_aligner_reset_clears_monotonic_clamp():
    aligner = LslClockAligner(resync_threshold_sec=0.25)
    aligner.stamp_ns(5_000_000_000, first_lsl_ts_sec=50.0)  # last stamp 5.0 s
    aligner.reset()
    # A reconnect starts a fresh run: an earlier arrival re-anchors cleanly rather
    # than being clamped up to the stale pre-drop stamp.
    assert aligner.stamp_ns(2_000_000_000, first_lsl_ts_sec=10.0) == 2_000_000_000


def test_clock_aligner_rejects_nonpositive_threshold():
    # A zero/negative threshold would re-anchor on every chunk, silently degrading
    # to arrival-time stamping while use_lsl_timestamps is still on. Fail fast.
    with pytest.raises(ValueError, match="greater than 0"):
        LslClockAligner(resync_threshold_sec=0.0)
    with pytest.raises(ValueError, match="greater than 0"):
        LslClockAligner(resync_threshold_sec=-1.0)


def test_clock_aligner_constant_matches_one_second():
    assert NANOSECONDS_PER_SECOND == 1_000_000_000
