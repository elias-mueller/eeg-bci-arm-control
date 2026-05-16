import pytest

from eeg_bci_pipeline.intent_marker_mapping import style_for_intent


def test_left_and_right_intents_point_opposite_directions():
    left = style_for_intent("left_hand", 0.8)
    right = style_for_intent("right_hand", 0.8)

    assert left.yaw_rad == pytest.approx(3.141592653589793)
    assert right.yaw_rad == pytest.approx(0.0)
    assert left.length == pytest.approx(right.length)


def test_rest_uses_short_neutral_marker():
    style = style_for_intent("rest", 0.9)

    assert style.label == "rest"
    assert style.length < 0.5
    assert style.color_rgba[:3] == pytest.approx((0.6, 0.6, 0.6))


def test_marker_alpha_is_clamped_to_visible_range():
    low = style_for_intent("left_hand", 0.0)
    high = style_for_intent("left_hand", 2.0)

    assert low.color_rgba[3] == pytest.approx(0.35)
    assert high.color_rgba[3] == pytest.approx(1.0)


def test_unknown_intent_gets_fallback_style():
    style = style_for_intent(" blink ", 0.7)

    assert style.label == "blink"
    assert style.length == pytest.approx(0.35)
