"""Coverage for the typed ROS 2 parameter accessors in `node_params`.

These helpers wrap `node.get_parameter(name).value` so the loosely typed rclpy
boundary lives in one place. The tests declare real parameters on a real rclpy
node and assert each accessor returns the correctly typed value, including the
empty-list fallback in the list accessors (a declared-but-empty list must read
back as `[]`, not as a falsy union value that leaks past the call site).

Skipped when ROS (rclpy) is unavailable, as in a bare pytest run.
"""

from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")

from eeg_bci_pipeline.node_params import (  # noqa: E402
    bool_param,
    float_list_param,
)


@pytest.fixture
def ros_context():
    # Owns the rclpy lifecycle so a failure mid-test never leaks the context into
    # the next one.
    rclpy.init()
    try:
        yield
    finally:
        rclpy.shutdown()


@pytest.fixture
def node(ros_context):
    n = rclpy.create_node("np_test")
    try:
        yield n
    finally:
        n.destroy_node()


def test_bool_param_returns_declared_bool(node):
    node.declare_parameter("flag_true", True)
    node.declare_parameter("flag_false", False)

    true_value = bool_param(node, "flag_true")
    false_value = bool_param(node, "flag_false")

    assert true_value is True
    assert false_value is False


def test_float_list_param_returns_list_of_floats(node):
    node.declare_parameter("freqs", [1.0, 2.5, 3.0])

    result = float_list_param(node, "freqs")

    assert result == pytest.approx([1.0, 2.5, 3.0])
    assert all(isinstance(item, float) for item in result)


def test_float_list_param_empty_list_reads_back_empty(node):
    # An empty declared list is falsy, so the accessor must fall through to the
    # `[]` branch rather than returning the raw (possibly None) union value.
    from rcl_interfaces.msg import ParameterDescriptor
    from rclpy.parameter import Parameter

    node.declare_parameter(
        "no_freqs",
        [],
        ParameterDescriptor(type=Parameter.Type.DOUBLE_ARRAY.value),
    )

    result = float_list_param(node, "no_freqs")

    assert result == []
