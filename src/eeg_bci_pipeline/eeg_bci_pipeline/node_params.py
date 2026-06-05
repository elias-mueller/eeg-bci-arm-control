# pyright: basic
"""Typed accessors for ROS 2 node parameters.

rclpy types ``Parameter.value`` as a nullable union of every allowable parameter
type, so reading a declared parameter and converting it trips the type checker at
each call site even though a declared parameter always carries its declared type.
These helpers centralize the read and conversion so node setup stays readable and
the loosely typed rclpy boundary lives in one place. ``cast`` is a runtime no-op,
so each helper performs the same conversion the call sites did before.
"""

from __future__ import annotations

from typing import cast

from rclpy.node import Node


def str_param(node: Node, name: str) -> str:
    return str(node.get_parameter(name).value)


def int_param(node: Node, name: str) -> int:
    return int(cast(int, node.get_parameter(name).value))


def float_param(node: Node, name: str) -> float:
    return float(cast(float, node.get_parameter(name).value))


def bool_param(node: Node, name: str) -> bool:
    return bool(cast(bool, node.get_parameter(name).value))


def str_list_param(node: Node, name: str) -> list[str]:
    value = node.get_parameter(name).value
    return [str(item) for item in value] if value else []


def float_list_param(node: Node, name: str) -> list[float]:
    value = node.get_parameter(name).value
    return [float(item) for item in value] if value else []
