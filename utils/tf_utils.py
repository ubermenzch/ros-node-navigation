"""TF transform utilities."""

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class Offset2D:
    """2D transform offset: translation (tx, ty) and rotation (yaw_offset)."""
    tx: float
    ty: float
    yaw_offset: float


from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time


def compute_offset_at_same_point(
    point_in_parent_x: float, point_in_parent_y: float, point_in_parent_yaw: float,
    point_in_child_x: float, point_in_child_y: float, point_in_child_yaw: float
) -> Offset2D:
    """
    Compute the parent->child transform from a shared point known in both frames.

    Given the same physical point P expressed in both parent and child frames, compute
    the rigid transform parent_T_child. Applying this transform converts a point from
    the child frame to the parent frame:
        P_parent = parent_T_child * P_child

    Args:
        point_in_parent_x:   P's X in parent frame
        point_in_parent_y:   P's Y in parent frame
        point_in_parent_yaw: P's heading in parent frame (radians)
        point_in_child_x:    P's X in child frame
        point_in_child_y:    P's Y in child frame
        point_in_child_yaw:  P's heading in child frame (radians)

    Returns:
        Offset2D: tx, ty = child frame origin in parent frame;
                  yaw_offset = child frame heading relative to parent frame (radians)
    """
    yaw_offset = normalize_angle(point_in_parent_yaw - point_in_child_yaw)
    cos_yaw = math.cos(yaw_offset)
    sin_yaw = math.sin(yaw_offset)

    tx = point_in_parent_x - (cos_yaw * point_in_child_x - sin_yaw * point_in_child_y)
    ty = point_in_parent_y - (sin_yaw * point_in_child_x + cos_yaw * point_in_child_y)
    return Offset2D(tx=tx, ty=ty, yaw_offset=yaw_offset)


def transform_from_parent_to_child(
    parent_x: float, parent_y: float, parent_yaw: float,
    offset: Offset2D,
) -> Tuple[float, float, float]:
    """
    Transform a point from parent frame to child frame.

    Args:
        parent_x:     position in parent frame (X)
        parent_y:     position in parent frame (Y)
        parent_yaw:   heading in parent frame (radians)
        offset:       Offset2D containing tx, ty, yaw_offset

    Returns:
        (child_x, child_y, child_yaw): position in child frame
    """
    cos_offset = math.cos(offset.yaw_offset)
    sin_offset = math.sin(offset.yaw_offset)

    child_x = cos_offset * (parent_x - offset.tx) + sin_offset * (parent_y - offset.ty)
    child_y = -sin_offset * (parent_x - offset.tx) + cos_offset * (parent_y - offset.ty)
    child_yaw = normalize_angle(parent_yaw - offset.yaw_offset)

    return child_x, child_y, child_yaw


def transform_from_child_to_parent(
    child_x: float, child_y: float, child_yaw: float,
    offset: Offset2D,
) -> Tuple[float, float, float]:
    """
    Transform a point from child frame to parent frame.

    Args:
        child_x:     position in child frame (X)
        child_y:     position in child frame (Y)
        child_yaw:   heading in child frame (radians)
        offset:      Offset2D containing tx, ty, yaw_offset

    Returns:
        (parent_x, parent_y, parent_yaw): position in parent frame
    """
    cos_offset = math.cos(offset.yaw_offset)
    sin_offset = math.sin(offset.yaw_offset)

    parent_x = cos_offset * child_x - sin_offset * child_y + offset.tx
    parent_y = sin_offset * child_x + cos_offset * child_y + offset.ty
    parent_yaw = normalize_angle(child_yaw + offset.yaw_offset)

    return parent_x, parent_y, parent_yaw


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi)."""
    while angle < -math.pi:
        angle += 2 * math.pi
    while angle >= math.pi:
        angle -= 2 * math.pi
    return angle


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert yaw angle to quaternion (x, y, z, w).

    Args:
        yaw: rotation angle around Z axis (radians)

    Returns:
        (x, y, z, w) quaternion components
    """
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """
    Convert quaternion to yaw angle.

    Args:
        x, y, z, w: quaternion components

    Returns:
        yaw angle (radians), normalized to [-pi, pi)
    """
    return normalize_angle(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def offset_to_transform(
        offset: Offset2D,
        header_frame: str, child_frame: str,
        stamp_nanosec: int = 0
    ) -> TransformStamped:
        """
        Build a TransformStamped message from an Offset2D.

        Args:
            offset:        Offset2D containing tx, ty, yaw_offset
            header_frame: name of the parent frame
            child_frame:  name of the child frame
            stamp_nanosec: nanoseconds since epoch; 0 means leave unset (defaults to zero)

        Returns:
            geometry_msgs/TransformStamped message
        """
        t = TransformStamped()
        t.header.frame_id = header_frame
        t.child_frame_id = child_frame
        t.header.stamp = Time(sec=stamp_nanosec // 1_000_000_000, nanosec=stamp_nanosec % 1_000_000_000)

        t.transform.translation.x = offset.tx
        t.transform.translation.y = offset.ty
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(offset.yaw_offset / 2.0)
        t.transform.rotation.w = math.cos(offset.yaw_offset / 2.0)

        return t

def transform_to_offset(
        t: TransformStamped,
    ) -> Offset2D:
        """
        Extract Offset2D from a TransformStamped message.

        Args:
            t: geometry_msgs/TransformStamped message

        Returns:
            Offset2D: containing tx, ty, yaw_offset
        """
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        q = t.transform.rotation
        yaw_offset = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        return Offset2D(tx=tx, ty=ty, yaw_offset=yaw_offset)
