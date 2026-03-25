"""时间戳工具模块

提供 ROS builtin_interfaces/Time 与纳秒整数之间的相互转换，
以及获取当前系统时刻（纳秒）的能力。
"""

import time

from builtin_interfaces.msg import Time


class TimeUtils:
    """ROS 时间戳与纳秒整数互相转换"""

    @staticmethod
    def stamp_to_nanos(stamp) -> int:
        """ROS Time -> 纳秒（int64）"""
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    @staticmethod
    def nanos_to_stamp(nanos: int) -> Time:
        """纳秒（int64）-> builtin_interfaces/Time"""
        stamp = Time()
        stamp.sec = nanos // 1_000_000_000
        stamp.nanosec = nanos % 1_000_000_000
        return stamp

    @staticmethod
    def now_nanos() -> int:
        """获取当前系统时刻（纳秒，int64）"""
        return int(time.time() * 1_000_000_000)
