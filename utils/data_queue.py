#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用数据缓存队列模块

功能：
1. 基于时间戳的数据缓存队列
2. 自动清理过期数据
3. 支持查找最近时间戳的数据
4. 支持泛型数据格式

使用示例：
```python
# GNSS 数据队列
gnss_queue = DataQueue[Tuple[int, float, float, float, float]](timeout_seconds=2.0)
gnss_queue.append((stamp_nanos, utm_x, utm_y, lat, lon))

# 查找最近数据
nearest = gnss_queue.find_nearest(target_stamp_nanos)

# 获取最新数据
latest = gnss_queue.get_latest()

# 清理过期数据
gnss_queue.prune_expired(current_nanos)
```
"""

from typing import Generic, TypeVar, Optional, List, Tuple

T = TypeVar('T')


class DataFrame(Generic[T]):
    """
    数据帧包装器，包含时间戳和实际数据

    属性：
        stamp_nanos: 时间戳（纳秒）
        data: 数据内容
    """

    __slots__ = ('stamp_nanos', 'data')

    def __init__(self, stamp_nanos: int, data: T):
        self.stamp_nanos = stamp_nanos
        self.data = data

    def __iter__(self):
        """支持解包为 (stamp_nanos, *data) 元组"""
        if isinstance(self.data, tuple):
            return iter((self.stamp_nanos,) + self.data)
        return iter((self.stamp_nanos, self.data))

    def __repr__(self):
        return f"DataFrame(stamp={self.stamp_nanos}, data={self.data})"


class DataQueue(Generic[T]):
    """
    基于时间戳的数据缓存队列

    特性：
    - FIFO 队列，自动清理过期数据
    - 支持按时间戳查找最近数据
    - 支持泛型，可存储任意类型数据
    - 支持自定义超时时间
    """

    def __init__(self, timeout_seconds: float = 1.0, max_size: int = 0):
        """
        初始化数据队列

        Args:
            timeout_seconds: 数据超时时间（秒），超过此时间的数据将被清理
            max_size: 最大队列长度，0 表示不限制
        """
        self._queue: List[DataFrame[T]] = []
        self._timeout_nanos: int = int(timeout_seconds * 1e9)
        self._max_size: int = max_size

    @property
    def timeout_seconds(self) -> float:
        """获取超时时间（秒）"""
        return self._timeout_nanos / 1e9

    @timeout_seconds.setter
    def timeout_seconds(self, value: float):
        """设置超时时间（秒）"""
        self._timeout_nanos = int(value * 1e9)

    @property
    def size(self) -> int:
        """获取当前队列大小"""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """判断队列是否为空"""
        return len(self._queue) == 0

    def append(self, stamp_nanos: int, data: T) -> None:
        """
        添加数据到队列

        Args:
            stamp_nanos: 时间戳（纳秒）
            data: 数据内容
        """
        self._queue.append(DataFrame(stamp_nanos, data))

        # 如果超过最大长度，移除最旧的数据
        if self._max_size > 0 and len(self._queue) > self._max_size:
            self._queue.pop(0)

    def append_tuple(self, data: Tuple) -> None:
        """
        添加元组格式数据到队列

        Args:
            data: 元组格式 (stamp_nanos, *data_fields)
        """
        if not data:
            return
        self.append(data[0], data[1:] if len(data) > 1 else None)

    def get_latest(self) -> Optional[DataFrame[T]]:
        """
        获取最新（最近）的数据帧

        Returns:
            最新数据帧，如果队列为空则返回 None
        """
        return self._queue[-1] if self._queue else None

    def get_oldest(self) -> Optional[DataFrame[T]]:
        """
        获取最旧的数据帧

        Returns:
            最旧数据帧，如果队列为空则返回 None
        """
        return self._queue[0] if self._queue else None

    def find_nearest(self, target_stamp_nanos: int) -> Optional[DataFrame[T]]:
        """
        查找与目标时间戳最接近的数据帧

        Args:
            target_stamp_nanos: 目标时间戳（纳秒）

        Returns:
            最近的数据帧，如果队列为空则返回 None
        """
        if not self._queue:
            return None
        return min(self._queue, key=lambda frame: abs(frame.stamp_nanos - target_stamp_nanos))

    def prune_expired(self, current_nanos: int) -> int:
        """
        清理过期数据

        Args:
            current_nanos: 当前时间戳（纳秒）

        Returns:
            清理的数据帧数量
        """
        original_size = len(self._queue)
        self._queue = [
            frame for frame in self._queue
            if current_nanos - frame.stamp_nanos <= self._timeout_nanos
        ]
        return original_size - len(self._queue)

    def clear(self) -> None:
        """清空队列"""
        self._queue.clear()

    def get_all(self) -> List[DataFrame[T]]:
        """获取队列中所有数据帧的副本"""
        return list(self._queue)

    def to_tuples(self) -> List[Tuple]:
        """
        将队列转换为元组列表

        Returns:
            [(stamp_nanos, data), ...] 格式的列表
        """
        return [tuple(frame) for frame in self._queue]


