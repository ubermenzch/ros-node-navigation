#!/usr/bin/env python3
"""
频率统计工具类

用于统计对象/节点的实际工作频率，并与目标频率比较，
如果实际频率与目标频率偏差超过阈值，则记录警告日志。

使用方法:
    from utils.frequency_stats import FrequencyStats

    # 在初始化时创建
    self.freq_stats = FrequencyStats(
        object_name="planner_node",       # 对象名称
        target_frequency=5.0,             # 目标频率 Hz
        node_logger=self.logger,          # NodeLogger 实例（会同时输出到文件和控制台）
        window_size=10,                    # 滑动窗口大小
        warn_threshold=0.8                # 低于目标频率*阈值时警告 (0.8 = 80%)
    )

    # 在每次回调开始时调用
    self.freq_stats.tick()

    # 在回调结束时也可以调用（如果想统计处理耗时）
    # self.freq_stats.tock()
"""

import time
from collections import deque
from typing import Optional


class FrequencyStats:
    """
    频率统计工具类
    
    通过记录时间戳序列，计算实际工作频率，
    并在频率低于阈值时输出警告日志。
    """

    def __init__(
        self,
        object_name: str,
        target_frequency: Optional[float] = None,
        node_logger=None,
        window_size: int = 10,
        warn_threshold: float = 0.8,
        log_interval: float = 5.0
    ):
        """
        初始化频率统计器

        Args:
            object_name: 对象名称，用于日志输出
            target_frequency: 目标工作频率 (Hz)，可选。不传入时仅记录实际频率
            node_logger: NodeLogger 实例（可选），会同时输出到文件和控制台
            window_size: 滑动窗口大小，用于计算平均频率
            warn_threshold: 警告阈值，实际频率低于 target_frequency * warn_threshold 时警告
            log_interval: 日志输出间隔 (秒)
        """
        self.object_name = object_name
        self.target_frequency = target_frequency
        self.window_size = window_size
        self.warn_threshold = warn_threshold
        self.log_interval = log_interval

        self.node_logger = node_logger

        # 时间戳队列
        self.timestamps = deque(maxlen=window_size)

        # 上次日志输出时间
        self._last_log_time = 0.0

        # 计算的频率
        self._actual_frequency = 0.0

    def tick(self):
        """
        记录一次回调执行
        
        在回调函数开始时调用，记录当前时间戳。
        """
        current_time = time.monotonic()
        self.timestamps.append(current_time)
        self._check_and_log(current_time)

    def tock(self):
        """
        记录一次回调结束
        
        如果想统计回调处理耗时，可以在回调结束时调用。
        此方法仅用于记录耗时，不影响频率统计。
        """
        pass  # 目前不需要额外处理

    def _check_and_log(self, current_time: float):
        """
        检查频率并在必要时输出日志
        
        Args:
            current_time: 当前时间戳
        """
        # 只在达到最小样本数时计算频率
        if len(self.timestamps) < 2:
            return

        # 检查是否到达日志输出间隔
        if current_time - self._last_log_time < self.log_interval:
            return

        # 计算实际频率
        self._actual_frequency = self._calculate_frequency()

        log_interval = f'{self.log_interval:g}s'
        base_msg = (
            f'object={self.object_name} '
            f'actual={self._actual_frequency:.2f}Hz '
        )

        # 无目标频率时，仅记录实际频率
        if self.target_frequency is None:
            msg = (
                f'{base_msg}'
                f'window={self.window_size} '
                f'log_interval={log_interval}'
            )
            if self.node_logger:
                self.node_logger.info(msg, include_caller=False)
        else:
            # 计算偏差
            frequency_error = self.target_frequency - self._actual_frequency
            error_percent = (frequency_error / self.target_frequency) * 100 if self.target_frequency > 0 else 0

            # 判断是否低于阈值
            warn_threshold_freq = self.target_frequency * self.warn_threshold

            # 构建日志消息
            if self._actual_frequency < warn_threshold_freq:
                msg = (
                    f'{base_msg}'
                    f'target={self.target_frequency:.1f}Hz '
                    f'window={self.window_size} '
                    f'log_interval={log_interval} '
                    f'status=low '
                    f'deviation={error_percent:.1f}%'
                )

                if self.node_logger:
                    self.node_logger.warning(msg, include_caller=False)
            else:
                msg = (
                    f'{base_msg}'
                    f'target={self.target_frequency:.1f}Hz '
                    f'window={self.window_size} '
                    f'log_interval={log_interval}'
                )

                if self.node_logger:
                    self.node_logger.info(msg, include_caller=False)

        self._last_log_time = current_time

    def _calculate_frequency(self) -> float:
        """
        计算实际频率
        
        Returns:
            实际频率 (Hz)
        """
        if len(self.timestamps) < 2:
            return 0.0

        # 取窗口内最早和最晚的时间戳
        oldest = self.timestamps[0]
        newest = self.timestamps[-1]

        # 计算时间差
        time_diff = newest - oldest

        if time_diff <= 0:
            return 0.0

        # 计算频率 (样本数-1 是因为N个时间戳有N-1个间隔)
        samples = len(self.timestamps) - 1
        frequency = samples / time_diff

        return frequency

    def get_actual_frequency(self) -> float:
        """
        获取当前计算的实际频率
        
        Returns:
            实际频率 (Hz)
        """
        return self._actual_frequency

    def get_stats(self) -> dict:
        """
        获取详细统计信息
        
        Returns:
            包含频率统计信息的字典
        """
        return {
            'object_name': self.object_name,
            'target_frequency': self.target_frequency,
            'actual_frequency': self._actual_frequency,
            'warn_threshold': self.warn_threshold,
            'window_size': self.window_size,
            'sample_count': len(self.timestamps)
        }


def create_frequency_stats(
    object_name: str,
    config: dict,
    node_logger=None
) -> Optional['FrequencyStats']:
    """
    从配置字典创建频率统计器的便捷函数

    Args:
        object_name: 对象名称
        config: 配置字典，需要包含 'frequency' 键
        node_logger: NodeLogger 实例
    
    Returns:
        FrequencyStats 实例，如果配置中禁用了则返回 None
    """
    if 'frequency' not in config:
        return None

    frequency = config.get('frequency', 10.0)
    enabled = config.get('log_frequency_stats', True)

    if not enabled:
        return None

    return FrequencyStats(
        object_name=object_name,
        target_frequency=frequency,
        node_logger=node_logger,
        window_size=config.get('frequency_stats_window', 10),
        warn_threshold=config.get('frequency_stats_warn_threshold', 0.8),
        log_interval=config.get('frequency_stats_log_interval', 5.0)
    )
