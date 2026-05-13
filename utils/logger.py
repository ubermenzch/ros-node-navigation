#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具类

提供带函数名前缀的日志功能，自动获取调用函数的名称。
支持同时输出到文件和控制台（ROS logger）。

用法:
    from utils.logger import NodeLogger

    logger = NodeLogger(
        node_name='my_node',
        log_dir='/path/to/logs',
        log_timestamp='20240101_120000',
        enabled=True
    )

    logger.info('This message will have function name prefix')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.debug('Debug message')
"""

import os
import logging
import inspect
from typing import Optional
from datetime import datetime


class NodeLogger:
    """
    带函数名前缀的日志工具类

    自动获取调用函数的名称，并添加到日志消息前缀中。
    支持同时输出到文件和控制台（ROS logger）。
    """

    def __init__(
        self,
        node_name: str,
        log_dir: Optional[str] = None,
        log_timestamp: Optional[str] = None,
        enabled: bool = True,
        ros_logger=None,
        level: int = logging.INFO
    ):
        """
        初始化日志工具类

        Args:
            node_name: 节点名称，用于 logger 名称和日志文件名
            log_dir: 日志目录，None 时使用默认目录
            log_timestamp: 日志时间戳，None 时自动生成
            enabled: 是否启用文件日志
            ros_logger: ROS logger，用于同时输出到 ROS 控制台
            level: 日志级别
        """
        self.node_name = node_name
        self.log_timestamp = log_timestamp if log_timestamp else datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = self._resolve_log_dir(log_dir)
        self.enabled = enabled
        self.ros_logger = ros_logger

        self._logger = logging.getLogger(node_name)
        self._logger.setLevel(level)
        self._logger.handlers.clear()
        self._logger.propagate = False

        if enabled:
            os.makedirs(self.log_dir, exist_ok=True)

            log_file = os.path.join(self.log_dir, f'{node_name}_log_{self.log_timestamp}.log')

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def _resolve_log_dir(self, log_dir: Optional[str]) -> str:
        """Resolve the directory used for file logs."""
        if log_dir:
            return os.path.abspath(os.path.expanduser(log_dir))

        return os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'logs',
            f'navigation_{self.log_timestamp}',
        ))

    @property
    def logger(self) -> logging.Logger:
        """返回底层的 logging.Logger 实例"""
        return self._logger

    @property
    def log_file(self) -> Optional[str]:
        """返回日志文件路径"""
        if self.enabled and self.log_dir:
            return os.path.join(self.log_dir, f'{self.node_name}_log_{self.log_timestamp}.log')
        return None

    def _format_message(self, msg: str, include_caller: bool = True) -> str:
        """带函数名前缀的日志，自动获取调用函数的名称"""
        if not include_caller:
            return msg

        frame = inspect.currentframe()
        if frame is not None:
            caller_frame = frame.f_back.f_back
            if caller_frame is not None:
                func_name = caller_frame.f_code.co_name
                msg = f'{{{func_name}}} {msg}'
        return msg

    def debug(self, msg: str, include_caller: bool = True):
        """记录 debug 级别日志"""
        formatted_msg = self._format_message(msg, include_caller)
        self._logger.debug(formatted_msg)
        if self.ros_logger:
            self.ros_logger.debug(formatted_msg)

    def info(self, msg: str, include_caller: bool = True):
        """记录 info 级别日志"""
        formatted_msg = self._format_message(msg, include_caller)
        self._logger.info(formatted_msg)
        if self.ros_logger:
            self.ros_logger.info(formatted_msg)

    def warning(self, msg: str, include_caller: bool = True):
        """记录 warning 级别日志"""
        formatted_msg = self._format_message(msg, include_caller)
        self._logger.warning(formatted_msg)
        if self.ros_logger:
            self.ros_logger.warning(formatted_msg)

    def error(self, msg: str, include_caller: bool = True):
        """记录 error 级别日志"""
        formatted_msg = self._format_message(msg, include_caller)
        self._logger.error(formatted_msg)
        if self.ros_logger:
            self.ros_logger.error(formatted_msg)

    def log_init(self, init_info: list):
        """
        记录初始化信息

        Args:
            init_info: 初始化信息列表，每项为一行字符串
        """
        for line in init_info:
            self.info(line)
