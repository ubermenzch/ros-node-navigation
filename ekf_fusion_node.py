#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化定位融合节点（适用于：只有世界系经纬度 + 世界系 yaw，无可信协方差）

功能：
1. 记录启动后接收到的第一个有效 /rtk_fix，经纬度转 UTM，作为 map 原点
2. 发布 utm -> map 静态 TF（零旋转）
3. 持续接收 GNSS / ODOM / 世界系 yaw
   - GNSS: 提供 map 下绝对位置
   - 世界系 yaw: 优先来自 RTK 计算输出；RTK 无法提供时，可配置为使用启动朝东后的 odom yaw
   - ODOM: 提供 odom 坐标系下的位姿
4. 输出 /navigation/map_pose (PoseStamped, frame_id=map)

坐标系约定（ENU / 地理坐标系）：
- map 坐标系：与 UTM 坐标系对齐
- 朝向角度 yaw（弧度）：
  - 0°   = 朝东 (East)
  - 90°  = 朝北 (North)
  - 180° = 朝西 (West)
  - 270° = 朝南 (South)

融合策略：
- odom_map_pose: odom 位姿转换到 map 坐标系
- gnss_map_pose_topic: RTK 位姿转换到 map 坐标系
- 最终 map_pose = (1-alpha) * odom_map_pose + alpha * gnss_map_pose_topic
"""

import math
import os
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile

from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Quaternion
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
import numpy as np
import pyproj

from utils.config_loader import get_config
from utils.frequency_stats import FrequencyStats
from utils.time_utils import TimeUtils
from utils.logger import NodeLogger
from utils.data_queue import DataQueue
from utils import tf_utils


class EKFFusionNode(Node):
    """
    为了兼容原有启动方式，类名保留 EKFFusionNode。
    实际实现为"简化融合器"，使用位姿直接融合方式。
    """

    def __init__(self, log_dir: str = None, log_timestamp: str = None):
        super().__init__('ekf_fusion_node')

        # 使用传入的日志目录和时间戳，或生成新的
        self.log_dir = log_dir
        self.log_timestamp = log_timestamp if log_timestamp is not None else datetime.now().strftime('%Y%m%d_%H%M%S')

        # 加载配置
        config = get_config()
        ekf_config = config.get_ekf_config()

        subscriptions = ekf_config.get('subscriptions', {})
        publications = ekf_config.get('publications', {})

        # 订阅话题
        self.gnss_topic = subscriptions.get('gnss_topic', '/rtk_fix')
        # RTK 计算输出的世界系 yaw 话题，虽然 topic 名叫 rtk_imu，但这里主要使用 orientation yaw。
        self.gnss_yaw_topic = subscriptions.get('gnss_yaw_topic', '/rtk_imu')
        self.odom_topic = subscriptions.get('odom_topic', '/utlidar/robot_odom')

        # 发布话题
        self.map_pose_topic = publications.get('map_pose_topic', '/navigation/map_pose')
        self.gnss_map_pose_topic = publications.get(
            'gnss_map_pose_topic',
            publications.get('gnss_pose_topic', '/navigation/gnss_pose')
        )

        # 运行参数
        self.frequency = ekf_config.get('frequency', 10.0)
        self.gnss_timeout = ekf_config.get('gnss_timeout', 2.0)
        self.odom_timeout = ekf_config.get('odom_timeout', 1.0)
        self.gnss_yaw_timeout = ekf_config.get('gnss_yaw_timeout', 5.0)

        # 世界系 yaw 来源：
        # - false：使用 gnss_yaw_topic 提供的 RTK 世界系朝向。
        # - true：不订阅 gnss_yaw_topic，将 odom yaw 当作世界系 yaw。
        #         仅在 RTK 无法提供绝对 yaw，且启动时机器狗朝正东时使用。
        self.use_odom_yaw_as_world_yaw = bool(ekf_config.get('use_odom_yaw_as_world_yaw', False))

        # 融合模式：默认保持原有 GNSS 锚点 + odom 增量方式
        self.use_2d_ekf = bool(ekf_config.get('use_2d_ekf', False))
        ekf_2d_config = ekf_config.get('ekf_2d', {}) or {}
        self.default_gnss_position_variance = float(ekf_2d_config.get('default_gnss_position_variance', 0.0025))
        self.default_yaw_variance = float(ekf_2d_config.get('default_yaw_variance', 1.2e-05))
        self.rtk_uncertainty_scale = float(ekf_2d_config.get('rtk_uncertainty_scale', 1.0))
        if not math.isfinite(self.rtk_uncertainty_scale) or self.rtk_uncertainty_scale <= 0.0:
            self.rtk_uncertainty_scale = 1.0
        self.gnss_position_variance_floor = float(ekf_2d_config.get('gnss_position_variance_floor', 1.0e-4))
        self.yaw_variance_floor = float(ekf_2d_config.get('yaw_variance_floor', 1.0e-8))
        self.process_position_variance_min = float(ekf_2d_config.get('process_position_variance_min', 1.0e-6))
        self.process_position_variance_per_meter = float(ekf_2d_config.get('process_position_variance_per_meter', 1.0e-2))
        self.process_yaw_variance_min = float(ekf_2d_config.get('process_yaw_variance_min', 1.0e-8))
        self.process_yaw_variance_per_rad = float(ekf_2d_config.get('process_yaw_variance_per_rad', 1.0e-2))
        self.max_position_mahalanobis = float(ekf_2d_config.get('max_position_mahalanobis', 16.0))
        self.max_yaw_mahalanobis = float(ekf_2d_config.get('max_yaw_mahalanobis', 16.0))
        self.rtk_rejection_stats_window = float(ekf_2d_config.get('rtk_rejection_stats_window', 30.0))
        if not math.isfinite(self.rtk_rejection_stats_window) or self.rtk_rejection_stats_window <= 0.0:
            self.rtk_rejection_stats_window = 30.0
        self.rtk_rejection_stats_log_interval = float(ekf_2d_config.get('rtk_rejection_stats_log_interval', 5.0))
        if not math.isfinite(self.rtk_rejection_stats_log_interval) or self.rtk_rejection_stats_log_interval < 0.0:
            self.rtk_rejection_stats_log_interval = 5.0

        self.log_enabled = bool(ekf_config.get('log_enabled', True))

        self.tf_init_completed = False    # tf初始化是否已完成

        # 数据采集队列（用于等待期间的加权平均）
        self._gnss_queue = DataQueue(timeout_seconds=self.gnss_timeout)
        self._gnss_yaw_queue = DataQueue(timeout_seconds=self.gnss_yaw_timeout)
        self._odom_queue = DataQueue(timeout_seconds=self.odom_timeout)

        # UTM转换相关
        self.utm_zone: Optional[int] = None
        self.utm_transformer = None
        self.utm_to_map_offset: tf_utils.Offset2D = None   # utm->map transform

        # 传感器数据
        # 日志
        self._init_logger(self.log_enabled)

        # 频率统计
        self._init_frequency_stats()

        # 订阅者
        self.gnss_sub = self.create_subscription(
            NavSatFix,
            self.gnss_topic,
            self.gnss_callback,
            1
        )
        if self.use_odom_yaw_as_world_yaw:
            self.gnss_yaw_sub = None
        else:
            self.gnss_yaw_sub = self.create_subscription(
                Imu,
                self.gnss_yaw_topic,
                self.gnss_yaw_callback,
                1
            )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            1
        )

        self.map_pose_pub = self.create_publisher(PoseStamped, self.map_pose_topic, 1)
        self.gnss_pose_pub = self.create_publisher(PoseStamped, self.gnss_map_pose_topic, 1)

        # 静态 TF 广播器（用于 utm->map）
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # 动态 TF 广播器（用于 map->odom 和 odom->base_link）
        self.tf_broadcaster = TransformBroadcaster(self)

        # map->odom 偏移量（在收到首个有效 GNSS 后设置）
        self.map_to_odom_offset: tf_utils.Offset2D = tf_utils.Offset2D(tx=0.0, ty=0.0, yaw_offset=0.0)

        # 2D EKF 状态：[map_x, map_y, yaw]
        self.ekf_state: Optional[np.ndarray] = None
        self.ekf_covariance: Optional[np.ndarray] = None
        self.ekf_last_odom_stamp: Optional[int] = None
        self.ekf_last_odom_pose: Optional[tuple[float, float, float]] = None
        self.ekf_last_gnss_update_stamp: Optional[int] = None
        self.ekf_last_yaw_update_stamp: Optional[int] = None
        self._synthetic_yaw_frame_id = 'odom_synthetic_yaw'
        self._rtk_observation_events = deque()
        self._last_rtk_rejection_stats_log_nanos = 0
        # GNSS yaw 到达时刻的 odom 航向快照
        self.odom_snapshot_yaw: Optional[float] = None
        # GNSS yaw 到达时刻的 odom 位置快照 (x, y)
        self.odom_in_map_snapshot_pos: Optional[tuple[float, float]] = None
        # 定时融合
        period = 1.0 / max(self.frequency, 1e-3)
        self.timer = self.create_timer(period, self.fuse)

    def _init_logger(self, enabled: bool = True):
        """初始化日志系统"""
        self.node_logger = NodeLogger(
            node_name='ekf_fusion_node',
            log_dir=self.log_dir,
            log_timestamp=self.log_timestamp,
            enabled=enabled,
            ros_logger=self.get_logger()
        )
        self.logger = self.node_logger

        init_info = [
            'Fusion Node initialized',
            f'  工作频率: {self.frequency} Hz',
            f'  GNSS 话题: {self.gnss_topic}',
            (
                '  世界系 yaw 来源: odom yaw（要求启动时朝正东）'
                if self.use_odom_yaw_as_world_yaw
                else f'  世界系 yaw 来源: RTK yaw ({self.gnss_yaw_topic})'
            ),
            f'  ODOM 话题: {self.odom_topic}',
            f'  融合模式: {"2D_EKF" if self.use_2d_ekf else "LEGACY_DR"}',
            f'  RTK不确定性倍率: {self.rtk_uncertainty_scale}',
            f'  RTK拒绝率统计窗口: {self.rtk_rejection_stats_window}s',
            f'  GNSS 超时: {self.gnss_timeout}s',
            f'  ODOM 超时: {self.odom_timeout}s',
            f'  世界系 yaw 超时: {self.gnss_yaw_timeout}s',
            f'  日志文件: {self.node_logger.log_file}',
            f'  日志启用: {enabled}',
        ]

        self.node_logger.log_init(init_info)

    def _init_frequency_stats(self):
        """初始化各传感器和融合循环的频率统计"""
        # 融合循环频率（已知目标频率）
        self.fusion_freq_stats = FrequencyStats(
            object_name='ekf_fusion',
            target_frequency=self.frequency,
            node_logger=self.logger,
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )
        # 传感器频率（目标频率未知，仅记录实际频率）
        self.gnss_freq_stats = FrequencyStats(
            object_name='gnss',
            node_logger=self.logger,
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )
        self.rtk_yaw_freq_stats = FrequencyStats(
            object_name='rtk_yaw',
            node_logger=self.logger,
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )
        self.odom_freq_stats = FrequencyStats(
            object_name='odom',
            node_logger=self.logger,
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )

    def gnss_to_utm(self, lat: float, lon: float) -> Tuple[Optional[float], Optional[float]]:
        """经纬度转 UTM"""
        try:
            zone = int((lon + 180) / 6) + 1

            if self.utm_transformer is None or self.utm_zone != zone:
                proj_str = f"+proj=utm +zone={zone} +datum=WGS84"
                if lat < 0:
                    proj_str += " +south"

                self.utm_transformer = pyproj.Transformer.from_crs(
                    "EPSG:4326",
                    proj_str,
                    always_xy=True
                )
                self.utm_zone = zone
                self.logger.info(f'UTM transformer initialized for zone {zone}')

            utm_x, utm_y = self.utm_transformer.transform(lon, lat)
            return (utm_x, utm_y)

        except Exception as e:
            self.logger.error(f'UTM conversion failed: {e}')
            return (None, None)

    def _valid_variance(self, value, default_value: float, floor_value: float) -> float:
        """读取方差配置/消息值，遇到未知或无效值时回退到默认值。"""
        try:
            variance = float(value)
            if not math.isfinite(variance) or variance <= 0.0:
                variance = float(default_value)
        except (TypeError, ValueError):
            variance = float(default_value)

        if not math.isfinite(variance) or variance <= 0.0:
            variance = float(floor_value)
        return max(variance, float(floor_value))

    def _scale_rtk_variance(self, variance: float, floor_value: float) -> float:
        """按总体 RTK 不确定性倍率放大观测方差。"""
        scaled = float(variance) * self.rtk_uncertainty_scale
        return max(scaled, float(floor_value))

    def _gnss_position_variances(self, msg: NavSatFix) -> Tuple[float, float]:
        """从 NavSatFix 读取 ENU 平面位置方差，未知时使用配置默认值。"""
        cov_type_unknown = getattr(NavSatFix, 'COVARIANCE_TYPE_UNKNOWN', 0)
        if msg.position_covariance_type == cov_type_unknown:
            default_var = self._valid_variance(
                self.default_gnss_position_variance,
                self.default_gnss_position_variance,
                self.gnss_position_variance_floor,
            )
            scaled_var = self._scale_rtk_variance(default_var, self.gnss_position_variance_floor)
            return scaled_var, scaled_var

        cov = msg.position_covariance
        var_x = cov[0] if len(cov) > 0 else self.default_gnss_position_variance
        var_y = cov[4] if len(cov) > 4 else self.default_gnss_position_variance
        valid_x = self._valid_variance(var_x, self.default_gnss_position_variance, self.gnss_position_variance_floor)
        valid_y = self._valid_variance(var_y, self.default_gnss_position_variance, self.gnss_position_variance_floor)
        return (
            self._scale_rtk_variance(valid_x, self.gnss_position_variance_floor),
            self._scale_rtk_variance(valid_y, self.gnss_position_variance_floor),
        )

    def _yaw_variance(self, msg: Imu, scale_rtk: bool = True) -> float:
        """从 Imu.orientation_covariance 读取 yaw 方差，未知时使用配置默认值。"""
        cov = msg.orientation_covariance
        yaw_var = self.default_yaw_variance
        if len(cov) > 8 and cov[0] != -1.0:
            yaw_var = cov[8]
        valid_yaw_var = self._valid_variance(yaw_var, self.default_yaw_variance, self.yaw_variance_floor)
        if scale_rtk:
            return self._scale_rtk_variance(valid_yaw_var, self.yaw_variance_floor)
        return valid_yaw_var

    def _gnss_frame_variances(self, frame) -> Tuple[float, float]:
        data = frame.data
        if isinstance(data, tuple) and len(data) >= 6:
            return data[4], data[5]
        default_var = self._valid_variance(
            self.default_gnss_position_variance,
            self.default_gnss_position_variance,
            self.gnss_position_variance_floor,
        )
        scaled_var = self._scale_rtk_variance(default_var, self.gnss_position_variance_floor)
        return scaled_var, scaled_var

    def _yaw_frame_values(self, frame) -> Tuple[float, float, bool]:
        data = frame.data
        if isinstance(data, tuple):
            yaw = data[0]
            variance = data[1] if len(data) > 1 else self._scale_rtk_variance(
                self.default_yaw_variance,
                self.yaw_variance_floor,
            )
            odom_yaw_source = bool(data[2]) if len(data) > 2 else False
            return yaw, variance, odom_yaw_source
        return data, self._scale_rtk_variance(self.default_yaw_variance, self.yaw_variance_floor), False

    def _prune_rtk_observation_events(self, current_nanos: int):
        cutoff_nanos = current_nanos - int(self.rtk_rejection_stats_window * 1e9)
        while self._rtk_observation_events and self._rtk_observation_events[0][0] < cutoff_nanos:
            self._rtk_observation_events.popleft()

    def _record_rtk_observation(self, kind: str, rejected: bool, mahalanobis: float):
        """记录 RTK/GNSS 观测门限结果，用于滑动窗口拒绝率统计。"""
        current_nanos = TimeUtils.now_nanos()
        self._rtk_observation_events.append(
            (current_nanos, kind, bool(rejected), float(mahalanobis))
        )
        self._prune_rtk_observation_events(current_nanos)

        if self.rtk_rejection_stats_log_interval <= 0.0:
            return
        log_interval_nanos = int(self.rtk_rejection_stats_log_interval * 1e9)
        if (
            self._last_rtk_rejection_stats_log_nanos > 0 and
            current_nanos - self._last_rtk_rejection_stats_log_nanos < log_interval_nanos
        ):
            return

        self._last_rtk_rejection_stats_log_nanos = current_nanos
        self._log_rtk_rejection_stats()

    def _rtk_rejection_summary(self, kind: Optional[str] = None) -> Tuple[int, int, float, float]:
        events = [
            event for event in self._rtk_observation_events
            if kind is None or event[1] == kind
        ]
        total = len(events)
        rejected = sum(1 for event in events if event[2])
        ratio = (rejected / total * 100.0) if total > 0 else 0.0
        max_mahalanobis = max((event[3] for event in events), default=0.0)
        return total, rejected, ratio, max_mahalanobis

    def _log_rtk_rejection_stats(self):
        total, rejected, ratio, max_m = self._rtk_rejection_summary()
        pos_total, pos_rejected, pos_ratio, pos_max_m = self._rtk_rejection_summary('position')
        yaw_total, yaw_rejected, yaw_ratio, yaw_max_m = self._rtk_rejection_summary('yaw')

        self.logger.info(
            f'[RTK Rejection Stats] '
            f'window={self.rtk_rejection_stats_window:.1f}s | '
            f'rtk_uncertainty_scale={self.rtk_uncertainty_scale:.3g} | '
            f'total={rejected}/{total} ({ratio:.1f}%), max_mahalanobis={max_m:.3f} | '
            f'position={pos_rejected}/{pos_total} ({pos_ratio:.1f}%), max={pos_max_m:.3f} | '
            f'yaw={yaw_rejected}/{yaw_total} ({yaw_ratio:.1f}%), max={yaw_max_m:.3f}'
        )

    def gnss_callback(self, msg: NavSatFix):
        """GNSS 回调：入队 + 清理过期帧 + 处理数据"""
        current_nanos = TimeUtils.now_nanos()

        lat = msg.latitude
        lon = msg.longitude

        gnss_valid = msg.status.status >= 0

        # 入队（带时间戳，初始化和正常运行都入队）
        utm_x=None
        utm_y=None
        if gnss_valid:
            utm_x, utm_y = self.gnss_to_utm(lat, lon)
            if utm_x is not None and utm_y is not None:
                var_x, var_y = self._gnss_position_variances(msg)
                self._gnss_queue.append(current_nanos, (utm_x, utm_y, lat, lon, var_x, var_y))
            else:
                return
        else:
            self.logger.error(f'GNSS invalid: {msg.status.status}')
            return

        # 若初始化还未完成
        if not self.tf_init_completed:
            return

        # legacy 模式中，GNSS 回调只更新 map->odom 偏移和 odom 锚点。
        # map->odom 动态 TF 统一在 fuse 循环中发布。
        if not self.use_2d_ekf:
            gnss_yaw_frame = self._gnss_yaw_queue.find_nearest(current_nanos)
            odom_frame = self._odom_queue.find_nearest(current_nanos)
            if gnss_yaw_frame and odom_frame:
                gnss_yaw, _, _ = self._yaw_frame_values(gnss_yaw_frame)
                odom_x, odom_y, odom_yaw = odom_frame.data
                gnss_map_x, gnss_map_y, _ = tf_utils.transform_from_parent_to_child(
                    utm_x, utm_y, 0.0, self.utm_to_map_offset
                )
                # 计算map_to_odom_offset
                self.map_to_odom_offset = tf_utils.compute_offset_at_same_point(
                    gnss_map_x, gnss_map_y, gnss_yaw,
                    odom_x, odom_y, odom_yaw
                )
                # 记录此刻 odom 在 map 坐标系中的位置
                odom_in_map_x, odom_in_map_y, _ = tf_utils.transform_from_child_to_parent(
                    odom_x, odom_y, 0.0, self.map_to_odom_offset
                )
                self.odom_in_map_snapshot_pos = (odom_in_map_x, odom_in_map_y)
                # map->odom 动态 TF 统一在 legacy fuse 循环中发布

        self.gnss_freq_stats.tick()

    def tf_init(self) -> bool:
        """
        TF初始化函数：在等待期间收集数据后，直接平均建立tf变换。

        工作流程：
        1. 直接平均得到平均gnss，建立utm->map和地图原点
        2. 直接平均得到平均世界系朝向
        3. 直接平均得到平均odom数据，建立map->odom

        Returns:
            True: 初始化成功
            False: 初始化失败（数据不足）
        """
        if self.tf_init_completed:
            self.logger.warning('tf_init already completed, skipping')
            return True


        try:
            # ============================================================
            # 步骤1：建立并发布utm->map
            # ============================================================
            if self._gnss_queue.is_empty:
                return False

            # 计算直接平均GNSS
            sum_utm_x = 0.0
            sum_utm_y = 0.0
            sum_lat = 0.0
            sum_lon = 0.0

            for frame in self._gnss_queue.get_all():
                sum_utm_x += frame.data[0]
                sum_utm_y += frame.data[1]
                sum_lat += frame.data[2]
                sum_lon += frame.data[3]

            num_samples = self._gnss_queue.size
            avg_utm_x = sum_utm_x / num_samples
            avg_utm_y = sum_utm_y / num_samples
            avg_lat = sum_lat / num_samples
            avg_lon = sum_lon / num_samples

            self.logger.info(
                f'tf_init GNSS: collected={num_samples} samples, '
                f'avg=UTM({avg_utm_x:.3f}, {avg_utm_y:.3f})'
            )

            # 建立utm->map和地图原点
            self.utm_to_map_offset = tf_utils.compute_offset_at_same_point(
                avg_utm_x, avg_utm_y, 0.0,
                0.0, 0.0, 0.0
            )
            self.logger.info(f'Map origin set: UTM({avg_utm_x:.3f}, {avg_utm_y:.3f}), GNSS(lat={avg_lat:.8f}, lon={avg_lon:.8f})')
            self.get_logger().info(f'Map origin set: UTM({avg_utm_x:.3f}, {avg_utm_y:.3f}), GNSS(lat={avg_lat:.8f}, lon={avg_lon:.8f})')
            self.static_tf_broadcaster.sendTransform(
                tf_utils.offset_to_transform(self.utm_to_map_offset, 'utm', 'map', TimeUtils.now_nanos())
            )
            self.logger.info(
                f'tf_init completed successfully: '
                f'utm->map origin: UTM({avg_utm_x:.3f}, {avg_utm_y:.3f})'
            )
            self.get_logger().info('tf_init completed successfully')

            self.tf_init_completed = True
            return True

        except Exception as e:
            return False


    def odom_callback(self, msg: Odometry):
        """ODOM 回调：读取速度、位置、朝向，并发布 odom->base_link TF"""
        current_nanos = TimeUtils.now_nanos()

        # ODOM时间戳使用接收时刻（不使用数据源时间戳）
        odom_stamp = current_nanos

        odom_pose_x = msg.pose.pose.position.x
        odom_pose_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        odom_pose_yaw = tf_utils.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # 入队（带时间戳，初始化和正常运行都入队）
        self._odom_queue.append(odom_stamp, (odom_pose_x, odom_pose_y, odom_pose_yaw))

        offset = tf_utils.compute_offset_at_same_point(odom_pose_x, odom_pose_y, odom_pose_yaw, 0.0, 0.0, 0.0)
        t = tf_utils.offset_to_transform(offset, 'odom', 'base_link', stamp_nanosec=odom_stamp)
        self.tf_broadcaster.sendTransform(t)

        if self.use_odom_yaw_as_world_yaw:
            imu_msg = Imu()
            imu_msg.header.stamp = TimeUtils.nanos_to_stamp(odom_stamp)
            imu_msg.header.frame_id = self._synthetic_yaw_frame_id
            quat = tf_utils.yaw_to_quaternion(odom_pose_yaw)
            imu_msg.orientation.x = quat[0]
            imu_msg.orientation.y = quat[1]
            imu_msg.orientation.z = quat[2]
            imu_msg.orientation.w = quat[3]
            self.gnss_yaw_callback(imu_msg)

        self.odom_freq_stats.tick()

    def gnss_yaw_callback(self, msg: Imu):
        """世界系 yaw 回调：RTK yaw 或 odom-as-world-yaw 模式伪造的 Imu。"""
        current_nanos = TimeUtils.now_nanos()

        q = msg.orientation
        gnss_yaw = tf_utils.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        gnss_yaw_stamp = current_nanos
        odom_yaw_source = msg.header.frame_id == self._synthetic_yaw_frame_id
        yaw_var = self._yaw_variance(msg, scale_rtk=not odom_yaw_source)

        # 入队（带时间戳，初始化和正常运行都入队）
        self._gnss_yaw_queue.append(gnss_yaw_stamp, (gnss_yaw, yaw_var, odom_yaw_source))

        # tf_init完成后：同步更新 odom_snapshot_yaw（用于 fusion 中的 yaw DR）
        if not self.tf_init_completed:
            return

        odom_frame = self._odom_queue.find_nearest(gnss_yaw_stamp)
        if odom_frame:
            self.odom_snapshot_yaw = odom_frame.data[2]

        if not odom_yaw_source:
            self.rtk_yaw_freq_stats.tick()

    def fuse(self):
        """融合主入口：根据配置选择 legacy DR 或 2D EKF。"""
        if self.use_2d_ekf:
            self._fuse_ekf()
        else:
            self._fuse_legacy()

    def _fuse_legacy(self):
        """
        基于航位推算（DR）的融合策略。

        位置：GNSS 到达时发布 utm->map 静态 TF，之后 map->odom 由 odom 里程计持续更新
        朝向：世界系 yaw 源（RTK yaw 或 odom-as-world-yaw）+ odom yaw 增量

        所有增量计算均在 map 坐标系中进行，确保坐标系一致性。
        """


        if not self.tf_init_completed:
            return

        if not self.utm_to_map_offset or not self.map_to_odom_offset:
            return

        current_nanos = TimeUtils.now_nanos()

        # 清理过期帧
        self._gnss_queue.prune_expired(current_nanos)
        self._odom_queue.prune_expired(current_nanos)
        self._gnss_yaw_queue.prune_expired(current_nanos)

        # 从队列获取最新数据
        gnss_frame = self._gnss_queue.get_latest()
        odom_frame = self._odom_queue.get_latest()
        gnss_yaw_frame = self._gnss_yaw_queue.get_latest()
        if odom_frame is None or gnss_frame is None or gnss_yaw_frame is None:
            self.logger.error(f'No valid data found for fusion')
            return
        odom_pose_x, odom_pose_y, odom_pose_yaw = odom_frame.data
        gnss_utm_x, gnss_utm_y = gnss_frame.data[0], gnss_frame.data[1]
        gnss_yaw, _, _ = self._yaw_frame_values(gnss_yaw_frame)

        # 航位推算计算
        # yaw 锚点有效才能进行融合
        if self.odom_snapshot_yaw is None or self.odom_in_map_snapshot_pos is None:
            self.logger.error(f'No valid odom snapshot yaw or odom in map snapshot pos found for fusion')
            return

        map_x: float = 0.0
        map_y: float = 0.0
        yaw: float = 0.0
        fusion_components: list = []

        # ============================================================
        # 位置 DR：GNSS 锚点 + odom 增量（在 map 坐标系中计算）
        # ============================================================

        # 将当前 odom 位置转换到 map 坐标系
        current_odom_in_map_x, current_odom_in_map_y, _ = tf_utils.transform_from_child_to_parent(
            odom_pose_x, odom_pose_y, 0.0, self.map_to_odom_offset
        )
        gnss_map_x, gnss_map_y, _ = tf_utils.transform_from_parent_to_child(
            gnss_utm_x, gnss_utm_y, 0.0, self.utm_to_map_offset
        )
        map_x = gnss_map_x+current_odom_in_map_x-self.odom_in_map_snapshot_pos[0]
        map_y = gnss_map_y+current_odom_in_map_y-self.odom_in_map_snapshot_pos[1]  
        fusion_components.append('POS_DR')

        # 发布 GNSS 原始位姿（用于可视化对比）
        self._publish_gnss_pose(gnss_map_x, gnss_map_y, gnss_yaw)

        # ============================================================
        # 航向：世界系 yaw 源（RTK yaw 或 odom-as-world-yaw）+ odom yaw 增量
        # ============================================================

        delta_odom_yaw = tf_utils.normalize_angle(odom_pose_yaw - self.odom_snapshot_yaw)
        yaw = tf_utils.normalize_angle(gnss_yaw + delta_odom_yaw)
        fusion_components.append('YAW_DR')

        fusion_mode = '+'.join(fusion_components) if fusion_components else 'INVALID'

        t = tf_utils.offset_to_transform(
            self.map_to_odom_offset,
            'map',
            'odom',
            stamp_nanosec=current_nanos,
        )
        self.tf_broadcaster.sendTransform(t)

        self.publish_fusion_result(map_x, map_y, yaw)

        self.logger.debug(
            f'Fusion | {fusion_mode} | '
            f'Pos=({map_x:.3f}, {map_y:.3f}) | '
            f'Yaw={math.degrees(yaw):.1f} deg'
        )

        self.fusion_freq_stats.tick()

    def _fuse_ekf(self):
        """2D EKF 融合：odom 预测，GNSS 位置和 RTK yaw 观测更新。"""
        if not self.tf_init_completed:
            return

        if not self.utm_to_map_offset:
            return

        current_nanos = TimeUtils.now_nanos()

        # 清理过期帧
        self._gnss_queue.prune_expired(current_nanos)
        self._odom_queue.prune_expired(current_nanos)
        self._gnss_yaw_queue.prune_expired(current_nanos)

        odom_frame = self._odom_queue.get_latest()
        gnss_frame = self._gnss_queue.get_latest()
        gnss_yaw_frame = self._gnss_yaw_queue.get_latest()

        if odom_frame is None:
            self.logger.error('No valid odom found for 2D EKF fusion')
            return

        if gnss_frame is not None and gnss_yaw_frame is not None:
            gnss_map_x, gnss_map_y, _ = tf_utils.transform_from_parent_to_child(
                gnss_frame.data[0], gnss_frame.data[1], 0.0, self.utm_to_map_offset
            )
            gnss_yaw, _, _ = self._yaw_frame_values(gnss_yaw_frame)
            self._publish_gnss_pose(gnss_map_x, gnss_map_y, gnss_yaw)

        if self.ekf_state is None:
            if gnss_frame is None or gnss_yaw_frame is None:
                self.logger.error('No valid GNSS/yaw data found for 2D EKF initialization')
                return
            if not self._initialize_ekf(gnss_frame, gnss_yaw_frame, odom_frame):
                return
        else:
            self._predict_ekf_with_odom(odom_frame)
            if gnss_frame is not None:
                self._update_ekf_position(gnss_frame)
            if gnss_yaw_frame is not None:
                self._update_ekf_yaw(gnss_yaw_frame)

        self._broadcast_map_to_odom_from_ekf(odom_frame, current_nanos)
        self.publish_fusion_result(
            float(self.ekf_state[0]),
            float(self.ekf_state[1]),
            float(self.ekf_state[2]),
        )

        self.logger.debug(
            f'Fusion | 2D_EKF | '
            f'Pos=({self.ekf_state[0]:.3f}, {self.ekf_state[1]:.3f}) | '
            f'Yaw={math.degrees(self.ekf_state[2]):.1f} deg'
        )

        self.fusion_freq_stats.tick()

    def _initialize_ekf(self, gnss_frame, gnss_yaw_frame, odom_frame) -> bool:
        """使用最新 GNSS 位置、yaw 和 odom 快照初始化 EKF。"""
        try:
            gnss_map_x, gnss_map_y, _ = tf_utils.transform_from_parent_to_child(
                gnss_frame.data[0], gnss_frame.data[1], 0.0, self.utm_to_map_offset
            )
            gnss_yaw, yaw_var, odom_yaw_source = self._yaw_frame_values(gnss_yaw_frame)
            var_x, var_y = self._gnss_frame_variances(gnss_frame)

            self.ekf_state = np.array(
                [gnss_map_x, gnss_map_y, tf_utils.normalize_angle(gnss_yaw)],
                dtype=float,
            )
            self.ekf_covariance = np.diag([var_x, var_y, yaw_var]).astype(float)

            self.ekf_last_odom_stamp = odom_frame.stamp_nanos
            self.ekf_last_odom_pose = tuple(odom_frame.data)
            self.ekf_last_gnss_update_stamp = gnss_frame.stamp_nanos
            self.ekf_last_yaw_update_stamp = gnss_yaw_frame.stamp_nanos

            self.logger.info(
                f'2D EKF initialized: '
                f'Pos=({gnss_map_x:.3f}, {gnss_map_y:.3f}), '
                f'Yaw={math.degrees(gnss_yaw):.1f} deg, '
                f'P=diag({var_x:.6g}, {var_y:.6g}, {yaw_var:.6g}), '
                f'yaw_source={"odom_as_world" if odom_yaw_source else "rtk"}'
            )
            return True
        except Exception as e:
            self.logger.error(f'2D EKF initialization failed: {e}')
            return False

    def _predict_ekf_with_odom(self, odom_frame) -> bool:
        """用 odom 两帧之间的增量预测 EKF 状态。"""
        if self.ekf_state is None or self.ekf_covariance is None:
            return False

        if self.ekf_last_odom_stamp == odom_frame.stamp_nanos:
            return False

        current_odom_pose = tuple(odom_frame.data)
        if self.ekf_last_odom_pose is None:
            self.ekf_last_odom_pose = current_odom_pose
            self.ekf_last_odom_stamp = odom_frame.stamp_nanos
            return False

        prev_odom_x, prev_odom_y, prev_odom_yaw = self.ekf_last_odom_pose
        odom_x, odom_y, odom_yaw = current_odom_pose

        dx_odom = odom_x - prev_odom_x
        dy_odom = odom_y - prev_odom_y
        delta_odom_yaw = tf_utils.normalize_angle(odom_yaw - prev_odom_yaw)

        yaw_offset = tf_utils.normalize_angle(float(self.ekf_state[2]) - prev_odom_yaw)
        cos_yaw = math.cos(yaw_offset)
        sin_yaw = math.sin(yaw_offset)

        dx_map = cos_yaw * dx_odom - sin_yaw * dy_odom
        dy_map = sin_yaw * dx_odom + cos_yaw * dy_odom

        f_jacobian = np.eye(3)
        f_jacobian[0, 2] = -sin_yaw * dx_odom - cos_yaw * dy_odom
        f_jacobian[1, 2] = cos_yaw * dx_odom - sin_yaw * dy_odom

        distance = math.hypot(dx_odom, dy_odom)
        q_pos = max(self.process_position_variance_min, self.process_position_variance_per_meter * distance)
        q_yaw = max(self.process_yaw_variance_min, self.process_yaw_variance_per_rad * abs(delta_odom_yaw))
        process_noise = np.diag([q_pos, q_pos, q_yaw])

        self.ekf_state[0] += dx_map
        self.ekf_state[1] += dy_map
        self.ekf_state[2] = tf_utils.normalize_angle(float(self.ekf_state[2]) + delta_odom_yaw)
        self.ekf_covariance = f_jacobian @ self.ekf_covariance @ f_jacobian.T + process_noise
        self.ekf_covariance = 0.5 * (self.ekf_covariance + self.ekf_covariance.T)

        self.ekf_last_odom_pose = current_odom_pose
        self.ekf_last_odom_stamp = odom_frame.stamp_nanos
        return True

    def _update_ekf_position(self, gnss_frame) -> bool:
        """使用 GNSS map 平面位置更新 EKF。"""
        if self.ekf_state is None or self.ekf_covariance is None:
            return False

        if self.ekf_last_gnss_update_stamp == gnss_frame.stamp_nanos:
            return False

        gnss_map_x, gnss_map_y, _ = tf_utils.transform_from_parent_to_child(
            gnss_frame.data[0], gnss_frame.data[1], 0.0, self.utm_to_map_offset
        )
        var_x, var_y = self._gnss_frame_variances(gnss_frame)
        innovation = np.array([gnss_map_x - self.ekf_state[0], gnss_map_y - self.ekf_state[1]], dtype=float)
        h_jacobian = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        measurement_noise = np.diag([var_x, var_y])

        self.ekf_last_gnss_update_stamp = gnss_frame.stamp_nanos
        mahalanobis = self._mahalanobis_distance(innovation, h_jacobian, measurement_noise)
        if mahalanobis is None:
            return False
        rejected = mahalanobis > self.max_position_mahalanobis
        self._record_rtk_observation('position', rejected, mahalanobis)
        if rejected:
            self.logger.warning(
                f'2D EKF rejected GNSS position observation: '
                f'mahalanobis={mahalanobis:.3f} > {self.max_position_mahalanobis:.3f}'
            )
            return False

        return self._apply_ekf_update(h_jacobian, innovation, measurement_noise)

    def _update_ekf_yaw(self, gnss_yaw_frame) -> bool:
        """使用 RTK yaw 更新 EKF；odom-as-world-yaw 模式只参与初始化，之后由 odom 预测延续。"""
        if self.ekf_state is None or self.ekf_covariance is None:
            return False

        if self.ekf_last_yaw_update_stamp == gnss_yaw_frame.stamp_nanos:
            return False

        gnss_yaw, yaw_var, odom_yaw_source = self._yaw_frame_values(gnss_yaw_frame)
        self.ekf_last_yaw_update_stamp = gnss_yaw_frame.stamp_nanos
        if odom_yaw_source:
            return False

        innovation = np.array([tf_utils.normalize_angle(gnss_yaw - self.ekf_state[2])], dtype=float)
        h_jacobian = np.array([[0.0, 0.0, 1.0]], dtype=float)
        measurement_noise = np.array([[yaw_var]], dtype=float)

        mahalanobis = self._mahalanobis_distance(innovation, h_jacobian, measurement_noise)
        if mahalanobis is None:
            return False
        rejected = mahalanobis > self.max_yaw_mahalanobis
        self._record_rtk_observation('yaw', rejected, mahalanobis)
        if rejected:
            self.logger.warning(
                f'2D EKF rejected yaw observation: '
                f'mahalanobis={mahalanobis:.3f} > {self.max_yaw_mahalanobis:.3f}'
            )
            return False

        return self._apply_ekf_update(h_jacobian, innovation, measurement_noise)

    def _mahalanobis_distance(self, innovation: np.ndarray, h_jacobian: np.ndarray, measurement_noise: np.ndarray) -> Optional[float]:
        try:
            innovation_covariance = h_jacobian @ self.ekf_covariance @ h_jacobian.T + measurement_noise
            inv_covariance = np.linalg.inv(innovation_covariance)
            distance = float(innovation.T @ inv_covariance @ innovation)
            return distance if math.isfinite(distance) else None
        except np.linalg.LinAlgError as e:
            self.logger.warning(f'2D EKF innovation covariance not invertible: {e}')
            return None

    def _apply_ekf_update(self, h_jacobian: np.ndarray, innovation: np.ndarray, measurement_noise: np.ndarray) -> bool:
        """Joseph form 更新，降低协方差数值误差。"""
        try:
            innovation_covariance = h_jacobian @ self.ekf_covariance @ h_jacobian.T + measurement_noise
            kalman_gain = self.ekf_covariance @ h_jacobian.T @ np.linalg.inv(innovation_covariance)
            identity = np.eye(3)
            gain_h = kalman_gain @ h_jacobian

            self.ekf_state = self.ekf_state + kalman_gain @ innovation
            self.ekf_state[2] = tf_utils.normalize_angle(float(self.ekf_state[2]))
            self.ekf_covariance = (
                (identity - gain_h) @ self.ekf_covariance @ (identity - gain_h).T
                + kalman_gain @ measurement_noise @ kalman_gain.T
            )
            self.ekf_covariance = 0.5 * (self.ekf_covariance + self.ekf_covariance.T)
            return True
        except np.linalg.LinAlgError as e:
            self.logger.warning(f'2D EKF update failed: {e}')
            return False

    def _broadcast_map_to_odom_from_ekf(self, odom_frame, stamp_nanosec: int):
        """用 EKF map 位姿和当前 odom 位姿发布一致的 map->odom TF。"""
        if self.ekf_state is None:
            return

        odom_x, odom_y, odom_yaw = odom_frame.data
        self.map_to_odom_offset = tf_utils.compute_offset_at_same_point(
            float(self.ekf_state[0]),
            float(self.ekf_state[1]),
            float(self.ekf_state[2]),
            odom_x,
            odom_y,
            odom_yaw,
        )
        t = tf_utils.offset_to_transform(
            self.map_to_odom_offset,
            'map',
            'odom',
            stamp_nanosec=stamp_nanosec,
        )
        self.tf_broadcaster.sendTransform(t)

    def publish_fusion_result(self, map_x: float, map_y: float, yaw: float):
        """发布 map 位姿"""
        if not self.utm_to_map_offset:
            return

        # 使用当前时间作为时间戳
        pose_stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())

        pose_msg = PoseStamped()
        pose_msg.header.stamp = pose_stamp
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = map_x
        pose_msg.pose.position.y = map_y
        pose_msg.pose.position.z = 0.0

        quat = tf_utils.yaw_to_quaternion(yaw)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.map_pose_pub.publish(pose_msg)

    def _publish_gnss_pose(self, gnss_x: float, gnss_y: float, gnss_yaw: float):
        """发布 GNSS 原始位姿（用于可视化对比）"""
        pose_stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())

        pose_msg = PoseStamped()
        pose_msg.header.stamp = pose_stamp
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = gnss_x
        pose_msg.pose.position.y = gnss_y
        pose_msg.pose.position.z = 0.0

        quat = tf_utils.yaw_to_quaternion(gnss_yaw)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.gnss_pose_pub.publish(pose_msg)

def run_ekf_fusion_node(log_dir: str = None, log_timestamp: str = None, args=None):
    """运行节点

    Args:
        log_dir: 日志目录
        log_timestamp: 日志时间戳
        args: ROS 参数
    """
    rclpy.init(args=args)
    node = EKFFusionNode(log_dir=log_dir, log_timestamp=log_timestamp)

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # 启动前短暂等待收集初始数据
    node.get_logger().info('Waiting for initial sensor data collection...')
    start_time = TimeUtils.now_nanos()
    timeout_ns = 3_000_000_000

    while rclpy.ok():
        current_time = TimeUtils.now_nanos()
        if current_time - start_time > timeout_ns:
            break
        executor.spin_once(timeout_sec=0.1)

    # 等待结束后执行TF初始化
    node.get_logger().info('Initial data collection complete. Running tf_init...')
    if not node.tf_init():
        node.get_logger().error('tf_init failed! Node will continue but may not function correctly.')

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def main():
    """独立运行入口（使用默认日志目录）"""
    run_ekf_fusion_node()


if __name__ == '__main__':
    main()
