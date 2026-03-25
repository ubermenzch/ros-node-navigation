#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化定位融合节点（适用于：只有世界系经纬度 + 世界系朝向，无可信协方差）

功能：
1. 记录启动后接收到的第一个有效 /rtk_fix，经纬度转 UTM，作为 map 原点
2. 发布 utm -> map 静态 TF（零旋转）
3. 持续接收 GPS / ODOM / 世界系朝向
   - GPS: 提供 map 下绝对位置
   - 世界系朝向: 直接提供 map 下绝对朝向（来自 RTK 计算输出）
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
- gps_map_pose: RTK 位姿转换到 map 坐标系
- 最终 map_pose = (1-alpha) * odom_map_pose + alpha * gps_map_pose
"""

import math
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile

from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from builtin_interfaces.msg import Time

# UTM 库
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    logging.warning("pyproj not installed, UTM conversion will not work")

from config_loader import get_config
from frequency_stats import FrequencyStats
from utils.time_utils import TimeUtils


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi]"""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(q: Quaternion) -> float:
    """四元数转 yaw"""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """yaw 转四元数（绕 z 轴旋转）"""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


@dataclass
class SensorData:
    """最新传感器数据（仅用于回调中提取原始数据，不做融合状态存储）"""
    # GPS
    gps_stamp: int = 0          # int64 纳秒
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt: float = 0.0

    # ODOM（仅使用速度）
    odom_stamp: int = 0         # int64 纳秒
    odom_vx: float = 0.0
    odom_vy: float = 0.0
    odom_vyaw: float = 0.0

    # ODOM 完整数据（用于 TF 发布）
    odom_pose_x: float = 0.0  # odom 坐标系下的 x
    odom_pose_y: float = 0.0  # odom 坐标系下的 y
    odom_pose_yaw: float = 0.0  # odom 坐标系下的 yaw

    # 世界系朝向（来自 RTK 计算输出）
    world_orientation_stamp: int = 0  # int64 纳秒
    world_orientation_yaw: float = 0.0


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
        self.gps_topic = subscriptions.get('gps_topic', '/rtk_fix')
        # 世界系方向话题（RTK计算输出的世界系姿态，虽然topic名叫rtk_imu但是世界系方向）
        self.world_orientation_topic = subscriptions.get('world_orientation_topic', '/rtk_imu')
        self.odom_topic = subscriptions.get('odom_topic', '/utlidar/robot_odom')

        # 发布话题
        self.map_pose_topic = publications.get('map_pose_topic', '/navigation/map_pose')
        self.gps_map_pose_topic = publications.get('gps_pose_topic', '/navigation/gps_pose')

        # 运行参数
        self.frequency = ekf_config.get('frequency', 10.0)
        self.gps_timeout = ekf_config.get('gps_timeout', 2.0)
        self.odom_timeout = ekf_config.get('odom_timeout', 1.0)
        self.world_orientation_timeout = ekf_config.get('world_orientation_timeout', 5.0)

        # 启动时是否朝正东
        # 当设为 true 时，机器人在启动时默认朝向正东，此时里程计 yaw=0 对应世界系 yaw=0
        # 当设为 false 时，使用 world_orientation_topic 提供世界系朝向
        self.face_east_on_startup = bool(ekf_config.get('face_east_on_startup', False))

        self.log_frequency_stats = bool(ekf_config.get('log_frequency_stats', True))
        self.log_enabled = bool(ekf_config.get('log_enabled', True))

        # TF初始化状态
        self.tf_init_flag = False        # 是否已完成tf初始化
        self.tf_init_completed = False    # tf初始化是否已完成

        # 数据采集队列（用于等待期间的加权平均）
        self._gps_queue = []             # [(utm_x, utm_y, lat, lon), ...]
        self._world_yaw_queue = []        # [yaw, ...]（不带卫星数，直接平均）
        self._odom_queue = []            # [(odom_x, odom_y, odom_yaw), ...]

        # UTM转换相关
        self.utm_zone: Optional[int] = None
        self.utm_transformer = None
        self.map_origin_utm = None   # {'easting': x, 'northing': y}
        self.map_origin_set = False
        self.map_to_odom_published = False
        self.utm_to_map_published = False

        # 传感器数据
        self.latest_sensor_data = SensorData()

        # 最近消息到达时间（本地时间，用于超时判断）
        self.last_gps_time = 0  # int64 纳秒

        # 日志
        self._init_logger(self.log_enabled)

        # 频率统计
        self._init_frequency_stats()

        # 订阅者
        self.gps_sub = self.create_subscription(
            NavSatFix,
            self.gps_topic,
            self.gps_callback,
            1
        )
        if self.face_east_on_startup:
            self.world_orientation_sub = None
        else:
            self.world_orientation_sub = self.create_subscription(
                Imu,
                self.world_orientation_topic,
                self.world_orientation_callback,
                1
            )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            1
        )

        # 发布者 - 使用 transient local 确保晚加入的订阅者(如 rviz)能获取最新数据
        qos_transient_local = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pose_pub = self.create_publisher(PoseStamped, self.map_pose_topic, qos_transient_local)
        self.gps_pose_pub = self.create_publisher(PoseStamped, self.gps_map_pose_topic, 10)

        # 静态 TF 广播器
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # 动态 TF 广播器（用于 odom->base_link）
        self.dynamic_tf_broadcaster = TransformBroadcaster(self)

        # map->odom 偏移量（在收到首个有效 GPS 后设置）
        self.map_to_odom_offset_x = 0.0
        self.map_to_odom_offset_y = 0.0
        self.map_to_odom_offset_yaw = 0.0  # odom 初始朝向

        # GPS 到达时记录锚点
        # GPS 在 map 坐标系的绝对位置
        self.gps_map_x: Optional[float] = None
        self.gps_map_y: Optional[float] = None
        # GPS 到达时刻的 map 坐标系快照（用于 DR 增量计算）
        self.odom_in_map_snapshot_x: Optional[float] = None
        self.odom_in_map_snapshot_y: Optional[float] = None

        # world_orientation 到达时刻的 odom 航向快照
        self.odom_snapshot_yaw: Optional[float] = None
        # 定时融合
        period = 1.0 / max(self.frequency, 1e-3)
        self.timer = self.create_timer(period, self.fuse)

        # 主循环频率统计
        self.node_freq_stats = FrequencyStats(
            node_name='ekf_fusion_node',
            target_frequency=self.frequency,
            logger=self.logger,
            ros_logger=self.get_logger(),
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )

    def _init_logger(self, enabled: bool = True):
        """初始化日志系统"""
        # 使用传入的日志目录或创建新的
        if self.log_dir is not None:
            log_dir = self.log_dir
        else:
            ts = self.log_timestamp
            log_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'logs',
                f'navigation_{ts}'
            )
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'ekf_fusion_node_log_{self.log_timestamp}.log')

        self.logger = logging.getLogger('ekf_fusion_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        if enabled:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        init_info = [
            'Fusion Node initialized',
            f'  工作频率: {self.frequency} Hz',
            f'  GPS 话题: {self.gps_topic}',
            f'  世界系朝向话题: {self.world_orientation_topic}',
            f'  ODOM 话题: {self.odom_topic}',
            f'  GPS 超时: {self.gps_timeout}s',
            f'  ODOM 超时: {self.odom_timeout}s',
            f'  世界朝向超时: {self.world_orientation_timeout}s',
            f'  日志文件: {log_file}',
            f'  日志启用: {enabled}',
        ]

        for line in init_info:
            self.logger.info(line)
            self.get_logger().info(line)

    def _init_frequency_stats(self):
        """初始化频率统计"""
        self.freq_stats = {
            'gps':  {'last_time': 0, 'count': 0, 'last_log_time': 0},
            'imu':  {'last_time': 0, 'count': 0, 'last_log_time': 0},
            'odom': {'last_time': 0, 'count': 0, 'last_log_time': 0},
        }

    def _update_frequency_stats(self, sensor_type: str):
        current_nanos = TimeUtils.now_nanos()
        current_sec = current_nanos / 1e9
        stats = self.freq_stats[sensor_type]
        if stats['last_time'] > 0:
            dt = current_sec - stats['last_time'] / 1e9
            if dt > 0:
                stats['count'] += 1
        stats['last_time'] = current_nanos

    def _log_frequency_stats(self):
        if not self.log_frequency_stats:
            return

        current_nanos = TimeUtils.now_nanos()
        current_sec = current_nanos / 1e9
        log_interval = 10.0

        for sensor_type in ['gps', 'imu', 'odom']:
            stats = self.freq_stats[sensor_type]
            last_log_sec = stats['last_log_time'] / 1e9
            if current_sec - last_log_sec >= log_interval:
                avg_freq = stats['count'] / log_interval
                self.logger.info(
                    f'{sensor_type.upper()} Frequency: {avg_freq:.2f} Hz (samples={stats["count"]})'
                )
                stats['count'] = 0
                stats['last_log_time'] = current_nanos

    def gps_to_utm(self, lat: float, lon: float) -> Tuple[Optional[float], Optional[float]]:
        """经纬度转 UTM"""
        if not HAS_PYPROJ:
            self.logger.error('pyproj not installed, cannot convert GPS to UTM')
            return (None, None)

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

    def gps_callback(self, msg: NavSatFix):
        """GPS 回调：记录 GPS 在 map 坐标系的绝对位置作为位置锚点"""
        current_nanos = TimeUtils.now_nanos()
        self._update_frequency_stats('gps')

        # GPS时间戳使用接收时刻（不使用数据源时间戳）
        gps_stamp = current_nanos

        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        gps_valid = (
            msg.status.status >= 0 and
            math.isfinite(lat) and
            math.isfinite(lon) and
            not (abs(lat) < 1e-12 and abs(lon) < 1e-12)
        )

        self.latest_sensor_data.gps_stamp = gps_stamp
        self.latest_sensor_data.gps_lat = lat
        self.latest_sensor_data.gps_lon = lon
        self.latest_sensor_data.gps_alt = alt if math.isfinite(alt) else 0.0
        self.last_gps_time = current_nanos

        # ============================================================
        # 初始化期间：收集数据用于直接平均
        # ============================================================
        if gps_valid and not self.tf_init_completed and not self.tf_init_flag:
            utm_x, utm_y = self.gps_to_utm(lat, lon)
            if utm_x is not None and utm_y is not None:
                self._gps_queue.append((utm_x, utm_y, lat, lon))
                self.logger.debug(
                    f'GPS collected for tf_init: '
                    f'UTM({utm_x:.3f}, {utm_y:.3f}), queue_size={len(self._gps_queue)}'
                )
            return  # 初始化期间不执行后续逻辑

        # tf_init完成后：使用原有逻辑
        if self.tf_init_completed:
            # 每次有效GPS且 map原点已设置时，更新位置锚点
            self._update_position_anchor(lat, lon)

    def _update_position_anchor(self, lat: float, lon: float):
        """
        更新位置锚点：记录当前 GPS 在 map 坐标系下的位置，
        以及此时 odom 在 map 坐标系下的位置快照。
        """
        if not( self.map_origin_set and self.utm_to_map_published):
            return

        utm_x, utm_y = self.gps_to_utm(lat, lon)
        if utm_x is None or utm_y is None:
            return

        # GPS 在 map 坐标系下的位置
        gps_map_x, gps_map_y, _ = self._transform_utm_to_map(utm_x, utm_y, 0.0)

        odom_pose_x = self.latest_sensor_data.odom_pose_x
        odom_pose_y = self.latest_sensor_data.odom_pose_y

        self.gps_map_x = gps_map_x
        self.gps_map_y = gps_map_y

        if self.map_to_odom_published:
            odom_in_map_snapshot_x, odom_in_map_snapshot_y, _ = self._transform_odom_to_map(
                odom_pose_x, odom_pose_y, 0.0
            )
            self.odom_in_map_snapshot_x = odom_in_map_snapshot_x
            self.odom_in_map_snapshot_y = odom_in_map_snapshot_y

            self.logger.debug(
                f'GPS anchor updated: gps_map=({gps_map_x:.3f}, {gps_map_y:.3f}), '
                f'map_snapshot=({odom_in_map_snapshot_x:.3f}, {odom_in_map_snapshot_y:.3f})'
            )
        else:
            self.logger.warning('GPS anchor skipped: map->odom not published yet')

    def _set_map_origin(self, utm_x: float, utm_y: float, lat: float, lon: float):
        """设置 map 原点"""
        self.map_origin_utm = {
            'easting': utm_x,
            'northing': utm_y,
        }

        msg = (
            f'Map origin set: '
            f'UTM({utm_x:.3f}, {utm_y:.3f}), '
            f'GPS(lat={lat:.8f}, lon={lon:.8f})'
        )
        self.logger.info(msg)
        self.get_logger().info(msg)
        self.map_origin_set = True
        self._publish_utm_to_map_transform(utm_x, utm_y)

    def _publish_utm_to_map_transform(self, utm_x: float, utm_y: float):
        """
        发布 utm -> map 静态 TF

        语义：
        - 父坐标系: utm
        - 子坐标系: map
        - map 原点在 utm 中的位置 = (utm_x, utm_y)
        - map 与 utm 轴对齐，零旋转

        因此：
        p_utm = p_map + [utm_x, utm_y]
        p_map = p_utm - [utm_x, utm_y]
        """
        t = TransformStamped()
        t.header.stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
        t.header.frame_id = 'utm'
        t.child_frame_id = 'map'

        t.transform.translation.x = utm_x
        t.transform.translation.y = utm_y
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

        self.utm_to_map_published = True

    def tf_init(self) -> bool:
        """
        TF初始化函数：在等待期间收集数据后，直接平均建立tf变换。

        工作流程：
        1. 直接平均得到平均gps，建立utm->map和地图原点
        2. 直接平均得到平均世界系朝向
        3. 直接平均得到平均odom数据，建立map->odom

        Returns:
            True: 初始化成功
            False: 初始化失败（数据不足）
        """
        if self.tf_init_completed:
            self.logger.warning('tf_init already completed, skipping')
            return True

        if self.tf_init_flag:
            self.logger.warning('tf_init already in progress, skipping')
            return False

        self.tf_init_flag = True

        try:
            # ============================================================
            # 步骤1：处理GPS数据 - 直接平均
            # ============================================================
            if not self._gps_queue:
                self.logger.error('tf_init failed: No GPS data collected')
                self.tf_init_flag = False
                return False

            # 计算直接平均GPS
            sum_utm_x = 0.0
            sum_utm_y = 0.0
            sum_lat = 0.0
            sum_lon = 0.0

            for utm_x, utm_y, lat, lon in self._gps_queue:
                sum_utm_x += utm_x
                sum_utm_y += utm_y
                sum_lat += lat
                sum_lon += lon

            num_samples = len(self._gps_queue)
            avg_utm_x = sum_utm_x / num_samples
            avg_utm_y = sum_utm_y / num_samples
            avg_lat = sum_lat / num_samples
            avg_lon = sum_lon / num_samples

            self.logger.info(
                f'tf_init GPS: collected={num_samples} samples, '
                f'avg=UTM({avg_utm_x:.3f}, {avg_utm_y:.3f})'
            )

            # 建立utm->map和地图原点
            self._set_map_origin(avg_utm_x, avg_utm_y, avg_lat, avg_lon)

            # ============================================================
            # 步骤2：处理世界系朝向 - 直接平均
            # ============================================================
            if not self._world_yaw_queue:
                self.logger.error('tf_init failed: No world orientation data collected')
                self.tf_init_flag = False
                return False

            # 使用向量平均处理角度
            sin_sum = 0.0
            cos_sum = 0.0
            for yaw in self._world_yaw_queue:
                sin_sum += math.sin(yaw)
                cos_sum += math.cos(yaw)
            avg_world_yaw = math.atan2(sin_sum, cos_sum)

            self.logger.info(
                f'tf_init World Yaw: collected={len(self._world_yaw_queue)} samples, '
                f'avg_yaw={math.degrees(avg_world_yaw):.1f} deg'
            )

            # ============================================================
            # 步骤3：处理ODOM数据 - 直接平均
            # ============================================================
            if not self._odom_queue:
                self.logger.error('tf_init failed: No odom data collected')
                self.tf_init_flag = False
                return False

            # 计算平均odom位姿
            avg_odom_x = 0.0
            avg_odom_y = 0.0
            for odom_x, odom_y, _ in self._odom_queue:
                avg_odom_x += odom_x
                avg_odom_y += odom_y
            avg_odom_x /= len(self._odom_queue)
            avg_odom_y /= len(self._odom_queue)

            # 使用向量平均处理角度
            sin_sum_yaw = 0.0
            cos_sum_yaw = 0.0
            for _, _, odom_yaw in self._odom_queue:
                sin_sum_yaw += math.sin(odom_yaw)
                cos_sum_yaw += math.cos(odom_yaw)
            avg_odom_yaw = math.atan2(sin_sum_yaw, cos_sum_yaw)

            self.logger.info(
                f'tf_init Odom: collected={len(self._odom_queue)} samples, '
                f'avg_pose=({avg_odom_x:.3f}, {avg_odom_y:.3f}), '
                f'avg_yaw={math.degrees(avg_odom_yaw):.1f} deg'
            )

            # ============================================================
            # 步骤4：建立map->odom变换
            # ============================================================
            # 计算map->odom的偏移量
            # yaw_offset = world_yaw - odom_yaw
            # tx = -(cos(yaw_offset) * odom_x - sin(yaw_offset) * odom_y)
            # ty = -(sin(yaw_offset) * odom_x + cos(yaw_offset) * odom_y)
            self.map_to_odom_offset_yaw = normalize_angle(avg_world_yaw - avg_odom_yaw)

            cos_yaw = math.cos(self.map_to_odom_offset_yaw)
            sin_yaw = math.sin(self.map_to_odom_offset_yaw)

            self.map_to_odom_offset_x = -(cos_yaw * avg_odom_x - sin_yaw * avg_odom_y)
            self.map_to_odom_offset_y = -(sin_yaw * avg_odom_x + cos_yaw * avg_odom_y)

            # 发布map->odom静态TF
            t = TransformStamped()
            t.header.stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
            t.header.frame_id = 'map'
            t.child_frame_id = 'odom'

            t.transform.translation.x = self.map_to_odom_offset_x
            t.transform.translation.y = self.map_to_odom_offset_y
            t.transform.translation.z = 0.0

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = math.sin(self.map_to_odom_offset_yaw / 2.0)
            t.transform.rotation.w = math.cos(self.map_to_odom_offset_yaw / 2.0)

            self.tf_broadcaster.sendTransform(t)

            self.map_to_odom_published = True

            self.logger.info(
                f'tf_init completed successfully: '
                f'utm->map: UTM({avg_utm_x:.3f}, {avg_utm_y:.3f}), '
                f'map->odom: offset=({self.map_to_odom_offset_x:.3f}, {self.map_to_odom_offset_y:.3f}), '
                f'yaw={math.degrees(self.map_to_odom_offset_yaw):.1f} deg'
            )
            self.get_logger().info('tf_init completed successfully')

            # 清理采集队列
            self._gps_queue.clear()
            self._world_yaw_queue.clear()
            self._odom_queue.clear()

            self.tf_init_completed = True
            return True

        except Exception as e:
            self.logger.error(f'tf_init failed with exception: {e}')
            self.tf_init_flag = False
            return False

    def _transform_odom_to_map(
        self, odom_x: float, odom_y: float, odom_yaw: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        将 odom 坐标系中的位姿转换到 map 坐标系

        Args:
            odom_x: odom 坐标系下的 X 坐标
            odom_y: odom 坐标系下的 Y 坐标
            odom_yaw: odom 坐标系下的朝向角（弧度）

        Returns:
            (map_x, map_y, map_yaw): map 坐标系下的位姿
            如果 map_to_odom_published 为 False，返回 (None, None, None)

        转换公式：
            P_map = R(yaw_offset) @ P_odom + T_offset
            yaw_map = yaw_odom + yaw_offset

        坐标系约定（ENU）：
            X: 东向, Y: 北向
            yaw: 0°=东, 90°=北, 180°=西, 270°=南
        """
        if not self.map_to_odom_published:
            self.logger.warning('Cannot transform ODOM to map: map_to_odom not published yet')
            return (None, None, None)

        cos_offset = math.cos(self.map_to_odom_offset_yaw)
        sin_offset = math.sin(self.map_to_odom_offset_yaw)

        map_x = cos_offset * odom_x - sin_offset * odom_y + self.map_to_odom_offset_x
        map_y = sin_offset * odom_x + cos_offset * odom_y + self.map_to_odom_offset_y
        map_yaw = normalize_angle(odom_yaw + self.map_to_odom_offset_yaw)

        return map_x, map_y, map_yaw

    def _transform_utm_to_map(
        self, utm_x: float, utm_y: float, yaw: float = 0.0
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        将 UTM 坐标系中的位姿转换到 map 坐标系

        Args:
            utm_x: UTM 坐标系下的东向坐标（米）
            utm_y: UTM 坐标系下的北向坐标（米）
            yaw: UTM/map 坐标系下的朝向角（弧度，默认0）

        Returns:
            (map_x, map_y, map_yaw): map 坐标系下的位姿
            如果 utm_to_map_published 为 False，返回 (None, None, None)

        说明：
            map 原点 = 首次 GPS 位置对应的 UTM 坐标
            map 与 UTM 轴对齐，无旋转

        坐标系约定（ENU）：
            X: 东向, Y: 北向
            yaw: 0°=东, 90°=北, 180°=西, 270°=南
        """
        if not self.utm_to_map_published:
            self.logger.warning('Cannot transform UTM to map: utm_to_map not published yet')
            return (None, None, None)

        map_x = utm_x - self.map_origin_utm['easting']
        map_y = utm_y - self.map_origin_utm['northing']
        return map_x, map_y, yaw

    def odom_callback(self, msg: Odometry):
        """ODOM 回调：读取速度、位置、朝向，并发布 odom->base_link TF"""
        current_nanos = TimeUtils.now_nanos()
        self._update_frequency_stats('odom')

        # ODOM时间戳使用接收时刻（不使用数据源时间戳）
        odom_stamp = current_nanos

        odom_pose_x = msg.pose.pose.position.x
        odom_pose_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        odom_pose_yaw = normalize_angle(quaternion_to_yaw(q))

        self.latest_sensor_data.odom_stamp = odom_stamp
        self.latest_sensor_data.odom_vx = msg.twist.twist.linear.x
        self.latest_sensor_data.odom_vy = msg.twist.twist.linear.y
        self.latest_sensor_data.odom_vyaw = msg.twist.twist.angular.z
        self.latest_sensor_data.odom_pose_x = odom_pose_x
        self.latest_sensor_data.odom_pose_y = odom_pose_y
        self.latest_sensor_data.odom_pose_yaw = odom_pose_yaw

        if self.face_east_on_startup:
            imu_msg = Imu()
            imu_msg.header.stamp = TimeUtils.nanos_to_stamp(current_nanos)
            imu_msg.orientation = yaw_to_quaternion(odom_pose_yaw)
            self.world_orientation_callback(imu_msg)

        # ============================================================
        # 初始化期间：收集odom数据用于直接平均
        # ============================================================
        if not self.tf_init_completed and not self.tf_init_flag:
            self._odom_queue.append((odom_pose_x, odom_pose_y, odom_pose_yaw))
            self.logger.debug(
                f'ODOM collected for tf_init: pose=({odom_pose_x:.3f}, {odom_pose_y:.3f}), '
                f'yaw={math.degrees(odom_pose_yaw):.1f}, queue_size={len(self._odom_queue)}'
            )
            return  # 初始化期间不执行后续逻辑

        # tf_init完成后：使用原有逻辑
        if self.tf_init_completed:
            self._publish_odom_to_base_link_tf(odom_pose_x, odom_pose_y, odom_pose_yaw, TimeUtils.nanos_to_stamp(current_nanos))

    def world_orientation_callback(self, msg: Imu):
        """世界系朝向回调：记录世界系绝对朝向作为朝向锚点"""
        current_nanos = TimeUtils.now_nanos()
        self._update_frequency_stats('imu')

        # 世界系朝向时间戳使用接收时刻（不使用数据源时间戳）
        world_orientation_stamp = current_nanos

        q = msg.orientation
        world_orientation_yaw = normalize_angle(quaternion_to_yaw(q))

        self.latest_sensor_data.world_orientation_stamp = world_orientation_stamp
        self.latest_sensor_data.world_orientation_yaw = world_orientation_yaw

        # ============================================================
        # 初始化期间：收集世界系朝向数据用于直接平均
        # ============================================================
        if not self.tf_init_completed and not self.tf_init_flag:
            self._world_yaw_queue.append(world_orientation_yaw)
            self.logger.debug(
                f'World yaw collected for tf_init: yaw={math.degrees(world_orientation_yaw):.1f}, '
                f'queue_size={len(self._world_yaw_queue)}'
            )
            return  # 初始化期间不执行后续逻辑

        # tf_init完成后：使用原有逻辑
        if self.tf_init_completed:
            odom_stamp = self.latest_sensor_data.odom_stamp

            if odom_stamp > 0:
                odom_pose_yaw = self.latest_sensor_data.odom_pose_yaw
                self.odom_snapshot_yaw = odom_pose_yaw
                self.logger.debug(
                    f'Odom yaw anchor updated: odom_pose_yaw={math.degrees(odom_pose_yaw):.1f} deg'
                )

    def fuse(self):
        """
        基于航位推算（DR）的融合策略。

        位置：GPS 到达时记录锚点 (gps_map, odom_in_map_snapshot)，之后用 odom 增量更新
        朝向：直接用 world_orientation_yaw + map_to_odom_offset_yaw

        所有增量计算均在 map 坐标系中进行，确保坐标系一致性。
        """
        self.node_freq_stats.tick()
        current_nanos = TimeUtils.now_nanos()

        if not self.tf_init_completed:
            return

        if not self.map_origin_set:
            return

        odom_pose_x = self.latest_sensor_data.odom_pose_x
        odom_pose_y = self.latest_sensor_data.odom_pose_y
        odom_pose_yaw = self.latest_sensor_data.odom_pose_yaw
        odom_stamp = self.latest_sensor_data.odom_stamp

        if odom_stamp <= 0:
            return

        # odom 数据时效检查（超时仍使用，但记录警告）
        if current_nanos - odom_stamp > self.odom_timeout * 1e9:
            self.logger.warning(f'Odom data stale: age={(current_nanos - odom_stamp) / 1e9:.2f}s > timeout={self.odom_timeout}s')
            # 仍然使用该数据（已是最新的）

        # 航位推算计算
        # 必须两个 anchor 都有效才能进行融合
        anchor_set = (
            self.gps_map_x is not None and
            self.gps_map_y is not None and
            self.odom_in_map_snapshot_x is not None and
            self.odom_in_map_snapshot_y is not None and
            self.odom_snapshot_yaw is not None and
            self.latest_sensor_data.world_orientation_yaw is not None
        )

        if not anchor_set:
            return

        map_x: float = 0.0
        map_y: float = 0.0
        yaw: float = 0.0
        fusion_components: list = []

        # ============================================================
        # 位置 DR：GPS 锚点 + odom 增量（在 map 坐标系中计算）
        # ============================================================
        gps_stamp = self.latest_sensor_data.gps_stamp
        if current_nanos - gps_stamp > self.gps_timeout * 1e9:
            self.logger.warning(
                f'GPS data stale: age={(current_nanos - gps_stamp) / 1e9:.2f}s > timeout={self.gps_timeout}s'
            )
            # 仍然使用 DR 计算的位置（已是最新的）
        # 将当前 odom 位置转换到 map 坐标系
        if not self.map_to_odom_published:
            self.logger.warning('DR computation skipped: map->odom not published yet')
            return

        current_odom_in_map_x, current_odom_in_map_y, _ = self._transform_odom_to_map(odom_pose_x, odom_pose_y, 0.0)
        # 计算增量（两个 map 坐标系位置的差）
        delta_x = current_odom_in_map_x - self.odom_in_map_snapshot_x
        delta_y = current_odom_in_map_y - self.odom_in_map_snapshot_y
        # map_x = self.gps_map_x + delta_x
        # map_y = self.gps_map_y + delta_y
        #纯里程计测试
        map_x = current_odom_in_map_x
        map_y = current_odom_in_map_y
        fusion_components.append('POS_DR')

        # 发布 GPS 原始位姿（用于可视化对比）
        self._publish_gps_pose(self.gps_map_x, self.gps_map_y, self.latest_sensor_data.world_orientation_yaw)

        # ============================================================
        # 航向：直接用 world_orientation_yaw + map_to_odom_offset_yaw
        # ============================================================
        world_orientation_stamp = self.latest_sensor_data.world_orientation_stamp
        if current_nanos - world_orientation_stamp > self.world_orientation_timeout * 1e9:
            self.logger.warning(
                f'World orientation stale: age={(current_nanos - world_orientation_stamp) / 1e9:.2f}s > '
                f'timeout={self.world_orientation_timeout}s'
            )

        delta_odom_yaw = normalize_angle(odom_pose_yaw - self.odom_snapshot_yaw)
        yaw = normalize_angle(self.latest_sensor_data.world_orientation_yaw + delta_odom_yaw)
        fusion_components.append('YAW_DR')

        fusion_mode = '+'.join(fusion_components) if fusion_components else 'INVALID'

        self.publish_fusion_result(map_x, map_y, yaw)

        self.logger.debug(
            f'Fusion | {fusion_mode} | '
            f'Pos=({map_x:.3f}, {map_y:.3f}) | '
            f'Yaw={math.degrees(yaw):.1f} deg'
        )

        self._log_frequency_stats()

    def publish_fusion_result(self, map_x: float, map_y: float, yaw: float):
        """发布 map 位姿"""
        # map_pose 发布需要 map_origin_set
        if not self.map_origin_set:
            return

        # 使用当前时间作为时间戳
        pose_stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())

        pose_msg = PoseStamped()
        pose_msg.header.stamp = pose_stamp
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = map_x
        pose_msg.pose.position.y = map_y
        pose_msg.pose.position.z = 0.0

        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        pose_msg.pose.orientation = q

        self.map_pose_pub.publish(pose_msg)

    def _publish_gps_pose(self, gps_x: float, gps_y: float, gps_yaw: float):
        """发布 GPS 原始位姿（用于可视化对比）"""
        pose_stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())

        pose_msg = PoseStamped()
        pose_msg.header.stamp = pose_stamp
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = gps_x
        pose_msg.pose.position.y = gps_y
        pose_msg.pose.position.z = 0.0

        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(gps_yaw / 2.0)
        q.w = math.cos(gps_yaw / 2.0)
        pose_msg.pose.orientation = q

        self.gps_pose_pub.publish(pose_msg)

    def _publish_odom_to_base_link_tf(self, odom_pose_x: float, odom_pose_y: float, odom_pose_yaw: float, stamp: Time):
        """
        发布 odom -> base_link 动态 TF

        从 odom 回调接收到的位姿数据直接发布为 odom->base_link 的 TF 变换。
        时间戳使用 odom 消息的时间戳以保持一致性。
        """
        # 构建 TF 变换
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        # 设置平移
        t.transform.translation.x = odom_pose_x
        t.transform.translation.y = odom_pose_y
        t.transform.translation.z = 0.0

        # 设置旋转
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(odom_pose_yaw / 2.0)
        t.transform.rotation.w = math.cos(odom_pose_yaw / 2.0)

        # 使用动态广播器发布
        self.dynamic_tf_broadcaster.sendTransform(t)


def run_ekf_fusion_node(args=None):
    """运行节点"""
    rclpy.init(args=args)
    node = EKFFusionNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    # 启动前短暂等待收集初始数据
    node.get_logger().info('Waiting for initial sensor data collection...')
    start_time = node.get_clock().now().nanoseconds / 1e9
    timeout = 3.0

    while rclpy.ok():
        current_time = node.get_clock().now().nanoseconds / 1e9
        if current_time - start_time > timeout:
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
