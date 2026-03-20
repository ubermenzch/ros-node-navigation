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
from builtin_interfaces.msg import Time as RosTime

# UTM 库
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    logging.warning("pyproj not installed, UTM conversion will not work")

from config_loader import get_config
from frequency_stats import FrequencyStats


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
    gps_stamp: float = 0.0
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt: float = 0.0

    # ODOM（仅使用速度）
    odom_stamp: float = 0.0
    odom_vx: float = 0.0
    odom_vy: float = 0.0
    odom_vyaw: float = 0.0

    # ODOM 完整数据（用于 TF 发布）
    odom_pose_x: float = 0.0  # odom 坐标系下的 x
    odom_pose_y: float = 0.0  # odom 坐标系下的 y
    odom_pose_yaw: float = 0.0  # odom 坐标系下的 yaw

    # 世界系朝向（来自 RTK 计算输出）
    world_orientation_stamp: float = 0.0
    world_orientation_yaw: float = 0.0


class EKFFusionNode(Node):
    """
    为了兼容原有启动方式，类名保留 EKFFusionNode。
    实际实现为"简化融合器"，使用位姿直接融合方式。
    """

    def __init__(self, log_dir: str = None, timestamp: str = None):
        super().__init__('ekf_fusion_node')

        # 使用传入的日志目录和时间戳，或生成新的
        self.log_dir = log_dir
        self.timestamp = timestamp if timestamp is not None else datetime.now().strftime('%Y%m%d_%H%M%S')

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
        self.last_gps_time = 0.0

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
            ts = self.timestamp
            log_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'logs',
                f'navigation_{ts}'
            )
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'ekf_fusion_node_log_{self.timestamp}.log')

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
            'Fusion Node initialized (DR-based fusion)',
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
            'gps':  {'last_time': 0.0, 'count': 0, 'last_log_time': 0.0},
            'imu':  {'last_time': 0.0, 'count': 0, 'last_log_time': 0.0},
            'odom': {'last_time': 0.0, 'count': 0, 'last_log_time': 0.0},
        }

    def _sec_to_stamp(self, t: float) -> RosTime:
        """秒数转 ROS 时间戳"""
        stamp = RosTime()
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        return stamp

    def _update_frequency_stats(self, sensor_type: str):
        current_time = self.get_clock().now().nanoseconds / 1e9
        stats = self.freq_stats[sensor_type]
        if stats['last_time'] > 0:
            dt = current_time - stats['last_time']
            if dt > 0:
                stats['count'] += 1
        stats['last_time'] = current_time

    def _log_frequency_stats(self):
        if not self.log_frequency_stats:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        log_interval = 10.0

        for sensor_type in ['gps', 'imu', 'odom']:
            stats = self.freq_stats[sensor_type]
            if current_time - stats['last_log_time'] >= log_interval:
                avg_freq = stats['count'] / log_interval
                self.logger.info(
                    f'{sensor_type.upper()} Frequency: {avg_freq:.2f} Hz (samples={stats["count"]})'
                )
                stats['count'] = 0
                stats['last_log_time'] = current_time

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
        current_time = self.get_clock().now().nanoseconds / 1e9
        self._update_frequency_stats('gps')

        if msg.header.stamp.sec > 0:
            gps_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            gps_stamp = current_time

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
        self.last_gps_time = current_time

        # 首次有效GPS，设定 map 原点并发布 utm -> map 静态TF
        if gps_valid and not self.map_origin_set:
            utm_x, utm_y = self.gps_to_utm(lat, lon)
            if utm_x is not None and utm_y is not None:
                self._set_map_origin(utm_x, utm_y, lat, lon)

        # 每次有效GPS且 map原点已设置时，更新位置锚点
        if gps_valid and self.map_origin_set and self.utm_to_map_published:
            utm_x, utm_y = self.gps_to_utm(lat, lon)
            if utm_x is not None and utm_y is not None:
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
        """设置 map 原点，并发布 map->odom 静态 TF"""
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
        t.header.stamp = self.get_clock().now().to_msg()
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

    def _publish_map_to_odom_transform(self):
        """
        发布 map -> odom 静态 TF

        基于世界系朝向锚点和当前 odom 位姿计算并发布此 TF。

        语义：
        - 父坐标系: map
        - 子坐标系: odom

        变换公式：
        - yaw_offset = world_orientation_yaw - odom_yaw
        - tx = -(cos(yaw_offset) * odom_x - sin(yaw_offset) * odom_y)
        - ty = -(sin(yaw_offset) * odom_x + cos(yaw_offset) * odom_y)
        """
        odom_stamp = self.latest_sensor_data.odom_stamp
        odom_pose_x = self.latest_sensor_data.odom_pose_x
        odom_pose_y = self.latest_sensor_data.odom_pose_y
        odom_pose_yaw = self.latest_sensor_data.odom_pose_yaw
        world_orientation_yaw = self.latest_sensor_data.world_orientation_yaw
        world_orientation_stamp = self.latest_sensor_data.world_orientation_stamp

        if odom_stamp <= 0:
            self.logger.warning('Cannot publish map->odom TF: odom not available yet')
            return
        if world_orientation_stamp <= 0:
            self.logger.warning('Cannot publish map->odom TF: world orientation not available yet')
            return
            
        self.map_to_odom_offset_yaw = normalize_angle(world_orientation_yaw - odom_pose_yaw)

        cos_yaw = math.cos(self.map_to_odom_offset_yaw)
        sin_yaw = math.sin(self.map_to_odom_offset_yaw)

        self.map_to_odom_offset_x = -(cos_yaw * odom_pose_x - sin_yaw * odom_pose_y)
        self.map_to_odom_offset_y = -(sin_yaw * odom_pose_x + cos_yaw * odom_pose_y)

        # 构建并发布 TF（平移部分直接取负）
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
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
            f'Published map->odom TF: offset=({self.map_to_odom_offset_x:.3f}, {self.map_to_odom_offset_y:.3f}), '
            f'yaw={math.degrees(self.map_to_odom_offset_yaw):.1f} deg'
        )
        self.get_logger().info(
            f'Published map->odom TF: offset=({self.map_to_odom_offset_x:.3f}, {self.map_to_odom_offset_y:.3f}), '
            f'yaw={math.degrees(self.map_to_odom_offset_yaw):.1f} deg'
        )

    def odom_callback(self, msg: Odometry):
        """ODOM 回调：读取速度、位置、朝向，并发布 odom->base_link TF"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        self._update_frequency_stats('odom')

        if msg.header.stamp.sec > 0:
            odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            odom_stamp = current_time

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
            imu_msg.header.stamp = msg.header.stamp
            imu_msg.orientation = yaw_to_quaternion(odom_pose_yaw)
            self.world_orientation_callback(imu_msg)

        if self.map_origin_set and not self.map_to_odom_published:
            self._publish_map_to_odom_transform()

        self._publish_odom_to_base_link_tf(odom_pose_x, odom_pose_y, odom_pose_yaw, msg.header.stamp)

    def world_orientation_callback(self, msg: Imu):
        """世界系朝向回调：记录世界系绝对朝向作为朝向锚点"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        self._update_frequency_stats('imu')

        if msg.header.stamp.sec > 0:
            world_orientation_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            world_orientation_stamp = current_time

        q = msg.orientation
        world_orientation_yaw = normalize_angle(quaternion_to_yaw(q))

        self.latest_sensor_data.world_orientation_stamp = world_orientation_stamp
        self.latest_sensor_data.world_orientation_yaw = world_orientation_yaw

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
        current_time = self.get_clock().now().nanoseconds / 1e9

        if not self.map_origin_set:
            return

        odom_pose_x = self.latest_sensor_data.odom_pose_x
        odom_pose_y = self.latest_sensor_data.odom_pose_y
        odom_pose_yaw = self.latest_sensor_data.odom_pose_yaw
        odom_stamp = self.latest_sensor_data.odom_stamp

        if odom_stamp <= 0:
            return

        # odom 数据时效检查（超时仍使用，但记录警告）
        if current_time - odom_stamp > self.odom_timeout:
            self.logger.warning(f'Odom data stale: age={current_time - odom_stamp:.2f}s > timeout={self.odom_timeout}s')
            # 仍然使用该数据（已是最新的）

        # 航位推算计算
        # 必须两个 anchor 都有效才能进行融合
        location_anchor_set = (
            self.gps_map_x is not None and
            self.odom_in_map_snapshot_x is not None and
            self.odom_in_map_snapshot_y is not None
        )
        yaw_anchor_set = self.odom_snapshot_yaw is not None

        if not (location_anchor_set and yaw_anchor_set):
            return

        map_x: float = 0.0
        map_y: float = 0.0
        yaw: float = 0.0
        fusion_components: list = []

        # ============================================================
        # 位置 DR：GPS 锚点 + odom 增量（在 map 坐标系中计算）
        # ============================================================
        gps_stamp = self.latest_sensor_data.gps_stamp
        if current_time - gps_stamp > self.gps_timeout:
            self.logger.warning(
                f'GPS data stale: age={current_time - gps_stamp:.2f}s > timeout={self.gps_timeout}s'
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
        map_x = self.gps_map_x + delta_x
        map_y = self.gps_map_y + delta_y
        fusion_components.append('POS_DR')

        # 发布 GPS 原始位姿（用于可视化对比）
        self._publish_gps_pose(self.gps_map_x, self.gps_map_y, self.latest_sensor_data.world_orientation_yaw)

        # ============================================================
        # 航向：直接用 world_orientation_yaw + map_to_odom_offset_yaw
        # ============================================================
        world_orientation_stamp = self.latest_sensor_data.world_orientation_stamp
        if current_time - world_orientation_stamp > self.world_orientation_timeout:
            self.logger.warning(
                f'World orientation stale: age={current_time - world_orientation_stamp:.2f}s > '
                f'timeout={self.world_orientation_timeout}s'
            )

        delta_odom_yaw = normalize_angle(odom_pose_yaw - self.odom_snapshot_yaw)
        yaw = normalize_angle(self.latest_sensor_data.world_orientation_yaw + delta_odom_yaw)
        fusion_components.append('YAW_DR')

        fusion_mode = '+'.join(fusion_components) if fusion_components else 'INVALID'

        self.publish_fusion_result(map_x, map_y, yaw, fusion_mode)

        self.logger.debug(
            f'Fusion | {fusion_mode} | '
            f'Pos=({map_x:.3f}, {map_y:.3f}) | '
            f'Yaw={math.degrees(yaw):.1f} deg'
        )

        self._log_frequency_stats()

    def publish_fusion_result(self, map_x: float, map_y: float, yaw: float, fusion_mode: str = 'unknown'):
        """发布 map 位姿"""
        # map_pose 发布需要 map_origin_set
        if not self.map_origin_set:
            return

        # 使用当前时间作为时间戳
        pose_stamp = self.get_clock().now().to_msg()

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
        pose_stamp = self.get_clock().now().to_msg()

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

    def _publish_odom_to_base_link_tf(self, odom_pose_x: float, odom_pose_y: float, odom_pose_yaw: float, stamp: RosTime):
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

    # 启动前短暂等待一次初始数据
    node.get_logger().info('Waiting briefly for initial sensor data...')
    start_time = node.get_clock().now().nanoseconds / 1e9
    timeout = 3.0

    while rclpy.ok():
        current_time = node.get_clock().now().nanoseconds / 1e9
        if current_time - start_time > timeout:
            break
        executor.spin_once(timeout_sec=0.1)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    run_ekf_fusion_node()
