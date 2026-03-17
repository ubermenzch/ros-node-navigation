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
   - ODOM: 提供 base_link 下速度，用于两次GPS之间的短时传播
4. 输出 /navigation/map_pose (PoseStamped, frame_id=map)

说明：
- 本节点不是“标准 EKF”，而是更适合当前传感器条件的工程化简化融合器
- yaw 优先直接使用世界系朝向（RTK 计算输出）
- x,y,yaw 由 ODOM 传播，并用 GPS 周期性校正
"""

import math
import os
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

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


@dataclass
class RobotState:
    """机器人状态（工程化简化融合）"""
    x: float = 0.0       # map坐标系下X（东）
    y: float = 0.0       # map坐标系下Y（北）
    yaw: float = 0.0     # map坐标系下航向角（弧度）
    vx: float = 0.0      # base_link系线速度x（前）
    vy: float = 0.0      # base_link系线速度y（左）
    vyaw: float = 0.0    # 角速度z


@dataclass
class SensorData:
    """最新传感器数据"""
    # GPS
    gps_stamp: float = 0.0
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt: float = 0.0
    gps_available: bool = False

    # ODOM（仅使用速度）
    odom_stamp: float = 0.0
    odom_vx: float = 0.0
    odom_vy: float = 0.0
    odom_vyaw: float = 0.0
    odom_available: bool = False

    # ODOM 完整数据（用于 TF 发布）
    odom_pose_x: float = 0.0  # odom 坐标系下的 x
    odom_pose_y: float = 0.0  # odom 坐标系下的 y
    odom_pose_yaw: float = 0.0  # odom 坐标系下的 yaw

    # 世界系朝向（来自 RTK 计算输出）
    world_orientation_stamp: float = 0.0
    world_orientation_yaw: float = 0.0
    world_orientation_yaw_rate: float = 0.0
    world_orientation_acc_x: float = 0.0
    world_orientation_acc_y: float = 0.0
    world_orientation_available: bool = False


class EKFFusionNode(Node):
    """
    为了兼容原有启动方式，类名保留 EKFFusionNode。
    实际实现为“简化融合器”，不是标准EKF。
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

        # 运行参数
        self.frequency = ekf_config.get('frequency', 10.0)
        self.gps_timeout = ekf_config.get('gps_timeout', 2.0)
        self.odom_timeout = ekf_config.get('odom_timeout', 1.0)
        self.world_orientation_timeout = ekf_config.get('world_orientation_timeout', 0.5)

        # 融合参数
        # gps_alpha: GPS融合权重 [0,1]
        #   - 位置: gps_alpha 控制 GPS 位置与 odom 传播的融合权重
        #   - 朝向: gps_alpha 控制世界系朝向与 odom 角速度积分的融合权重
        #   - 1.0 = 完全信任 GPS/世界系朝向, 0.0 = 完全信任 odom
        self.gps_alpha = float(ekf_config.get('gps_alpha', 0.30))
        self.gps_jump_reject = float(ekf_config.get('gps_jump_reject', 5.0))   # 单次GPS跳变拒绝阈值（米）
        self.yaw_jump_reject = float(ekf_config.get('yaw_jump_reject', 0.5))   # 单次朝向跳变拒绝阈值（弧度）
        self.log_frequency_stats = bool(ekf_config.get('log_frequency_stats', True))
        self.log_enabled = bool(ekf_config.get('log_enabled', True))

        # UTM转换相关
        self.utm_zone: Optional[int] = None
        self.utm_transformer = None
        self.map_origin_utm = None   # {'easting': x, 'northing': y}
        self.map_origin_set = False
        self.map_origin_lock = threading.Lock()

        # 传感器数据
        self.latest_sensor_data = SensorData()
        self.sensor_lock = threading.Lock()

        # 最近消息到达时间（本地时间，用于超时判断）
        self.last_gps_time = 0.0
        self.last_odom_time = 0.0
        self.last_world_orientation_time = 0.0

        # 状态
        self.state = RobotState()
        self.state_lock = threading.Lock()

        # 融合时序
        self.last_fuse_time: Optional[float] = None
        self.last_processed_gps_stamp: float = 0.0   # 防止同一帧GPS重复校正

        # 日志
        self._init_logger(self.log_enabled)

        # 频率统计
        self._init_frequency_stats()

        # 订阅者
        self.gps_sub = self.create_subscription(
            NavSatFix,
            self.gps_topic,
            self.gps_callback,
            10
        )
        self.world_orientation_sub = self.create_subscription(
            Imu,
            self.world_orientation_topic,
            self.world_orientation_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        # 发布者
        self.map_pose_pub = self.create_publisher(PoseStamped, self.map_pose_topic, 10)

        # 静态 TF 广播器
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # 动态 TF 广播器（用于 odom->base_link）
        self.dynamic_tf_broadcaster = TransformBroadcaster(self)

        # map->odom 偏移量（在收到首个有效 GPS 后设置）
        self.map_to_odom_offset_x = 0.0
        self.map_to_odom_offset_y = 0.0
        self.map_to_odom_offset_yaw = 0.0  # odom 初始朝向
        self.first_gps_received = False  # 是否已收到首个有效 GPS

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
            'Fusion Node initialized (simplified fusion, not standard EKF)',
            f'  工作频率: {self.frequency} Hz',
            f'  GPS 话题: {self.gps_topic}',
            f'  世界系朝向话题: {self.world_orientation_topic}',
            f'  ODOM 话题: {self.odom_topic}',
            f'  GPS 超时: {self.gps_timeout}s',
            f'  世界系朝向超时: {self.world_orientation_timeout}s',
            f'  ODOM 超时: {self.odom_timeout}s',
            f'  GPS/世界系朝向融合权重 (alpha): {self.gps_alpha}',
            f'  GPS 跳变拒绝阈值: {self.gps_jump_reject} m',
            f'  朝向跳变拒绝阈值: {self.yaw_jump_reject} rad ({math.degrees(self.yaw_jump_reject):.1f} deg)',
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
        """GPS 回调"""
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

        with self.sensor_lock:
            self.latest_sensor_data.gps_stamp = gps_stamp
            self.latest_sensor_data.gps_lat = lat
            self.latest_sensor_data.gps_lon = lon
            self.latest_sensor_data.gps_alt = alt if math.isfinite(alt) else 0.0
            self.latest_sensor_data.gps_available = gps_valid
            self.last_gps_time = current_time

        # 首次有效GPS，设定 map 原点并发布 utm -> map 静态TF
        if gps_valid and not self.map_origin_set:
            utm_x, utm_y = self.gps_to_utm(lat, lon)
            if utm_x is not None and utm_y is not None:
                self._set_map_origin(utm_x, utm_y, lat, lon)

    def _set_map_origin(self, utm_x: float, utm_y: float, lat: float, lon: float):
        """设置 map 原点，并发布 map->odom 静态 TF"""
        with self.map_origin_lock:
            if self.map_origin_set:
                return

            self.map_origin_utm = {
                'easting': utm_x,
                'northing': utm_y,
            }
            self.map_origin_set = True

            self._publish_utm_to_map_transform(utm_x, utm_y)

            msg = (
                f'Map origin set: '
                f'UTM({utm_x:.3f}, {utm_y:.3f}), '
                f'GPS(lat={lat:.8f}, lon={lon:.8f})'
            )
            self.logger.info(msg)
            self.get_logger().info(msg)

            # 原点建立后，当前位置初始化为 map 原点
            with self.state_lock:
                self.state.x = 0.0
                self.state.y = 0.0

            # 在收到首个有效 GPS 后，设置 map->odom 偏移并发布静态 TF
            self._publish_map_to_odom_transform()

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

    def _publish_map_to_odom_transform(self):
        """
        发布 map -> odom 静态 TF

        在收到首个有效 GPS 后（设置 map 原点时），根据当前 odom 位置和朝向发布此 TF。

        语义：
        - 父坐标系: map
        - 子坐标系: odom

        严格逆变换计算：
        如果 odom->base_link 在首帧时的位姿是 (x0, y0, yaw0)，
        那么 map->odom 的变换需要将 base_link 位置转换到 map 坐标系后再取负。

        具体计算：
        - 旋转: -yaw0 (负的首帧 odom 朝向)
        - 平移: -R(-yaw0) @ [x0, y0] (平移也要跟旋转联动)
        """
        # 检查 odom 是否已经收到并更新过
        with self.sensor_lock:
            odom_available = self.latest_sensor_data.odom_available
            odom_pose_x = self.latest_sensor_data.odom_pose_x
            odom_pose_y = self.latest_sensor_data.odom_pose_y
            odom_pose_yaw = self.latest_sensor_data.odom_pose_yaw

        if not odom_available:
            self.logger.warning('Cannot publish map->odom TF: odom not available yet')
            return

        # 设置 map->odom 偏移（存储用于后续使用）
        self.map_to_odom_offset_x = odom_pose_x
        self.map_to_odom_offset_y = odom_pose_y
        self.map_to_odom_offset_yaw = odom_pose_yaw  # 存储初始朝向
        self.first_gps_received = True

        # 构建并发布 TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        # 严格的逆变换计算：
        # 旋转：-yaw0
        yaw_offset = -self.map_to_odom_offset_yaw

        # 平移：-R(-yaw0) @ [x0, y0]
        # R(-yaw0) = [[cos(-yaw0), -sin(-yaw0)], [sin(-yaw0), cos(-yaw0)]]
        #           = [[cos(yaw0), sin(yaw0)], [-sin(yaw0), cos(yaw0)]]
        # 因为是求逆，所以用 -yaw0 的旋转矩阵
        cos_yaw = math.cos(-self.map_to_odom_offset_yaw)
        sin_yaw = math.sin(-self.map_to_odom_offset_yaw)

        # [tx, ty] = -R(-yaw0) @ [x0, y0]
        tx = -(cos_yaw * self.map_to_odom_offset_x - sin_yaw * self.map_to_odom_offset_y)
        ty = -(sin_yaw * self.map_to_odom_offset_x + cos_yaw * self.map_to_odom_offset_y)

        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = 0.0

        # map->odom: odom 相对于 map 的旋转（负的初始朝向）
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw_offset / 2.0)
        t.transform.rotation.w = math.cos(yaw_offset / 2.0)

        self.tf_broadcaster.sendTransform(t)

        self.logger.info(
            f'Published map->odom TF: offset=({self.map_to_odom_offset_x:.3f}, {self.map_to_odom_offset_y:.3f}), '
            f'yaw={math.degrees(self.map_to_odom_offset_yaw):.1f} deg, '
            f'corrected_offset=({tx:.3f}, {ty:.3f})'
        )
        self.get_logger().info(
            f'Published map->odom TF: offset=({self.map_to_odom_offset_x:.3f}, {self.map_to_odom_offset_y:.3f}), '
            f'yaw={math.degrees(self.map_to_odom_offset_yaw):.1f} deg'
        )

    def odom_callback(self, msg: Odometry):
        """ODOM 回调：读取速度、位置和朝向"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        self._update_frequency_stats('odom')

        if msg.header.stamp.sec > 0:
            odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            odom_stamp = current_time

        # 提取位置
        odom_pose_x = msg.pose.pose.position.x
        odom_pose_y = msg.pose.pose.position.y

        # 提取朝向（从四元数）
        q = msg.pose.pose.orientation
        odom_pose_yaw = 0.0
        if q.x != 0.0 or q.y != 0.0 or q.z != 0.0 or q.w != 0.0:
            odom_pose_yaw = quaternion_to_yaw(q)

        with self.sensor_lock:
            self.latest_sensor_data.odom_stamp = odom_stamp
            self.latest_sensor_data.odom_vx = msg.twist.twist.linear.x
            self.latest_sensor_data.odom_vy = msg.twist.twist.linear.y
            self.latest_sensor_data.odom_vyaw = msg.twist.twist.angular.z
            self.latest_sensor_data.odom_pose_x = odom_pose_x
            self.latest_sensor_data.odom_pose_y = odom_pose_y
            self.latest_sensor_data.odom_pose_yaw = odom_pose_yaw
            self.latest_sensor_data.odom_available = True
            self.last_odom_time = current_time

    def world_orientation_callback(self, msg: Imu):
        """世界系朝向回调：读取 RTK 计算输出的世界系 yaw"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        self._update_frequency_stats('imu')

        if msg.header.stamp.sec > 0:
            world_orientation_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            world_orientation_stamp = current_time

        q = msg.orientation
        world_orientation_yaw = 0.0
        if q.x != 0.0 or q.y != 0.0 or q.z != 0.0 or q.w != 0.0:
            world_orientation_yaw = quaternion_to_yaw(q)

        with self.sensor_lock:
            self.latest_sensor_data.world_orientation_stamp = world_orientation_stamp
            self.latest_sensor_data.world_orientation_yaw = world_orientation_yaw
            self.latest_sensor_data.world_orientation_yaw_rate = msg.angular_velocity.z
            self.latest_sensor_data.world_orientation_acc_x = msg.linear_acceleration.x
            self.latest_sensor_data.world_orientation_acc_y = msg.linear_acceleration.y
            self.latest_sensor_data.world_orientation_available = True
            self.last_world_orientation_time = current_time

    def fuse(self):
        """执行一次简化融合"""
        self.node_freq_stats.tick()

        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.last_fuse_time is None:
            dt = 1.0 / max(self.frequency, 1e-3)
        else:
            dt = current_time - self.last_fuse_time
            dt = max(1e-3, min(dt, 0.5))  # 防止时间异常导致传播过大
        self.last_fuse_time = current_time

        with self.sensor_lock:
            gps_stamp = self.latest_sensor_data.gps_stamp
            gps_available = self.latest_sensor_data.gps_available
            gps_lat = self.latest_sensor_data.gps_lat
            gps_lon = self.latest_sensor_data.gps_lon

            odom_available = self.latest_sensor_data.odom_available
            odom_vx = self.latest_sensor_data.odom_vx
            odom_vy = self.latest_sensor_data.odom_vy
            odom_vyaw = self.latest_sensor_data.odom_vyaw

        world_orientation_available = self.latest_sensor_data.world_orientation_available
        world_orientation_yaw = self.latest_sensor_data.world_orientation_yaw
        world_orientation_yaw_rate = self.latest_sensor_data.world_orientation_yaw_rate

        gps_fresh = (current_time - self.last_gps_time) <= self.gps_timeout
        odom_fresh = (current_time - self.last_odom_time) <= self.odom_timeout
        world_orientation_fresh = (current_time - self.last_world_orientation_time) <= self.world_orientation_timeout

        fusion_components = []
        gps_applied_this_cycle = False
        alpha = max(0.0, min(1.0, self.gps_alpha))  # 统一融合权重

        with self.state_lock:
            # 1) 朝向融合：世界系朝向 vs odom 角速度积分
            if world_orientation_available and world_orientation_fresh:
                # 先用 odom 积分计算预测朝向
                odom_yaw_pred = normalize_angle(self.state.yaw + odom_vyaw * dt) if (odom_available and odom_fresh) else self.state.yaw

                # 计算世界系朝向与 odom 预测的跳变差值
                yaw_err = abs(normalize_angle(world_orientation_yaw - odom_yaw_pred))

                # 第一帧校正不拒绝；后续若跳变过大则拒绝
                first_yaw_correction = (abs(self.state.yaw) < 1e-9)

                if first_yaw_correction or yaw_err <= self.yaw_jump_reject:
                    # 使用 gps_alpha 融合世界系朝向和 odom 积分
                    if alpha >= 1.0:
                        # 完全信任世界系朝向
                        self.state.yaw = normalize_angle(world_orientation_yaw)
                        self.state.vyaw = world_orientation_yaw_rate
                    else:
                        # 融合：alpha * 世界系朝向 + (1-alpha) * odom 积分
                        self.state.yaw = normalize_angle(alpha * world_orientation_yaw + (1 - alpha) * odom_yaw_pred)
                        # 角速度也做融合
                        self.state.vyaw = alpha * world_orientation_yaw_rate + (1 - alpha) * (odom_vyaw if (odom_available and odom_fresh) else 0.0)
                    fusion_components.append('ORI')
                else:
                    # 朝向跳变过大，拒绝世界系朝向，使用 odom 积分
                    self.get_logger().warn(
                        f'Reject yaw jump: err={math.degrees(yaw_err):.2f} deg, '
                        f'world_ori={math.degrees(world_orientation_yaw):.1f} deg, '
                        f'odom_pred={math.degrees(odom_yaw_pred):.1f} deg'
                    )
                    self.state.yaw = odom_yaw_pred
                    self.state.vyaw = odom_vyaw if (odom_available and odom_fresh) else 0.0
                    fusion_components.append('ODOM_YAW')
            elif odom_available and odom_fresh:
                # 无世界系朝向时，用 odom 角速度积分
                self.state.yaw = normalize_angle(self.state.yaw + odom_vyaw * dt)
                self.state.vyaw = odom_vyaw
                fusion_components.append('ODOM_YAW')

            # 2) 使用 ODOM 速度在当前 yaw 下做短时传播（作为预测）
            if odom_available and odom_fresh:
                self.state.vx = odom_vx
                self.state.vy = odom_vy

                c = math.cos(self.state.yaw)
                s = math.sin(self.state.yaw)

                self.state.x += (self.state.vx * c - self.state.vy * s) * dt
                self.state.y += (self.state.vx * s + self.state.vy * c) * dt

                fusion_components.append('ODOM')

            # 3) 仅对“新到的一帧GPS”做一次绝对位置校正
            if (
                gps_available and
                gps_fresh and
                self.map_origin_set and
                gps_stamp > self.last_processed_gps_stamp
            ):
                self.last_processed_gps_stamp = gps_stamp

                utm_x, utm_y = self.gps_to_utm(gps_lat, gps_lon)
                if utm_x is not None and utm_y is not None:
                    gps_map_x = utm_x - self.map_origin_utm['easting']
                    gps_map_y = utm_y - self.map_origin_utm['northing']

                    err = math.hypot(gps_map_x - self.state.x, gps_map_y - self.state.y)

                    # 第一帧校正不拒绝；后续若跳变过大则拒绝
                    first_gps_correction = (self.last_processed_gps_stamp == gps_stamp and abs(self.state.x) < 1e-9 and abs(self.state.y) < 1e-9)

                    if first_gps_correction or err <= self.gps_jump_reject:
                        # 使用 gps_alpha 融合 GPS 位置和 odom 传播位置
                        # alpha=1: 完全信任 GPS, alpha=0: 完全信任 odom
                        self.state.x = (1.0 - alpha) * self.state.x + alpha * gps_map_x
                        self.state.y = (1.0 - alpha) * self.state.y + alpha * gps_map_y

                        gps_applied_this_cycle = True
                        fusion_components.append('GPS')
                    else:
                        self.get_logger().warn(
                            f'Reject GPS jump: err={err:.2f} m, '
                            f'gps_map=({gps_map_x:.2f}, {gps_map_y:.2f}), '
                            f'state=({self.state.x:.2f}, {self.state.y:.2f})'
                        )

        fusion_mode = '+'.join(fusion_components) if fusion_components else 'invalid'

        self.publish_fusion_result(fusion_mode)

        with self.state_lock:
            self.logger.info(
                f'Fusion | {fusion_mode} | '
                f'Pos=({self.state.x:.3f}, {self.state.y:.3f}) | '
                f'Yaw={math.degrees(self.state.yaw):.1f} deg | '
                f'GPS_applied={gps_applied_this_cycle}'
            )

        self._log_frequency_stats()

    def publish_fusion_result(self, fusion_mode: str = 'unknown'):
        """发布 map 位姿和 odom->base_link TF"""
        # odom->base_link 应该在收到 odom 数据后立即发布，与 map_origin_set 解耦
        # 只要有有效的 odom 数据就发布
        self._publish_odom_to_base_link_tf()

        # map_pose 发布需要 map_origin_set
        if not self.map_origin_set:
            return

        with self.state_lock:
            map_x = self.state.x
            map_y = self.state.y
            yaw = self.state.yaw

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
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

    def _publish_odom_to_base_link_tf(self):
        """
        发布 odom -> base_link 动态 TF

        从 /utildar/robot_odom 获取当前 odom 坐标系下的位置和朝向，
        发布为 odom->base_link 的 TF 变换。
        """
        with self.sensor_lock:
            odom_pose_x = self.latest_sensor_data.odom_pose_x
            odom_pose_y = self.latest_sensor_data.odom_pose_y
            odom_pose_yaw = self.latest_sensor_data.odom_pose_yaw

        # 构建 TF 变换
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
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

    def initialize_from_sensors(self):
        """用当前已有传感器数据初始化状态（可选）"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        with self.sensor_lock:
            sensor_data = self.latest_sensor_data

        with self.state_lock:
            if sensor_data.world_orientation_available and (current_time - self.last_world_orientation_time < self.world_orientation_timeout):
                self.state.yaw = normalize_angle(sensor_data.world_orientation_yaw)
                self.state.vyaw = sensor_data.world_orientation_yaw_rate
                self.get_logger().info(
                    f'Initialized yaw from world orientation: {math.degrees(self.state.yaw):.2f} deg'
                )

            if sensor_data.odom_available and (current_time - self.last_odom_time < self.odom_timeout):
                self.state.vx = sensor_data.odom_vx
                self.state.vy = sensor_data.odom_vy
                self.state.vyaw = sensor_data.odom_vyaw
                self.get_logger().info(
                    f'Initialized velocity from ODOM: vx={self.state.vx:.3f}, vy={self.state.vy:.3f}, vyaw={self.state.vyaw:.3f}'
                )


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

    node.initialize_from_sensors()

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