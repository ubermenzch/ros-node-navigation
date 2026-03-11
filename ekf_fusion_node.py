#!/usr/bin/env python3
"""
EKF融合定位节点
接收GPS、IMU、ODOM数据，进行融合定位，输出融合后的坐标。
融合后的坐标以地图中心为坐标系原点。
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String, Float64
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Odometry
import json
import threading
import time
import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
import os
import logging
from datetime import datetime
from typing import Optional, Deque

from config_loader import get_config
from frequency_stats import FrequencyStats



@dataclass
class RobotState:
    """机器人状态向量 [x, y, yaw, vx, vy, vyaw]"""
    x: float = 0.0          # 地图坐标系X (米)
    y: float = 0.0          # 地图坐标系Y (米)
    yaw: float = 0.0        # 航向角 (弧度)
    vx: float = 0.0        # X方向速度 (米/秒)
    vy: float = 0.0        # Y方向速度 (米/秒)
    vyaw: float = 0.0      # 角速度 (弧度/秒)


@dataclass
class SensorData:
    """传感器数据结构"""
    timestamp: float = 0.0
    # GPS数据
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt: float = 0.0
    gps_cov_lat: float = 0.0  # 纬度协方差
    gps_cov_lon: float = 0.0  # 经度协方差
    gps_available: bool = False

    # ODOM数据
    odom_x: float = 0.0
    odom_y: float = 0.0
    odom_yaw: float = 0.0
    odom_vx: float = 0.0
    odom_vy: float = 0.0
    odom_vyaw: float = 0.0
    odom_available: bool = False

    # IMU数据
    imu_yaw: float = 0.0          # 从四元数提取的航向角
    imu_yaw_rate: float = 0.0     # 角速度
    imu_acc_x: float = 0.0        # X方向加速度
    imu_acc_y: float = 0.0        # Y方向加速度
    imu_available: bool = False


@dataclass
class TimestampedData:
    """带时间戳的数据"""
    timestamp: float = 0.0
    data: any = None


@dataclass
class SensorDataBuffer:
    """传感器数据缓冲区（用于时间戳匹配）"""
    odom_history: Deque[TimestampedData] = field(default_factory=lambda: deque(maxlen=100))
    imu_history: Deque[TimestampedData] = field(default_factory=lambda: deque(maxlen=100))


class EKFFusionNode(Node):
    """
    EKF融合定位节点

    输入:
    - GPS: /rtk_fix (NavSatFix)
    - ODOM: /unitree_ros2/odom (Odometry)
    - IMU: /imu/data (Imu)

    输出:
    - 融合定位: /fusion_pose (String - JSON格式)
    - GPS权重: /gps_weight (Float64)

    坐标系: 地图坐标系（以gps_waypoint_node生成的地图中心为原点）
    """

    def __init__(self):
        super().__init__('ekf_fusion_node')

        # 加载配置文件
        config = get_config()
        ekf_config = config.get_ekf_config()

        # 获取参数（优先从config.yaml，失败则使用默认值）
        subscriptions = ekf_config.get('subscriptions', {})
        publications = ekf_config.get('publications', {})

        # 订阅话题
        self.gps_topic = subscriptions.get('gps_topic', '/rtk_fix')
        self.odom_topic = subscriptions.get('odom_topic', '/unitree_ros2/odom')
        self.imu_topic = subscriptions.get('imu_topic', '/imu/data')

        # 发布话题
        self.fusion_pose_topic = publications.get('fusion_pose_topic', '/fusion_pose')
        self.gps_weight_topic = publications.get('gps_weight_topic', '/gps_weight')

        self.frequency = ekf_config.get('frequency', 10.0)
        self.gps_timeout = ekf_config.get('gps_timeout', 2.0)
        self.odom_timeout = ekf_config.get('odom_timeout', 1.0)
        self.imu_timeout = ekf_config.get('imu_timeout', 0.5)

        self.gps_cov_threshold = ekf_config.get('gps_position_covariance_threshold', 0.5)
        self.gps_satellite_min = ekf_config.get('gps_satellite_min', 6)

        self.process_noise_pos = ekf_config.get('process_noise_position', 0.1)
        self.process_noise_yaw = ekf_config.get('process_noise_yaw', 0.05)
        self.odom_noise = ekf_config.get('odom_measurement_noise', 0.1)
        self.gps_noise_good = ekf_config.get('gps_measurement_noise_good', 0.1)
        self.gps_noise_bad = ekf_config.get('gps_measurement_noise_bad', 10.0)
        self.imu_noise_yaw = ekf_config.get('imu_measurement_noise_yaw', 0.1)

        self.log_frequency_stats = ekf_config.get('log_frequency_stats', True)

        # 融合模式配置
        self.use_odom_imu_fusion = ekf_config.get('use_odom_imu_fusion', True)

        # 初始化日志系统
        self._init_logger()

        # 传感器数据缓冲区（最新数据）
        self.latest_sensor_data = SensorData()
        self.sensor_lock = threading.Lock()

        # 传感器数据历史缓冲区（用于时间戳匹配）
        self.sensor_buffer = SensorDataBuffer()
        self.buffer_lock = threading.Lock()

        # 传感器数据时间戳
        self.last_gps_time = 0.0
        self.last_odom_time = 0.0
        self.last_imu_time = 0.0

        # 频率统计
        self._init_frequency_stats()

        # EKF状态
        self.state = RobotState()
        self.state_lock = threading.Lock()

        # 订阅者
        self.gps_sub = self.create_subscription(
            NavSatFix,
            self.gps_topic,
            self.gps_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            10
        )

        # 发布者
        self.fusion_pub = self.create_publisher(
            String,
            self.fusion_pose_topic,
            10
        )

        self.gps_weight_pub = self.create_publisher(
            Float64,
            self.gps_weight_topic,
            10
        )

        # 定时器：按指定频率执行 EKF 融合
        period = 1.0 / max(self.frequency, 1e-3)
        self.timer = self.create_timer(period, self.fuse)

        # 节点主循环频率统计
        self.node_freq_stats = FrequencyStats(
            node_name='ekf_fusion_node',
            target_frequency=self.frequency,
            logger=self.logger,
            ros_logger=self.get_logger(),
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )

    def _init_logger(self):
        """初始化文件日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{timestamp}')

        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'ekf_fusion_node_log_{timestamp}.log')

        self.logger = logging.getLogger('ekf_fusion_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # 终端输出初始化信息（同时写入文件日志）
        init_info = [
            f'EKF Fusion Node initialized',
            f'  工作频率: {self.frequency} Hz',
            f'  GPS 话题: {self.gps_topic}',
            f'  ODOM 话题: {self.odom_topic}',
            f'  IMU 话题: {self.imu_topic}',
            f'  GPS 超时: {self.gps_timeout}s',
            f'  ODOM 超时: {self.odom_timeout}s',
            f'  IMU 超时: {self.imu_timeout}s',
            f'  使用 ODOM/IMU 融合: {self.use_odom_imu_fusion}',
            f'  详细日志已写入: {log_file}',
        ]

        for line in init_info:
            self.logger.info(line)  # 写入文件
            self.get_logger().info(line)  # 输出到终端

        # EKF协方差矩阵 (6x6)
        # [x, y, yaw, vx, vy, vyaw]
        self.P = np.diag([1.0, 1.0, 0.1, 0.5, 0.5, 0.1])

        # 状态转移矩阵
        self.F = np.eye(6)

        # 观测矩阵
        self.H_gps = np.array([
            [1, 0, 0, 0, 0, 0],  # x观测
            [0, 1, 0, 0, 0, 0],  # y观测
        ])

        self.H_odom = np.array([
            [0, 0, 0, 1, 0, 0],  # vx观测
            [0, 0, 0, 0, 1, 0],  # vy观测
            [0, 0, 0, 0, 0, 1],  # vyaw观测
        ])

        self.H_imu_yaw = np.array([
            [0, 0, 1, 0, 0, 0],  # yaw观测
        ])

    def _init_frequency_stats(self):
        """初始化频率统计变量"""
        self.freq_stats = {
            'gps': {
                'last_time': 0.0,
                'count': 0,
                'frequency': 0.0,
                'last_log_time': 0.0
            },
            'odom': {
                'last_time': 0.0,
                'count': 0,
                'frequency': 0.0,
                'last_log_time': 0.0
            },
            'imu': {
                'last_time': 0.0,
                'count': 0,
                'frequency': 0.0,
                'last_log_time': 0.0
            }
        }

    def _update_frequency_stats(self, sensor_type: str):
        """更新传感器频率统计"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        stats = self.freq_stats[sensor_type]

        if stats['last_time'] > 0:
            dt = current_time - stats['last_time']
            if dt > 0:
                stats['frequency'] = 1.0 / dt
                stats['count'] += 1

        stats['last_time'] = current_time

    def _log_frequency_stats(self):
        """周期性记录传感器频率统计"""
        if not self.log_frequency_stats:
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        # 每10秒记录一次频率统计
        log_interval = 10.0

        for sensor_type in ['gps', 'odom', 'imu']:
            stats = self.freq_stats[sensor_type]
            if current_time - stats['last_log_time'] >= log_interval:
                if stats['count'] > 0:
                    avg_freq = stats['count'] / log_interval
                    self.logger.info(
                        f'{sensor_type.upper()} Frequency: {avg_freq:.2f} Hz (samples: {stats["count"]})'
                    )
                    stats['count'] = 0
                    stats['last_log_time'] = current_time

    def gps_callback(self, msg: NavSatFix):
        """GPS数据回调"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # 解析GPS时间戳（使用消息头的时间戳）
        if msg.header.stamp.sec > 0:
            gps_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            gps_timestamp = current_time

        # 更新频率统计
        self._update_frequency_stats('gps')

        with self.sensor_lock:
            self.latest_sensor_data.gps_lat = msg.latitude
            self.latest_sensor_data.gps_lon = msg.longitude
            self.latest_sensor_data.gps_alt = msg.altitude if msg.altitude else 0.0
            self.latest_sensor_data.timestamp = gps_timestamp

            # 获取GPS位置协方差
            if len(msg.position_covariance) >= 6:
                self.latest_sensor_data.gps_cov_lat = msg.position_covariance[0]  # xx
                self.latest_sensor_data.gps_cov_lon = msg.position_covariance[4]  # yy

            # status >= 0 表示有效定位（-1 为无信号）；status=2 为 RTK 固定/浮点解
            self.latest_sensor_data.gps_available = (msg.status.status >= 0)

            self.last_gps_time = current_time

    def odom_callback(self, msg: Odometry):
        """ODOM数据回调"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # 解析ODOM时间戳
        if msg.header.stamp.sec > 0:
            odom_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            odom_timestamp = current_time

        # 更新频率统计
        self._update_frequency_stats('odom')

        # 提取ODOM数据
        odom_data = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'yaw': math.atan2(
                2.0 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z +
                       msg.pose.pose.orientation.x * msg.pose.pose.orientation.y),
                1.0 - 2.0 * (msg.pose.pose.orientation.y ** 2 + msg.pose.pose.orientation.z ** 2)
            ),
            'vx': msg.twist.twist.linear.x,
            'vy': msg.twist.twist.linear.y,
            'vyaw': msg.twist.twist.angular.z
        }

        # 存入历史缓冲区
        with self.buffer_lock:
            self.sensor_buffer.odom_history.append(
                TimestampedData(timestamp=odom_timestamp, data=odom_data)
            )

        with self.sensor_lock:
            self.latest_sensor_data.odom_x = odom_data['x']
            self.latest_sensor_data.odom_y = odom_data['y']
            self.latest_sensor_data.odom_yaw = odom_data['yaw']
            self.latest_sensor_data.odom_vx = odom_data['vx']
            self.latest_sensor_data.odom_vy = odom_data['vy']
            self.latest_sensor_data.odom_vyaw = odom_data['vyaw']
            self.latest_sensor_data.odom_available = True
            self.last_odom_time = current_time

    def imu_callback(self, msg: Imu):
        """IMU数据回调"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # 解析IMU时间戳
        if msg.header.stamp.sec > 0:
            imu_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            imu_timestamp = current_time

        # 更新频率统计
        self._update_frequency_stats('imu')

        # 提取IMU数据
        q = msg.orientation
        imu_yaw = 0.0
        if q.x != 0 or q.y != 0 or q.z != 0 or q.w != 0:
            imu_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

        imu_data = {
            'yaw': imu_yaw,
            'yaw_rate': msg.angular_velocity.z,
            'acc_x': msg.linear_acceleration.x,
            'acc_y': msg.linear_acceleration.y
        }

        # 存入历史缓冲区
        with self.buffer_lock:
            self.sensor_buffer.imu_history.append(
                TimestampedData(timestamp=imu_timestamp, data=imu_data)
            )

        with self.sensor_lock:
            self.latest_sensor_data.imu_yaw = imu_data['yaw']
            self.latest_sensor_data.imu_yaw_rate = imu_data['yaw_rate']
            self.latest_sensor_data.imu_acc_x = imu_data['acc_x']
            self.latest_sensor_data.imu_acc_y = imu_data['acc_y']
            self.latest_sensor_data.imu_available = True
            self.last_imu_time = current_time

    def get_gps_weight(self) -> float:
        """
        根据GPS信号质量计算权重

        返回值范围: 0.0 - 1.0
        0.0 表示GPS完全不可用
        1.0 表示GPS信号非常好
        """
        current_time = self.get_clock().now().nanoseconds / 1e9

        # 检查GPS是否超时
        if current_time - self.last_gps_time > self.gps_timeout:
            return 0.0

        with self.sensor_lock:
            if not self.latest_sensor_data.gps_available:
                return 0.0

            # 基于协方差计算权重
            cov = max(self.latest_sensor_data.gps_cov_lat,
                     self.latest_sensor_data.gps_cov_lon)

            if cov <= 0:
                return 0.5

            # 将协方差转换为权重
            # 协方差越小，权重越高
            if cov < self.gps_cov_threshold:
                # 信号好: 权重 0.7 - 1.0
                weight = 1.0 - (cov / self.gps_cov_threshold) * 0.3
            else:
                # 信号差: 权重 0.0 - 0.3
                weight = max(0.0, 0.3 - (cov - self.gps_cov_threshold) * 0.01)

            return weight

    def predict(self, dt: float):
        """
        EKF预测步骤

        基于运动模型预测状态和协方差
        注意：odom提供的是base_link坐标系的速度，需要转换到世界坐标系
        """
        if dt <= 0:
            return

        # 计算当前航向的正弦和余弦
        c = math.cos(self.state.yaw)
        s = math.sin(self.state.yaw)

        # 更新状态转移矩阵 (将base_link坐标系速度转换到世界坐标系)
        # x方向位移 = (vx*cos(yaw) - vy*sin(yaw)) * dt
        # y方向位移 = (vx*sin(yaw) + vy*cos(yaw)) * dt
        self.F[0, 3] = c * dt  # dx/d(vx)
        self.F[0, 4] = -s * dt  # dx/d(vy)
        self.F[1, 3] = s * dt   # dy/d(vx)
        self.F[1, 4] = c * dt   # dy/d(vy)
        self.F[2, 5] = dt       # dyaw/d(vyaw)

        # 状态预测 - 将base_link坐标系速度转换到世界坐标系
        dx = (self.state.vx * c - self.state.vy * s) * dt
        dy = (self.state.vx * s + self.state.vy * c) * dt
        self.state.x += dx
        self.state.y += dy
        self.state.yaw += self.state.vyaw * dt

        # 归一化yaw到[-pi, pi]
        self.state.yaw = math.atan2(math.sin(self.state.yaw), math.cos(self.state.yaw))

        # 协方差预测: P_pred = F * P * F' + Q
        Q = np.diag([
            self.process_noise_pos * dt,
            self.process_noise_pos * dt,
            self.process_noise_yaw * dt,
            self.process_noise_pos,
            self.process_noise_pos,
            self.process_noise_yaw
        ])

        self.P = self.F @ self.P @ self.F.T + Q

    def update_gps(self, gps_x: float, gps_y: float, noise: float):
        """
        GPS校正步骤

        Args:
            gps_x: GPS X坐标 (地图坐标系)
            gps_y: GPS Y坐标 (地图坐标系)
            noise: GPS测量噪声
        """
        # 观测噪声矩阵
        R_gps = np.eye(2) * noise

        # 观测模型
        z = np.array([gps_x, gps_y])
        z_pred = np.array([self.state.x, self.state.y])

        # 卡尔曼增益
        S = self.H_gps @ self.P @ self.H_gps.T + R_gps
        K = self.P @ self.H_gps.T @ np.linalg.inv(S)

        # 状态更新
        innovation = z - z_pred
        self.state.x += K[0, 0] * innovation[0] + K[0, 1] * innovation[1]
        self.state.y += K[1, 0] * innovation[0] + K[1, 1] * innovation[1]

        # 协方差更新
        I = np.eye(6)
        self.P = (I - K @ self.H_gps) @ self.P

    def update_odom(self, vx: float, vy: float, vyaw: float):
        """
        ODOM校正步骤

        Args:
            vx: X方向速度
            vy: Y方向速度
            vyaw: 角速度
        """
        # 观测噪声矩阵
        R_odom = np.eye(3) * self.odom_noise

        # 观测模型
        z = np.array([vx, vy, vyaw])
        z_pred = np.array([self.state.vx, self.state.vy, self.state.vyaw])

        # 卡尔曼增益
        S = self.H_odom @ self.P @ self.H_odom.T + R_odom
        K = self.P @ self.H_odom.T @ np.linalg.inv(S)

        # 状态更新
        innovation = z - z_pred
        self.state.vx += K[3, 0] * innovation[0] + K[3, 1] * innovation[1] + K[3, 2] * innovation[2]
        self.state.vy += K[4, 0] * innovation[0] + K[4, 1] * innovation[1] + K[4, 2] * innovation[2]
        self.state.vyaw += K[5, 0] * innovation[0] + K[5, 1] * innovation[1] + K[5, 2] * innovation[2]

        # 协方差更新
        I = np.eye(6)
        self.P = (I - K @ self.H_odom) @ self.P

    def update_imu_yaw(self, imu_yaw: float):
        """
        IMU航向角校正步骤

        Args:
            imu_yaw: IMU航向角
        """
        # 观测噪声矩阵
        R_imu = np.eye(1) * self.imu_noise_yaw

        # 观测模型
        z = np.array([imu_yaw])
        z_pred = np.array([self.state.yaw])

        # 计算 innovation（考虑角度Wrap）
        innovation = z_pred - z
        # 归一化到[-pi, pi]
        innovation = math.atan2(math.sin(innovation[0]), math.cos(innovation[0]))
        innovation = np.array([innovation])

        # 卡尔曼增益
        S = self.H_imu_yaw @ self.P @ self.H_imu_yaw.T + R_imu
        K = self.P @ self.H_imu_yaw.T @ np.linalg.inv(S)

        # 状态更新
        self.state.yaw -= K[2, 0] * innovation[0]

        # 归一化yaw到[-pi, pi]
        self.state.yaw = math.atan2(math.sin(self.state.yaw), math.cos(self.state.yaw))

        # 协方差更新
        I = np.eye(6)
        self.P = (I - K @ self.H_imu_yaw) @ self.P

    def find_matched_data(self, target_timestamp: float, sensor_type: str) -> Optional[dict]:
        """
        找到与目标时间戳匹配的数据
        返回距离目标时间戳最近且不超过失效时间阈值的数据
        """
        if sensor_type == 'odom':
            history = self.sensor_buffer.odom_history
            timeout = self.odom_timeout
        elif sensor_type == 'imu':
            history = self.sensor_buffer.imu_history
            timeout = self.imu_timeout
        else:
            return None

        with self.buffer_lock:
            if not history:
                return None

            # 找到时间差最小且在阈值内的数据
            best_data = None
            best_time_diff = float('inf')

            for item in history:
                time_diff = abs(item.timestamp - target_timestamp)
                if time_diff < best_time_diff and time_diff <= timeout:
                    best_time_diff = time_diff
                    best_data = item.data

            return best_data

    def fuse(self):
        """执行一次EKF融合（有什么用什么）"""
        # 记录频率统计
        self.node_freq_stats.tick()

        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = 1.0 / self.frequency

        # 获取最新GPS数据和时间戳
        with self.sensor_lock:
            latest_gps_timestamp = self.latest_sensor_data.timestamp
            gps_available = self.latest_sensor_data.gps_available
            gps_lat = self.latest_sensor_data.gps_lat
            gps_lon = self.latest_sensor_data.gps_lon

        # 检查GPS是否超时
        gps_timeout = current_time - self.last_gps_time > self.gps_timeout
        gps_stale = latest_gps_timestamp == 0.0 or gps_timeout

        # 融合状态记录
        fusion_components = []

        # 只使用GPS模式：跳过EKF，直接使用GPS值输出
        if not self.use_odom_imu_fusion:
            if not gps_stale and gps_available:
                # 直接使用GPS值作为定位输出
                with self.state_lock:
                    self.state.x = gps_lat
                    self.state.y = gps_lon
                fusion_components.append('GPS')
                self.get_logger().debug(f"GPS direct output: lat={gps_lat:.8f}, lon={gps_lon:.8f}")
            else:
                self.get_logger().warn("GPS not available for direct output")
        else:
            # 融合模式：使用EKF融合
            # 预测步骤
            self.predict(dt)

            # 获取GPS权重
            gps_weight = self.get_gps_weight()

            # 尝试融合GPS数据（GPS时间戳有效且信号质量足够）
            # 注：不再在 ekf_fusion_node 中做 gps -> map 坐标转换，由 map_node 负责
            if not gps_stale and gps_available:
                # 根据信号质量调整噪声
                gps_noise = min(self.gps_noise_good / gps_weight, self.gps_noise_bad)
            
                if gps_weight > 0.1:
                    # GPS 融合到 odom 坐标系，不需要坐标转换
                    self.update_gps(0.0, 0.0, gps_noise)  # GPS 在 odom 坐标系中的初始偏移为 0
                    fusion_components.append('GPS')

            # 获取最新ODOM和IMU数据
            with self.sensor_lock:
                odom_available = self.latest_sensor_data.odom_available
                odom_vx = self.latest_sensor_data.odom_vx
                odom_vy = self.latest_sensor_data.odom_vy
                odom_vyaw = self.latest_sensor_data.odom_vyaw

                imu_available = self.latest_sensor_data.imu_available
                imu_yaw = self.latest_sensor_data.imu_yaw
                imu_yaw_rate = self.latest_sensor_data.imu_yaw_rate

            # 检查ODOM和IMU是否超时
            odom_timeout = current_time - self.last_odom_time > self.odom_timeout
            imu_timeout = current_time - self.last_imu_time > self.imu_timeout

            # 融合ODOM速度数据
            if odom_available and not odom_timeout:
                self.update_odom(odom_vx, odom_vy, odom_vyaw)
                fusion_components.append('ODOM')

            # 融合IMU航向角数据
            if imu_available and not imu_timeout:
                self.update_imu_yaw(imu_yaw)
                fusion_components.append('IMU')

        # 确定融合模式标识
        if not fusion_components:
            fusion_mode = 'invalid'
        else:
            fusion_mode = '+'.join(fusion_components)

        # 发布GPS权重
        weight_msg = Float64()
        weight_msg.data = 1.0 if not self.use_odom_imu_fusion else self.get_gps_weight()
        self.gps_weight_pub.publish(weight_msg)

        # 发布融合结果（带融合模式标记）
        self.publish_fusion_result(fusion_mode)

        # 记录融合日志
        with self.state_lock:
            if fusion_mode == 'invalid':
                self.logger.warn(
                    f"Fusion | INVALID | GPS_ts: {latest_gps_timestamp:.3f}, "
                    f"Current: {current_time:.3f}, Diff: {current_time - latest_gps_timestamp:.3f}s"
                )
            else:
                self.logger.info(
                    f"Fusion | {fusion_mode} | "
                    f"Pos: ({self.state.x:.3f}, {self.state.y:.3f}) | "
                    f"Yaw: {math.degrees(self.state.yaw):.1f}deg"
                )

        # 周期性记录频率统计
        self._log_frequency_stats()

    def publish_fusion_result(self, fusion_mode: str = 'unknown'):
        """发布融合定位结果：/fusion_pose (odom坐标系)
        
        注：/map_pose 由 map_node 负责发布（接收 /fusion_pose 后转换）
        """
        with self.state_lock:
            is_valid = fusion_mode != 'invalid'

            # 原始融合结果（odom坐标系）
            result = {
                'x': self.state.x,
                'y': self.state.y,
                'yaw': self.state.yaw,
                'vx': self.state.vx,
                'vy': self.state.vy,
                'vyaw': self.state.vyaw,
                'timestamp': self.get_clock().now().nanoseconds / 1e9,
                'valid': is_valid,
                'fusion_mode': fusion_mode
            }

        # 发布 /fusion_pose (odom坐标系)
        msg = String()
        msg.data = json.dumps(result)
        self.fusion_pub.publish(msg)

        # 注：/map_pose 由 map_node 负责发布（接收 /fusion_pose 后转换）

    def initialize_from_sensors(self):
        """使用传感器数据初始化EKF状态"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        with self.sensor_lock:
            sensor_data = self.latest_sensor_data

        # 注：坐标转换由 map_node 负责，ekf_fusion_node 只负责发布 /fusion_pose (odom坐标系)

        # 根据配置决定是否使用odom和imu初始化
        if self.use_odom_imu_fusion:
            # 使用IMU初始化航向
            if sensor_data.imu_available:
                if current_time - self.last_imu_time < self.imu_timeout:
                    self.state.yaw = sensor_data.imu_yaw
                    self.state.vyaw = sensor_data.imu_yaw_rate
                    self.get_logger().info(f'Initialized from IMU: yaw={math.degrees(self.state.yaw):.2f}deg')

            # 使用ODOM初始化速度
            if sensor_data.odom_available:
                if current_time - self.last_odom_time < self.odom_timeout:
                    self.state.vx = sensor_data.odom_vx
                    self.state.vy = sensor_data.odom_vy
        else:
            self.get_logger().info('ODOM/IMU fusion disabled, skipping odom/imu initialization')


def run_ekf_fusion_node(args=None):
    """运行EKF融合定位节点"""
    rclpy.init(args=args)
    node = EKFFusionNode()

    # 等待传感器数据初始化
    node.get_logger().info('Waiting for sensor data to initialize...')
    timeout = 10.0  # 最多等待10秒
    start_time = node.get_clock().now().nanoseconds / 1e9

    while rclpy.ok():
        current_time = node.get_clock().now().nanoseconds / 1e9

        # 检查是否有足够的传感器数据
        with node.sensor_lock:
            has_gps = node.latest_sensor_data.gps_available
            has_odom = node.latest_sensor_data.odom_available
            has_imu = node.latest_sensor_data.imu_available

        if has_gps and has_odom and has_imu:
            break

        if current_time - start_time > timeout:
            node.get_logger().warn('Timeout waiting for sensor data, starting anyway...')
            break

        time.sleep(0.1)

    # 初始化EKF状态
    node.initialize_from_sensors()

    # 创建定时器执行融合
    period = 1.0 / node.frequency

    timer = node.create_timer(period, node.fuse)

    # 使用SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)

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
