#!/usr/bin/env python3
"""
激光雷达Costmap节点
1. 接收激光雷达扫描数据，过滤高度范围
2. 稀疏化去除噪声点
3. 压平到2D并截取感兴趣区域
4. 转换为局部costmap并发布到topic，供地图节点使用
5. 计算并发布RL观测数据
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
import json
import threading
import time
import math
import numpy as np
import os
import logging
from datetime import datetime

from config_loader import get_config


class LidarcostmapNode(Node):
    """
    激光雷达Costmap节点
    
    功能：
    1. 接收激光雷达扫描数据
    2. 过滤高度范围（保留最低和最高高度之间的点）
    3. 稀疏化降采样
    4. 压平到2D，截取感兴趣区域
    5. 转换为局部costmap并发布到topic
    6. 发布RL观测数据
    """

    def __init__(self):
        super().__init__('lidar_costmap_node')

        # 加载配置文件
        config = get_config()
        lidar_config = config.get('lidar_costmap_node', {})

        # 初始化日志系统
        self._init_logger(lidar_config.get('log_enabled', True))

        # 获取订阅和发布话题配置
        subscriptions = lidar_config.get('subscriptions', {})
        publications = lidar_config.get('publications', {})

        # 室外模式：使用 map_pose
        pose_topic = '/map_pose'
        pose_format = 'json'

        # 订阅话题
        self.scan_topic = subscriptions.get('scan_topic', '/scan')

        # 发布话题
        self.obs_output_topic = publications.get('obs_output_topic', '/lidar_obs')
        self.local_costmap_topic = publications.get('local_costmap_topic', '/local_costmap')

        self.scan_range = lidar_config.get('scan_range', 10.0)  # 扫描范围 (米)
        self.min_height = lidar_config.get('min_height', -0.5)   # 最低高度
        self.max_height = lidar_config.get('max_height', 2.0)   # 最高高度

        self.downsampling_factor = lidar_config.get('downsampling_factor', 3)

        self.update_frequency = lidar_config.get('update_frequency', 5.0)
        self.costmap_resolution = lidar_config.get('costmap_resolution', 0.1)
        self.costmap_value = lidar_config.get('costmap_value', 100)

        # RL输入参数
        self.bin_num = lidar_config.get('bin_num', 20)  # 前方180度的分区数量
        self.obs_output_topic = lidar_config.get('obs_output_topic', '/lidar_obs')

        # 状态
        self.latest_scan = None
        self.scan_lock = threading.Lock()

        # 订阅者 - LaserScan
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )

        # 发布者 - RL观测数据
        self.obs_pub = self.create_publisher(
            String,
            self.obs_output_topic,
            10
        )

        # 发布者 - 局部costmap（供地图节点使用）
        self.local_costmap_pub = self.create_publisher(
            String,
            self.local_costmap_topic,
            10
        )

        self.logger.info('LiDAR Costmap Node initialized')
        self.logger.info(f'  Scan topic: {self.scan_topic}')
        self.logger.info(f'  Scan range: {self.scan_range}m')
        self.logger.info(f'  Height range: {self.min_height}m to {self.max_height}m')
        self.logger.info(f'  Downsampling: {self.downsampling_factor}')
        self.logger.info(f'  Update frequency: {self.update_frequency} Hz')
        self.logger.info(f'  Costmap resolution: {self.costmap_resolution}m')
        self.logger.info(f'  Bin num: {self.bin_num}')
        self.logger.info(f'  Obs output topic: {self.obs_output_topic}')
        self.logger.info(f'  Local costmap topic: {self.local_costmap_topic}')

        # 定时器：按指定频率处理激光雷达数据
        period = 1.0 / max(self.update_frequency, 1e-3)
        self.timer = self.create_timer(period, self.process_scan)

    def _init_logger(self, enabled: bool):
        """初始化文件日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'lidar_costmap_node_log_{timestamp}.log')

        self.logger = logging.getLogger('lidar_costmap_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        if enabled:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

            self.logger.info(f'LiDAR Costmap Node started, log file: {log_file}')

    def scan_callback(self, msg: LaserScan):
        """激光扫描回调"""
        with self.scan_lock:
            self.latest_scan = msg

    def process_scan(self):
        """处理激光扫描数据"""
        with self.scan_lock:
            if self.latest_scan is None:
                return None
            scan = self.latest_scan

        # 1. 提取点（LaserScan是2D，需要转换为3D点）
        # 假设LaserScan是单线激光雷达，返回的是2D扫描
        # 这里我们需要处理的是3D点云，但ROS中常用的是PointCloud2
        # 对于LaserScan，我们直接使用其range和angle信息

        ranges = np.array(scan.ranges)
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)

        # 过滤无效距离
        valid_mask = (ranges > scan.range_min) & (ranges < scan.range_max)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]

        if len(ranges) == 0:
            return None

        # 2. 稀疏化降采样
        if self.downsampling_factor > 1:
            indices = np.arange(0, len(ranges), self.downsampling_factor)
            ranges = ranges[indices]
            angles = angles[indices]

        # 3. 压平到2D（对于2D激光雷达，直接使用range和angle）
        # 计算2D坐标（相对于雷达）
        # 假设雷达坐标系：前方为X正方向，左侧为Y正方向（ROS标准）
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)

        # 4. 截取感兴趣区域
        # 左边 scan_range，右边 scan_range，前方 scan_range
        # 即：X方向 -scan_range ~ +scan_range，Y方向 0 ~ scan_range
        # 这是一个宽为 2*scan_range，高为 scan_range 的长方形
        x_min = -self.scan_range
        x_max = self.scan_range
        y_min = 0.0
        y_max = self.scan_range

        # 过滤在感兴趣区域内的点
        mask = (x_coords >= x_min) & (x_coords <= x_max) & \
               (y_coords >= y_min) & (y_coords <= y_max)

        x_filtered = x_coords[mask]
        y_filtered = y_coords[mask]

        if len(x_filtered) == 0:
            return None

        return x_filtered, y_filtered

    def compute_lidar_obs(self, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        """
        计算前方180度的分区最近障碍物距离

        从右到左逆时针扫描前方180度，分成bin_num个分区
        右边(y>0, x>0) -> 前方(y>0) -> 左边(y>0, x<0)

        Args:
            x_points: 过滤后的X坐标
            y_points: 过滤后的Y坐标

        Returns:
            obs_min: 每个分区的最近障碍物距离数组
        """
        if len(x_points) == 0:
            return np.full(self.bin_num, self.scan_range)

        # 计算每个点的角度（相对于机器狗前方）
        # 前方为0度，右边为-90度，左边为+90度
        # 但这里是从ROI中取的点：x范围[-scan_range, scan_range]，y范围[0, scan_range]
        angles = np.arctan2(y_points, x_points)  # -pi ~ pi

        # 只取前方180度：-pi/2 ~ pi/2
        # 即：左边极限角度90度，右边极限角度-90度
        front_mask = (angles >= -math.pi / 2) & (angles <= math.pi / 2)

        front_x = x_points[front_mask]
        front_y = y_points[front_mask]
        front_angles = angles[front_mask]

        if len(front_x) == 0:
            return np.full(self.bin_num, self.scan_range)

        # 计算每个点到机器狗的距离
        distances = np.sqrt(front_x ** 2 + front_y ** 2)

        # 将角度映射到分区索引
        # -pi/2 -> 0, 0 -> bin_num/2, pi/2 -> bin_num
        # 归一化角度到[0, 1]
        normalized_angles = (front_angles + math.pi / 2) / math.pi  # 0 ~ 1
        bin_indices = (normalized_angles * (self.bin_num - 1)).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.bin_num - 1)

        # 计算每个分区的最近障碍物距离
        obs_min = np.full(self.bin_num, self.scan_range)

        for i in range(self.bin_num):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                obs_min[i] = np.min(distances[bin_mask])

        return obs_min

    def publish_lidar_obs(self, obs_min: np.ndarray):
        """发布RL观测数据"""
        # 转换为列表并归一化（除以scan_range）
        obs_normalized = (obs_min / self.scan_range).tolist()

        msg = String()
        msg.data = json.dumps({
            'obs_min': obs_normalized,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        })

        self.obs_pub.publish(msg)

    def publish_local_costmap(self, costmap: np.ndarray):
        """发布局部costmap到topic，供地图节点使用"""
        # 转换为1D列表
        costmap_flat = costmap.flatten().tolist()
        
        height, width = costmap.shape
        
        msg = String()
        msg.data = json.dumps({
            'costmap': costmap_flat,
            'width': width,
            'height': height,
            'resolution': self.costmap_resolution,
            'value': self.costmap_value,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        })

        self.local_costmap_pub.publish(msg)

    def points_to_costmap(self, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        """将2D点转换为costmap"""
        # 计算costmap大小
        # 宽: 2*scan_range (X方向: -scan_range ~ +scan_range)
        # 高: scan_range (Y方向: 0 ~ scan_range)
        width = int(2 * self.scan_range / self.costmap_resolution)
        height = int(self.scan_range / self.costmap_resolution)

        # 创建空白costmap（0表示空闲）
        costmap = np.zeros((height, width), dtype=np.int8)

        # 将点转换为网格坐标
        # X: -scan_range ~ scan_range -> 0 ~ width
        # Y: 0 ~ scan_range -> height ~ 0 (注意Y轴方向)
        grid_x = ((x_points + self.scan_range) / self.costmap_resolution).astype(int)
        grid_y = ((self.scan_range - y_points) / self.costmap_resolution).astype(int)

        # 过滤边界外的点
        valid_mask = (grid_x >= 0) & (grid_x < width) & \
                     (grid_y >= 0) & (grid_y < height)

        grid_x = grid_x[valid_mask]
        grid_y = grid_y[valid_mask]

        # 标记障碍物
        for gx, gy in zip(grid_x, grid_y):
            costmap[gy, gx] = self.costmap_value

        return costmap

    def update(self):
        """执行一次更新"""
        result = self.process_scan()

        if result is None:
            return

        x_filtered, y_filtered = result

        # 1. 转换为costmap并发布到topic供地图节点使用
        if len(x_filtered) > 0:
            costmap = self.points_to_costmap(x_filtered, y_filtered)
            self.publish_local_costmap(costmap)

        # 2. 计算RL观测数据并发布
        obs_min = self.compute_lidar_obs(x_filtered, y_filtered)
        self.publish_lidar_obs(obs_min)


def main(args=None):
    rclpy.init(args=args)

    node = LidarcostmapNode()

    # 创建定时器
    period = 1.0 / node.update_frequency
    timer = node.create_timer(period, node.update)

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
    main()
