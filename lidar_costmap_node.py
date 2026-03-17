#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光雷达 Costmap 节点（全程使用 ROS 标准坐标系）

ROS 标准坐标系：
- x: 前方
- y: 左侧
- z: 上方

流程：
1. 节点启动时预计算射线路径
2. 订阅 PointCloud2，每次只保存最新点云
3. 按工作频率循环：若最新点云时间戳失效，则跳过
4. 高度过滤
5. 降采样
6. 截取 ROI
7. 转换为 costmap（先投影，再沿射线向后填充）
8. 计算 obs_min_distance
9. 将 2D costmap 和 obs_min_distance 发布到指定 topic
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import PointCloud2, LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Quaternion
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
from frequency_stats import FrequencyStats
from builtin_interfaces.msg import Time as RosTime


class LidarCostmapNode(Node):
    def __init__(self, log_dir: str = None, timestamp: str = None):
        super().__init__('lidar_costmap_node')

        # 使用传入的日志目录和时间戳，或生成新的
        self.log_dir = log_dir
        self.timestamp = timestamp if timestamp is not None else datetime.now().strftime('%Y%m%d_%H%M%S')

        # 加载配置
        config = get_config()
        lidar_config = config.get('lidar_costmap_node', {})
        subscriptions = lidar_config.get('subscriptions', {})
        publications = lidar_config.get('publications', {})

        # 订阅 PointCloud2
        self.pointcloud_topic = subscriptions.get(
            'pointcloud_topic',
            subscriptions.get('scan_topic', '/livox/lidar')
        )

        # 发布话题
        self.obs_output_topic = publications.get('obs_output_topic', '/lidar_obs')
        self.local_costmap_topic = publications.get('local_costmap_topic', '/local_costmap')

        # 参数
        self.scan_range = float(lidar_config.get('scan_range', 10.0))
        self.min_height = float(lidar_config.get('min_height', 0.0))
        self.max_height = float(lidar_config.get('max_height', 2.0))

        # -1 / 0 / 1 表示不降采样；>=2 表示每隔 factor 个点取 1 个
        self.downsampling_factor = int(lidar_config.get('downsampling_factor', -1))

        # 工作频率 (Hz)
        self.frequency = float(lidar_config.get('frequency', 10.0))
        # 从公共配置获取分辨率
        common_config = config.get('common', {})
        self.costmap_resolution = float(common_config.get('resolution', 0.05))

        # 其他参数
        self.costmap_value = int(lidar_config.get('costmap_value', 100))
        self.bin_num = int(lidar_config.get('bin_num', 20))
        self.ray_angle_step_deg = float(lidar_config.get('ray_angle_step_deg', 1.0))

        # 点云时间戳失效阈值（秒）
        self.cloud_timeout = float(lidar_config.get('cloud_timeout', 1.0))

        # 状态：只维护最新点云（ROS 标准坐标系）
        self.latest_points = None               # numpy array, shape (N, 3)
        self.latest_cloud_stamp = None          # float seconds
        self.scan_lock = threading.Lock()

        # 用于节流 warning，避免刷屏
        self._last_warn_time = {}

        # 订阅者
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            10
        )

        # 发布者
        self.obs_pub = self.create_publisher(
            LaserScan,
            self.obs_output_topic,
            10
        )

        self.local_costmap_pub = self.create_publisher(
            OccupancyGrid,
            self.local_costmap_topic,
            10
        )

        # 初始化日志
        self._init_logger(lidar_config.get('log_enabled', True))

        # 节点启动时预计算射线路径（ROS 标准坐标系）
        t0 = time.perf_counter()
        self.ray_paths = self.precompute_ray_paths(
            self.scan_range,
            self.costmap_resolution,
            self.ray_angle_step_deg
        )
        t1 = time.perf_counter()

        self.logger.info(
            f'Precomputed ray paths: {len(self.ray_paths)} rays, time cost: {t1 - t0:.6f} s'
        )
        self.get_logger().info(
            f'Precomputed ray paths: {len(self.ray_paths)} rays, time cost: {t1 - t0:.6f} s'
        )

        # 定时器：按指定频率执行一次完整流程
        period = 1.0 / max(self.frequency, 1e-3)
        self.timer = self.create_timer(period, self.update)

        # 初始化频率统计
        self.freq_stats = FrequencyStats(
            node_name='lidar_costmap_node',
            target_frequency=self.frequency,
            logger=self.logger,
            ros_logger=self.get_logger(),
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )
    def _sec_to_stamp(self, t: float) -> RosTime:
        stamp = RosTime()
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        stamp.sec = sec
        stamp.nanosec = nanosec
        return stamp
    def _init_logger(self, enabled: bool):
        """初始化文件日志系统"""
        self.log_enabled = enabled

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

        log_file = os.path.join(log_dir, f'lidar_costmap_node_log_{self.timestamp}.log')

        self.logger = logging.getLogger('lidar_costmap_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False

        if enabled:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        init_info = [
            'LiDAR Costmap Node initialized',
            f'  订阅点云: {self.pointcloud_sub.topic}',
            f'  发布 obs_min_distance: {self.obs_pub.topic}',
            f'  发布局部 costmap: {self.local_costmap_pub.topic}',
            '  坐标系: ROS 标准坐标系 (x前, y左, z上)',
            f'  扫描范围: {self.scan_range} m',
            f'  高度范围: {self.min_height} m ~ {self.max_height} m',
            f'  降采样因子: {self.downsampling_factor}',
            f'  costmap 分辨率: {self.costmap_resolution} m',
            f'  costmap 值: {self.costmap_value}',
            f'  bin 数量: {self.bin_num}',
            f'  射线角度间隔: {self.ray_angle_step_deg} deg',
            f'  工作频率: {self.frequency} Hz',
            f'  点云超时阈值: {self.cloud_timeout} s',
            f'  耗时日志: {"启用" if self.log_enabled else "禁用"}',
            f'  详细日志文件: {log_file}',
        ]

        for line in init_info:
            self.logger.info(line)
            self.get_logger().info(line)

    def _stamp_to_sec(self, stamp) -> float:
        """ROS 时间戳转秒"""
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _warn_throttled(self, key: str, message: str, interval_sec: float = 1.0):
        """节流 warning，避免高频刷屏"""
        now = time.monotonic()
        last = self._last_warn_time.get(key, 0.0)
        if now - last >= interval_sec:
            self._last_warn_time[key] = now
            self.logger.warning(message)
            self.get_logger().warning(message)

    def pointcloud2_to_xyz_array(self, msg: PointCloud2) -> np.ndarray:
        offsets = {}
        for f in msg.fields:
            if f.name in ('x', 'y', 'z'):
                offsets[f.name] = f.offset

        if len(offsets) != 3:
            raise ValueError("PointCloud2 missing x/y/z")

        n = msg.width * msg.height
        if n == 0:
            return np.empty((0, 3), dtype=np.float32)

        endian = '>' if msg.is_bigendian else '<'
        dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [endian + 'f4', endian + 'f4', endian + 'f4'],
            'offsets': [offsets['x'], offsets['y'], offsets['z']],
            'itemsize': msg.point_step,
        })

        arr = np.frombuffer(msg.data, dtype=dtype, count=n)
        points = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32, copy=False)

        valid = np.isfinite(points).all(axis=1)
        valid &= ~((points[:, 0] == 0.0) & (points[:, 1] == 0.0) & (points[:, 2] == 0.0))

        return points[valid]

    def pointcloud_callback(self, msg: PointCloud2):
        """
        点云回调：
        只维护最新一帧点云（ROS 标准坐标系）
        """
        points = self.pointcloud2_to_xyz_array(msg)

        cloud_stamp = self._stamp_to_sec(msg.header.stamp)
        if cloud_stamp <= 0.0:
            cloud_stamp = self.get_clock().now().nanoseconds / 1e9

        with self.scan_lock:
            self.latest_points = points
            self.latest_cloud_stamp = cloud_stamp

    def precompute_ray_paths(self, scan_range: float, resolution: float, angle_step_deg: float):
        """
        预计算每条射线会经过哪些 grid cell。
        只依赖 scan_range / resolution / angle_step_deg，节点启动时算一次即可。

        ROS 标准坐标系角度定义：
        - 前方 +x = 0°
        - 右边 = -90°（负 y）
        - 左边 = +90°（正 y）
        """
        rows = int(scan_range / resolution)         # 前向 x 方向
        cols = int(2 * scan_range / resolution)     # 横向 y 方向

        angle_step_rad = np.deg2rad(angle_step_deg)
        angles = np.arange(-np.pi / 2, np.pi / 2 + angle_step_rad * 0.5, angle_step_rad)

        radial_step = resolution * 0.5
        r_values = np.arange(0.0, scan_range + radial_step, radial_step)

        ray_paths = []

        for theta in angles:
            # ROS 标准坐标系：
            # x 前方，y 左侧
            x = r_values * np.cos(theta)
            y = r_values * np.sin(theta)

            # 列表示横向 y：[-scan_range, +scan_range)
            grid_col = np.floor((y + scan_range) / resolution).astype(int)

            # 行表示前向 x：[0, scan_range)，x 越大越靠上
            grid_row = rows - 1 - np.floor(x / resolution).astype(int)

            valid = (
                (grid_col >= 0) & (grid_col < cols) &
                (grid_row >= 0) & (grid_row < rows)
            )
            grid_col = grid_col[valid]
            grid_row = grid_row[valid]

            if len(grid_col) == 0:
                ray_paths.append((
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32)
                ))
                continue

            cells = np.stack([grid_row, grid_col], axis=1)

            # 去掉重复格子，保留顺序
            unique_cells = [cells[0]]
            for i in range(1, len(cells)):
                if not np.array_equal(cells[i], cells[i - 1]):
                    unique_cells.append(cells[i])

            unique_cells = np.array(unique_cells, dtype=np.int32)
            ray_paths.append((unique_cells[:, 0], unique_cells[:, 1]))

        return ray_paths

    def filter_height(self, points: np.ndarray) -> np.ndarray:
        """高度过滤"""
        if len(points) == 0:
            return points
        mask = (points[:, 2] >= self.min_height) & (points[:, 2] <= self.max_height)
        return points[mask]

    def downsample(self, points: np.ndarray) -> np.ndarray:
        """
        降采样：
        - factor = -1 / 0 / 1：不降采样
        - factor >= 2：每隔 factor 个点取 1 个
        """
        factor = self.downsampling_factor

        if factor == -1 or factor == 0 or factor == 1:
            return points
        if len(points) == 0:
            return points

        indices = np.arange(0, len(points), factor)
        return points[indices]

    def filter_roi(self, points: np.ndarray) -> np.ndarray:
        """
        截取 ROI（ROS 标准坐标系）：
        - x 方向: [0, scan_range)
        - y 方向: [-scan_range, +scan_range)
        """
        if len(points) == 0:
            return points

        mask = (
            (points[:, 0] >= 0.0) &
            (points[:, 0] < self.scan_range) &
            (points[:, 1] >= -self.scan_range) &
            (points[:, 1] <  self.scan_range)
        )
        return points[mask]

    def project_points_to_initial_costmap(self, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        """
        将点云直接投影到初步 2D costmap 上（ROS 标准坐标系）

        costmap 大小：
        - 高 = scan_range / resolution
        - 宽 = 2 * scan_range / resolution

        坐标范围：
        - x: [0, scan_range)
        - y: [-scan_range, +scan_range)

        数组定义：
        - 行(row): 前向 x，x 越大越靠上
        - 列(col): 横向 y，从右(-scan_range)到左(+scan_range)
        """
        rows = int(self.scan_range / self.costmap_resolution)
        cols = int(2 * self.scan_range / self.costmap_resolution)

        costmap = np.zeros((rows, cols), dtype=np.uint8)

        if len(x_points) == 0:
            return costmap

        grid_col = np.floor((y_points + self.scan_range) / self.costmap_resolution).astype(int)
        grid_row = rows - 1 - np.floor(x_points / self.costmap_resolution).astype(int)

        valid_mask = (
            (grid_col >= 0) & (grid_col < cols) &
            (grid_row >= 0) & (grid_row < rows)
        )

        grid_col = grid_col[valid_mask]
        grid_row = grid_row[valid_mask]

        costmap[grid_row, grid_col] = self.costmap_value
        return costmap

    def fill_costmap_by_rays_fast(self, initial_costmap: np.ndarray) -> np.ndarray:
        """
        使用预计算好的射线路径做填充：
        一条射线碰到第一个障碍后，后续格子全部设为障碍。
        """
        final_costmap = initial_costmap.copy()
        obstacle_mask = initial_costmap > 0

        for grid_row_path, grid_col_path in self.ray_paths:
            if len(grid_col_path) == 0:
                continue

            hits = obstacle_mask[grid_row_path, grid_col_path]

            if np.any(hits):
                first_hit_idx = np.argmax(hits)
                final_costmap[
                    grid_row_path[first_hit_idx:],
                    grid_col_path[first_hit_idx:]
                ] = self.costmap_value

        return final_costmap

    def points_to_costmap(self, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        """
        先投影成初步 costmap，再做射线填充，得到最终 costmap
        """
        initial_costmap = self.project_points_to_initial_costmap(x_points, y_points)
        final_costmap = self.fill_costmap_by_rays_fast(initial_costmap)
        return final_costmap

    def compute_lidar_obs(self, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        """
        计算前方 180 度各分区最近障碍物距离（ROS 标准坐标系）

        角度定义：
        - 前方(+x) = 0°
        - 右边 = -90°
        - 左边 = +90°
        """
        if len(x_points) == 0:
            return np.full(self.bin_num, self.scan_range, dtype=np.float32)

        angles = np.arctan2(y_points, x_points)

        front_mask = (
            (angles >= -math.pi / 2) &
            (angles <=  math.pi / 2) &
            (x_points >= 0.0)
        )

        front_x = x_points[front_mask]
        front_y = y_points[front_mask]
        front_angles = angles[front_mask]

        if len(front_x) == 0:
            return np.full(self.bin_num, self.scan_range, dtype=np.float32)

        distances = np.sqrt(front_x ** 2 + front_y ** 2)

        normalized_angles = (front_angles + math.pi / 2) / math.pi
        bin_indices = np.floor(normalized_angles * self.bin_num).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.bin_num - 1)

        obs_min_distance = np.full(self.bin_num, self.scan_range, dtype=np.float32)

        for i in range(self.bin_num):
            bin_mask = (bin_indices == i)
            if np.any(bin_mask):
                obs_min_distance[i] = np.min(distances[bin_mask])

        return obs_min_distance

    def publish_lidar_obs(self, obs_min_distance: np.ndarray, source_timestamp: float):
        """
        发布 obs_min_distance
        使用 LaserScan 消息类型，frame_id = 'base_link'
        """
        msg = LaserScan()
        msg.header.stamp = self._sec_to_stamp(source_timestamp)
        msg.header.frame_id = 'base_link'
        
        # 设置角度范围
        msg.angle_min = -math.pi / 2  # -90度
        msg.angle_max = math.pi / 2   # 90度
        msg.angle_increment = math.pi / self.bin_num
        msg.time_increment = 0.0
        msg.scan_time = 0.1  # 假设扫描时间
        msg.range_min = 0.0
        msg.range_max = self.scan_range
        msg.ranges = obs_min_distance.tolist()
        msg.intensities = []
        
        self.obs_pub.publish(msg)

    def publish_local_costmap(self, costmap: np.ndarray, source_timestamp: float):
        """
        发布局部 costmap（修复版）
        使用 OccupancyGrid 消息类型，frame_id = 'base_link'

        内部 costmap 定义：
        - shape = (rows_x, cols_y)
        - row 对应前向 x，且 row 越小表示 x 越大（越靠前）
        - col 对应横向 y，且 col 越大表示 y 越大（越靠左）

        目标 OccupancyGrid 定义：
        - width 对应 x 方向
        - height 对应 y 方向
        - origin 为左下角，即 (min_x, min_y)
        - 地图覆盖范围：
            x: [0, scan_range)
            y: [-scan_range, +scan_range)
        - 因此 base_link 位于脚下那条长边的中心
        """
        rows_x, cols_y = costmap.shape

        msg = OccupancyGrid()
        msg.header.stamp = self._sec_to_stamp(source_timestamp)
        msg.header.frame_id = 'base_link'

        # OccupancyGrid 语义：
        # width  -> x 方向格子数
        # height -> y 方向格子数
        msg.info.resolution = self.costmap_resolution
        msg.info.width = rows_x
        msg.info.height = cols_y

        # 地图左下角坐标
        # x 从 0 开始向前延伸
        # y 从 -scan_range 到 +scan_range
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = -self.scan_range
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0

        # 转为 OccupancyGrid 数据格式
        # 内部 costmap[row_x, col_y]
        # -> OccupancyGrid[my, mx]
        #
        # 先 flip(axis=0) 让 x 从近到远变成从小到大，
        # 再转置成 (y, x) 排列
        costmap_int = np.clip(costmap.astype(np.int8), 0, 100)
        occ_grid = np.flip(costmap_int, axis=0).T   # shape: (height_y, width_x)

        msg.data = occ_grid.reshape(-1).tolist()

        self.local_costmap_pub.publish(msg)

    def update(self):
        """
        按工作频率执行一次完整流程：
        1. 取最新点云
        2. 时间戳校验
        3. 高度过滤
        4. 降采样
        5. ROI
        6. costmap
        7. obs_min_distance
        8. 发布
        """
        # 记录频率统计
        self.freq_stats.tick()

        cycle_start = time.perf_counter()

        with self.scan_lock:
            if self.latest_points is None or self.latest_cloud_stamp is None:
                self._warn_throttled('no_cloud', 'No PointCloud2 received yet, skip this update.', 2.0)
                return

            points = self.latest_points
            cloud_stamp = self.latest_cloud_stamp

        # 时间戳失效检测
        now_sec = self.get_clock().now().nanoseconds / 1e9
        cloud_age = max(0.0, now_sec - cloud_stamp)

        if cloud_age > self.cloud_timeout:
            self._warn_throttled(
                'stale_cloud',
                f'Latest point cloud is stale: age={cloud_age:.3f}s > timeout={self.cloud_timeout:.3f}s, skip.',
                1.0
            )
            return

        # 4. 高度过滤
        t0 = time.perf_counter()
        points_filtered = self.filter_height(points)
        t1 = time.perf_counter()
        time_height = t1 - t0

        # 5. 降采样
        t0 = time.perf_counter()
        points_downsampled = self.downsample(points_filtered)
        t1 = time.perf_counter()
        time_downsample = t1 - t0

        # 6. ROI
        t0 = time.perf_counter()
        points_roi = self.filter_roi(points_downsampled)
        t1 = time.perf_counter()
        time_roi = t1 - t0

        if len(points_roi) == 0:
            x_filtered = np.array([], dtype=np.float32)
            y_filtered = np.array([], dtype=np.float32)
        else:
            x_filtered = points_roi[:, 0]
            y_filtered = points_roi[:, 1]

        # 7. costmap
        t0 = time.perf_counter()
        costmap = self.points_to_costmap(x_filtered, y_filtered)
        t1 = time.perf_counter()
        time_costmap = t1 - t0

        # 8. obs_min_distance
        t0 = time.perf_counter()
        obs_min_distance = self.compute_lidar_obs(x_filtered, y_filtered)
        t1 = time.perf_counter()
        time_obs = t1 - t0

        # 9. 发布
        t0 = time.perf_counter()
        self.publish_local_costmap(costmap, cloud_stamp)
        self.publish_lidar_obs(obs_min_distance, cloud_stamp)
        t1 = time.perf_counter()
        time_publish = t1 - t0

        total_time = time.perf_counter() - cycle_start

        if self.log_enabled:
            log_msg = (
                f'Update done | '
                f'raw={len(points)} | '
                f'height={len(points_filtered)} | '
                f'downsample={len(points_downsampled)} | '
                f'roi={len(points_roi)} | '
                f'costmap={costmap.shape[1]}x{costmap.shape[0]} | '
                f'cloud_age={cloud_age:.3f}s | '
                f't_height={time_height:.6f}s | '
                f't_downsample={time_downsample:.6f}s | '
                f't_roi={time_roi:.6f}s | '
                f't_costmap={time_costmap:.6f}s | '
                f't_obs={time_obs:.6f}s | '
                f't_publish={time_publish:.6f}s | '
                f't_total={total_time:.6f}s'
            )
            self.logger.info(log_msg)


def main(args=None):
    rclpy.init(args=args)

    node = LidarCostmapNode()
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