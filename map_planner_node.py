#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_planner_node.py

将 map_node.py、planner_node.py、shared_map_storage.py 合并为一个节点。

功能：
1. 接收 anav_v3 下发的 GPS 航点
2. 将航点 GPS 转换为 map 坐标
3. 使用“机器人当前 map 坐标 + 航点 map 坐标”生成道路地图
4. 地图保存在节点内部，并发布完整地图给 rviz
5. 订阅 map_pose，维护未到达航点指针
6. 订阅 local_costmap，更新全局地图，并发布增量更新
7. 以机器人当前位置为起点、当前未到达航点为终点，在全局地图上进行 A* 规划
8. 对规划路径进行稀疏化
9. 将稀疏后的路径转换到 base_link 坐标系后发布，供 controller_node 跟随

全局地图内部存储约定：
- 使用 OccupancyGrid 一致的栅格方向：
  row(gy)=0 -> 地图底边（最小 y）
  row(gy) 增大 -> y 增大
  col(gx)=0 -> 地图左边（最小 x）
  col(gx) 增大 -> x 增大

坐标系约定：
- map 坐标系：x 正方向 = East，y 正方向 = North
- base_link 坐标系：x 前，y 左
"""

import json
import math
import heapq
import threading
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile

import tf2_ros
from nav_msgs.msg import OccupancyGrid, Path
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from std_msgs.msg import String

# UTM 库
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    logging.warning("pyproj not installed, GPS->UTM conversion will not work")

from config_loader import get_config
from frequency_stats import FrequencyStats


@dataclass
class MapMetadata:
    """地图元数据"""
    resolution: float
    width: int
    height: int
    origin_x: float   # 地图左下角 x（米）
    origin_y: float   # 地图左下角 y（米）
    robot_x: float    # 生成地图时机器人的 map x（米）
    robot_y: float    # 生成地图时机器人的 map y（米）
    gps_points: List[dict]

    origin_lat: float = 0.0
    origin_lon: float = 0.0
    meters_per_degree_lat: float = 111320.0
    meters_per_degree_lon: float = 111320.0


class MapPlannerNode(Node):
    """地图与规划合并节点"""

    def __init__(self, log_dir: str = None, timestamp: str = None):
        super().__init__('map_planner_node')

        self.log_dir = log_dir
        self.timestamp = timestamp if timestamp is not None else datetime.now().strftime('%Y%m%d_%H%M%S')

        config = get_config()
        common_config = config.get('common', {})
        node_config = config.get('map_planner_node', {})

        subscriptions = node_config.get('subscriptions', {})
        publications = node_config.get('publications', {})

        # 公共参数
        self.resolution = float(common_config.get('resolution', 0.05))

        # 地图生成参数
        self.square_size = float(node_config.get('square_size', 20.0))
        self.road_width = float(node_config.get('road_width', 1.0))

        # 规划参数
        self.map_publish_frequency = float(node_config.get('map_publish_frequency', 10.0))
        self.map_pose_timeout = float(node_config.get('map_pose_timeout', 1.0))
        self.local_costmap_timeout = float(node_config.get('local_costmap_timeout', 0.3))
        self.pose_timeout = float(node_config.get('pose_timeout', 2.0))
        self.max_distance_between = float(node_config.get('max_distance_between', 0.5))
        self.allow_diagonal = bool(node_config.get('allow_diagonal', False))
        self.arrival_threshold = float(node_config.get('arrival_threshold', 1.0))
        self.obstacle_threshold = int(node_config.get('obstacle_threshold', 50))

        # 膨胀参数（用于膨胀地图和规划，单位：米）
        self.inflation_margin = float(node_config.get('inflation_margin', 0.3))
        self.utm_lookup_timeout = float(node_config.get('utm_lookup_timeout', 1.0))

        # 话题
        self.gps_path_topic = subscriptions.get('gps_path_topic', '/navigation_control')
        self.map_pose_topic = subscriptions.get('map_pose_topic', '/navigation/map_pose')
        self.local_costmap_topic = subscriptions.get('local_costmap_topic', '/navigation/local_costmap')

        self.map_topic = publications.get('map_topic', '/map')
        self.map_update_topic = publications.get('map_update_topic', '/navigation/map_update')
        self.inflated_map_topic = publications.get('inflated_map_topic', '/navigation/inflated_map')
        self.inflated_map_update_topic = publications.get('inflated_map_update_topic', '/navigation/inflated_map_update')
        self.nav_map_points_topic = publications.get('nav_map_points_topic', '/navigation/nav_map_points')
        self.path_topic = publications.get('path_topic', '/planned_path')
        self.path_map_topic = publications.get('path_map_topic', '/planned_path_map')

        # 锁
        self.pose_lock = threading.Lock()
        self.path_lock = threading.Lock()
        self.tf_lock = threading.Lock()

        # 内部地图存储（代替 shared_map_storage）
        self.map_data: Optional[np.ndarray] = None          # shape=(height, width), row=bottom->top
        self.map_metadata: Optional[MapMetadata] = None
        self.last_update_bbox: Optional[Tuple[int, int, int, int]] = None  # (min_col, min_row, max_col, max_row)
        self.first_map_published = False

        # 膨胀地图存储
        self.inflated_map_data: Optional[np.ndarray] = None  # 膨胀后的地图，shape=(height, width)
        self.first_inflated_map_published = False

        # 局部地图坐标缓存（用于向量化更新）
        self._local_grid_cache: dict = {}

        # 位姿状态
        self.latest_map_pose = None
        self.last_map_pose_time = 0.0

        # local costmap 时间戳
        self.last_local_costmap_time = 0.0
        self.local_costmap_timestamp = None

        # 路径任务状态
        self.nav_gps_points: List[dict] = []
        self.nav_map_points: List[dict] = []
        self.batch_id = ''
        self.batch_number = 0
        self.unreached_index = 0
        self.task_completed = False

        # UTM 变换
        self.utm_zone: Optional[int] = None
        self.utm_transformer = None

        # TF 监听
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 发布者 - 使用 transient local 确保晚加入的订阅者(如 rviz)能获取最新数据
        qos_transient_local = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, qos_transient_local)
        self.map_update_pub = self.create_publisher(OccupancyGridUpdate, self.map_update_topic, 1)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, self.inflated_map_topic, qos_transient_local)
        self.inflated_map_update_pub = self.create_publisher(OccupancyGridUpdate, self.inflated_map_update_topic, 1)
        self.nav_map_points_pub = self.create_publisher(Path, self.nav_map_points_topic, qos_transient_local)
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.path_map_pub = self.create_publisher(Path, self.path_map_topic, 1)

        # 订阅者
        self.gps_path_sub = self.create_subscription(
            String,
            self.gps_path_topic,
            self.gps_path_callback,
            1
        )

        self.map_pose_sub = self.create_subscription(
            PoseStamped,
            self.map_pose_topic,
            self.map_pose_callback,
            1
        )

        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.local_costmap_topic,
            self.local_costmap_callback,
            1
        )

        # 日志
        self._init_logger(node_config.get('log_enabled', True))

        # 频率统计
        self.map_freq_stats = FrequencyStats(
            node_name='map_planner_node_map',
            target_frequency=self.map_publish_frequency,
            logger=self.logger,
            ros_logger=self.get_logger(),
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )

        # 定时器 - 仅用于频率统计，不定期发布完整地图
        # 完整地图只在生成新地图时发布一次
        self.map_timer = self.create_timer(
            1.0 / max(self.map_publish_frequency, 1e-3),
            self.map_freq_stats.tick
        )

        init_info = [
            'MapPlanner Node initialized',
            f'  订阅 GPS 路径: {self.gps_path_topic}',
            f'  订阅 map_pose: {self.map_pose_topic}',
            f'  订阅 local_costmap: {self.local_costmap_topic}',
            f'  发布地图: {self.map_topic}',
            f'  发布地图增量更新: {self.map_update_topic}',
            f'  发布膨胀地图: {self.inflated_map_topic}',
            f'  发布膨胀地图增量更新: {self.inflated_map_update_topic}',
            f'  发布导航 map 点: {self.nav_map_points_topic}',
            f'  发布规划路径: {self.path_topic}',
            f'  分辨率: {self.resolution}m',
            f'  道路宽度: {self.road_width}m',
            f'  膨胀余量: {self.inflation_margin}m',
            f'  膨胀形状: 正方形（边长={self.inflation_margin * 2:.2f}m）',
            f'  运行模式: 事件驱动 (local_costmap 触发规划)',
            f'  完整地图: 仅在新地图生成时发布一次',
        ]
        for line in init_info:
            self.logger.info(line)
            self.get_logger().info(line)

    # ==================== 日志 ====================

    def _init_logger(self, enabled: bool):
        """初始化日志系统"""
        if self.log_dir is not None:
            log_dir = self.log_dir
        else:
            ts = self.timestamp
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{ts}')
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'map_planner_node_log_{self.timestamp}.log')

        self.logger = logging.getLogger('map_planner_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        if enabled:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    # ==================== 基础状态检查 ====================

    def _has_map(self) -> bool:
        return self.map_data is not None and self.map_metadata is not None

    def _get_inflation_radius_cells(self, resolution: float) -> int:
        """计算膨胀半径（格子数），膨胀正方形半边长 = inflation_margin"""
        return max(0, int(math.ceil(self.inflation_margin / resolution)))
    
    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        radius: int,
        width: int,
        height: int
    ) -> Tuple[int, int, int, int]:
        """将 bbox 向四周扩展 radius 个格子，并裁剪到地图范围内"""
        min_col, min_row, max_col, max_row = bbox
        return (
            max(0, min_col - radius),
            max(0, min_row - radius),
            min(width - 1, max_col + radius),
            min(height - 1, max_row + radius),
        )
    # ==================== TF 与坐标转换 ====================

    def gps_to_utm(self, lat: float, lon: float) -> Tuple[Optional[float], Optional[float]]:
        """经纬度转 UTM"""
        if not HAS_PYPROJ:
            self.logger.error('pyproj not installed, cannot convert GPS to UTM')
            return (None, None)

        try:
            zone = int((lon + 180) / 6) + 1

            if self.utm_transformer is None or self.utm_zone != zone:
                proj_str = f'+proj=utm +zone={zone} +datum=WGS84'
                if lat < 0:
                    proj_str += ' +south'

                self.utm_transformer = pyproj.Transformer.from_crs(
                    "EPSG:4326",
                    proj_str,
                    always_xy=True
                )
                self.utm_zone = zone
                self.logger.info(f'UTM transformer initialized for zone {zone}')

            utm_x, utm_y = self.utm_transformer.transform(lon, lat)
            return utm_x, utm_y

        except Exception as e:
            self.logger.error(f'UTM conversion failed: {e}')
            return None, None

    def get_utm_to_map_transform(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        查询 map->utm 变换（lookup_transform('utm', 'map')）
        返回：
            trans_x, trans_y, yaw
        其中：
            p_utm = t + R(yaw) * p_map
            因此：
            p_map = R(-yaw) * (p_utm - t)
        """
        with self.tf_lock:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'utm',
                    'map',
                    rclpy.time.Time(),
                    timeout=Duration(seconds=self.utm_lookup_timeout)
                )

                trans = transform.transform.translation
                rot = transform.transform.rotation

                yaw = math.atan2(
                    2.0 * (rot.w * rot.z + rot.x * rot.y),
                    1.0 - 2.0 * (rot.y ** 2 + rot.z ** 2)
                )

                return trans.x, trans.y, yaw

            except tf2_ros.LookupException as e:
                self.logger.warning(f'TF lookup failed (utm <- map not ready): {e}')
                return None, None, None
            except tf2_ros.ConnectivityException as e:
                self.logger.warning(f'TF connectivity error: {e}')
                return None, None, None
            except tf2_ros.ExtrapolationException as e:
                self.logger.warning(f'TF extrapolation error: {e}')
                return None, None, None
            except Exception as e:
                self.logger.error(f'Failed to get utm<-map transform: {e}')
                return None, None, None

    def gps_to_map_coords(
        self,
        lat: float,
        lon: float,
        trans_x: float,
        trans_y: float,
        yaw: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        GPS -> UTM -> map

        已知：
            p_utm = t + R(yaw) * p_map
        所以：
            p_map = R(-yaw) * (p_utm - t)
        """
        utm_x, utm_y = self.gps_to_utm(lat, lon)
        if utm_x is None or utm_y is None:
            return None, None

        dx = utm_x - trans_x
        dy = utm_y - trans_y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        map_x = cos_yaw * dx + sin_yaw * dy
        map_y = -sin_yaw * dx + cos_yaw * dy

        return map_x, map_y

    def world_to_grid(
        self,
        x: float,
        y: float,
        metadata: MapMetadata
    ) -> Tuple[int, int]:
        """世界坐标(map) -> 栅格坐标"""
        gx = int(math.floor((x - metadata.origin_x) / metadata.resolution))
        gy = int(math.floor((y - metadata.origin_y) / metadata.resolution))
        return gx, gy

    def grid_to_world(
        self,
        gx: int,
        gy: int,
        metadata: MapMetadata
    ) -> Tuple[float, float]:
        """栅格坐标 -> 世界坐标(map)，返回栅格中心点"""
        x = metadata.origin_x + (gx + 0.5) * metadata.resolution
        y = metadata.origin_y + (gy + 0.5) * metadata.resolution
        return x, y

    def is_inside_grid(self, gx: int, gy: int, metadata: MapMetadata) -> bool:
        return 0 <= gx < metadata.width and 0 <= gy < metadata.height

    # ==================== 回调 ====================

    def map_pose_callback(self, msg: PoseStamped):
        """接收机器人 map 坐标系位姿"""
        try:
            yaw = math.atan2(
                2.0 * (msg.pose.orientation.w * msg.pose.orientation.z +
                       msg.pose.orientation.x * msg.pose.orientation.y),
                1.0 - 2.0 * (msg.pose.orientation.y ** 2 + msg.pose.orientation.z ** 2)
            )

            with self.pose_lock:
                self.latest_map_pose = {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'yaw': yaw,
                    'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    'valid': True
                }
                self.last_map_pose_time = self.latest_map_pose['timestamp']

        except Exception as e:
            self.logger.error(f'Failed to parse map_pose: {e}')

    def gps_path_callback(self, msg: String):
        """
        接收下发的导航 GPS 点

        格式：
        {
            "action": 1,
            "mode": 1,
            "batchId": "batch_xxx",
            "points": [
                {"latitude": lat1, "longitude": lon1},
                ...
            ]
        }
        """
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.logger.error(f'Failed to parse GPS path message: {e}')
            return

        points = data.get('points', [])
        if not points:
            self.logger.warning('Received empty GPS path message')
            return

        with self.path_lock:
            self.nav_gps_points = points
            self.batch_id = data.get('batchId', '')
            self.batch_number += 1

        self.logger.info(
            f'Received new GPS path: {len(points)} points, '
            f'batchId={self.batch_id}, batch_number={self.batch_number}'
        )

        success = self.generate_map_and_nav_points()
        if not success:
            self.logger.warning('Failed to generate map from new GPS path')

    def local_costmap_callback(self, msg: OccupancyGrid):
        if msg.header.frame_id != 'base_link':
            self.logger.warning(f'Unexpected local_costmap frame_id: {msg.header.frame_id}')
        self.last_local_costmap_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.local_costmap_timestamp = msg.header.stamp

        current_time = self.get_clock().now().nanoseconds / 1e9
        cloud_age = current_time - self.last_local_costmap_time
        if cloud_age > self.local_costmap_timeout:
            self.logger.warning(f'local_costmap timeout: {cloud_age:.2f}s > {self.local_costmap_timeout:.2f}s')
            return

        if not self._has_map():
            return

        # 检查 map_pose 超时
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_map_pose_time <= 0.0 or current_time - self.last_map_pose_time > self.map_pose_timeout:
            self.logger.warning(f'map_pose timeout or not received')
            return

        with self.pose_lock:
            pose = self.latest_map_pose

        if pose is None or not pose.get('valid', False):
            return

        try:
            width = int(msg.info.width)
            height = int(msg.info.height)
            resolution = float(msg.info.resolution)

            if width <= 0 or height <= 0:
                return

            costmap = np.array(msg.data, dtype=np.int16).reshape((height, width))

            # 如果你的上游 local_costmap 语义仍和旧版一致，保留这一句
            # costmap = np.flip(costmap, axis=0)

            origin_x = float(msg.info.origin.position.x)
            origin_y = float(msg.info.origin.position.y)

            map_updated = self.update_global_map_from_local_costmap(
                local_costmap=costmap,
                local_resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
                robot_x=pose['x'],
                robot_y=pose['y'],
                robot_yaw=pose['yaw']
            )

            if map_updated:
                update_bbox = self.last_update_bbox
                self.publish_map_update(update_bbox)
                self.publish_inflated_map_update(update_bbox)
                self.last_update_bbox = None
                self.plan_once()

        except Exception as e:
            self.logger.error(f'Failed to process local_costmap: {e}')

    # ==================== 地图生成 ====================

    def interpolate_polyline(self, points: List[Tuple[float, float]], step: float) -> List[Tuple[float, float]]:
        """按固定步长对折线做线性插值"""
        if len(points) <= 1:
            return list(points)

        step = max(step, 1e-3)
        result = [points[0]]

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)

            if dist < 1e-9:
                continue

            n = max(1, int(math.ceil(dist / step)))
            for j in range(1, n + 1):
                t = j / n
                result.append((x1 + t * dx, y1 + t * dy))

        return result

    def stamp_circle_free(
        self,
        grid: np.ndarray,
        gx: int,
        gy: int,
        radius_cells: int
    ):
        """将以 (gx, gy) 为中心的圆形区域标记为空闲(0)"""
        h, w = grid.shape
        r2 = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            yy = gy + dy
            if yy < 0 or yy >= h:
                continue
            remain = r2 - dy * dy
            if remain < 0:
                continue
            dx_max = int(math.floor(math.sqrt(remain)))
            x0 = max(0, gx - dx_max)
            x1 = min(w - 1, gx + dx_max)
            grid[yy, x0:x1 + 1] = 0

    def draw_road_on_grid(
        self,
        grid: np.ndarray,
        metadata: MapMetadata,
        path_points: List[Tuple[float, float]]
    ):
        """根据路径点在栅格图上画道路（道路内置为 0，外部保持 100）"""
        if not path_points:
            return

        dense_points = self.interpolate_polyline(
            path_points,
            step=max(self.resolution * 0.5, 0.05)
        )

        radius_cells = max(1, int(math.ceil((self.road_width * 0.5) / metadata.resolution)))

        for x, y in dense_points:
            gx, gy = self.world_to_grid(x, y, metadata)
            if self.is_inside_grid(gx, gy, metadata):
                self.stamp_circle_free(grid, gx, gy, radius_cells)

    def generate_map_and_nav_points(self) -> bool:
        """根据当前 GPS 航点生成全局地图，并计算导航 map 点"""
        # 检查 map_pose 超时
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_map_pose_time <= 0.0 or current_time - self.last_map_pose_time > self.map_pose_timeout:
            self.logger.warning('No valid map_pose yet, cannot generate map')
            return False

        trans_x, trans_y, yaw = self.get_utm_to_map_transform()
        if trans_x is None:
            self.logger.warning('utm<-map transform not available, cannot generate map')
            return False

        with self.pose_lock:
            pose = self.latest_map_pose

        if pose is None or not pose.get('valid', False):
            self.logger.warning('No valid map pose')
            return False

        robot_map_x = float(pose['x'])
        robot_map_y = float(pose['y'])

        with self.path_lock:
            gps_points = list(self.nav_gps_points)

        if not gps_points:
            self.logger.warning('No navigation GPS points')
            return False

        # GPS 航点 -> map 航点
        nav_map_points = []
        for wp in gps_points:
            try:
                lat = float(wp['latitude'])
                lon = float(wp['longitude'])
            except Exception:
                continue

            map_x, map_y = self.gps_to_map_coords(lat, lon, trans_x, trans_y, yaw)
            if map_x is None or map_y is None:
                self.logger.warning(f'Failed to convert GPS ({lat}, {lon}) to map coords')
                continue

            nav_map_points.append({'x': map_x, 'y': map_y})

        if not nav_map_points:
            self.logger.error('No valid nav_map_points after conversion')
            return False

        # 地图范围：机器人当前位置 + 所有航点
        all_points_xy = [(robot_map_x, robot_map_y)] + [(p['x'], p['y']) for p in nav_map_points]
        xs = [p[0] for p in all_points_xy]
        ys = [p[1] for p in all_points_xy]

        min_x = min(xs) - self.square_size
        max_x = max(xs) + self.square_size
        min_y = min(ys) - self.square_size
        max_y = max(ys) + self.square_size

        width = max(1, int(math.ceil((max_x - min_x) / self.resolution)))
        height = max(1, int(math.ceil((max_y - min_y) / self.resolution)))

        origin_x = min_x
        origin_y = min_y

        # 初始化为障碍物 100，道路区域改成 0
        grid = np.full((height, width), 100, dtype=np.int8)

        lat_origin = float(gps_points[0]['latitude'])
        lon_origin = float(gps_points[0]['longitude'])
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(lat_origin))

        metadata = MapMetadata(
            resolution=self.resolution,
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y,
            robot_x=robot_map_x,
            robot_y=robot_map_y,
            gps_points=gps_points.copy(),
            origin_lat=lat_origin,
            origin_lon=lon_origin,
            meters_per_degree_lat=meters_per_degree_lat,
            meters_per_degree_lon=meters_per_degree_lon,
        )

        # 使用“机器人当前位置 + 航点”连成路
        full_path = [(robot_map_x, robot_map_y)] + [(p['x'], p['y']) for p in nav_map_points]
        self.draw_road_on_grid(grid, metadata, full_path)

        # 确保机器人与各航点所在格也是空闲
        for x, y in full_path:
            gx, gy = self.world_to_grid(x, y, metadata)
            if self.is_inside_grid(gx, gy, metadata):
                grid[gy, gx] = 0

        self.map_data = grid
        self.map_metadata = metadata
        self.last_update_bbox = None
        self.first_map_published = False

        with self.path_lock:
            self.nav_map_points = nav_map_points
            self.unreached_index = 0
            self.task_completed = False

        self.logger.info(
            f'Map generated: size={width}x{height}, resolution={self.resolution:.3f}m, '
            f'nav_points={len(nav_map_points)}, robot=({robot_map_x:.2f}, {robot_map_y:.2f})'
        )

        self.publish_nav_map_points()
        self.publish_map(force_full=True)

        # 对全图进行膨胀并发布
        self.perform_full_inflation()

        return True

    # ==================== 局部地图更新全局地图 ====================

    def get_local_grid_centers(
        self, height: int, width: int, resolution: float, origin_x: float, origin_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取局部地图每个格子中心的本地坐标（带缓存）"""
        key = (height, width, resolution, origin_x, origin_y)
        cached = self._local_grid_cache.get(key)
        if cached is not None:
            return cached

        rows, cols = np.indices((height, width), dtype=np.float32)
        local_x = origin_x + (cols + 0.5) * resolution
        local_y = origin_y + (rows + 0.5) * resolution

        self._local_grid_cache[key] = (local_x, local_y)
        return local_x, local_y

    def update_global_map_from_local_costmap(
        self,
        local_costmap: np.ndarray,
        local_resolution: float,
        origin_x: float,
        origin_y: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float
    ) -> bool:
        """
        根据局部 costmap 覆盖更新全局地图（向量化实现）

        更新策略：
        - 只要 local_costmap 和全局地图相交，重叠区域内以 local_costmap 为准
        - 0~100 的已知值直接覆盖
        - -1(unknown) 不覆盖

        冲突处理（规则 A：保守型，障碍优先）：
        - 多个局部格映射到同一全局格时，取最大值
        - 保证障碍物不会被误覆盖

        OccupancyGrid 标准语义：
        - local_costmap.shape = (height, width)
        - occ_grid[row, col] 对应世界坐标 (origin_x + col*res, origin_y + row*res)
        - 坐标系为 base_link
        """
        if self.map_data is None or self.map_metadata is None:
            return False

        metadata = self.map_metadata

        if abs(local_resolution - metadata.resolution) > 1e-6:
            self.logger.warning(
                f'Local/global resolution mismatch: local={local_resolution}, global={metadata.resolution}'
            )
            return False

        height, width = local_costmap.shape

        # 只处理 known 格子
        known_mask = (local_costmap >= 0)
        if not np.any(known_mask):
            return False

        local_x, local_y = self.get_local_grid_centers(
            height, width, local_resolution, origin_x, origin_y
        )

        vals = local_costmap[known_mask].astype(np.int8)
        lx = local_x[known_mask]
        ly = local_y[known_mask]

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        # base_link -> map 坐标变换
        wx = robot_x + cos_yaw * lx - sin_yaw * ly
        wy = robot_y + sin_yaw * lx + cos_yaw * ly

        # 栅格化
        gx = np.floor((wx - metadata.origin_x) / metadata.resolution).astype(np.int32)
        gy = np.floor((wy - metadata.origin_y) / metadata.resolution).astype(np.int32)

        # 边界筛选
        inside = (
            (gx >= 0) & (gx < metadata.width) &
            (gy >= 0) & (gy < metadata.height)
        )
        if not np.any(inside):
            return False

        gx = gx[inside]
        gy = gy[inside]
        vals = vals[inside]

        # 处理 many-to-one 冲突：多个局部格映射到同一全局格
        # 规则 A：保守型，障碍优先（取最大值）
        flat_idx = gy * metadata.width + gx
        order = np.argsort(flat_idx)
        flat_idx_sorted = flat_idx[order]
        vals_sorted = vals[order]

        unique_idx, start_idx = np.unique(flat_idx_sorted, return_index=True)
        agg_vals = np.maximum.reduceat(vals_sorted, start_idx)

        gy_u = unique_idx // metadata.width
        gx_u = unique_idx % metadata.width

        self.map_data[gy_u, gx_u] = agg_vals

        min_col = int(gx_u.min())
        max_col = int(gx_u.max())
        min_row = int(gy_u.min())
        max_row = int(gy_u.max())
        self.last_update_bbox = (min_col, min_row, max_col, max_row)
        return True

    # ==================== 地图膨胀 ====================

    def inflate_square(
        self,
        map_data: np.ndarray,
        inflation_radius: int
    ) -> np.ndarray:
        """
        对地图进行正方形膨胀（优化版：只遍历障碍物点）

        Args:
            map_data: 原始地图数据，shape=(height, width)
            inflation_radius: 膨胀半径（格子数）

        Returns:
            膨胀后的地图，障碍物 >= obstacle_threshold ({self.obstacle_threshold})
        """
        if inflation_radius <= 0:
            return map_data.copy()

        inflated = map_data.copy()
        h, w = map_data.shape

        # 找到所有障碍物点（使用 obstacle_threshold 判断）
        obstacle_mask = map_data >= self.obstacle_threshold
        obstacle_coords = np.argwhere(obstacle_mask)

        if len(obstacle_coords) == 0:
            return inflated

        # 对每个障碍物点，标记其膨胀区域（带边界检查）
        half = inflation_radius

        for oy, ox in obstacle_coords:
            # 计算膨胀区域边界（带越界检查）
            x0 = max(0, ox - half)
            x1 = min(w - 1, ox + half)
            y0 = max(0, oy - half)
            y1 = min(h - 1, oy + half)

            inflated[y0:y1 + 1, x0:x1 + 1] = 100  # OccupancyGrid 标准：100 = 障碍

        return inflated

    def inflate_map_full(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        对完整地图进行膨胀

        Returns:
            (inflated_map, bounding_box) - 膨胀后的地图和膨胀区域包围盒
            包围盒：(min_col, min_row, max_col, max_row)
        """
        if self.map_data is None or self.map_metadata is None:
            return None, None

        metadata = self.map_metadata
        inflation_radius = self._get_inflation_radius_cells(metadata.resolution)

        if inflation_radius <= 0:
            return self.map_data.copy(), (0, 0, metadata.width - 1, metadata.height - 1)

        inflated = self.inflate_square(
            self.map_data,
            inflation_radius
        )

        bbox = (0, 0, metadata.width - 1, metadata.height - 1)
        return inflated, bbox

    def perform_full_inflation(self):
        """对全图进行膨胀并发布膨胀地图"""
        if not self._has_map():
            return

        inflated_map, bbox = self.inflate_map_full()
        if inflated_map is None:
            return

        self.inflated_map_data = inflated_map
        self.first_inflated_map_published = False

        self.publish_inflated_map(force_full=True)
        self.logger.info('Full map inflation completed and published')

    # ==================== 膨胀地图发布 ====================

    def build_inflated_map_msg(self) -> Optional[OccupancyGrid]:
        """构建膨胀地图消息"""
        if self.inflated_map_data is None or self.map_metadata is None:
            return None

        metadata = self.map_metadata
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'

        grid_msg.info.resolution = float(metadata.resolution)
        grid_msg.info.width = int(metadata.width)
        grid_msg.info.height = int(metadata.height)

        origin_pose = Pose()
        origin_pose.position.x = float(metadata.origin_x)
        origin_pose.position.y = float(metadata.origin_y)
        origin_pose.position.z = 0.0
        origin_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        grid_msg.info.origin = origin_pose

        grid_msg.data = self.inflated_map_data.flatten().tolist()
        return grid_msg

    def publish_inflated_map(
        self,
        force_full: bool = False,
        update_bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        """发布膨胀地图（完整或增量）"""
        if self.inflated_map_data is None or self.map_metadata is None:
            return

        if force_full or not self.first_inflated_map_published:
            grid_msg = self.build_inflated_map_msg()
            if grid_msg is None:
                return
            self.inflated_map_pub.publish(grid_msg)
            self.first_inflated_map_published = True
            self.logger.info(
                f'Published full inflated map: {grid_msg.info.width}x{grid_msg.info.height}'
            )
        elif update_bbox is not None:
            self.publish_inflated_map_update(update_bbox)

    def publish_inflated_map_update(self, update_bbox: Optional[Tuple[int, int, int, int]]):
        """
        发布膨胀地图增量更新

        正确做法：
        - 原始地图变化区域 = A
        - 膨胀地图需要更新的输出区域 = A + r
        - 为了正确计算该输出区域，需要读取的原始地图输入区域 = A + 2r

        其中 r 为膨胀半径（格子数）。
        """
        if self.map_data is None or self.map_metadata is None or update_bbox is None:
            return

        if self.inflated_map_data is None:
            # 膨胀图还没准备好时，退化为全图重建
            self.perform_full_inflation()
            return

        metadata = self.map_metadata
        r = self._get_inflation_radius_cells(metadata.resolution)

        # r = 0 时，膨胀图等于原图，直接复用原 update_bbox
        if r <= 0:
            min_col, min_row, max_col, max_row = update_bbox
            sub = self.map_data[min_row:max_row + 1, min_col:max_col + 1].copy()
            self.inflated_map_data[min_row:max_row + 1, min_col:max_col + 1] = sub

            update_msg = OccupancyGridUpdate()
            update_msg.header.stamp = self.get_clock().now().to_msg()
            update_msg.header.frame_id = 'map'
            update_msg.x = int(min_col)
            update_msg.y = int(min_row)
            update_msg.width = int(max_col - min_col + 1)
            update_msg.height = int(max_row - min_row + 1)
            update_msg.data = sub.flatten().astype(int).tolist()
            self.inflated_map_update_pub.publish(update_msg)
            return

        # A: 原始地图变化区域
        a_bbox = update_bbox

        # 输出更新区域：A + r
        out_bbox = self._expand_bbox(a_bbox, r, metadata.width, metadata.height)
        out_min_col, out_min_row, out_max_col, out_max_row = out_bbox

        # 输入依赖区域：A + 2r = (A + r) + r
        in_bbox = self._expand_bbox(out_bbox, r, metadata.width, metadata.height)
        in_min_col, in_min_row, in_max_col, in_max_row = in_bbox

        in_sub_map = self.map_data[in_min_row:in_max_row + 1, in_min_col:in_max_col + 1]

        # 在输入区域上做一次局部膨胀
        inflated_in_sub = self.inflate_square(
            in_sub_map,
            r
        )

        # 从 inflated_in_sub 中裁出我们真正要更新的输出区域 A+r
        crop_x0 = out_min_col - in_min_col
        crop_y0 = out_min_row - in_min_row
        crop_x1 = crop_x0 + (out_max_col - out_min_col + 1)
        crop_y1 = crop_y0 + (out_max_row - out_min_row + 1)

        inflated_out_sub = inflated_in_sub[crop_y0:crop_y1, crop_x0:crop_x1]

        # 写回膨胀地图
        self.inflated_map_data[out_min_row:out_max_row + 1,
                            out_min_col:out_max_col + 1] = inflated_out_sub

        # 发布增量更新（只发布输出区域 A+r）
        update_msg = OccupancyGridUpdate()
        update_msg.header.stamp = self.get_clock().now().to_msg()
        update_msg.header.frame_id = 'map'
        update_msg.x = int(out_min_col)
        update_msg.y = int(out_min_row)
        update_msg.width = int(out_max_col - out_min_col + 1)
        update_msg.height = int(out_max_row - out_min_row + 1)
        update_msg.data = inflated_out_sub.flatten().astype(int).tolist()

        self.inflated_map_update_pub.publish(update_msg)

        self.logger.debug(
            f'Published inflated map update | '
            f'A={a_bbox} | OUT(A+r)={out_bbox} | IN(A+2r)={in_bbox} | '
            f'size={update_msg.width}x{update_msg.height}'
        )

    def build_full_map_msg(self) -> Optional[OccupancyGrid]:
        if self.map_data is None or self.map_metadata is None:
            return None

        metadata = self.map_metadata
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'

        grid_msg.info.resolution = float(metadata.resolution)
        grid_msg.info.width = int(metadata.width)
        grid_msg.info.height = int(metadata.height)

        origin_pose = Pose()
        origin_pose.position.x = float(metadata.origin_x)
        origin_pose.position.y = float(metadata.origin_y)
        origin_pose.position.z = 0.0
        origin_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        grid_msg.info.origin = origin_pose

        # 内部存储方向已经与 OccupancyGrid 一致
        grid_msg.data = self.map_data.flatten().tolist()
        return grid_msg

    def publish_map(
        self,
        force_full: bool = False,
        update_bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        """发布完整地图或增量更新"""
        if not self._has_map():
            return

        if force_full or not self.first_map_published:
            grid_msg = self.build_full_map_msg()
            if grid_msg is None:
                return
            self.map_pub.publish(grid_msg)
            self.first_map_published = True
            self.logger.info(
                f'Published full map: {grid_msg.info.width}x{grid_msg.info.height}'
            )
        elif update_bbox is not None:
            self.publish_map_update(update_bbox)

    def publish_map_update(self, update_bbox: Optional[Tuple[int, int, int, int]]):
        """发布增量更新"""
        if self.map_data is None or self.map_metadata is None or update_bbox is None:
            return

        metadata = self.map_metadata
        min_col, min_row, max_col, max_row = update_bbox

        width = max_col - min_col + 1
        height = max_row - min_row + 1

        update_msg = OccupancyGridUpdate()
        update_msg.header.stamp = self.get_clock().now().to_msg()
        update_msg.header.frame_id = 'map'

        update_msg.x = int(min_col)
        update_msg.y = int(min_row)
        update_msg.width = int(width)
        update_msg.height = int(height)

        submap = self.map_data[min_row:max_row + 1, min_col:max_col + 1]
        update_msg.data = submap.flatten().astype(int).tolist()

        self.map_update_pub.publish(update_msg)

    def publish_nav_map_points(self):
        """发布导航点的 map 坐标，供 rviz 可视化"""
        with self.path_lock:
            nav_points = list(self.nav_map_points)

        if not nav_points:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for p in nav_points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(p['x'])
            pose.pose.position.y = float(p['y'])
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.nav_map_points_pub.publish(msg)
        self.logger.info(f'Published {len(nav_points)} nav_map_points')

    # ==================== 规划 ====================

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def find_nearest_free_cell(
        self,
        cell: Tuple[int, int],
        width: int,
        height: int,
        max_radius: int = 10
    ) -> Optional[Tuple[int, int]]:
        """
        若起点或终点落在障碍物上，则在膨胀地图上搜索最近可行点。

        Args:
            cell: 目标格子坐标 (gx, gy)
            width, height: 地图尺寸
            max_radius: 最大搜索半径（格子数）

        Returns:
            最近可行点的坐标，如果未找到则返回 None
        """
        if self.inflated_map_data is None:
            return None

        cx, cy = cell

        # 首先检查给定点本身是否可通过（膨胀地图上检查）
        # 可通过 = 值 < obstacle_threshold
        if 0 <= cx < width and 0 <= cy < height:
            if self.inflated_map_data[cy, cx] < self.obstacle_threshold:
                return cell

        best = None
        best_dist = float('inf')

        for r in range(1, max_radius + 1):
            x0 = max(0, cx - r)
            x1 = min(width - 1, cx + r)
            y0 = max(0, cy - r)
            y1 = min(height - 1, cy + r)

            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    # 可通过 = 值 < obstacle_threshold
                    if self.inflated_map_data[yy, xx] < self.obstacle_threshold:
                        continue
                    dist = math.hypot(xx - cx, yy - cy)
                    if dist < best_dist:
                        best_dist = dist
                        best = (xx, yy)

            if best is not None:
                return best

        return None

    def astar_planning(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        width: int,
        height: int,
        allow_diagonal: bool
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* 路径规划（在膨胀地图上规划，移除矩形碰撞检测）

        因为地图已经膨胀成点机器人可走图，A* 只需检查一个格子即可。
        """
        if self.inflated_map_data is None:
            self.logger.warning('Inflated map not available for planning')
            return None

        if allow_diagonal:
            moves = [
                (0, 1), (1, 0), (0, -1), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]
            move_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
        else:
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            move_costs = [1.0, 1.0, 1.0, 1.0]

        # 使用膨胀地图检查起点（障碍 = 值 >= obstacle_threshold）
        if self.inflated_map_data[start[1], start[0]] >= self.obstacle_threshold:
            self.logger.warning('Start position collides with obstacle (inflated map)')
            return None

        # 使用膨胀地图检查终点（障碍 = 值 >= obstacle_threshold）
        if self.inflated_map_data[goal[1], goal[0]] >= self.obstacle_threshold:
            self.logger.warning('Goal position collides with obstacle (inflated map)')
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            closed_set.add(current)

            for move, move_cost in zip(moves, move_costs):
                nx = current[0] + move[0]
                ny = current[1] + move[1]

                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue

                # 直接检查膨胀地图上该点是否为障碍（障碍 = 值 >= obstacle_threshold）
                if self.inflated_map_data[ny, nx] >= self.obstacle_threshold:
                    continue

                # 对角线移动时，检查两个相邻格子是否为空闲（防止穿过角落）
                if allow_diagonal and move[0] != 0 and move[1] != 0:
                    if self.inflated_map_data[current[1] + move[1], current[0]] >= self.obstacle_threshold or \
                       self.inflated_map_data[current[1], current[0] + move[0]] >= self.obstacle_threshold:
                        continue

                neighbor = (nx, ny)
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None

    def sparsify_path(self, path: List[Tuple[int, int]], max_cells: int) -> List[Tuple[int, int]]:
        """
        与原 planner_node 一致的稀疏化方式：
        - path 不包含起点
        - 每隔 max_cells 个格子取一个点
        - 终点必须保留
        """
        if len(path) <= 1:
            return path

        max_cells = max(1, int(max_cells))
        sparse = []

        for i in range(0, len(path), max_cells):
            sparse.append(path[i])

        if sparse[-1] != path[-1]:
            sparse.append(path[-1])

        return sparse

    def publish_sparse_path_in_base_link(self, waypoints_map: List[dict]):
        """将 map 坐标系路径转换到 base_link 坐标系后发布"""
        with self.pose_lock:
            pose = self.latest_map_pose

        if pose is None or not pose.get('valid', False):
            self.logger.warning('No valid robot pose, cannot publish planned path')
            return

        robot_x = float(pose['x'])
        robot_y = float(pose['y'])
        robot_yaw = float(pose['yaw'])

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        for wp in waypoints_map:
            dx = float(wp['x']) - robot_x
            dy = float(wp['y']) - robot_y

            # map -> base_link
            base_x = cos_yaw * dx + sin_yaw * dy
            base_y = -sin_yaw * dx + cos_yaw * dy

            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.pose.position.x = base_x
            pose_msg.pose.position.y = base_y
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            msg.poses.append(pose_msg)

        self.path_pub.publish(msg)
        self.logger.info(f'Published sparse path: {len(msg.poses)} waypoints in base_link frame')

    def publish_sparse_path_in_map(self, waypoints_map: List[dict]):
        """发布 map 坐标系下的稀疏路径，供 rviz 可视化使用"""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for wp in waypoints_map:
            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.pose.position.x = float(wp['x'])
            pose_msg.pose.position.y = float(wp['y'])
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            msg.poses.append(pose_msg)

        self.path_map_pub.publish(msg)
        self.logger.info(f'Published sparse path: {len(msg.poses)} waypoints in map frame')

    def publish_empty_path(self):
        """任务完成时发布空路径"""
        stamp = self.get_clock().now().to_msg()

        msg_base = Path()
        msg_base.header.stamp = stamp
        msg_base.header.frame_id = 'base_link'
        self.path_pub.publish(msg_base)

        msg_map = Path()
        msg_map.header.stamp = stamp
        msg_map.header.frame_id = 'map'
        self.path_map_pub.publish(msg_map)

    def plan_once(self):
        """执行一次规划"""
        if not self._has_map():
            return

        current_time = self.get_clock().now().nanoseconds / 1e9

        with self.pose_lock:
            pose = self.latest_map_pose

        if pose is None or not pose.get('valid', False):
            return

        pose_timestamp = pose.get('timestamp', 0.0)
        if current_time - pose_timestamp > self.pose_timeout:
            self.logger.warning('Robot pose timestamp expired')
            return

        robot_x = float(pose['x'])
        robot_y = float(pose['y'])

        with self.path_lock:
            if self.task_completed:
                return

            if not self.nav_map_points:
                return

            # 维护未到达航点指针
            while self.unreached_index < len(self.nav_map_points):
                goal_point = self.nav_map_points[self.unreached_index]
                dist = math.hypot(goal_point['x'] - robot_x, goal_point['y'] - robot_y)
                if dist <= self.arrival_threshold:
                    self.logger.info(
                        f'Waypoint {self.unreached_index} reached '
                        f'(dist={dist:.3f}m <= {self.arrival_threshold:.3f}m)'
                    )
                    self.unreached_index += 1
                else:
                    break

            if self.unreached_index >= len(self.nav_map_points):
                self.task_completed = True
                self.publish_empty_path()
                self.logger.info('All waypoints reached, task completed')
                return

            goal_point = self.nav_map_points[self.unreached_index]
            goal_map_x = float(goal_point['x'])
            goal_map_y = float(goal_point['y'])

        map_data = self.map_data
        metadata = self.map_metadata
        inflated_map = self.inflated_map_data

        if map_data is None or metadata is None:
            return

        if inflated_map is None:
            self.logger.warning('Inflated map not available for planning')
            return

        start_gx, start_gy = self.world_to_grid(robot_x, robot_y, metadata)
        goal_gx, goal_gy = self.world_to_grid(goal_map_x, goal_map_y, metadata)

        start_gx = max(0, min(metadata.width - 1, start_gx))
        start_gy = max(0, min(metadata.height - 1, start_gy))
        goal_gx = max(0, min(metadata.width - 1, goal_gx))
        goal_gy = max(0, min(metadata.height - 1, goal_gy))

        max_snap_radius = max(1, int(math.ceil(self.arrival_threshold / metadata.resolution)))

        start_cell = self.find_nearest_free_cell(
            (start_gx, start_gy), metadata.width, metadata.height, max_snap_radius
        )
        goal_cell = self.find_nearest_free_cell(
            (goal_gx, goal_gy), metadata.width, metadata.height, max_snap_radius
        )

        if start_cell is None:
            self.logger.warning('No free start cell found')
            return
        if goal_cell is None:
            self.logger.warning('No free goal cell found')
            return

        path = self.astar_planning(
            start=start_cell,
            goal=goal_cell,
            width=metadata.width,
            height=metadata.height,
            allow_diagonal=self.allow_diagonal
        )

        if not path or len(path) <= 1:
            self.logger.warning('No path found')
            return

        max_cells = int(self.max_distance_between / metadata.resolution) if metadata.resolution > 0 else 1
        sparse_grid_path = self.sparsify_path(path[1:], max_cells)

        waypoints_map = []
        for gx, gy in sparse_grid_path:
            wx, wy = self.grid_to_world(gx, gy, metadata)
            waypoints_map.append({'x': wx, 'y': wy})

        self.publish_sparse_path_in_base_link(waypoints_map)
        self.publish_sparse_path_in_map(waypoints_map)

    # ==================== 定时器 ====================

    # 注意：完整地图只在 generate_map_and_nav_points() 中生成新地图时发布一次
    # 增量地图更新在 local_costmap_callback 中每次收到局部 costmap 时发布


def main(args=None):
    rclpy.init(args=args)

    node = MapPlannerNode()

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