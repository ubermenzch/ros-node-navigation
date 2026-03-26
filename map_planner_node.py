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
from collections import deque
from typing import Optional, Tuple, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.duration import Duration
from builtin_interfaces.msg import Time
from utils.time_utils import TimeUtils

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

    def __init__(self, log_dir: str = None, log_timestamp: str = None):
        super().__init__('map_planner_node')

        self.log_dir = log_dir
        self.log_timestamp = log_timestamp if log_timestamp is not None else datetime.now().strftime('%Y%m%d_%H%M%S')
        config = get_config()
        common_config = config.get('common', {})
        node_config = config.get('map_planner_node', {})

        # 日志初始化（尽早初始化以便记录后续信息）
        self._init_logger(node_config.get('log_enabled', True))

        subscriptions = node_config.get('subscriptions', {})
        publications = node_config.get('publications', {})

        # 公共参数
        self.resolution = float(common_config.get('resolution', 0.05))

        # 地图生成参数
        self.square_size = float(node_config.get('square_size', 20.0))
        self.road_width = float(node_config.get('road_width', 1.0))
        self.road_sample_step = float(node_config.get('road_sample_step', 0.025))

        # 规划参数
        self.publish_full_map = bool(node_config.get('publish_full_map', True))
        self.map_pose_timeout = float(node_config.get('map_pose_timeout', 1.0))
        self.local_costmap_timeout = float(node_config.get('local_costmap_timeout', 0.3))
        self.max_distance_between = float(node_config.get('max_distance_between', 0.5))
        self.allow_diagonal = bool(node_config.get('allow_diagonal', False))
        self.arrival_threshold = float(node_config.get('arrival_threshold', 1.0))
        arrival_check_frequency = float(node_config.get('arrival_check_frequency', 2.0))
        self.arrival_check_interval = 1.0 / arrival_check_frequency if arrival_check_frequency > 0 else 0.5
        self.obstacle_threshold = int(node_config.get('obstacle_threshold', 50))

        # 膨胀参数（用于膨胀地图和规划，单位：米）
        self.inflation_margin = float(node_config.get('inflation_margin', 0.3))
        self.inflation_enabled = bool(node_config.get('inflation_enabled', True))
        self.inflation_radius_cells = self._get_inflation_radius_cells(self.resolution)
        if self.inflation_radius_cells <= 0:
            self.inflation_enabled = False
            self.logger.info('Inflation disabled: inflation_radius_cells <= 0')
        self.utm_lookup_timeout = 1.0

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

        # 内部地图存储（代替 shared_map_storage）
        self.map_data: Optional[np.ndarray] = None          # shape=(height, width), row=bottom->top
        self.map_metadata: Optional[MapMetadata] = None

        # 膨胀地图存储
        self.inflated_map_data: Optional[np.ndarray] = None  # 膨胀后的地图，shape=(height, width)

        # 局部地图坐标缓存（用于向量化更新）
        self._local_grid_cache: dict = {}

        # 位姿状态 - 使用队列缓存 map_pose，按时间戳匹配 local_costmap
        self.map_pose_queue: deque = deque(maxlen=100)  # 最多缓存100条 pose

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

        # 发布者 - 普通发布者（不重新发送最后一帧）
        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, 1)
        self.map_update_pub = self.create_publisher(OccupancyGridUpdate, self.map_update_topic, 1)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, self.inflated_map_topic, 1)
        self.inflated_map_update_pub = self.create_publisher(OccupancyGridUpdate, self.inflated_map_update_topic, 1)
        self.nav_map_points_pub = self.create_publisher(Path, self.nav_map_points_topic, 1)
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.path_map_pub = self.create_publisher(Path, self.path_map_topic, 1)

        # 调试地图发布者
        self.debug_map_pub = self.create_publisher(OccupancyGrid, '/navigation/debug_map', 1)

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

        # 频率统计（统计 local_costmap_callback 执行频率）
        self.map_freq_stats = FrequencyStats(
            node_name='map_planner_node_map',
            logger=self.logger,
            ros_logger=self.get_logger(),
            window_size=10,
            log_interval=5.0
        )

        # 初始化航点到达检查定时器
        self._init_arrival_check_timer()

        init_info = [
            'MapPlanner Node initialized',
            f'  订阅 GPS 路径: {self.gps_path_topic}',
            f'  订阅 map_pose: {self.map_pose_topic}',
            f'  订阅 local_costmap: {self.local_costmap_topic}',
            f'  发布地图: {self.map_topic}',
            f'  发布地图增量更新: {self.map_update_topic}' if not self.publish_full_map else None,
            f'  发布膨胀地图: {self.inflated_map_topic}',
            f'  发布膨胀地图增量更新: {self.inflated_map_update_topic}' if not self.publish_full_map else None,
            f'  发布导航 map 点: {self.nav_map_points_topic}',
            f'  发布规划路径: {self.path_topic}',
            f'  分辨率: {self.resolution}m',
            f'  道路宽度: {self.road_width}m',
            f'  膨胀余量: {self.inflation_margin}m',
            f'  膨胀形状: 正方形（边长={self.inflation_margin * 2:.2f}m）',
            f'  膨胀地图功能: {"开启" if self.inflation_enabled else "关闭"}',
            f'  航点到达阈值: {self.arrival_threshold}m',
            f'  航点检查间隔: {self.arrival_check_interval}s',
            f'  运行模式: 事件驱动 (local_costmap 触发规划)',
            f'  地图发布方式: {"完整 OccupancyGrid" if self.publish_full_map else "OccupancyGridUpdate 增量"}',
        ]
        for line in init_info:
            if line is not None:
                self.logger.info(line)
                self.get_logger().info(line)

    # ==================== 日志 ====================

    def _init_logger(self, enabled: bool):
        """初始化日志系统"""
        if self.log_dir is not None:
            log_dir = self.log_dir
        else:
            ts = self.log_timestamp
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{ts}')
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'map_planner_node_log_{self.log_timestamp}.log')

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
        if self.map_data is None:
            self.logger.warning('Map not ready: map_data is None')
            return False
        if self.map_metadata is None:
            self.logger.warning('Map not ready: map_metadata is None')
            return False
        # 仅在膨胀功能开启时检查膨胀地图
        if self.inflation_enabled and self.inflated_map_data is None:
            self.logger.warning('Map not ready: inflated_map_data is None')
            return False
        return True

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
        try:
            transform = self.tf_buffer.lookup_transform(
                'utm',
                'map',
                TimeUtils.nanos_to_stamp(TimeUtils.now_nanos()),
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
        """接收机器人 map 坐标系位姿，存入队列"""
        try:
            yaw = math.atan2(
                2.0 * (msg.pose.orientation.w * msg.pose.orientation.z +
                       msg.pose.orientation.x * msg.pose.orientation.y),
                1.0 - 2.0 * (msg.pose.orientation.y ** 2 + msg.pose.orientation.z ** 2)
            )

            pose_entry = {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'yaw': yaw,
                'timestamp': TimeUtils.stamp_to_nanos(msg.header.stamp),
                'valid': True
            }

            # 添加到队列
            self.map_pose_queue.append(pose_entry)

            # 清理过期数据（超出最大缓存时间）
            self._clean_map_pose_queue()

        except Exception as e:
            self.logger.error(f'Failed to parse map_pose: {e}')

    def _clean_map_pose_queue(self):
        """清理超出缓存时间的 pose 数据"""
        if not self.map_pose_queue:
            return

        current_nanos = TimeUtils.now_nanos()
        timeout_nanos = self.map_pose_timeout * 1e9
        cutoff_time = current_nanos - timeout_nanos

        # 从队首移除所有超时的 pose
        while self.map_pose_queue and self.map_pose_queue[0]['timestamp'] < cutoff_time:
            self.map_pose_queue.popleft()

    def get_closest_map_pose(self, target_timestamp: int) -> Optional[dict]:
        """
        从队列中获取时间戳最接近 target_timestamp 的 map_pose

        Args:
            target_timestamp: 目标时间戳（纳秒）

        Returns:
            最接近的 pose 数据，如果没有有效数据则返回 None
        """
        if not self.map_pose_queue:
            return None

        # 查找时间差最小的 pose
        best_pose = None
        best_diff = float('inf')

        for pose in self.map_pose_queue:
            diff = abs(pose['timestamp'] - target_timestamp)
            if diff < best_diff:
                best_diff = diff
                best_pose = pose

        return best_pose

    @property
    def latest_map_pose(self) -> Optional[dict]:
        """获取队列中最新的 map_pose（兼容属性，供不涉及时间戳匹配的代码使用）"""
        if self.map_pose_queue:
            return self.map_pose_queue[-1]
        return None

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
        callback_start_nanos = TimeUtils.now_nanos()
        
        # 统计回调执行频率（节点由本回调驱动）
        freq_stats_start = TimeUtils.now_nanos()
        self.map_freq_stats.tick()
        freq_stats_elapsed = (TimeUtils.now_nanos() - freq_stats_start) / 1e6

        if not self._has_map():
            self.logger.warning('local_costmap_callback: map not ready, skipped')
            return
        
        map_check_end = TimeUtils.now_nanos()
        map_check_elapsed = (map_check_end - callback_start_nanos) / 1e6
        
        if msg.header.frame_id != 'base_link':
            self.logger.warning(f'Unexpected local_costmap frame_id: {msg.header.frame_id}')
        
        frame_check_elapsed = (TimeUtils.now_nanos() - map_check_end) / 1e6
        
        current_nanos = TimeUtils.now_nanos()
        cloud_age_sec = (current_nanos - TimeUtils.stamp_to_nanos(msg.header.stamp)) / 1e9
        if cloud_age_sec > self.local_costmap_timeout:
            self.logger.warning(f'local_costmap timeout: {cloud_age_sec:.2f}s > {self.local_costmap_timeout:.2f}s')
            return

        # 从队列中获取与 local_costmap 时间戳最接近的 map_pose
        costmap_timestamp = TimeUtils.stamp_to_nanos(msg.header.stamp)
        pose = self.get_closest_map_pose(costmap_timestamp)

        pose_lookup_elapsed = (TimeUtils.now_nanos() - current_nanos) / 1e6

        if pose is None or not pose.get('valid', False):
            self.logger.warning(f'map_pose queue empty or not received')
            return

        # 检查 pose 时间戳与 costmap 的差异
        pose_time_diff = abs(pose['timestamp'] - costmap_timestamp) / 1e9
        if pose_time_diff > self.map_pose_timeout:
            self.logger.warning(f'map_pose time diff too large: {pose_time_diff:.3f}s > {self.map_pose_timeout:.3f}s')
            return

        try:
            parse_start = TimeUtils.now_nanos()
            width = int(msg.info.width)
            height = int(msg.info.height)
            resolution = float(msg.info.resolution)

            if width <= 0 or height <= 0:
                return

            costmap = np.array(msg.data, dtype=np.int16).reshape((height, width))

            origin_x = float(msg.info.origin.position.x)
            origin_y = float(msg.info.origin.position.y)
            parse_elapsed = (TimeUtils.now_nanos() - parse_start) / 1e6

            # 更新全局地图
            update_map_start = TimeUtils.now_nanos()
            map_updated, update_box = self.update_global_map_from_local_costmap(
                local_costmap=costmap,
                local_resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
                robot_x=pose['x'],
                robot_y=pose['y'],
                robot_yaw=pose['yaw']
            )
            #print(f'update_box: {update_box}')
            update_map_elapsed = (TimeUtils.now_nanos() - update_map_start) / 1e6

            if map_updated:
                # 发布地图更新
                pub_map_start = TimeUtils.now_nanos()
                if self.publish_full_map:
                    # 发布完整地图
                    full_grid = self.build_full_map_msg()
                    if full_grid is not None:
                        self.map_pub.publish(full_grid)
                else:
                    # 发布增量更新
                    self.publish_map_update(update_box, msg.header.stamp)
                pub_map_elapsed = (TimeUtils.now_nanos() - pub_map_start) / 1e6

                # 仅在膨胀功能开启时更新膨胀地图
                if self.inflation_enabled:
                    # 更新膨胀地图
                    inflate_start = TimeUtils.now_nanos()
                    inflated_update_box = self.update_inflated_map_from_bbox(update_box)
                    inflate_elapsed = (TimeUtils.now_nanos() - inflate_start) / 1e6
                    
                    # 发布膨胀地图更新
                    pub_inflate_start = TimeUtils.now_nanos()
                    if self.publish_full_map:
                        # 发布完整膨胀地图
                        full_inflated_grid = self.build_inflated_map_msg()
                        if full_inflated_grid is not None:
                            self.inflated_map_pub.publish(full_inflated_grid)
                    else:
                        # 发布增量更新
                        self.publish_inflated_map_bbox_update(inflated_update_box, msg.header.stamp)
                    pub_inflate_elapsed = (TimeUtils.now_nanos() - pub_inflate_start) / 1e6
                else:
                    inflate_elapsed = 0.0
                    pub_inflate_elapsed = 0.0

                # 执行路径规划
                plan_start = TimeUtils.now_nanos()
                self.plan_once(msg.header.stamp)
                plan_elapsed = (TimeUtils.now_nanos() - plan_start) / 1e6
                
                total_elapsed = (TimeUtils.now_nanos() - callback_start_nanos) / 1e6
                
                self.logger.info(
                    f'local_costmap_callback timing: '
                    f'total={total_elapsed:.2f}ms | '
                    f'freq_stats={freq_stats_elapsed:.2f}ms | '
                    f'map_check={map_check_elapsed:.2f}ms | '
                    f'frame_check={frame_check_elapsed:.2f}ms | '
                    f'pose_lookup={pose_lookup_elapsed:.2f}ms | '
                    f'parse={parse_elapsed:.2f}ms | '
                    f'update_map={update_map_elapsed:.2f}ms | '
                    f'pub_map={pub_map_elapsed:.2f}ms | '
                    f'inflate={inflate_elapsed:.2f}ms | '
                    f'pub_inflate={pub_inflate_elapsed:.2f}ms | '
                    f'plan={plan_elapsed:.2f}ms'
                )
            else:
                self.logger.debug('local_costmap_callback: map not updated')

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
        """
        根据路径点在栅格图上画道路
        - 每隔 road_sample_step 米在路径上采样一个点
        - 以每个采样点为中心画圆形区域（半径 = road_width / 2）
        - 圆形区域内的栅格设为空闲 (0)，外部保持原值
        """
        if not path_points:
            return

        dense_points = self.interpolate_polyline(
            path_points,
            step=max(self.road_sample_step, self.resolution * 0.5)
        )

        radius_cells = max(1, int(math.ceil((self.road_width * 0.5) / metadata.resolution)))
        radius_sq = radius_cells * radius_cells

        if len(dense_points) == 0:
            return

        # 向量化：将所有点一次性转换为栅格坐标
        pts = np.array(dense_points, dtype=np.float64)
        gx_all = np.floor((pts[:, 0] - metadata.origin_x) / metadata.resolution).astype(np.int32)
        gy_all = np.floor((pts[:, 1] - metadata.origin_y) / metadata.resolution).astype(np.int32)

        h, w = grid.shape

        # 边界筛选
        valid = (gx_all >= 0) & (gx_all < w) & (gy_all >= 0) & (gy_all < h)
        gx_all = gx_all[valid]
        gy_all = gy_all[valid]

        if len(gx_all) == 0:
            return

        # 计算道路覆盖范围（矩形膨胀 + 圆形约束）
        min_gx = gx_all.min() - radius_cells
        max_gx = gx_all.max() + radius_cells
        min_gy = gy_all.min() - radius_cells
        max_gy = gy_all.max() + radius_cells

        # 裁剪到地图范围
        x0 = max(0, min_gx)
        x1 = min(w - 1, max_gx)
        y0 = max(0, min_gy)
        y1 = min(h - 1, max_gy)

        if x0 > x1 or y0 > y1:
            return

        # 构建局部坐标网格
        sub_w = x1 - x0 + 1
        sub_h = y1 - y0 + 1
        cols, rows = np.meshgrid(
            np.arange(x0, x1 + 1, dtype=np.int32),
            np.arange(y0, y1 + 1, dtype=np.int32)
        )

        # 计算每个局部格到最近道路点的距离平方（向量化最近点查找）
        road_grid_x = gx_all[:, np.newaxis, np.newaxis]  # (N, 1, 1)
        road_grid_y = gy_all[:, np.newaxis, np.newaxis]  # (N, 1, 1)
        dist_sq = (cols - road_grid_x) ** 2 + (rows - road_grid_y) ** 2  # (N, sub_h, sub_w)
        min_dist_sq = dist_sq.min(axis=0)  # (sub_h, sub_w)

        # 标记道路区域：距离在 radius_cells 以内的设为 0
        road_mask = min_dist_sq <= radius_sq
        grid[y0:y1 + 1, x0:x1 + 1][road_mask] = 0

    def generate_map_and_nav_points(self) -> bool:
        """根据当前 GPS 航点生成全局地图，并计算导航 map 点"""
        # 检查 map_pose 超时
        current_nanos = TimeUtils.now_nanos()
        pose = self.latest_map_pose
        if pose is None or not pose.get('valid', False):
            self.logger.warning('No valid map_pose yet, cannot generate map')
            return False
        pose_timestamp = pose.get('timestamp', 0)
        if current_nanos - pose_timestamp > self.map_pose_timeout * 1e9:
            self.logger.warning('map_pose timestamp expired')
            return False

        trans_x, trans_y, yaw = self.get_utm_to_map_transform()
        if trans_x is None:
            self.logger.warning('utm<-map transform not available, cannot generate map')
            return False

        robot_map_x = float(pose['x'])
        robot_map_y = float(pose['y'])

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

        self.map_data = grid
        self.map_metadata = metadata

        self.nav_map_points = nav_map_points
        self.unreached_index = 0
        self.task_completed = False

        self.logger.info(
            f'Map generated: size={width}x{height}, resolution={self.resolution:.3f}m, '
            f'nav_points={len(nav_map_points)}, robot=({robot_map_x:.2f}, {robot_map_y:.2f})'
        )

        self.publish_nav_map_points()
        #发送未膨胀的全局地图用于可视化初始化
        grid_msg = self.build_full_map_msg()
        if grid_msg is not None:
            self.map_pub.publish(grid_msg)
            self.logger.info(
                f'Published full map: {grid_msg.info.width}x{grid_msg.info.height}'
            )

        # 对全图进行膨胀并发布（仅在膨胀功能开启时）
        if self.inflation_enabled and self.map_data is not None and self.map_metadata is not None:
            self.inflated_map_data = self.inflate_square(self.map_data, self.inflation_radius_cells)
            grid_msg = self.build_inflated_map_msg()
            if grid_msg is not None:
                self.inflated_map_pub.publish(grid_msg)
                self.logger.info('Published inflated map')

        return True

    def _publish_debug_map(
        self,
        min_col: int,
        max_col: int,
        min_row: int,
        max_row: int
    ) -> None:
        """
        发布调试地图：将包围盒区域提取为独立的 OccupancyGrid 发布

        Args:
            min_col/max_col: 列索引范围（对应 x 方向）
            min_row/max_row: 行索引范围（对应 y 方向）
        """
        if not self._has_map():
            return

        metadata = self.map_metadata
        data = self.map_data

        # 提取子区域（包含边界）
        sub_h = max_row - min_row + 1
        sub_w = max_col - min_col + 1
        sub_data = data[min_row:max_row + 1, min_col:max_col + 1].copy()

        # 构建 OccupancyGrid
        debug_grid = OccupancyGrid()
        debug_grid.header.stamp = self.get_clock().now().to_msg()
        debug_grid.header.frame_id = 'map'

        debug_grid.info.resolution = metadata.resolution
        debug_grid.info.width = sub_w
        debug_grid.info.height = sub_h
        debug_grid.info.origin.position.x = metadata.origin_x + min_col * metadata.resolution
        debug_grid.info.origin.position.y = metadata.origin_y + min_row * metadata.resolution
        debug_grid.info.origin.position.z = 0.0
        debug_grid.info.origin.orientation.w = 1.0

        debug_grid.data = sub_data.flatten().tolist()

        self.debug_map_pub.publish(debug_grid)
        self.logger.debug(
            f'DEBUG: Published debug map [{sub_h}x{sub_w}] at world '
            f'({debug_grid.info.origin.position.x:.3f}, {debug_grid.info.origin.position.y:.3f})'
        )

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
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
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
        if not self._has_map():
            return (False, None)

        metadata = self.map_metadata

        if abs(local_resolution - metadata.resolution) > 1e-6:
            self.logger.warning(
                f'Local/global resolution mismatch: local={local_resolution}, global={metadata.resolution}'
            )
            return (False, None)

        height, width = local_costmap.shape

        # 只处理已知格子（0 或 100），跳过 -1（未知）
        known_mask = (local_costmap == 0) | (local_costmap == 100)
        if not np.any(known_mask):
            return (False, None)

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
            return (False, None)

        gx = gx[inside]
        gy = gy[inside]
        vals = vals[inside]

        # 处理 many-to-one 冲突，但只在“本次 local_costmap 映射结果内部”归并
        # 语义：
        # - local_costmap 中 -1 已经过滤，不参与覆盖
        # - 若某个全局格被本次多个局部格命中：
        #     * 只要有一个 100，则该格写 100
        #     * 否则写 0
        # - 然后把归并结果直接覆盖到全局地图，不与旧值取 max

        flat_idx = gy.astype(np.int64) * metadata.width + gx.astype(np.int64)

        unique_flat, inverse = np.unique(flat_idx, return_inverse=True)

        # 本次更新里，同一目标格是否命中过障碍
        has_obstacle = np.zeros(len(unique_flat), dtype=bool)
        np.logical_or.at(has_obstacle, inverse, vals == 100)

        # 归并后的写回值：有障碍则 100，否则 0
        merged_vals = np.zeros(len(unique_flat), dtype=np.int8)
        merged_vals[has_obstacle] = 100

        # 还原为二维索引
        merged_gx = (unique_flat % metadata.width).astype(np.int32)
        merged_gy = (unique_flat // metadata.width).astype(np.int32)

        # 直接覆盖写回
        self.map_data[merged_gy, merged_gx] = merged_vals


        min_col = int(gx.min())
        max_col = int(gx.max())
        min_row = int(gy.min())
        max_row = int(gy.max())
        update_box = (min_col, min_row, max_col, max_row)

        # 调试：发布包围盒区域的全分辨率地图
        self._publish_debug_map(min_col, max_col, min_row, max_row)

        return (True, update_box)

    # ==================== 地图膨胀 ====================

    def inflate_square(
        self,
        map_data: np.ndarray,
        inflation_radius: int
    ) -> np.ndarray:
        """
        对地图进行正方形膨胀（NumPy向量化优化版 - O(h*w)复杂度）

        Args:
            map_data: 原始地图数据，shape=(height, width)
            inflation_radius: 膨胀半径（格子数）

        Returns:
            膨胀后的地图，障碍物 >= obstacle_threshold ({self.obstacle_threshold})
        """
        if inflation_radius <= 0:
            return map_data.copy()

        h, w = map_data.shape
        r = inflation_radius

        # 创建障碍物二值掩码
        obstacle_mask = (map_data >= self.obstacle_threshold).astype(np.uint8)

        if np.sum(obstacle_mask) == 0:
            return map_data.copy()

        # 使用累积和算法实现高效的矩形区域膨胀
        # 原理：对每个障碍物点，用差分数组标记其膨胀区域，
        # 然后通过两次累加（水平和垂直）得到最终掩码
        diff = np.zeros((h + 2, w + 2), dtype=np.int8)

        # 获取障碍物坐标
        obstacle_coords = np.argwhere(obstacle_mask)

        if len(obstacle_coords) == 0:
            return map_data.copy()

        # 向量化：一次性计算所有障碍物的膨胀边界
        oy_all = obstacle_coords[:, 0]  # 行索引 (y方向)
        ox_all = obstacle_coords[:, 1]  # 列索引 (x方向)

        x0_all = np.maximum(0, ox_all - r)
        x1_all = np.minimum(w - 1, ox_all + r)
        y0_all = np.maximum(0, oy_all - r)
        y1_all = np.minimum(h - 1, oy_all + r)

        # 向量化更新差分数组的四个角
        diff[y0_all + 1, x0_all + 1] += 1
        diff[y0_all + 1, x1_all + 2] -= 1
        diff[y1_all + 2, x0_all + 1] -= 1
        diff[y1_all + 2, x1_all + 2] += 1

        # 水平方向累加
        cumsum_h = np.cumsum(diff, axis=1)
        # 垂直方向累加
        cumsum_v = np.cumsum(cumsum_h, axis=0)

        # 提取有效区域并转换为布尔掩码
        inflated_mask = cumsum_v[1:h + 1, 1:w + 1] > 0

        inflated = map_data.copy()
        inflated[inflated_mask] = 100

        return inflated

    # ==================== 膨胀地图发布 ====================

    def build_inflated_map_msg(self) -> Optional[OccupancyGrid]:
        """构建膨胀地图消息"""
        if not self._has_map():
            return None

        metadata = self.map_metadata
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
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

        # 使用 ravel() 而非 flatten()：避免不必要的内存拷贝
        grid_msg.data = self.inflated_map_data.ravel().tolist()
        return grid_msg

    def update_inflated_map_from_bbox(
        self,
        update_bbox: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        根据原始地图变化区域，更新 self.inflated_map_data。
        只做数据更新，不发布消息。

        返回：
            膨胀地图中真正被更新的 bbox（即 A+r），
            如果未更新则返回 None。
        """
        if not self.inflation_enabled or update_bbox is None or not self._has_map():
            return None

        metadata = self.map_metadata
        r = self.inflation_radius_cells

        # A: 原始地图变化区域
        a_bbox = update_bbox

        # 输出更新区域：A + r
        out_bbox = self._expand_bbox(a_bbox, r, metadata.width, metadata.height)
        out_min_col, out_min_row, out_max_col, out_max_row = out_bbox

        # 输入依赖区域：A + 2r
        in_bbox = self._expand_bbox(out_bbox, r, metadata.width, metadata.height)
        in_min_col, in_min_row, in_max_col, in_max_row = in_bbox

        in_sub_map = self.map_data[in_min_row:in_max_row + 1, in_min_col:in_max_col + 1]

        # 在输入区域上局部膨胀
        inflated_in_sub = self.inflate_square(in_sub_map, r)

        # 从 inflated_in_sub 中裁出真正需要写回的输出区域 A+r
        crop_x0 = out_min_col - in_min_col
        crop_y0 = out_min_row - in_min_row
        crop_x1 = crop_x0 + (out_max_col - out_min_col + 1)
        crop_y1 = crop_y0 + (out_max_row - out_min_row + 1)

        inflated_out_sub = inflated_in_sub[crop_y0:crop_y1, crop_x0:crop_x1]

        # 写回膨胀地图
        self.inflated_map_data[
            out_min_row:out_max_row + 1,
            out_min_col:out_max_col + 1
        ] = inflated_out_sub

        self.logger.debug(
            f'Updated inflated map region | '
            f'A={a_bbox} | OUT(A+r)={out_bbox} | IN(A+2r)={in_bbox} | '
            f'size={out_max_col - out_min_col + 1}x{out_max_row - out_min_row + 1}'
        )

        return out_bbox


    def publish_inflated_map_bbox_update(
        self,
        update_bbox: Optional[Tuple[int, int, int, int]],
        stamp: Optional[Time] = None
    ):
        """
        发布膨胀地图某个 bbox 的增量更新。
        只负责发布，不负责更新 self.inflated_map_data。
        """
        if update_bbox is None or not self._has_map():
            return

        min_col, min_row, max_col, max_row = update_bbox
        sub = self.inflated_map_data[min_row:max_row + 1, min_col:max_col + 1]

        update_msg = OccupancyGridUpdate()
        update_msg.header.stamp = (
            stamp if stamp is not None
            else TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
        )
        update_msg.header.frame_id = 'map'
        update_msg.x = int(min_col)
        update_msg.y = int(min_row)
        update_msg.width = int(max_col - min_col + 1)
        update_msg.height = int(max_row - min_row + 1)
        update_msg.data = sub.flatten().astype(int).tolist()

        self.inflated_map_update_pub.publish(update_msg)

        self.logger.debug(
            f'Published inflated map bbox update | '
            f'bbox={update_bbox} | size={update_msg.width}x{update_msg.height}'
        )

    def build_full_map_msg(self) -> Optional[OccupancyGrid]:
        if self.map_data is None or self.map_metadata is None:
            return None

        metadata = self.map_metadata
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
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

        # map内部存储方向已经与 OccupancyGrid 一致
        # 使用 ravel() 而非 flatten()：ravel 返回视图（连续时）或浅拷贝，开销更小
        grid_msg.data = self.map_data.ravel().tolist()
        return grid_msg

    def publish_map_update(
        self,
        update_bbox: Optional[Tuple[int, int, int, int]],
        stamp: Optional[Time] = None
    ):
        """发布增量更新"""
        if update_bbox is None or not self._has_map():
            return

        metadata = self.map_metadata
        min_col, min_row, max_col, max_row = update_bbox

        width = max_col - min_col + 1
        height = max_row - min_row + 1

        update_msg = OccupancyGridUpdate()
        update_msg.header.stamp = stamp if stamp is not None else TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
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
        nav_points = list(self.nav_map_points)

        if not nav_points:
            return

        msg = Path()
        msg.header.stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
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
        max_radius: int = 10,
        planning_map: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int]]:
        """
        若起点或终点落在障碍物上，则搜索最近可行点（BFS优化版）。

        Args:
            cell: 目标格子坐标 (gx, gy)
            width, height: 地图尺寸
            max_radius: 最大搜索半径（格子数）
            planning_map: 用于规划的地图（膨胀地图或原始地图）

        Returns:
            最近可行点的坐标，如果未找到则返回 None
        """
        if not self._has_map():
            return None

        check_map = planning_map if planning_map is not None else self.inflated_map_data
        if check_map is None:
            return None

        cx, cy = cell

        if 0 <= cx < width and 0 <= cy < height:
            if check_map[cy, cx] < self.obstacle_threshold:
                return cell

        visited = set()
        queue = deque([(cx, cy, 0)])
        visited.add((cx, cy))

        while queue:
            x, y, dist = queue.popleft()
            if dist > max_radius:
                continue

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                visited.add((nx, ny))

                if check_map[ny, nx] < self.obstacle_threshold:
                    return (nx, ny)

                queue.append((nx, ny, dist + 1))

        return None

    def astar_planning(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        width: int,
        height: int,
        allow_diagonal: bool,
        planning_map: Optional[np.ndarray] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* 路径规划（支持双向 A* 优化，适用于长距离路径）

        Args:
            planning_map: 用于规划的地图（膨胀地图或原始地图）

        因为地图已经膨胀成点机器人可走图，A* 只需检查一个格子即可。
        若使用原始地图，则需要考虑机器人尺寸带来的碰撞。
        """
        if not self._has_map():
            self.logger.warning('Map not ready for planning')
            return None

        check_map = planning_map if planning_map is not None else self.inflated_map_data
        if check_map is None:
            self.logger.warning('Planning map not available')
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

        if check_map[start[1], start[0]] >= self.obstacle_threshold:
            self.logger.warning('Start position collides with obstacle')
            return None

        if check_map[goal[1], goal[0]] >= self.obstacle_threshold:
            self.logger.warning('Goal position collides with obstacle')
            return None

        if start == goal:
            return [start]

        return self._bidirectional_astar(start, goal, width, height, moves, move_costs, check_map)

    def _bidirectional_astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        width: int,
        height: int,
        moves: List[Tuple[int, int]],
        move_costs: List[float],
        check_map: np.ndarray
    ) -> Optional[List[Tuple[int, int]]]:
        """双向 A* 实现"""

        # Forward search (from start to goal)
        fwd_open = []
        heapq.heappush(fwd_open, (0.0, start))
        fwd_g = {start: 0.0}
        fwd_came_from = {start: None}
        fwd_closed = set()

        # Backward search (from goal to start)
        bwd_open = []
        heapq.heappush(bwd_open, (0.0, goal))
        bwd_g = {goal: 0.0}
        bwd_came_from = {goal: None}
        bwd_closed = set()

        allow_diagonal = len(moves) > 4
        best_path = None
        best_estimate = float('inf')

        while fwd_open or bwd_open:
            if fwd_open and bwd_open:
                fwd_f = fwd_open[0][0]
                bwd_f = bwd_open[0][0]
                if best_path is not None and min(fwd_f, bwd_f) >= best_estimate:
                    break

            if fwd_open:
                fwd_f, fwd_current = heapq.heappop(fwd_open)
                if fwd_current in fwd_closed:
                    continue
                fwd_closed.add(fwd_current)

                if fwd_current in bwd_g:
                    total = fwd_g[fwd_current] + bwd_g[fwd_current]
                    if best_path is None or total < best_estimate:
                        best_estimate = total
                        best_path = self._reconstruct_bidirectional_path(
                            fwd_came_from, bwd_came_from, fwd_current
                        )

                for move, move_cost in zip(moves, move_costs):
                    nx, ny = fwd_current[0] + move[0], fwd_current[1] + move[1]
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if check_map[ny, nx] >= self.obstacle_threshold:
                        continue
                    if allow_diagonal and move[0] != 0 and move[1] != 0:
                        if check_map[fwd_current[1] + move[1], fwd_current[0]] >= self.obstacle_threshold or \
                           check_map[fwd_current[1], fwd_current[0] + move[0]] >= self.obstacle_threshold:
                            continue

                    neighbor = (nx, ny)
                    if neighbor in fwd_closed:
                        continue

                    tentative_g = fwd_g[fwd_current] + move_cost
                    if neighbor not in fwd_g or tentative_g < fwd_g[neighbor]:
                        fwd_g[neighbor] = tentative_g
                        fwd_came_from[neighbor] = fwd_current
                        h = self.heuristic(neighbor, goal)
                        heapq.heappush(fwd_open, (tentative_g + h, neighbor))

            if bwd_open:
                bwd_f, bwd_current = heapq.heappop(bwd_open)
                if bwd_current in bwd_closed:
                    continue
                bwd_closed.add(bwd_current)

                if bwd_current in fwd_g:
                    total = fwd_g[bwd_current] + bwd_g[bwd_current]
                    if best_path is None or total < best_estimate:
                        best_estimate = total
                        best_path = self._reconstruct_bidirectional_path(
                            fwd_came_from, bwd_came_from, bwd_current
                        )

                if best_path is not None and bwd_f >= best_estimate:
                    continue

                # Reverse moves for backward search
                for move, move_cost in zip(moves, move_costs):
                    nx, ny = bwd_current[0] + move[0], bwd_current[1] + move[1]
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if check_map[ny, nx] >= self.obstacle_threshold:
                        continue
                    if allow_diagonal and move[0] != 0 and move[1] != 0:
                        if check_map[bwd_current[1] + move[1], bwd_current[0]] >= self.obstacle_threshold or \
                           check_map[bwd_current[1], bwd_current[0] + move[0]] >= self.obstacle_threshold:
                            continue

                    neighbor = (nx, ny)
                    if neighbor in bwd_closed:
                        continue

                    tentative_g = bwd_g[bwd_current] + move_cost
                    if neighbor not in bwd_g or tentative_g < bwd_g[neighbor]:
                        bwd_g[neighbor] = tentative_g
                        bwd_came_from[neighbor] = bwd_current
                        h = self.heuristic(neighbor, start)
                        heapq.heappush(bwd_open, (tentative_g + h, neighbor))

        return best_path

    def _reconstruct_bidirectional_path(
        self,
        fwd_came_from: dict,
        bwd_came_from: dict,
        meeting_node: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """从双向搜索的交汇点重建路径"""
        # Forward path: start -> meeting_node
        forward_path = []
        node = meeting_node
        while node is not None:
            forward_path.append(node)
            node = fwd_came_from.get(node)

        forward_path.reverse()

        # Backward path: meeting_node -> goal (skip meeting_node)
        backward_path = []
        node = bwd_came_from.get(meeting_node)
        while node is not None:
            backward_path.append(node)
            node = bwd_came_from.get(node)

        return forward_path + backward_path

    def sparsify_path(self, path: List[Tuple[int, int]], max_cells: int) -> List[Tuple[int, int]]:
        """
        稀疏化路径：
        - path 不包含起点
        - 每隔 max_cells-1 个格子取一个点
        - 终点必须保留
        """
        if len(path) <= 1:
            return path

        max_cells = max(1, int(max_cells))
        sparse = []

        for i in range(1, len(path), max_cells):
            sparse.append(path[i])

        if not sparse or sparse[-1] != path[-1]:
            sparse.append(path[-1])

        return sparse

    def publish_sparse_path_in_base_link(
        self,
        waypoints_map: List[dict],
        stamp: Optional[Time] = None
    ):
        """将 map 坐标系路径转换到 base_link 坐标系后发布"""
        # 从队列中获取与时间戳最接近的 map_pose
        if stamp is not None:
            target_nanos = TimeUtils.stamp_to_nanos(stamp)
            pose = self.get_closest_map_pose(target_nanos)
            if pose is None or not pose.get('valid', False):
                self.logger.warning('No valid robot pose, cannot publish planned path')
                return
            pose_time_diff = abs(pose['timestamp'] - target_nanos) / 1e9
            if pose_time_diff > self.map_pose_timeout:
                self.logger.warning(f'Robot pose time diff too large: {pose_time_diff:.3f}s > {self.map_pose_timeout:.3f}s')
                return
        else:
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
        msg.header.stamp = stamp if stamp is not None else TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
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

    def publish_sparse_path_in_map(
        self,
        waypoints_map: List[dict],
        stamp: Optional[Time] = None
    ):
        """发布 map 坐标系下的稀疏路径，供 rviz 可视化使用"""
        msg = Path()
        msg.header.stamp = stamp if stamp is not None else TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
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
        stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())

        msg_base = Path()
        msg_base.header.stamp = stamp
        msg_base.header.frame_id = 'base_link'
        self.path_pub.publish(msg_base)

        msg_map = Path()
        msg_map.header.stamp = stamp
        msg_map.header.frame_id = 'map'
        self.path_map_pub.publish(msg_map)

    # ==================== 航点到达检查定时器 ====================

    def _init_arrival_check_timer(self):
        """初始化航点到达检查定时器"""
        self.arrival_check_timer = self.create_timer(
            self.arrival_check_interval,
            self.arrival_check_callback
        )
        self.logger.info(f'航点到达检查定时器已启动，间隔: {self.arrival_check_interval}s')

    def arrival_check_callback(self):
        """定时检查航点是否到达，使用最新的 map_pose"""
        if self.task_completed:
            return

        if not self.nav_map_points:
            return

        current_nanos = TimeUtils.now_nanos()
        pose = self.latest_map_pose

        if pose is None or not pose.get('valid', False):
            return

        pose_timestamp = pose.get('timestamp', 0)
        if current_nanos - pose_timestamp > self.map_pose_timeout * 1e9:
            self.logger.warning('arrival_check: Robot pose timestamp expired')
            return

        robot_x = float(pose['x'])
        robot_y = float(pose['y'])

        # 检查并更新未到达航点指针
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

    def plan_once(self, local_costmap_stamp: Optional[Time] = None):
        """执行一次规划；直接使用最新的 map_pose。"""
        if not self._has_map():
            return

        current_nanos = TimeUtils.now_nanos()

        pose = self.latest_map_pose
        if pose is None or not pose.get('valid', False):
            return
        pose_timestamp = pose.get('timestamp', 0)
        if current_nanos - pose_timestamp > self.map_pose_timeout * 1e9:
            self.logger.warning('Robot pose timestamp expired')
            return

        robot_x = float(pose['x'])
        robot_y = float(pose['y'])

        if self.task_completed:
            return

        if not self.nav_map_points:
            return

        goal_point = self.nav_map_points[self.unreached_index]
        goal_map_x = float(goal_point['x'])
        goal_map_y = float(goal_point['y'])

        map_data = self.map_data
        metadata = self.map_metadata
        # 选择规划地图：膨胀功能开启时使用膨胀地图，否则使用原始地图
        planning_map = self.inflated_map_data if self.inflation_enabled else self.map_data

        start_gx, start_gy = self.world_to_grid(robot_x, robot_y, metadata)
        goal_gx, goal_gy = self.world_to_grid(goal_map_x, goal_map_y, metadata)

        start_gx = max(0, min(metadata.width - 1, start_gx))
        start_gy = max(0, min(metadata.height - 1, start_gy))
        goal_gx = max(0, min(metadata.width - 1, goal_gx))
        goal_gy = max(0, min(metadata.height - 1, goal_gy))

        max_snap_radius = max(1, int(math.ceil(self.arrival_threshold / metadata.resolution)))

        start_cell = self.find_nearest_free_cell(
            (start_gx, start_gy), metadata.width, metadata.height, max_snap_radius,
            planning_map=planning_map
        )
        goal_cell = self.find_nearest_free_cell(
            (goal_gx, goal_gy), metadata.width, metadata.height, max_snap_radius,
            planning_map=planning_map
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
            allow_diagonal=self.allow_diagonal,
            planning_map=planning_map
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

        self.publish_sparse_path_in_base_link(waypoints_map, local_costmap_stamp)
        #self.publish_sparse_path_in_map(waypoints_map, local_costmap_stamp)

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