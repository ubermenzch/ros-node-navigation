#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_planner_node.py

将 map_node.py、planner_node.py、shared_map_storage.py 合并为一个节点。

功能：
1. 接收 anav_v3 下发的 GNSS 航点
2. 将航点 GNSS 转换为 map 坐标
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
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time
from utils.time_utils import TimeUtils
from utils.data_queue import DataQueue

import tf2_ros
from nav_msgs.msg import OccupancyGrid, Path
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import String

# UTM 库
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    logging.warning("pyproj not installed, GNSS->UTM conversion will not work")

from utils.config_loader import get_config
from utils.frequency_stats import FrequencyStats
from utils.logger import NodeLogger


@dataclass
class MapMetadata:
    """地图元数据"""
    resolution: float
    width: int
    height: int
    origin_x: float   # 地图左下角 x（米）
    origin_y: float   # 地图左下角 y（米）


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
        self.generate_road_on_init = self._get_bool_config(
            node_config,
            'generate_road_on_init',
            True
        )

        # 规划参数
        self.publish_full_map = bool(node_config.get('publish_full_map', True))
        self.publish_uninflated_map = bool(node_config.get('publish_uninflated_map', True))
        self.publish_debug_map = bool(node_config.get('publish_debug_map', False))
        self.map_pose_timeout = float(node_config.get('map_pose_timeout', 1.0))
        self.local_costmap_timeout = float(node_config.get('local_costmap_timeout', 0.3))
        self.queue_cleanup_frequency = float(node_config.get('queue_cleanup_frequency', 10.0))
        self.max_distance_between = float(node_config.get('max_distance_between', 0.5))
        self.dense_nav_points_max_distance = float(node_config.get('dense_nav_points_max_distance', 0.0))
        self.arrival_threshold = float(node_config.get('arrival_threshold', 1.0))
        self.global_path_rejoin_threshold = float(node_config.get('global_path_rejoin_threshold', 0.0))
        self.global_path_rejoin_goal_weight = max(
            0.0,
            min(1.0, float(node_config.get('global_path_rejoin_goal_weight', 0.1)))
        )
        arrival_check_frequency = float(node_config.get('arrival_check_frequency', 2.0))
        self.arrival_check_interval = 1.0 / arrival_check_frequency if arrival_check_frequency > 0 else 0.5
        self.planning_frequency = float(node_config.get('planning_frequency', 2.0))
        self.path_publish_frequency = float(node_config.get('path_publish_frequency', 10.0))
        self.obstacle_threshold = int(node_config.get('obstacle_threshold', 50))
        self.allow_diagonal_astar = bool(node_config.get('allow_diagonal_astar', False))
        # A* 单次最大节点扩展数，超出时返回已找到的最优路径（或 None）
        # 用于防止在大地图碎片障碍物场景下 A* 长时间阻塞规划线程
        self.max_astar_nodes = int(node_config.get('max_astar_nodes', 200000))
        self.use_bidirectional_astar = bool(node_config.get('use_bidirectional_astar', True))

        # 膨胀参数（用于膨胀地图和规划，单位：米）
        self.inflation_margin = float(node_config.get('inflation_margin', 0.3))
        self.inflation_enabled = bool(node_config.get('inflation_enabled', True))
        self.inflation_radius_cells = self._get_inflation_radius_cells(self.resolution)
        if self.inflation_radius_cells <= 0:
            self.inflation_enabled = False
            self.logger.info('Inflation disabled: inflation_radius_cells <= 0')
        self.utm_lookup_timeout = 1.0

        # 话题
        self.gnss_path_topic = subscriptions.get('gnss_path_topic', '/navigation_control')
        self.map_pose_topic = subscriptions.get('map_pose_topic', '/navigation/map_pose')
        self.local_costmap_topic = subscriptions.get('local_costmap_topic', '/navigation/local_costmap')

        self.map_topic = publications.get('map_topic', '/map')
        self.map_update_topic = publications.get('map_update_topic', '/navigation/map_update')
        self.inflated_map_topic = publications.get('inflated_map_topic', '/navigation/inflated_map')
        self.inflated_map_update_topic = publications.get('inflated_map_update_topic', '/navigation/inflated_map_update')
        self.nav_map_points_topic = publications.get('nav_map_points_topic', '/navigation/nav_map_points')
        self.path_topic = publications.get('path_topic', '/planned_path')
        self.path_map_topic = publications.get('path_map_topic', '/navigation/planned_path_map')
        self.target_marker_topic = publications.get('target_marker_topic', '/navigation/controller_target')

        # 内部地图存储（代替 shared_map_storage）
        self._state_lock = threading.RLock()
        self.map_data: Optional[np.ndarray] = None          # shape=(height, width), row=bottom->top
        self.map_metadata: Optional[MapMetadata] = None

        # 膨胀地图存储
        self.inflated_map_data: Optional[np.ndarray] = None  # 膨胀后的地图，shape=(height, width)

        # 局部地图坐标缓存（用于向量化更新）

        # 位姿状态 - 使用队列缓存 map_pose，按时间戳匹配 local_costmap
        self.map_pose_queue = DataQueue(timeout_seconds=self.map_pose_timeout)

        # 路径任务状态
        self.nav_gnss_points: List[dict] = []
        self.nav_map_points: List[dict] = []
        self.batch_id = ''
        self.batch_number = 0
        self.unreached_index = 0
        self.cached_waypoints_map: List[dict] = []
        self.cached_rviz_waypoints_map: List[dict] = []

        # ========== 并行执行：创建独立的 callback groups（共享地图状态用锁保护）==========
        self._data_group = MutuallyExclusiveCallbackGroup()      # gnss_path, map_pose 共用
        self._costmap_group = MutuallyExclusiveCallbackGroup()   # local_costmap 单独
        self._planning_group = MutuallyExclusiveCallbackGroup()  # 规划定时器单独
        self._path_publish_group = MutuallyExclusiveCallbackGroup()  # 高频路径发布定时器

        # UTM 变换
        self.utm_zone: Optional[int] = None
        self.utm_transformer = None

        # TF 监听
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 发布者
        self.map_pub = None
        self.map_update_pub = None
        if self.publish_uninflated_map:
            self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, 1)
            self.map_update_pub = self.create_publisher(OccupancyGridUpdate, self.map_update_topic, 1)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, self.inflated_map_topic, 1)
        self.inflated_map_update_pub = self.create_publisher(OccupancyGridUpdate, self.inflated_map_update_topic, 1)

        nav_map_points_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.nav_map_points_pub = self.create_publisher(Path, self.nav_map_points_topic, nav_map_points_qos)
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.path_map_pub = self.create_publisher(Path, self.path_map_topic, 1)
        self.target_marker_pub = self.create_publisher(Marker, self.target_marker_topic, 1)

        # 调试地图发布者。该地图尺寸/origin 会随局部更新变化，默认关闭以避免 RViz2 反复重建 Map 纹理。
        self.debug_map_pub = None
        if self.publish_debug_map:
            self.debug_map_pub = self.create_publisher(OccupancyGrid, '/navigation/debug_map', 1)

        # 订阅者（分配到独立 callback groups）
        self.gnss_path_sub = self.create_subscription(
            String,
            self.gnss_path_topic,
            self.gnss_path_callback,
            1,
            callback_group=self._data_group
        )

        self.map_pose_sub = self.create_subscription(
            PoseStamped,
            self.map_pose_topic,
            self.map_pose_callback,
            1,
            callback_group=self._data_group
        )

        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.local_costmap_topic,
            self.local_costmap_callback,
            1,
            callback_group=self._costmap_group
        )

        # 频率统计（统计 local_costmap_callback 执行频率）
        self.map_freq_stats = FrequencyStats(
            object_name='map_planner_node.local_costmap_callback',
            node_logger=self.logger,
            window_size=10,
            log_interval=5.0
        )

        # 定时清理所有缓存队列
        self.queue_cleanup_timer = None
        if self.queue_cleanup_frequency > 0:
            cleanup_interval = 1.0 / self.queue_cleanup_frequency
            self.queue_cleanup_timer = self.create_timer(
                cleanup_interval,
                self._queue_cleanup_timer_callback,
                callback_group=self._data_group
            )
            self.logger.info(
                f'队列清理定时器已启动，间隔: {cleanup_interval}s ({self.queue_cleanup_frequency}Hz)'
            )
        else:
            self.logger.info('队列清理定时器已禁用')

        # 初始化航点到达检查定时器
        self._init_arrival_check_timer()

        init_info = [
            'MapPlanner Node initialized',
            f'  订阅 GNSS 路径: {self.gnss_path_topic}',
            f'  订阅 map_pose: {self.map_pose_topic}',
            f'  订阅 local_costmap: {self.local_costmap_topic}',
            f'  发布未膨胀地图: {self.map_topic}' if self.publish_uninflated_map else '  发布未膨胀地图: 关闭',
            (
                f'  发布未膨胀地图增量更新: {self.map_update_topic}'
                if self.publish_uninflated_map and not self.publish_full_map
                else None
            ),
            f'  发布膨胀地图: {self.inflated_map_topic}',
            f'  发布膨胀地图增量更新: {self.inflated_map_update_topic}' if not self.publish_full_map else None,
            f'  发布导航 map 点: {self.nav_map_points_topic}',
            f'  发布规划路径: {self.path_topic}',
            f'  发布RViz规划路径(map): {self.path_map_topic}',
            f'  发布当前目标点Marker: {self.target_marker_topic}',
            f'  分辨率: {self.resolution}m',
            f'  道路宽度: {self.road_width}m',
            f'  初始化道路生成: {"开启" if self.generate_road_on_init else "关闭（空地图模式）"}',
            f'  膨胀余量: {self.inflation_margin}m',
            f'  膨胀形状: 正方形（边长={self.inflation_margin * 2:.2f}m）',
            f'  膨胀地图功能: {"开启" if self.inflation_enabled else "关闭"}',
            f'  航点到达阈值: {self.arrival_threshold}m',
            f'  全局路径回归阈值: {self.global_path_rejoin_threshold}m',
            f'  全局路径回归目标航点权重: {self.global_path_rejoin_goal_weight}',
            f'  航点检查间隔: {self.arrival_check_interval}s',
            f'  规划频率: {self.planning_frequency}Hz',
            f'  base_link路径发布频率: {self.path_publish_frequency}Hz',
            f'  队列清理频率: {self.queue_cleanup_frequency}Hz',
            f'  A* 算法模式: {"双向 A*" if self.use_bidirectional_astar else "单向 A*"}',
            f'  A* 移动模式: {"8连通（允许斜向）" if self.allow_diagonal_astar else "4连通（仅上下左右）"}',
            f'  运行模式: 定时A*规划 + 高频base_link路径发布',
            f'  地图发布方式: {"完整 OccupancyGrid" if self.publish_full_map else "OccupancyGridUpdate 增量"}',
            f'  地图发布内容: {"未膨胀地图 + 膨胀地图" if self.publish_uninflated_map else "仅膨胀地图"}',
            f'  调试局部地图发布: {"开启 /navigation/debug_map" if self.publish_debug_map else "关闭"}',
        ]
        self.node_logger.log_init([line for line in init_info if line is not None])

        # 初始化路径规划定时器
        planning_interval = 1.0 / self.planning_frequency if self.planning_frequency > 0 else 0.5
        self.planning_timer = self.create_timer(
            planning_interval,
            self._planning_timer_callback,
            callback_group=self._planning_group
        )
        self.logger.info(f'路径规划定时器已启动，间隔: {planning_interval}s ({self.planning_frequency}Hz)')

        self.path_publish_timer = None
        if self.path_publish_frequency > 0:
            path_publish_interval = 1.0 / self.path_publish_frequency
            self.path_publish_timer = self.create_timer(
                path_publish_interval,
                self._path_publish_timer_callback,
                callback_group=self._path_publish_group
            )
            self.logger.info(
                f'base_link路径发布定时器已启动，间隔: {path_publish_interval}s '
                f'({self.path_publish_frequency}Hz)'
            )
        else:
            self.logger.info('base_link路径发布定时器已禁用')

    # ==================== 日志 ====================

    def _init_logger(self, enabled: bool):
        """初始化日志系统"""
        self.node_logger = NodeLogger(
            node_name='map_planner_node',
            log_dir=self.log_dir,
            log_timestamp=self.log_timestamp,
            enabled=enabled,
            ros_logger=self.get_logger()
        )
        self.logger = self.node_logger

    # ==================== 基础状态检查 ====================

    @staticmethod
    def _get_bool_config(config: dict, key: str, default: bool) -> bool:
        value = config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ('true', '1', 'yes', 'on'):
                return True
            if normalized in ('false', '0', 'no', 'off'):
                return False
        return bool(value)

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

    def gnss_to_utm(self, lat: float, lon: float) -> Tuple[Optional[float], Optional[float]]:
        """经纬度转 UTM"""
        if not HAS_PYPROJ:
            self.logger.error('pyproj not installed, cannot convert GNSS to UTM')
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

    def gnss_to_map_coords(
        self,
        lat: float,
        lon: float,
        trans_x: float,
        trans_y: float,
        yaw: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        GNSS -> UTM -> map

        已知：
            p_utm = t + R(yaw) * p_map
        所以：
            p_map = R(-yaw) * (p_utm - t)
        """
        utm_x, utm_y = self.gnss_to_utm(lat, lon)
        if utm_x is None or utm_y is None:
            return None, None

        dx = utm_x - trans_x
        dy = utm_y - trans_y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        map_x = cos_yaw * dx + sin_yaw * dy
        map_y = -sin_yaw * dx + cos_yaw * dy

        return map_x, map_y

    def convert_gnss_points_to_map_points(
        self,
        gnss_points: List[dict],
        trans_x: float,
        trans_y: float,
        yaw: float
    ) -> List[dict]:
        """
        将 GNSS 航点列表转换为 map 坐标系下的航点列表

        Args:
            gnss_points: GNSS 航点列表，每个元素包含 'latitude' 和 'longitude'
            trans_x: UTM 到 map 的 x 偏移
            trans_y: UTM 到 map 的 y 偏移
            yaw: 航向角

        Returns:
            map 坐标系下的航点列表，转换失败的点会被跳过
        """
        nav_map_points = []
        for wp in gnss_points:
            try:
                lat = float(wp['latitude'])
                lon = float(wp['longitude'])
            except Exception:
                continue

            map_x, map_y = self.gnss_to_map_coords(lat, lon, trans_x, trans_y, yaw)
            if map_x is None or map_y is None:
                self.logger.warning(f'Failed to convert GNSS ({lat}, {lon}) to map coords')
                continue

            nav_map_points.append({'x': map_x, 'y': map_y})

        return nav_map_points

    def densify_nav_points(
        self,
        points: List[dict],
        max_distance: float
    ) -> List[dict]:
        """
        密集化导航航点，确保相邻两点之间的距离不超过 max_distance

        Args:
            points: 原始航点列表，每个元素包含 'x' 和 'y'
            max_distance: 相邻点之间的最大允许距离（米）

        Returns:
            密集化后的航点列表
        """
        if max_distance <= 0 or len(points) <= 1:
            return list(points)

        result = [points[0]]
        for i in range(1, len(points)):
            prev = result[-1]
            curr = points[i]
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dist = math.hypot(dx, dy)

            if dist > max_distance:
                n = int(math.ceil(dist / max_distance))
                for j in range(1, n):
                    t = j / n
                    result.append({
                        'x': prev['x'] + t * dx,
                        'y': prev['y'] + t * dy
                    })
            result.append(curr)

        return result

    def _project_point_to_segment(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float
    ) -> Tuple[float, float, float, float]:
        """返回点 P 到线段 AB 的最近点、线段参数 t 和距离。"""
        dx = bx - ax
        dy = by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return ax, ay, 0.0, math.hypot(px - ax, py - ay)

        t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        distance = math.hypot(px - closest_x, py - closest_y)
        return closest_x, closest_y, t, distance

    def _select_astar_goal_from_global_segment(
        self,
        robot_x: float,
        robot_y: float,
        current_goal: dict
    ) -> Tuple[float, float, str, Optional[Tuple[float, float, float, float]]]:
        """
        根据机器人到当前全局航段的偏离距离选择 A* 目标点。

        当偏离超过 global_path_rejoin_threshold 时，规划到当前航段最近点
        与当前目标航点的加权融合点；
        否则直接规划到当前目标航点。
        """
        current_x = float(current_goal['x'])
        current_y = float(current_goal['y'])
        if self.global_path_rejoin_threshold <= 0:
            return current_x, current_y, 'waypoint', None

        if self.unreached_index > 0:
            previous_goal = self.nav_map_points[self.unreached_index - 1]
        else:
            return current_x, current_y, 'waypoint', None

        prev_x = float(previous_goal['x'])
        prev_y = float(previous_goal['y'])
        closest_x, closest_y, t, distance = self._project_point_to_segment(
            robot_x, robot_y, prev_x, prev_y, current_x, current_y
        )
        projection_info = (closest_x, closest_y, t, distance)

        if distance > self.global_path_rejoin_threshold:
            waypoint_weight = self.global_path_rejoin_goal_weight
            projection_weight = 1.0 - waypoint_weight
            blended_x = projection_weight * closest_x + waypoint_weight * current_x
            blended_y = projection_weight * closest_y + waypoint_weight * current_y
            return blended_x, blended_y, 'global_path_blend', projection_info

        return current_x, current_y, 'waypoint', projection_info

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
            }

            # 添加到队列
            self.map_pose_queue.append(TimeUtils.stamp_to_nanos(msg.header.stamp), pose_entry)

        except Exception as e:
            self.logger.error(f'Failed to parse map_pose: {e}')

    def _queue_cleanup_timer_callback(self):
        """定时清理所有缓存队列中的过期数据"""
        current_nanos = TimeUtils.now_nanos()
        removed_map_pose = self.map_pose_queue.prune_expired(current_nanos)
        if removed_map_pose > 0:
            self.logger.debug(f'Pruned expired queue frames: map_pose_queue={removed_map_pose}')

    def is_task_completed(self) -> bool:
        """检查任务是否已完成（所有航点都已到达）"""
        return self.unreached_index >= len(self.nav_map_points)

    def gnss_path_callback(self, msg: String):
        """
        接收下发的导航 GNSS 点

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
            self.logger.error(f'Failed to parse GNSS path message: {e}')
            return

        points = data.get('points', [])
        if not points:
            self.logger.warning('Received empty GNSS path message')
            return

        self.nav_gnss_points = points
        self.batch_id = data.get('batchId', '')
        self.batch_number += 1

        self.logger.info(
            f'Received new GNSS path: {len(points)} points, '
            f'batchId={self.batch_id}, batch_number={self.batch_number}'
        )
        self.publish_empty_path()

        # GNSS 航点 -> map 航点
        trans_x, trans_y, yaw = self.get_utm_to_map_transform()
        if trans_x is None:
            self.logger.warning('utm<-map transform not available, cannot convert GNSS points')
            return

        self.nav_map_points = self.convert_gnss_points_to_map_points(
            list(self.nav_gnss_points), trans_x, trans_y, yaw
        )


        # 密集化航点
        if self.dense_nav_points_max_distance > 0 and len(self.nav_map_points) > 1:
            self.nav_map_points = self.densify_nav_points(
                self.nav_map_points, self.dense_nav_points_max_distance
            )
        
        # 先发布 nav_map_points（用于 rviz 可视化）
        self.publish_nav_map_points()
        # 生成地图（传入已经处理好的 nav_map_points）
        success = self.generate_map_and_nav_points(self.nav_map_points)
        if not success:
            self.logger.warning('Failed to generate map from new GNSS path')

    def local_costmap_callback(self, msg: OccupancyGrid):
        callback_start_nanos = TimeUtils.now_nanos()

        if self.map_data is None:
            self.logger.warning('local_costmap_callback: map_data is None, skipped')
            return
        if self.map_metadata is None:
            self.logger.warning('local_costmap_callback: map_metadata is None, skipped')
            return
        if self.inflation_enabled and self.inflated_map_data is None:
            self.logger.warning('local_costmap_callback: inflated_map_data is None, skipped')
            return
        if msg.header.frame_id != 'base_link':
            self.logger.warning(f'Unexpected local_costmap frame_id: {msg.header.frame_id}')
            return
        
        current_nanos = TimeUtils.now_nanos()
        costmap_timestamp = TimeUtils.stamp_to_nanos(msg.header.stamp)
        cloud_age_sec = (current_nanos - costmap_timestamp) / 1e9
        if cloud_age_sec > self.local_costmap_timeout:
            self.logger.warning(f'local_costmap timeout: {cloud_age_sec:.2f}s > {self.local_costmap_timeout:.2f}s')
            return

        pose_frame = self.map_pose_queue.find_nearest(costmap_timestamp)

        if pose_frame is None:
            self.logger.warning(f'map_pose queue empty or not received')
            return

        try:
            width = int(msg.info.width)
            height = int(msg.info.height)
            resolution = float(msg.info.resolution)

            if width <= 0 or height <= 0:
                return

            costmap = np.array(msg.data, dtype=np.int16).reshape((height, width))

            origin_x = float(msg.info.origin.position.x)
            origin_y = float(msg.info.origin.position.y)

            with self._state_lock:
                update_map_start = TimeUtils.now_nanos()
                map_updated, update_box = self.update_global_map_from_local_costmap(
                    local_costmap=costmap,
                    local_resolution=resolution,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    robot_x=pose_frame.data['x'],
                    robot_y=pose_frame.data['y'],
                    robot_yaw=pose_frame.data['yaw']
                )
                update_map_elapsed = (TimeUtils.now_nanos() - update_map_start) / 1e6

                if map_updated:
                    pub_map_elapsed = 0.0
                    if self.publish_uninflated_map:
                        pub_map_start = TimeUtils.now_nanos()
                        if self.publish_full_map:
                            full_grid = self.build_map_msg(self.map_data, self.map_metadata)
                            if full_grid is not None:
                                self.map_pub.publish(full_grid)
                        else:
                            min_col, min_row, max_col, max_row = update_box
                            submap = self.map_data[min_row:max_row + 1, min_col:max_col + 1]
                            update_msg = self.build_map_update_msg(update_box, submap, msg.header.stamp)
                            if update_msg is not None:
                                self.map_update_pub.publish(update_msg)
                        pub_map_elapsed = (TimeUtils.now_nanos() - pub_map_start) / 1e6

                    if self.inflation_enabled:
                        inflate_start = TimeUtils.now_nanos()
                        inflated_update_box = self.update_inflated_map_from_bbox(update_box)
                        inflate_elapsed = (TimeUtils.now_nanos() - inflate_start) / 1e6
                        
                        pub_inflate_start = TimeUtils.now_nanos()
                        if self.publish_full_map:
                            full_inflated_grid = self.build_map_msg(self.inflated_map_data, self.map_metadata)
                            if full_inflated_grid is not None:
                                self.inflated_map_pub.publish(full_inflated_grid)
                        else:
                            if inflated_update_box is not None:
                                min_col, min_row, max_col, max_row = inflated_update_box
                                sub = self.inflated_map_data[min_row:max_row + 1, min_col:max_col + 1]
                                update_msg = self.build_map_update_msg(inflated_update_box, sub, msg.header.stamp)
                                if update_msg is not None:
                                    self.inflated_map_update_pub.publish(update_msg)
                        pub_inflate_elapsed = (TimeUtils.now_nanos() - pub_inflate_start) / 1e6
                    else:
                        inflate_elapsed = 0.0
                        pub_inflate_elapsed = 0.0

                    total_elapsed = TimeUtils.now_nanos() - callback_start_nanos
                    other_elapsed = total_elapsed - update_map_elapsed * 1e6 - pub_map_elapsed * 1e6 - inflate_elapsed * 1e6 - pub_inflate_elapsed * 1e6
                    
                    self.logger.info(
                        f'local_costmap_callback timing: '
                        f'total={total_elapsed / 1e6:.2f}ms | '
                        f'update_map={update_map_elapsed:.2f}ms | '
                        f'pub_map={pub_map_elapsed:.2f}ms | '
                        f'inflate={inflate_elapsed:.2f}ms | '
                        f'pub_inflate={pub_inflate_elapsed:.2f}ms | '
                        f'other={other_elapsed / 1e6:.2f}ms'
                    )
                else:
                    self.logger.debug('local_costmap_callback: map not updated')

            self.map_freq_stats.tick()

        except Exception as e:
            self.logger.error(f'Failed to process local_costmap: {e}')

    # ==================== 地图生成 ====================

    def draw_road_on_grid(
        self,
        grid: np.ndarray,
        metadata: MapMetadata,
        path_points: List[Tuple[float, float]]
    ):
        """
        根据路径点在栅格图上画道路 —— 圆形邮票 (stamp) 法

        算法原理：
        - 沿路径以一定步长采样（保证相邻圆有重叠即可，无需到分辨率级）
        - 预计算一个圆形 mask (2R+1)×(2R+1)
        - 对每个采样点，在其周围用圆形 mask 标记空闲区域
        - 通过 np.unique 去重，避免在同一格子反复盖章

        复杂度：O(N × R²)，N=采样点数，R=半径(格子数)
        内存：O(R²)，只需要预计算圆形 mask
        """
        if not path_points or len(path_points) < 2:
            return

        h, w = grid.shape
        radius_cells = max(1, int(math.ceil((self.road_width * 0.5) / metadata.resolution)))

        # 步长：保证相邻圆重叠即可。取 max(配置, resolution)，并 cap 到 radius_cells*resolution
        # 防止用户把 road_sample_step 设得过大造成路面出现间隙
        max_safe_stride = radius_cells * metadata.resolution
        stride_m = min(
            max_safe_stride,
            max(self.road_sample_step, metadata.resolution)
        )

        dense_points = self.interpolate_polyline(path_points, step=stride_m)
        if not dense_points:
            return

        # 一次性转栅格坐标
        pts = np.asarray(dense_points, dtype=np.float64)
        gx_all = np.floor((pts[:, 0] - metadata.origin_x) / metadata.resolution).astype(np.int64)
        gy_all = np.floor((pts[:, 1] - metadata.origin_y) / metadata.resolution).astype(np.int64)

        # 只保留圆与地图有交集的点（圆心可在地图外）
        keep = (
            (gx_all + radius_cells >= 0) & (gx_all - radius_cells < w) &
            (gy_all + radius_cells >= 0) & (gy_all - radius_cells < h)
        )
        gx_all = gx_all[keep]
        gy_all = gy_all[keep]
        if gx_all.size == 0:
            return

        # 去重，避免在同一格反复盖章
        coords = np.unique(np.stack([gy_all, gx_all], axis=1), axis=0)

        # 预计算圆形 mask（半径 radius_cells，尺寸 (2R+1)×(2R+1)）
        rng = np.arange(-radius_cells, radius_cells + 1)
        yy, xx = np.meshgrid(rng, rng, indexing='ij')
        circle_mask = (xx * xx + yy * yy) <= radius_cells * radius_cells

        # 逐点贴章
        for gy, gx in coords:
            y0 = max(0, gy - radius_cells)
            y1 = min(h, gy + radius_cells + 1)
            x0 = max(0, gx - radius_cells)
            x1 = min(w, gx + radius_cells + 1)
            if y0 >= y1 or x0 >= x1:
                continue

            # 圆形 mask 中对应被裁掉的边界
            my0 = y0 - (gy - radius_cells)
            mx0 = x0 - (gx - radius_cells)
            my1 = my0 + (y1 - y0)
            mx1 = mx0 + (x1 - x0)

            sub_mask = circle_mask[my0:my1, mx0:mx1]
            sub = grid[y0:y1, x0:x1]
            sub[sub_mask] = 0

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

    def generate_map_and_nav_points(self, nav_map_points: List[dict]) -> bool:
        """根据传入的 nav_map_points 生成全局地图

        Args:
            nav_map_points: 已经转换好的 map 坐标航点列表
        """
        # 检查 map_pose 超时
        current_nanos = TimeUtils.now_nanos()
        pose_frame = self.map_pose_queue.get_latest()
        if pose_frame is None:
            self.logger.warning('No valid map_pose yet, cannot generate map')
            return False
        if current_nanos - pose_frame.stamp_nanos > self.map_pose_timeout * 1e9:
            self.logger.warning('map_pose timestamp expired')
            return False

        if not nav_map_points:
            self.logger.warning('No navigation map points')
            return False

        robot_map_x = float(pose_frame.data['x'])
        robot_map_y = float(pose_frame.data['y'])

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

        metadata = MapMetadata(
            resolution=self.resolution,
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y,
        )

        if self.generate_road_on_init:
            # 初始化为障碍物 100，道路区域改成 0
            grid = np.full((height, width), 100, dtype=np.int8)

            # 使用“机器人当前位置 + 航点”连成路
            full_path = [(robot_map_x, robot_map_y)] + [(p['x'], p['y']) for p in nav_map_points]
            self.draw_road_on_grid(grid, metadata, full_path)
            init_mode = 'road'
        else:
            # 快速初始化：全图可通行，后续由局部 costmap 写入障碍物并按需膨胀
            grid = np.zeros((height, width), dtype=np.int8)
            init_mode = 'empty'

        if self.generate_road_on_init and self.inflation_enabled:
            inflated_grid = self.inflate_square(grid, self.inflation_radius_cells)
        else:
            inflated_grid = grid.copy()

        with self._state_lock:
            self.map_metadata = metadata
            self.map_data = grid
            self.inflated_map_data = inflated_grid
            self.unreached_index = 0

        self.logger.info(
            f'Map generated: size={width}x{height}, resolution={self.resolution:.3f}m, '
            f'nav_points={len(nav_map_points)}, robot=({robot_map_x:.2f}, {robot_map_y:.2f}), '
            f'init_mode={init_mode}'
        )

        # 发送未膨胀的全局地图用于可视化初始化
        if self.publish_uninflated_map:
            grid_msg = self.build_map_msg(grid, metadata)
            if grid_msg is not None:
                self.map_pub.publish(grid_msg)
                self.logger.info(
                    f'Published full map: {grid_msg.info.width}x{grid_msg.info.height}'
                )
        else:
            self.logger.info('Skipped publishing uninflated map: publish_uninflated_map=false')

        # 发布初始规划地图。generate_road_on_init=false 时发布全 0 空地图，
        # 避免 RViz 继续显示上一轮道路地图；后续局部更新仍会正常膨胀。
        grid_msg = self.build_map_msg(inflated_grid, metadata)
        if grid_msg is not None:
            self.inflated_map_pub.publish(grid_msg)
            if self.generate_road_on_init and self.inflation_enabled:
                self.logger.info('Published inflated map')
            elif self.generate_road_on_init:
                self.logger.info('Published initial map without inflation')
            else:
                self.logger.info('Published empty initial map: generate_road_on_init=false')

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
        if not self.publish_debug_map or self.debug_map_pub is None:
            return
        if self.map_data is None:
            self.logger.debug('_publish_debug_map: map not ready')
            return
        if self.map_metadata is None:
            self.logger.debug('_publish_debug_map: map_metadata not ready')
            return
        metadata = self.map_metadata
        data = self.map_data

        # 提取子区域（包含边界）
        sub_h = max_row - min_row + 1
        sub_w = max_col - min_col + 1
        sub_data = data[min_row:max_row + 1, min_col:max_col + 1].copy()

        # 构建 OccupancyGrid
        debug_grid = OccupancyGrid()
        debug_grid.header.stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
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
            f'[Debug Map] action=publish size={sub_h}x{sub_w} '
            f'origin=({debug_grid.info.origin.position.x:.3f},{debug_grid.info.origin.position.y:.3f})'
        )

    # ==================== 局部地图更新全局地图 ====================

    def get_local_grid_centers(
        self, height: int, width: int, resolution: float, origin_x: float, origin_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """获取局部地图每个格子中心的本地坐标"""
        rows, cols = np.indices((height, width), dtype=np.float32)
        local_x = origin_x + (cols + 0.5) * resolution
        local_y = origin_y + (rows + 0.5) * resolution
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
        if self.map_data is None:
            self.logger.debug('update_global_map_from_local_costmap: map not ready')
            return (False, None)
        if self.map_metadata is None:
            self.logger.debug('update_global_map_from_local_costmap: map_metadata not ready')
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
        # diff[y0_all + 1, x0_all + 1] += 1
        # diff[y0_all + 1, x1_all + 2] -= 1
        # diff[y1_all + 2, x0_all + 1] -= 1
        # diff[y1_all + 2, x1_all + 2] += 1
        # opus4.7提到array的自增运算有bug（unbuffered），可能导致二维前缀和缺少停止标签，
        # 进而导致膨胀区域泄露，这里换成慢一点但无bug的运算，看能否解决大面积障碍物bug
        np.add.at(diff,(y0_all+1,x0_all+1),1)
        np.add.at(diff,(y0_all+1,x1_all+2),-1)
        np.add.at(diff,(y1_all+2,x0_all+1),-1)
        np.add.at(diff,(y1_all+2,x1_all+2),1)

        # 水平方向累加
        cumsum_h = np.cumsum(diff, axis=1)
        # 垂直方向累加
        cumsum_v = np.cumsum(cumsum_h, axis=0)

        # 提取有效区域并转换为布尔掩码
        inflated_mask = cumsum_v[1:h + 1, 1:w + 1] > 0

        inflated = map_data.copy()
        inflated[inflated_mask] = 100

        return inflated

    # ==================== 膨胀地图更新 ====================

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
        if self.inflated_map_data is None:
            self.logger.debug('update_inflated_map_from_bbox: inflation_map_data not ready')
            return None
        if update_bbox is None:
            self.logger.debug('update_inflated_map_from_bbox: update_bbox is None')
            return None
        if self.map_data is None:
            self.logger.debug('update_inflated_map_from_bbox: map_data not ready')
            return None
        if self.map_metadata is None:
            self.logger.debug('update_inflated_map_from_bbox: map_metadata not ready')
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

    def build_map_msg(self, map_data: Optional[np.ndarray], map_metadata: MapMetadata) -> Optional[OccupancyGrid]:
        """构建 OccupancyGrid 消息

        Args:
            map_data: 地图数据
            map_metadata: 地图元数据
        """
        if map_data is None or map_metadata is None:
            return None

        width = int(map_metadata.width)
        height = int(map_metadata.height)
        resolution = float(map_metadata.resolution)

        if width <= 0 or height <= 0 or resolution <= 0.0:
            self.logger.error(
                f'Invalid map metadata: width={width}, height={height}, resolution={resolution}'
            )
            return None

        if not (
            math.isfinite(resolution) and
            math.isfinite(float(map_metadata.origin_x)) and
            math.isfinite(float(map_metadata.origin_y))
        ):
            self.logger.error(
                f'Invalid non-finite map metadata: '
                f'resolution={resolution}, origin=({map_metadata.origin_x}, {map_metadata.origin_y})'
            )
            return None

        data = np.asarray(map_data)
        if data.shape != (height, width):
            self.logger.error(
                f'OccupancyGrid shape mismatch: data.shape={data.shape}, '
                f'metadata={width}x{height}; skip publishing to protect RViz2'
            )
            return None

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
        grid_msg.header.frame_id = 'map'

        grid_msg.info.resolution = resolution
        grid_msg.info.width = width
        grid_msg.info.height = height

        origin_pose = Pose()
        origin_pose.position.x = float(map_metadata.origin_x)
        origin_pose.position.y = float(map_metadata.origin_y)
        origin_pose.position.z = 0.0
        origin_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        grid_msg.info.origin = origin_pose

        grid_msg.data = data.astype(np.int16, copy=False).ravel().astype(int).tolist()
        expected_len = width * height
        if len(grid_msg.data) != expected_len:
            self.logger.error(
                f'OccupancyGrid data length mismatch: len={len(grid_msg.data)}, '
                f'expected={expected_len}; skip publishing to protect RViz2'
            )
            return None
        return grid_msg

    def build_map_update_msg(
        self,
        update_bbox: Optional[Tuple[int, int, int, int]],
        submap: np.ndarray,
        stamp: Optional[Time] = None
    ) -> Optional[OccupancyGridUpdate]:
        """构建地图增量更新消息

        Args:
            update_bbox: 更新区域 (min_col, min_row, max_col, max_row)
            submap: 子地图数据
            stamp: 时间戳
        """
        if update_bbox is None or submap is None:
            return None

        min_col, min_row, max_col, max_row = update_bbox
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        data = np.asarray(submap)
        if width <= 0 or height <= 0 or data.shape != (height, width):
            self.logger.error(
                f'OccupancyGridUpdate shape mismatch: bbox={update_bbox}, '
                f'expected={height}x{width}, data.shape={data.shape}; skip publishing'
            )
            return None

        update_msg = OccupancyGridUpdate()
        update_msg.header.stamp = stamp if stamp is not None else TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
        update_msg.header.frame_id = 'map'
        update_msg.x = int(min_col)
        update_msg.y = int(min_row)
        update_msg.width = int(width)
        update_msg.height = int(height)
        update_msg.data = data.ravel().astype(int).tolist()
        expected_len = width * height
        if len(update_msg.data) != expected_len:
            self.logger.error(
                f'OccupancyGridUpdate data length mismatch: len={len(update_msg.data)}, '
                f'expected={expected_len}; skip publishing'
            )
            return None
        return update_msg

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
        if self.allow_diagonal_astar:
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _get_astar_moves(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        move_costs = [1.0, 1.0, 1.0, 1.0]

        if self.allow_diagonal_astar:
            moves.extend([(1, 1), (1, -1), (-1, -1), (-1, 1)])
            move_costs.extend([math.sqrt(2.0)] * 4)

        return moves, move_costs

    def _is_astar_move_valid(
        self,
        current: Tuple[int, int],
        move: Tuple[int, int],
        nx: int,
        ny: int,
        width: int,
        height: int,
        check_map: np.ndarray
    ) -> bool:
        if not (0 <= nx < width and 0 <= ny < height):
            return False
        if check_map[ny, nx] >= self.obstacle_threshold:
            return False

        dx, dy = move
        if dx != 0 and dy != 0:
            cx, cy = current
            if check_map[cy, cx + dx] >= self.obstacle_threshold:
                return False
            if check_map[cy + dy, cx] >= self.obstacle_threshold:
                return False

        return True

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
        if planning_map is None:
            self.logger.warning(f"find_nearest_free_cell: planning_map is None, 无法搜索最近可行点 (cell={cell})")
            return None

        cx, cy = cell

        if 0 <= cx < width and 0 <= cy < height:
            if planning_map[cy, cx] < self.obstacle_threshold:
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

                if planning_map[ny, nx] < self.obstacle_threshold:
                    self.logger.debug(f"find_nearest_free_cell: 找到最近可行点 (nx={nx}, ny={ny}, dist={dist}, 原始cell={cell})")
                    return (nx, ny)

                queue.append((nx, ny, dist + 1))

        self.logger.warning(f"find_nearest_free_cell: 在半径 {max_radius} 内未找到可行点 (cell={cell}, 地图尺寸: {width}x{height})")
        return None

    def astar_planning(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        width: int,
        height: int,
        planning_map: Optional[np.ndarray] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* 路径规划（支持双向 A* 优化，适用于长距离路径）

        根据 allow_diagonal_astar 配置使用 4-连通或 8-连通移动。

        Args:
            planning_map: 用于规划的地图（膨胀地图或原始地图）

        因为地图已经膨胀成点机器人可走图，A* 只需检查一个格子即可。
        若使用原始地图，则需要考虑机器人尺寸带来的碰撞。
        """
        if planning_map is None:
            self.logger.warning('Planning map not available')
            return None

        moves, move_costs = self._get_astar_moves()

        if planning_map[start[1], start[0]] >= self.obstacle_threshold:
            self.logger.warning('Start position collides with obstacle')
            return None

        if planning_map[goal[1], goal[0]] >= self.obstacle_threshold:
            self.logger.warning('Goal position collides with obstacle')
            return None

        if start == goal:
            return [start]

        if self.use_bidirectional_astar:
            return self._bidirectional_astar(start, goal, width, height, moves, move_costs, planning_map)
        else:
            return self._unidirectional_astar(start, goal, width, height, moves, move_costs, planning_map)

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
        """双向 A* 实现

        修复说明（相对于原版）：
        1. 终止条件改为 fwd_f + bwd_f >= best_estimate（标准双向 A* 最优性条件），
           原来的 min(fwd_f, bwd_f) >= best_estimate 过于保守，会导致大量多余探索。
        2. 前向搜索补充了与后向搜索对称的提前剪枝：
           找到最优路径后，当 fwd_f >= best_estimate 时跳过邻居扩展。
        3. 增加节点扩展上限 max_astar_nodes，防止在碎片障碍物场景下长时间阻塞：
           - 超限且已找到路径 → 返回当前最优路径（次优但可用）
           - 超限且未找到路径 → 返回 None
        """

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

        best_path = None
        best_estimate = float('inf')
        expanded = 0

        while fwd_open or bwd_open:
            # ── 节点扩展上限保护 ──────────────────────────────────────────
            if expanded >= self.max_astar_nodes:
                self.logger.warning(
                    f'A* node limit reached ({expanded} nodes), '
                    f'returning {"best path found so far" if best_path is not None else "None"}'
                )
                return best_path

            # ── 标准双向 A* 最优终止条件 ─────────────────────────────────
            if fwd_open and bwd_open:
                fwd_f = fwd_open[0][0]
                bwd_f = bwd_open[0][0]
                if best_path is not None and fwd_f + bwd_f >= best_estimate:
                    break

            # ── 前向扩展 ─────────────────────────────────────────────────
            if fwd_open:
                fwd_f, fwd_current = heapq.heappop(fwd_open)
                if fwd_current in fwd_closed:
                    continue
                fwd_closed.add(fwd_current)
                expanded += 1

                if fwd_current in bwd_g:
                    total = fwd_g[fwd_current] + bwd_g[fwd_current]
                    if best_path is None or total < best_estimate:
                        best_estimate = total
                        best_path = self._reconstruct_bidirectional_path(
                            fwd_came_from, bwd_came_from, fwd_current
                        )

                # 找到最优路径后，前向 f 值已超出上界，跳过扩展（与后向对称）
                if best_path is not None and fwd_f >= best_estimate:
                    continue

                for move, move_cost in zip(moves, move_costs):
                    nx, ny = fwd_current[0] + move[0], fwd_current[1] + move[1]
                    if not self._is_astar_move_valid(fwd_current, move, nx, ny, width, height, check_map):
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

            # ── 后向扩展 ─────────────────────────────────────────────────
            if bwd_open:
                bwd_f, bwd_current = heapq.heappop(bwd_open)
                if bwd_current in bwd_closed:
                    continue
                bwd_closed.add(bwd_current)
                expanded += 1

                if bwd_current in fwd_g:
                    total = fwd_g[bwd_current] + bwd_g[bwd_current]
                    if best_path is None or total < best_estimate:
                        best_estimate = total
                        best_path = self._reconstruct_bidirectional_path(
                            fwd_came_from, bwd_came_from, bwd_current
                        )

                # 找到最优路径后，后向 f 值已超出上界，跳过扩展
                if best_path is not None and bwd_f >= best_estimate:
                    continue

                for move, move_cost in zip(moves, move_costs):
                    nx, ny = bwd_current[0] + move[0], bwd_current[1] + move[1]
                    if not self._is_astar_move_valid(bwd_current, move, nx, ny, width, height, check_map):
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

    def _unidirectional_astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        width: int,
        height: int,
        moves: List[Tuple[int, int]],
        move_costs: List[float],
        check_map: np.ndarray
    ) -> Optional[List[Tuple[int, int]]]:
        """标准单向 A* 实现"""
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        g_score = {start: 0.0}
        came_from = {start: None}
        closed = set()
        expanded = 0

        while open_set:
            if expanded >= self.max_astar_nodes:
                self.logger.warning(
                    f'A* node limit reached ({expanded} nodes), '
                    f'returning {"best path found so far" if g_score else "None"}'
                )
                path = []
                node = goal if goal in came_from else None
                if node is None and g_score:
                    node = min(g_score, key=lambda n: g_score[n])
                while node is not None:
                    path.append(node)
                    node = came_from.get(node)
                return path if path else None

            f, current = heapq.heappop(open_set)
            if current in closed:
                continue
            closed.add(current)
            expanded += 1

            if current == goal:
                result = []
                node = current
                while node is not None:
                    result.append(node)
                    node = came_from.get(node)
                result.reverse()
                return result

            for move, move_cost in zip(moves, move_costs):
                nx, ny = current[0] + move[0], current[1] + move[1]
                if not self._is_astar_move_valid(current, move, nx, ny, width, height, check_map):
                    continue
                neighbor = (nx, ny)
                if neighbor in closed:
                    continue

                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    h = self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (tentative_g + h, neighbor))

        return None

    def sparsify_path(self, path: List[Tuple[int, int]], max_cells: int) -> List[Tuple[int, int]]:
        """
        稀疏化路径：
        - 始终保留输入路径的起点和终点
        - 从起点开始，每隔 max_cells 个格子取一个点
        - 终点必须保留
        """
        if len(path) <= 2:
            return path

        max_cells = max(1, int(max_cells))
        sparse = [path[0]]

        for i in range(max_cells, len(path) - 1, max_cells):
            sparse.append(path[i])

        if sparse[-1] != path[-1]:
            sparse.append(path[-1])

        return sparse

    def publish_sparse_path_in_base_link(
        self,
        waypoints_map: List[dict],
        rviz_waypoints_map: Optional[List[dict]] = None,
        stamp: Optional[Time] = None
    ):
        """缓存最新规划路径，发布 map 可视化路径，并立即发布一次 base_link 路径。"""
        msg_stamp = stamp if stamp is not None else TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
        rviz_waypoints_map = rviz_waypoints_map if rviz_waypoints_map is not None else waypoints_map
        self.cache_sparse_path(waypoints_map, rviz_waypoints_map)
        self.publish_sparse_path_map(rviz_waypoints_map, msg_stamp)
        self.publish_cached_path_in_base_link()

        self.logger.info(
            f'Cached sparse path: {len(waypoints_map)} waypoints in map frame, '
            f'RViz map path: {len(rviz_waypoints_map)} waypoints'
        )

    def cache_sparse_path(self, waypoints_map: List[dict], rviz_waypoints_map: List[dict]):
        """保存最近一次 A* 输出的 map 坐标系路径，供高频发布器实时转换。"""
        with self._state_lock:
            self.cached_waypoints_map = [
                {'x': float(wp['x']), 'y': float(wp['y'])}
                for wp in waypoints_map
            ]
            self.cached_rviz_waypoints_map = [
                {'x': float(wp['x']), 'y': float(wp['y'])}
                for wp in rviz_waypoints_map
            ]

    def publish_sparse_path_map(self, rviz_waypoints_map: List[dict], stamp: Time):
        """发布 map 坐标系下的规划路径，主要供 RViz2 可视化。"""
        msg_map = Path()
        msg_map.header.stamp = stamp
        msg_map.header.frame_id = 'map'

        for wp in rviz_waypoints_map:
            pose_msg = PoseStamped()
            pose_msg.header = msg_map.header
            pose_msg.pose.position.x = float(wp['x'])
            pose_msg.pose.position.y = float(wp['y'])
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            msg_map.poses.append(pose_msg)

        self.path_map_pub.publish(msg_map)

    def publish_cached_path_in_base_link(self):
        """使用最新 map_pose，将缓存的 map 路径转换为 base_link 后发布给 controller。"""
        with self._state_lock:
            waypoints_map = [
                {'x': wp['x'], 'y': wp['y']}
                for wp in self.cached_waypoints_map
            ]

        if not waypoints_map:
            return

        current_nanos = TimeUtils.now_nanos()
        pose_frame = self.map_pose_queue.get_latest()
        if pose_frame is None:
            self.logger.warning('No valid robot pose, cannot publish planned path')
            return
        if current_nanos - pose_frame.stamp_nanos > self.map_pose_timeout * 1e9:
            pose_age = (current_nanos - pose_frame.stamp_nanos) / 1e9
            self.logger.warning(
                f'Robot pose timestamp expired, cannot publish planned path: '
                f'{pose_age:.3f}s > {self.map_pose_timeout:.3f}s'
            )
            return

        robot_x = float(pose_frame.data['x'])
        robot_y = float(pose_frame.data['y'])
        robot_yaw = float(pose_frame.data['yaw'])

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        msg = Path()
        msg.header.stamp = TimeUtils.nanos_to_stamp(current_nanos)
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
        if waypoints_map:
            target = waypoints_map[0]
            self.publish_target_marker(target['x'], target['y'], msg.header.stamp)
        else:
            self.publish_target_marker()
        self.logger.debug(f'Published cached path: {len(msg.poses)} waypoints in base_link frame')

    def publish_target_marker(
        self,
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
        stamp: Optional[Time] = None
    ):
        """发布当前 map 路径首点 Marker；无目标时删除 Marker。"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = stamp if stamp is not None else TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
        marker.ns = 'target_point'
        marker.id = 0

        if target_x is None or target_y is None:
            marker.action = Marker.DELETE
            self.target_marker_pub.publish(marker)
            return

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(target_x)
        marker.pose.position.y = float(target_y)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.target_marker_pub.publish(marker)

    def clear_cached_path(self):
        """清空缓存路径，避免高频发布器继续发送旧目标。"""
        with self._state_lock:
            self.cached_waypoints_map = []
            self.cached_rviz_waypoints_map = []

    def publish_empty_path(self):
        """发布空路径，并清空缓存路径。"""
        self.clear_cached_path()

        stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())

        msg_base = Path()
        msg_base.header.stamp = stamp
        msg_base.header.frame_id = 'base_link'
        self.path_pub.publish(msg_base)
        self.publish_target_marker(stamp=stamp)

        msg_map = Path()
        msg_map.header.stamp = stamp
        msg_map.header.frame_id = 'map'
        self.path_map_pub.publish(msg_map)

    # ==================== 航点到达检查定时器 ====================

    def _init_arrival_check_timer(self):
        """初始化航点到达检查定时器"""
        self.arrival_check_timer = self.create_timer(
            self.arrival_check_interval,
            self.arrival_check_callback,
            callback_group=self._planning_group
        )
        self.logger.info(f'航点到达检查定时器已启动，间隔: {self.arrival_check_interval}s')

    def _planning_timer_callback(self):
        """定时执行路径规划（由定时器驱动，不依赖 local_costmap）"""
        if self.map_data is None:
            self.logger.debug('_planning_timer_callback: map_data not ready')
            return
        if self.nav_map_points is None:
            self.logger.debug('_planning_timer_callback: nav_map_points not ready')
            return
        if self.is_task_completed():
            self.logger.debug('_planning_timer_callback: task completed')
            return

        pose_frame = self.map_pose_queue.get_latest()
        if pose_frame is None:
            self.logger.debug('_planning_timer_callback: map_pose is None')
            return

        # 触发规划（使用当前时间戳）
        self.plan_once(TimeUtils.nanos_to_stamp(TimeUtils.now_nanos()))

    def _path_publish_timer_callback(self):
        """高频发布缓存路径在当前 base_link 坐标系下的位置。"""
        if self.is_task_completed():
            return
        self.publish_cached_path_in_base_link()

    def arrival_check_callback(self):
        """定时检查航点是否到达，使用最新的 map_pose"""
        if self.is_task_completed():
            return

        if self.nav_map_points is None:
            return

        current_nanos = TimeUtils.now_nanos()
        pose_frame = self.map_pose_queue.get_latest()

        if pose_frame is None:
            return

        if current_nanos - pose_frame.stamp_nanos > self.map_pose_timeout * 1e9:
            self.logger.warning('arrival_check: Robot pose timestamp expired')
            return

        robot_x = float(pose_frame.data['x'])
        robot_y = float(pose_frame.data['y'])

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
            self.publish_empty_path()
            self.logger.info('All waypoints reached, task completed')

    def plan_once(self, local_costmap_stamp: Optional[Time] = None):
        """执行一次规划；直接使用最新的 map_pose。"""
        if self.map_data is None:
            self.logger.debug('plan_once: map not ready')
            return
        if self.map_metadata is None:
            self.logger.debug('plan_once: map_metadata not ready')
            return
        if self.inflation_enabled and self.inflated_map_data is None:
            self.logger.debug('plan_once: inflated_map_data not ready')
            return
        current_nanos = TimeUtils.now_nanos()
        pose_frame = self.map_pose_queue.get_latest()
        if pose_frame is None:
            self.logger.debug('plan_once: map_pose is None')
            return
        if current_nanos - pose_frame.stamp_nanos > self.map_pose_timeout * 1e9:
            self.logger.warning('Robot pose timestamp expired')
            return

        robot_x = float(pose_frame.data['x'])
        robot_y = float(pose_frame.data['y'])

        goal_point = self.nav_map_points[self.unreached_index]
        waypoint_x = float(goal_point['x'])
        waypoint_y = float(goal_point['y'])
        goal_map_x, goal_map_y, goal_source, projection_info = self._select_astar_goal_from_global_segment(
            robot_x, robot_y, goal_point
        )

        # 记录起点到终点的直线距离（世界坐标）
        straight_line_dist = math.hypot(goal_map_x - robot_x, goal_map_y - robot_y)
        projection_log = ''
        if projection_info is not None:
            proj_x, proj_y, proj_t, off_path_dist = projection_info
            projection_log = (
                f' current_waypoint=({waypoint_x:.3f},{waypoint_y:.3f})'
                f' segment_projection=({proj_x:.3f},{proj_y:.3f},t={proj_t:.3f})'
                f' off_path_dist={off_path_dist:.3f}m'
                f' rejoin_threshold={self.global_path_rejoin_threshold:.3f}m'
                f' waypoint_weight={self.global_path_rejoin_goal_weight:.3f}'
            )
        self.logger.info(
            f'[A* Planning] start=({robot_x:.3f},{robot_y:.3f}) '
            f'goal=({goal_map_x:.3f},{goal_map_y:.3f}) '
            f'goal_source={goal_source} '
            f'straight_line_dist={straight_line_dist:.3f}m'
            f'{projection_log}'
        )

        metadata = self.map_metadata
        planning_map = self.inflated_map_data.copy() if self.inflation_enabled else self.map_data.copy()

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
            self.publish_empty_path()
            return
        if goal_cell is None:
            self.logger.warning('No free goal cell found')
            self.publish_empty_path()
            return

        path = self.astar_planning(
            start=start_cell,
            goal=goal_cell,
            width=metadata.width,
            height=metadata.height,
            planning_map=planning_map
        )

        if not path or len(path) <= 1:
            self.logger.warning('No path found')
            self.publish_empty_path()
            return

        # 计算规划路径的实际长度（世界坐标米），兼容斜向移动。
        planned_path_length = 0.0
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            planned_path_length += math.hypot(x1 - x0, y1 - y0) * metadata.resolution
        self.logger.info(
            f'[A* Result] planned_path_length={planned_path_length:.3f}m '
            f'grid_points={len(path)} '
            f'path_efficiency={straight_line_dist / planned_path_length:.3f}'
        )

        max_cells = int(self.max_distance_between / metadata.resolution) if metadata.resolution > 0 else 1
        sparse_grid_path = self.sparsify_path(path, max_cells)
        if sparse_grid_path and sparse_grid_path[0] == path[0]:
            sparse_grid_path = sparse_grid_path[1:]

        waypoints_map = []
        for gx, gy in sparse_grid_path:
            wx, wy = self.grid_to_world(gx, gy, metadata)
            waypoints_map.append({'x': wx, 'y': wy})

        rviz_waypoints_map = []
        for gx, gy in path:
            wx, wy = self.grid_to_world(gx, gy, metadata)
            rviz_waypoints_map.append({'x': wx, 'y': wy})

        self.publish_sparse_path_in_base_link(waypoints_map, rviz_waypoints_map, local_costmap_stamp)




def run_map_planner_node(log_dir: str = None, log_timestamp: str = None, args=None):
    """运行节点

    Args:
        log_dir: 日志目录
        log_timestamp: 日志时间戳
        args: ROS 参数
    """
    rclpy.init(args=args)

    node = MapPlannerNode(log_dir=log_dir, log_timestamp=log_timestamp)

    # 使用多线程执行器，支持四个线程组并行执行
    # 线程分配：
    #   线程1: _data_group (gnss_path, map_pose)
    #   线程2: _costmap_group (local_costmap)
    #   线程3: _planning_group (arrival_check, planning_timer)
    #   线程4: _path_publish_group (path_publish_timer)
    num_threads = 4
    executor = MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)

    node.get_logger().info(f'MultiThreadedExecutor started with {num_threads} threads')

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
    run_map_planner_node()


if __name__ == '__main__':
    main()
