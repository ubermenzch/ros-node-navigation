#!/usr/bin/env python3
"""
地图节点 (map_node)

负责：
1. 室外模式：接收 anav_v3 下发的 GPS 点，根据机器狗 GPS + 路径点 GPS 生成地图
2. 订阅 /navigation/map_pose，计算并发布 map_pose（现在由 ekf_fusion_node 发布）
3. 根据局部 costmap、map_pose 和朝向更新地图
4. 计算导航点的地图坐标并发布
5. 发布地图到 topic 供可视化和其他节点使用

坐标系约定（与系统设计保持一致）：
- map坐标系：X轴正方向 = 正东方（East），Y轴正方向 = 正北方（North）
- local_costmap frame_id: base_link
"""

import tf2_ros
from tf2_ros import StaticTransformBroadcaster
import json
import threading
import time
import math
import numpy as np
import os
import logging
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid, Path
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, TransformStamped, Vector3
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix

# UTM 库
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    logging.warning("pyproj not installed, UTM conversion will not work")

from shared_map_storage import MapMetadata, get_shared_map
from config_loader import get_config
from frequency_stats import FrequencyStats


class MapNode(Node):
    """
    地图节点
    
    功能：
    1. 室外模式：接收 anav_v3 下发的 GPS 点，生成地图
    2. 根据定位数据计算 map_pose
    3. 根据局部 costmap、map_pose 和朝向更新地图
    4. 计算导航点的地图坐标并发布
    5. 发布地图供 rviz2 可视化和其他节点使用
    """

    def __init__(self, log_dir: str = None, timestamp: str = None):
        super().__init__('map_node')

        # 使用传入的日志目录和时间戳，或生成新的
        self.log_dir = log_dir
        self.timestamp = timestamp if timestamp is not None else datetime.now().strftime('%Y%m%d_%H%M%S')

        # 加载配置文件
        config = get_config()
        
        # 获取 map_node 配置
        map_node_config = config.get('map_node', {})
        planner_config = config.get('planner_node', {})

        # 获取公共配置
        common_config = config.get('common', {})
        
        # 地图生成参数
        self.square_size = map_node_config.get('square_size', 20.0)
        self.road_width = map_node_config.get('road_width', 1.0)
        self.resolution = common_config.get('resolution', 0.05)

        # 话题配置
        publications = map_node_config.get('publications', {})
        subscriptions = map_node_config.get('subscriptions', {})

        # 发布话题
        self.map_topic = publications.get('map_topic', '/map')
        self.map_update_topic = publications.get('map_update_topic', '/navigation/map_update')
        self.nav_map_points_topic = publications.get('nav_map_points_topic', '/navigation/nav_map_points')
        # map_pose 现在由 ekf_fusion_node 发布
        # self.map_pose_topic = publications.get('map_pose_topic', '/map_pose')

        # 订阅话题
        self.gps_path_topic = subscriptions.get('gps_path_topic', '/navigation_control')
        self.map_pose_topic = subscriptions.get('map_pose_topic', '/navigation/map_pose')
        self.local_costmap_topic = subscriptions.get('local_costmap_topic', '/navigation/local_costmap')

        # 状态锁
        self.pose_lock = threading.Lock()
        self.path_lock = threading.Lock()

        # 机器人 map_pose（来自 ekf_fusion_node）
        self.latest_map_pose = None
        self.last_map_pose_time = 0.0
        self.map_pose_timeout = map_node_config.get('map_pose_timeout', 1.0)  # 从配置读取
        self.map_pose_valid = False  # map_pose 是否有效（未超时）

        # 局部 costmap
        self.last_local_costmap_time = 0.0
        self.local_costmap_timestamp = None  # 保存 local_costmap 的时间戳用于增量更新
        self.local_costmap_timeout = map_node_config.get('local_costmap_timeout', 0.3)  # 从配置读取

        # 导航 GPS 点
        self.nav_gps_points = []
        self.batch_id = ""
        self.batch_counter = 0  # 组号，从 0 开始递增

        # 工作频率
        self.frequency = map_node_config.get('frequency', 10.0)

        # 首次发布标志
        self.first_map_published = False

        # 地图原点 GPS (持续发布)
        self.current_map_origin_lat = None
        self.current_map_origin_lon = None

        # UT M变换相关
        self.utm_zone: Optional[int] = None
        self.utm_transformer = None
        self.map_origin_utm = None  # {'easting': x, 'northing': y}
        self.map_origin_set = False

        # TF 监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 线程锁
        self.tf_lock = threading.Lock()

        # 创建发布者 - 地图
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            self.map_topic,
            10
        )

        # 创建发布者 - 地图增量更新
        self.map_update_pub = self.create_publisher(
            OccupancyGridUpdate,
            self.map_update_topic,
            10
        )

        # 创建发布者 - 导航地图坐标
        self.nav_map_points_pub = self.create_publisher(
            Path,
            self.nav_map_points_topic,
            10
        )

        # map_pose 和 map_origin_gps 现在由 ekf_fusion_node 发布
        # 不再在 map_node 中创建这些发布者

        # 创建订阅者 - anav_v3 下发的 GPS 路径
        self.gps_path_sub = self.create_subscription(
            String,
            self.gps_path_topic,
            self.gps_path_callback,
            10
        )

        # 创建订阅者 - 机器人 map_pose（来自 ekf_fusion_node）
        self.map_pose_sub = self.create_subscription(
            PoseStamped,
            self.map_pose_topic,
            self.map_pose_callback,
            10
        )

        # 创建订阅者 - 局部 costmap
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            self.local_costmap_topic,
            self.local_costmap_callback,
            10
        )

        # 定时器：按指定频率发布地图和 map_pose
        period = 1.0 / max(self.frequency, 1e-3)
        self.timer = self.create_timer(period, self.update)

        # 初始化日志（在订阅/发布创建之后）
        log_enabled = map_node_config.get('log_enabled', True)
        self._init_logger(log_enabled)

        # 初始化频率统计
        self.freq_stats = FrequencyStats(
            node_name='map_node',
            target_frequency=self.frequency,
            logger=self.logger,
            ros_logger=self.get_logger(),
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )

    def _init_logger(self, enabled: bool):
        """初始化日志系统"""
        # 使用传入的日志目录或创建新的
        if self.log_dir is not None:
            log_dir = self.log_dir
        else:
            ts = self.timestamp
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{ts}')
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'map_node_log_{self.timestamp}.log')

        self.logger = logging.getLogger('map_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        if enabled:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

        # 终端输出初始化信息（同时写入文件日志）
        init_info = [
            'Map Node initialized',
            f'  订阅 GPS 路径: {self.gps_path_topic}',
            f'  订阅 map_pose: {self.map_pose_topic}',
            f'  发布地图: {self.map_topic}',
            f'  发布地图更新: {self.map_update_topic}',
            f'  分辨率: {self.resolution}m',
            f'  工作频率: {self.frequency} Hz',
        ]

        for line in init_info:
            self.logger.info(line)  # 写入文件
            self.get_logger().info(line)  # 输出到终端

    def map_pose_callback(self, msg: PoseStamped):
        """
        接收 /navigation/map_pose（来自 ekf_fusion_node）

        消息格式: PoseStamped, frame_id=map
        包含机器人在地图坐标系下的位置和朝向
        """
        try:
            with self.pose_lock:
                self.latest_map_pose = {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'yaw': math.atan2(
                        2.0 * (msg.pose.orientation.w * msg.pose.orientation.z +
                               msg.pose.orientation.x * msg.pose.orientation.y),
                        1.0 - 2.0 * (msg.pose.orientation.y ** 2 + msg.pose.orientation.z ** 2)
                    ),
                    'valid': True
                }
                self.last_map_pose_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.map_pose_valid = True  # 收到新数据，标记为有效
        except Exception as e:
            self.logger.error(f'Failed to parse map_pose: {e}')

    def gps_path_callback(self, msg: String):
        """
        接收下发的导航规划 GPS 点

        消息格式 (std_msgs/String, JSON):
        {
            "action": 1,
            "mode": 1,
            "batchId": "batch_xxx",
            "points": [
                {"latitude": lat1, "longitude": lng1},
                {"latitude": lat2, "longitude": lng2},
                ...
            ]
        }
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.logger.error(f'Failed to parse GPS path message: {e}')
            return
        except Exception as e:
            self.logger.error(f'Error processing GPS path message: {e}')
            return

        points = data.get('points', [])
        if not points:
            self.logger.warning('Received empty GPS path message')
            return

        # 检查 map_pose 是否有效
        with self.pose_lock:
            if self.latest_map_pose is None or not self.latest_map_pose.get('valid', False):
                self.logger.warning('No valid map_pose yet, cannot generate map for new path')
                return

        # 检查 utm->map 变换是否可用
        trans_x, trans_y, yaw = self.get_utm_to_map_transform()
        if trans_x is None:
            self.logger.warning('utm->map transform not available yet, cannot generate map')
            return

        with self.path_lock:
            self.nav_gps_points = points
            self.batch_id = data.get('batchId', '')

        self.logger.info(f'Received new GPS path with {len(points)} points, batchId: {self.batch_id}')

        # 生成地图
        self.generate_and_store_map()

        # 计算并发布导航地图坐标
        self.compute_and_publish_nav_map_points()

    def _gps_to_relative_coords(self, waypoints):
        """将 GPS 坐标转换为相对坐标（米），以第一个点为原点

        坐标系（与系统设计一致）：
        - X轴正方向 = 正东方（East，经度方向）
        - Y轴正方向 = 正北方（North，纬度方向）
        """
        if not waypoints:
            return []

        origin_lon = float(waypoints[0]['longitude'])
        origin_lat = float(waypoints[0]['latitude'])

        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(origin_lat))

        map_points = []
        for wp in waypoints:
            lon = float(wp['longitude'])
            lat = float(wp['latitude'])

            delta_lon = lon - origin_lon
            delta_lat = lat - origin_lat

            # X轴=东方（由经度变化得到），Y轴=北方（由纬度变化得到）
            x = delta_lon * meters_per_degree_lon
            y = delta_lat * meters_per_degree_lat

            map_points.append((x, y))

        return map_points

    def _interpolate_centerline(self, points, num_points_per_segment: int = 10):
        """使用线性插值生成中心线"""
        if len(points) < 2:
            return points

        interpolated = [points[0]]

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx * dx + dy * dy)

            n = max(2, int(dist / self.resolution) + 1)
            n = min(n, num_points_per_segment * 3)

            for j in range(1, n):
                t = j / n
                x = p1[0] + t * dx
                y = p1[1] + t * dy
                interpolated.append((x, y))

        if interpolated[-1] != points[-1]:
            interpolated.append(points[-1])

        return interpolated

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

    def get_utm_to_map_transform(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        获取 map -> utm 变换（即 map 原点在 utm 坐标系中的位置）

        注意：ekf_fusion_node 发布的是 utm -> map TF，父 utm，子 map
        所以我们要查 'utm' -> 'map' 的变换，得到 map 原点在 utm 中的位置

        Returns:
            (trans_x, trans_y, rotation) - map 原点在 utm 坐标系中的位置和旋转
            如果变换不可用则返回 (None, None, None)
        """
        with self.tf_lock:
            try:
                # 查 'utm' -> 'map'，得到 map 原点在 utm 中的位置
                # 这与 ekf_fusion_node 发布 utm->map 的语义一致
                transform = self.tf_buffer.lookup_transform(
                    'utm',
                    'map',
                    rclpy.time.Time(),
                    timeout=Duration(seconds=1.0)
                )

                trans = transform.transform.translation
                rot = transform.transform.rotation

                # 从四元数获取旋转角度
                yaw = math.atan2(
                    2.0 * (rot.w * rot.z + rot.x * rot.y),
                    1.0 - 2.0 * (rot.y ** 2 + rot.z ** 2)
                )

                # trans 是 map 原点在 utm 中的位置
                return (trans.x, trans.y, yaw)

            except tf2_ros.LookupException as e:
                self.logger.warning(f'TF lookup failed (utm<-map not available yet): {e}')
                return (None, None, None)
            except tf2_ros.ConnectivityException as e:
                self.logger.warning(f'TF connectivity error: {e}')
                return (None, None, None)
            except tf2_ros.ExtrapolationException as e:
                self.logger.warning(f'TF extrapolation error: {e}')
                return (None, None, None)
            except Exception as e:
                self.logger.error(f'Failed to get utm<-map transform: {e}')
                return (None, None, None)

    def gps_to_map_coords(self, lat: float, lon: float) -> Tuple[Optional[float], Optional[float]]:
        """
        将 GPS 坐标转换为 map 坐标系下的坐标

        转换流程：GPS -> UTM -> map
        - GPS -> UTM: 使用 pyproj 将经纬度转换为 UTM 坐标
        - UTM -> map: 使用 tf2 查询 utm<-map 变换，得到 map 原点在 utm 中的位置

        ekf_fusion_node 发布的是 utm -> map TF（父 utm，子 map），
        所以我们查 'utm' -> 'map' 得到 map 原点在 utm 中的位置 (trans_x, trans_y)
        变换公式：map_point = utm_point - map_origin_utm

        Returns:
            (x, y) - map 坐标系下的坐标，如果转换失败返回 (None, None)
        """
        # GPS -> UTM
        utm_x, utm_y = self.gps_to_utm(lat, lon)
        if utm_x is None or utm_y is None:
            return (None, None)

        # UTM -> map: 查询 'utm' -> 'map' 得到 map 原点在 utm 中的位置
        # ekf_fusion_node 发布 utm -> map (父 utm, 子 map)，平移为 map 原点在 utm 中的坐标
        trans_x, trans_y, yaw = self.get_utm_to_map_transform()
        if trans_x is None:
            return (None, None)

        # 应用变换：map_point = utm_point - map_origin_utm
        map_x = utm_x - trans_x
        map_y = utm_y - trans_y

        return (map_x, map_y)

    def _compute_road_mask(self, centerline_points, origin_x: float, origin_y: float, grid_size: int):
        """计算道路区域掩码

        坐标系（与系统设计一致）：
        - X轴正方向 = 正东方（East）
        - Y轴正方向 = 正北方（North）
        """
        mask = np.zeros((grid_size, grid_size), dtype=bool)

        if len(centerline_points) < 2:
            return mask

        half_interval = self.road_width / 2.0

        for i in range(len(centerline_points) - 1):
            p1 = centerline_points[i]
            p2 = centerline_points[i + 1]

            dx = p2[0] - p1[0]  # 东方方向增量 (X轴)
            dy = p2[1] - p1[1]  # 北方方向增量 (Y轴)
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1e-6:
                continue

            ux = dx / dist  # 道路方向单位向量（东）
            uy = dy / dist  # 道路方向单位向量（北）

            # 垂直方向：北向（90度旋转）
            px = -uy
            py = ux

            steps = max(1, int(dist / self.resolution))
            for j in range(steps + 1):
                t = j / steps
                cx = p1[0] + t * dx  # 中心点 X（东）
                cy = p1[1] + t * dy  # 中心点 Y（北）

                for dx_offset in [-half_interval, half_interval]:
                    for dy_offset in [-half_interval, half_interval]:
                        # 沿道路方向(dx_offset)和垂直方向(dy_offset)扩展
                        wx = cx + ux * dx_offset + px * dy_offset
                        wy = cy + uy * dx_offset + py * dy_offset

                        # X轴=东方→col增加，Y轴=北方→row减少
                        col = int((wx - origin_x) / self.resolution)
                        # 物理坐标 y（北方）到数组行号的转换：
                        # origin_y 是地图上边界（y 最大），numpy row=0 在顶部
                        # row = 0 对应物理 y = origin_y
                        # row = grid_size-1 对应物理 y = origin_y - grid_size*resolution
                        row = int((origin_y - wy) / self.resolution)

                        if 0 <= col < grid_size and 0 <= row < grid_size:
                            mask[row, col] = True

        return mask

    def generate_and_store_map(self):
        """根据导航 GPS 点生成地图，通过 utm->map TF 将 GPS 航点转换为 map 坐标"""
        # 检查 map_pose 是否有效（未超时）
        if not self._check_map_pose_timeout():
            self.logger.warning('map_pose invalid or timeout, cannot generate map')
            return

        with self.path_lock:
            if not self.nav_gps_points:
                self.logger.warning('No navigation GPS points, cannot generate map')
                return

        self.logger.info('Generating map from GPS path...')

        # 检查 utm<-map 变换是否可用
        trans_x, trans_y, yaw = self.get_utm_to_map_transform()
        if trans_x is None:
            self.logger.warning('utm<-map transform not available yet, cannot generate map')
            return

        self.logger.info(f'utm->map transform: translation=({trans_x:.2f}, {trans_y:.2f}), yaw={math.degrees(yaw):.1f} deg')

        # 将所有航点从 GPS 转换为 map 坐标
        map_points = []
        for wp in self.nav_gps_points:
            lat = float(wp['latitude'])
            lon = float(wp['longitude'])
            map_x, map_y = self.gps_to_map_coords(lat, lon)
            if map_x is None:
                self.logger.warning(f'Failed to convert GPS ({lat}, {lon}) to map coordinates')
                continue
            map_points.append((map_x, map_y))

        if not map_points:
            self.logger.error('No valid map points after conversion')
            return

        self.logger.info(f'Converted {len(map_points)} GPS points to map coordinates')

        # 获取当前 map_pose（机器人在 map 坐标系中的位置）
        with self.pose_lock:
            if self.latest_map_pose is not None and self.latest_map_pose.get('valid', False):
                robot_map_x = self.latest_map_pose.get('x', 0.0)
                robot_map_y = self.latest_map_pose.get('y', 0.0)
                robot_yaw = self.latest_map_pose.get('yaw', 0.0)
                self.logger.info(f'Robot map pose: ({robot_map_x:.2f}, {robot_map_y:.2f}), yaw={math.degrees(robot_yaw):.1f} deg')
            else:
                robot_map_x = 0.0
                robot_map_y = 0.0
                robot_yaw = 0.0
                self.logger.warning('No valid map_pose, using (0, 0) as robot position')

        # 将机器人位置也加入地图点，用于确定地图范围
        all_points = map_points + [(robot_map_x, robot_map_y)]

        # 计算地图范围
        max_x = max(p[0] for p in all_points)
        min_x = min(p[0] for p in all_points)
        max_y = max(p[1] for p in all_points)
        min_y = min(p[1] for p in all_points)

        map_width = max_x - min_x
        map_height = max_y - min_y

        padding = self.square_size
        map_width += 2 * padding
        map_height += 2 * padding

        grid_size_x = int(map_width / self.resolution)
        grid_size_y = int(map_height / self.resolution)
        grid_size = max(grid_size_x, grid_size_y)

        # 地图原点（左上角）
        origin_x = min_x - padding
        origin_y = max_y + padding

        # 保存地图原点 GPS（用于逆向计算）
        lat_origin = float(self.nav_gps_points[0]['latitude'])
        lon_origin = float(self.nav_gps_points[0]['longitude'])
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(lat_origin))

        self.logger.info(f'Map range: width={map_width:.2f}m, height={map_height:.2f}m')
        self.logger.info(f'Grid size: {grid_size}x{grid_size}')
        self.logger.info(f'Origin: ({origin_x:.2f}, {origin_y:.2f})')

        grid = np.zeros((grid_size, grid_size), dtype=np.int8)

        # 使用航点的地图坐标生成道路掩码
        # 将机器人当前位置插入到航点序列前面，形成完整路径
        full_path = [(robot_map_x, robot_map_y)] + map_points
        centerline = self._interpolate_centerline(full_path)
        road_mask = self._compute_road_mask(centerline, origin_x, origin_y, grid_size)

        grid[road_mask] = 0
        grid[~road_mask] = 100

        metadata = MapMetadata(
            resolution=self.resolution,
            width=grid_size,
            height=grid_size,
            origin_x=origin_x,
            origin_y=origin_y - grid_size * self.resolution,
            robot_x=robot_map_x - origin_x,
            robot_y=origin_y + grid_size * self.resolution - robot_map_y,
            gps_points=self.nav_gps_points.copy(),
            origin_lat=lat_origin,
            origin_lon=lon_origin,
            meters_per_degree_lat=meters_per_degree_lat,
            meters_per_degree_lon=meters_per_degree_lon,
            odom_offset_x=-robot_map_x,
            odom_offset_y=-robot_map_y,
            odom_offset_yaw=0.0,
        )

        shared_map = get_shared_map()
        shared_map.set_map(grid, metadata)

        # 新地图生成后需要重新发完整的 OccupancyGrid，而非增量更新
        self.first_map_published = False

        self.logger.info(
            f'Map generated and stored: {grid_size}x{grid_size}, resolution={self.resolution}m'
        )

    def compute_and_publish_nav_map_points(self):
        """计算导航点的地图坐标并发布

        使用与 generate_and_store_map() 相同的 GPS->UTM->map 转换逻辑，
        确保航点坐标与地图路径一致。
        """
        # 检查 utm->map 变换是否可用
        trans_x, trans_y, yaw = self.get_utm_to_map_transform()
        if trans_x is None:
            self.logger.warning('utm->map transform not available, cannot compute nav map points')
            return

        with self.path_lock:
            if not self.nav_gps_points:
                return

            nav_map_points = []
            for wp in self.nav_gps_points:
                lat = float(wp['latitude'])
                lon = float(wp['longitude'])
                # 使用与 generate_and_store_map() 相同的转换逻辑
                map_x, map_y = self.gps_to_map_coords(lat, lon)
                if map_x is None:
                    self.logger.warning(f'Failed to convert GPS ({lat}, {lon}) to map coordinates')
                    continue
                nav_map_points.append({'x': map_x, 'y': map_y})

            if not nav_map_points:
                self.logger.warning('Failed to compute any nav map points')
                return

            # 发布导航地图坐标
            msg = Path()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'
            
            for point in nav_map_points:
                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = 'map'
                pose.pose.position.x = point['x']
                pose.pose.position.y = point['y']
                pose.pose.position.z = 0.0
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = 0.0
                pose.pose.orientation.w = 1.0
                msg.poses.append(pose)
            
            self.nav_map_points_pub.publish(msg)

            self.logger.info(f'Published {len(nav_map_points)} nav map points, batch_number: {self.batch_counter}')

            # 递增组号
            self.batch_counter += 1

    def local_costmap_callback(self, msg: OccupancyGrid):
        """局部 costmap 回调，使用 map_pose 更新地图"""
        # 记录 local_costmap 接收时间和时间戳
        self.last_local_costmap_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.local_costmap_timestamp = msg.header.stamp

        # 检查 local_costmap 是否超时
        if not self._check_local_costmap_timeout():
            return

        # 检查 map_pose 是否有效（未超时）
        if not self._check_map_pose_timeout():
            return

        try:
            # 从 OccupancyGrid 获取 costmap 数据
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution

            # OccupancyGrid 数据是 0-100 的整数，-1 表示未知
            costmap_data = np.array(msg.data, dtype=np.int8)
            # 注意：这里保留 0-100 范围，因为 shared_map.update_local_region() 期望 0-100

            # 翻转 y 轴 (恢复原始方向)
            costmap = costmap_data.reshape((height, width))
            costmap = np.flip(costmap, axis=0)

            # 使用计算出的 map_pose
            with self.pose_lock:
                if self.latest_map_pose is None:
                    return
                pose = self.latest_map_pose

                if not pose.get('valid', False):
                    return

                robot_x = pose.get('x', 0.0)
                robot_y = pose.get('y', 0.0)
                robot_yaw = pose.get('yaw', 0.0)
            
            self._update_map(costmap, robot_x, robot_y, robot_yaw, resolution)
            
        except Exception as e:
            self.logger.error(f'Failed to process local costmap: {e}')

    def _update_map(self, local_costmap: np.ndarray, robot_x: float, robot_y: float, 
                   robot_yaw: float, resolution: float):
        """更新全局地图"""
        shared_map = get_shared_map()
        
        if not shared_map.has_map():
            return
        
        rotated_costmap = self._rotate_costmap(local_costmap, -robot_yaw)
        
        success = shared_map.update_local_region(
            rotated_costmap,
            robot_x, robot_y,
            resolution,
            100
        )
        
        if success:
            self.logger.debug(f'Updated map at ({robot_x:.2f}, {robot_y:.2f})')

    def _rotate_costmap(self, costmap: np.ndarray, angle: float) -> np.ndarray:
        """旋转 costmap（输入为 0-100 范围的整数）"""
        height, width = costmap.shape

        robot_grid_x = width // 2
        robot_grid_y = height - 1

        rotated = np.zeros_like(costmap)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # 使用 50 作为障碍物阈值（costmap 0-100 范围）
        obstacle_threshold = 50

        for y in range(height):
            for x in range(width):
                if costmap[y, x] >= obstacle_threshold:
                    dx = x - robot_grid_x
                    dy = robot_grid_y - y

                    new_dx = dx * cos_a - dy * sin_a
                    new_dy = dx * sin_a + dy * cos_a

                    new_x = int(new_dx + robot_grid_x)
                    new_y = int(robot_grid_y - new_dy)

                    if 0 <= new_x < width and 0 <= new_y < height:
                        rotated[new_y, new_x] = costmap[y, x]

        return rotated

    def publish_map(self):
        """发布地图到 topic"""
        shared_map = get_shared_map()
        map_data, metadata = shared_map.get_map()

        if map_data is None or metadata is None:
            return

        if not self.first_map_published:
            # 首次发布完整的 OccupancyGrid
            grid_msg = OccupancyGrid()

            grid_msg.header.stamp = self.get_clock().now().to_msg()
            grid_msg.header.frame_id = 'map'

            grid_msg.info.resolution = float(metadata.resolution)
            grid_msg.info.width = metadata.width
            grid_msg.info.height = metadata.height

            origin_pose = Pose()
            origin_pose.position.x = metadata.origin_x
            origin_pose.position.y = metadata.origin_y
            origin_pose.position.z = 0.0
            origin_pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            grid_msg.info.origin = origin_pose

            grid_data = np.zeros((metadata.height, metadata.width), dtype=np.int8)
            # 坐标系转换：
            # 内部：row=0 对应 y=origin_y（顶部），row 增加 y 减小
            # OccupancyGrid：y=0 在底部，y 增加向上
            # 需要翻转：OccupancyGrid[row] = 内部[height-1-row]
            for row in range(metadata.height):
                for col in range(metadata.width):
                    # 翻转 row：底部 row=max -> OccupancyGrid row=0
                    grid_data[row, col] = map_data[metadata.height - 1 - row, col]

            grid_msg.data = grid_data.flatten().tolist()

            self.map_pub.publish(grid_msg)
            self.first_map_published = True
            self.logger.info(f'Published full map: {metadata.width}x{metadata.height}')
        else:
            # 后续发布增量更新 OccupancyGridUpdate
            self.publish_map_update()

    def publish_map_update(self):
        """发布地图增量更新到 topic"""
        shared_map = get_shared_map()
        bbox = shared_map.get_last_update_bbox()

        if bbox is None:
            return

        map_data, metadata = shared_map.get_map()
        if map_data is None or metadata is None:
            return

        min_col, min_row, max_col, max_row = bbox
        width = max_col - min_col + 1
        height = max_row - min_row + 1

        # 构建更新消息
        # 坐标系说明：
        # - 内部：origin_y 是地图上边界（y 最大），row=0 对应 y=origin_y，row 增加 y 减小
        # - OccupancyGrid：y=0 在底部，y 增加向上
        # 转换：OccupancyGrid_y = (metadata.height - 1) - internal_row
        update_msg = OccupancyGridUpdate()
        # 使用 local_costmap 的时间戳，而不是发布时的时间戳
        if self.local_costmap_timestamp is not None:
            update_msg.header.stamp = self.local_costmap_timestamp
        else:
            update_msg.header.stamp = self.get_clock().now().to_msg()
        update_msg.header.frame_id = 'map'
        update_msg.x = min_col
        update_msg.y = (metadata.height - 1) - max_row   # 内部底部 row -> OccupancyGrid 底部 y=0
        update_msg.width = width
        update_msg.height = height

        # data 按 OccupancyGrid 顺序排列：从底行到顶行
        # 内部 row=max（底部）对应 OccupancyGrid y=0
        update_data = []
        for internal_row in range(max_row, min_row - 1, -1):
            for col in range(min_col, max_col + 1):
                if 0 <= internal_row < metadata.height and 0 <= col < metadata.width:
                    update_data.append(int(map_data[internal_row, col]))
                else:
                    update_data.append(-1)

        update_msg.data = update_data

        self.map_update_pub.publish(update_msg)
        shared_map.clear_last_update_bbox()

        self.logger.debug(f'Published map update: x={update_msg.x}, y={update_msg.y}, {width}x{height}')

    def _check_map_pose_timeout(self) -> bool:
        """检查 map_pose 是否超时，并更新有效状态标志"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_map_pose_time > 0:
            elapsed = current_time - self.last_map_pose_time
            if elapsed > self.map_pose_timeout:
                self.logger.warning(f'map_pose timeout: {elapsed:.2f}s > {self.map_pose_timeout}s')
                self.map_pose_valid = False
                return False
        return True

    def _check_local_costmap_timeout(self) -> bool:
        """检查 local_costmap 是否超时"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_local_costmap_time > 0:
            elapsed = current_time - self.last_local_costmap_time
            if elapsed > self.local_costmap_timeout:
                self.logger.warning(f'local_costmap timeout: {elapsed:.2f}s > {self.local_costmap_timeout}s')
                return False
        return True

    def update(self):
        """执行一次更新"""
        # 记录频率统计
        self.freq_stats.tick()

        # 检查 map_pose 是否有效（未超时）
        pose_valid = self._check_map_pose_timeout()

        # 只有在 map_pose 有效时才发布地图
        # 注意：map->odom TF 现在由 ekf_fusion_node 统一发布
        if pose_valid:
            # 发布地图
            self.publish_map()


def main(args=None):
    rclpy.init(args=args)

    node = MapNode()

    # 注意：定时器已在 MapNode.__init__ 中创建，这里不再重复创建

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
