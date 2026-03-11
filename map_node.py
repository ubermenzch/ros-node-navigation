#!/usr/bin/env python3
"""
地图节点 (map_node)

负责：
1. 室外模式：接收 anav_v3 下发的 GPS 点，根据机器狗 GPS + 路径点 GPS 生成地图
2. 订阅 /fusion_pose 和 /rtk_imu，计算并发布 map_pose
3. 根据局部 costmap、map_pose 和朝向更新地图
4. 计算导航点的地图坐标并发布
5. 发布地图到 topic 供可视化和其他节点使用

坐标系约定：
室外模式：Y轴正方向 = 正北方，X轴正方向 = 正东方
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import Pose, Quaternion, TransformStamped
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import String
import tf2_ros
import json
import threading
import time
import math
import numpy as np
import os
import logging
from datetime import datetime

from shared_map_storage import MapMetadata, get_shared_map
from config_loader import get_config


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

    def __init__(self):
        super().__init__('map_node')

        # 加载配置文件
        config = get_config()
        
        # 获取 map_node 配置
        map_node_config = config.get('map_node', {})
        planner_config = config.get('planner_node', {})

        # 地图生成参数
        self.square_size = planner_config.get('square_size', 20.0)
        self.square_interval = planner_config.get('square_interval', 1.0)
        self.resolution = planner_config.get('resolution', 0.1)

        # 话题配置
        publications = map_node_config.get('publications', {})
        subscriptions = map_node_config.get('subscriptions', {})

        # 发布话题
        self.map_topic = publications.get('map_topic', '/map')
        self.map_update_topic = publications.get('map_update_topic', '/map_update')
        self.nav_map_points_topic = publications.get('nav_map_points_topic', '/nav_map_points')
        self.map_pose_topic = publications.get('map_pose_topic', '/map_pose')

        # 订阅话题
        self.gps_path_topic = subscriptions.get('gps_path_topic', '/gps_path')
        self.robot_gps_topic = subscriptions.get('robot_gps_topic', '/rtk_fix')
        self.fusion_pose_topic = subscriptions.get('fusion_pose_topic', '/fusion_pose')
        self.rtk_imu_topic = subscriptions.get('rtk_imu_topic', '/rtk_imu')
        self.local_costmap_topic = subscriptions.get('local_costmap_topic', '/local_costmap')

        # 状态锁
        self.pose_lock = threading.Lock()
        self.path_lock = threading.Lock()

        # 融合定位（odom坐标系）
        self.latest_fusion_pose = None

        # RTK IMU 朝向（绝对方向：朝东0度，朝北90度）
        self.latest_imu_yaw = None
        self.last_imu_time = 0.0
        self.imu_timeout = 1.0  # 秒

        # 计算出的 map_pose
        self.latest_map_pose = None

        # 机器人 GPS
        self.latest_robot_gps = None

        # 导航 GPS 点
        self.nav_gps_points = []
        self.batch_id = ""
        self.batch_counter = 0  # 组号，从 0 开始递增

        # 地图更新频率
        self.update_frequency = map_node_config.get('update_frequency', 10.0)

        # 首次发布标志
        self.first_map_published = False

        # 地图原点 GPS (持续发布)
        self.current_map_origin_lat = None
        self.current_map_origin_lon = None

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
            String,
            self.nav_map_points_topic,
            10
        )
        
        # 创建发布者 - map_pose
        self.map_pose_pub = self.create_publisher(
            String,
            self.map_pose_topic,
            10
        )

        # 创建发布者 - 地图原点GPS（供 tf_publisher 计算 map->odom TF）
        self.map_origin_pub = self.create_publisher(
            String,
            '/navigation/map_origin_gps',
            10
        )
        
        # 创建订阅者 - anav_v3 下发的 GPS 路径
        self.gps_path_sub = self.create_subscription(
            String,
            self.gps_path_topic,
            self.gps_path_callback,
            10
        )
        
        # 创建订阅者 - 机器人 GPS
        self.robot_gps_sub = self.create_subscription(
            NavSatFix,
            self.robot_gps_topic,
            self.robot_gps_callback,
            10
        )
        
        # 创建订阅者 - 融合定位（odom坐标系）
        self.fusion_pose_sub = self.create_subscription(
            String,
            self.fusion_pose_topic,
            self.fusion_pose_callback,
            10
        )
        
        # 创建订阅者 - RTK IMU（绝对朝向）
        self.rtk_imu_sub = self.create_subscription(
            Imu,
            self.rtk_imu_topic,
            self.rtk_imu_callback,
            10
        )
        
        # 创建订阅者 - 局部 costmap
        self.local_costmap_sub = self.create_subscription(
            String,
            self.local_costmap_topic,
            self.local_costmap_callback,
            10
        )

        # 定时器：按指定频率发布地图和 map_pose
        period = 1.0 / max(self.update_frequency, 1e-3)
        self.timer = self.create_timer(period, self.update)

        # 初始化日志（在订阅/发布创建之后）
        log_enabled = map_node_config.get('log_enabled', True)
        self._init_logger(log_enabled)

    def _init_logger(self, enabled: bool):
        """初始化日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'map_node_log_{timestamp}.log')

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
            'Running in outdoor mode',
            f'  订阅 GPS 路径: {self.gps_path_topic}',
            f'  订阅机器人 GPS: {self.robot_gps_topic}',
            f'  订阅融合定位: {self.fusion_pose_topic}',
            f'  发布地图: {self.map_topic}',
            f'  发布地图 pose: {self.map_pose_topic}',
            f'  分辨率: {self.resolution}m',
            f'  更新频率: {self.update_frequency} Hz',
        ]

        for line in init_info:
            self.logger.info(line)  # 写入文件
            self.get_logger().info(line)  # 输出到终端

    def robot_gps_callback(self, msg: NavSatFix):
        """接收机器人 GPS"""
        self.latest_robot_gps = msg

    def fusion_pose_callback(self, msg: String):
        """
        接收融合定位 /fusion_pose（odom坐标系）
        
        消息格式 (JSON):
        {
            "x": 1.0,
            "y": 2.0,
            "yaw": 0.5,
            "vx": 0.1,
            "vy": 0.2,
            "vyaw": 0.01,
            "timestamp": 1234567890.123,
            "valid": true,
            "fusion_mode": "gps"
        }
        """
        try:
            data = json.loads(msg.data)
            with self.pose_lock:
                self.latest_fusion_pose = data
        except Exception as e:
            self.logger.error(f'Failed to parse fusion pose: {e}')

    def rtk_imu_callback(self, msg: Imu):
        """
        接收 RTK IMU 数据，获取绝对朝向（室外模式使用）
        
        /rtk_imu 提供的朝向：朝东0度，朝北90度（地球尺度绝对方向）
        注意：这个朝向与地图坐标系一致，因为地图Y轴正方向=正北方
        """
        try:
            # 从四元数提取 yaw
            q = msg.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            self.latest_imu_yaw = yaw
            self.last_imu_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception as e:
            self.logger.error(f'Failed to parse RTK IMU: {e}')

    def compute_and_publish_map_pose(self):
        """根据定位数据计算 map_pose 并发布"""
        # 室外模式：使用 fusion_pose + rtk_imu
        self._compute_map_pose_outdoor()

    def _compute_map_pose_outdoor(self):
        """室外模式：根据 fusion_pose 和 rtk_imu 计算 map_pose"""
        try:
            with self.pose_lock:
                if self.latest_fusion_pose is None:
                    return
                
                fusion_pose = self.latest_fusion_pose
                
                # 检查融合定位是否有效
                if not fusion_pose.get('valid', False):
                    return
                
                # 检查 IMU 是否超时
                current_time = self.get_clock().now().nanoseconds / 1e9
                if current_time - self.last_imu_time > self.imu_timeout:
                    self.logger.warning('IMU data timeout')
                    return
                
                # 获取 odom 坐标系位置
                odom_x = fusion_pose.get('x', 0.0)
                odom_y = fusion_pose.get('y', 0.0)
                
                # 获取绝对朝向（与地图坐标系一致）
                imu_yaw = self.latest_imu_yaw
                
                # 转换为地图坐标系
                shared_map = get_shared_map()
                if not shared_map.has_map():
                    return
                
                map_x, map_y = shared_map.odom_to_map(odom_x, odom_y)
                if map_x is None:
                    return
                
                # 构建 map_pose
                map_pose = {
                    'x': map_x,
                    'y': map_y,
                    'yaw': imu_yaw,  # 直接使用 IMU 的绝对朝向
                    'vx': fusion_pose.get('vx', 0.0),
                    'vy': fusion_pose.get('vy', 0.0),
                    'vyaw': fusion_pose.get('vyaw', 0.0),
                    'timestamp': fusion_pose.get('timestamp', current_time),
                    'valid': True
                }
                
                self.latest_map_pose = map_pose
                
                # 发布 map_pose
                msg = String()
                msg.data = json.dumps(map_pose)
                self.map_pose_pub.publish(msg)
                
                self.logger.debug(f'Published outdoor map_pose: ({map_x:.2f}, {map_y:.2f}), yaw={math.degrees(imu_yaw):.1f}deg')
                
        except Exception as e:
            self.logger.error(f'Failed to compute map_pose: {e}')

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

        if self.latest_robot_gps is None:
            self.logger.warning('No robot GPS yet, cannot generate map for new path')
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
        """将 GPS 坐标转换为相对坐标（米），以第一个点为原点"""
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

    def _compute_road_mask(self, centerline_points, origin_x: float, origin_y: float, grid_size: int):
        """计算道路区域掩码"""
        mask = np.zeros((grid_size, grid_size), dtype=bool)

        if len(centerline_points) < 2:
            return mask

        half_interval = self.square_interval / 2.0

        for i in range(len(centerline_points) - 1):
            p1 = centerline_points[i]
            p2 = centerline_points[i + 1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1e-6:
                continue

            ux = dx / dist
            uy = dy / dist

            px = -uy
            py = ux

            steps = max(1, int(dist / self.resolution))
            for j in range(steps + 1):
                t = j / steps
                cx = p1[0] + t * dx
                cy = p1[1] + t * dy

                for dx_offset in [-half_interval, half_interval]:
                    for dy_offset in [-half_interval, half_interval]:
                        wx = cx + px * dx_offset + ux * dy_offset
                        wy = cy + py * dy_offset + uy * dy_offset

                        col = int((wx - origin_x) / self.resolution)
                        row = int((origin_y + grid_size * self.resolution - wy) / self.resolution)

                        if 0 <= col < grid_size and 0 <= row < grid_size:
                            mask[row, col] = True

        return mask

    def generate_and_store_map(self):
        """根据 机器人GPS + 导航GPS点 生成地图"""
        if self.latest_robot_gps is None:
            self.logger.warning('No robot GPS position, cannot generate map')
            return

        with self.path_lock:
            if not self.nav_gps_points:
                self.logger.warning('No navigation GPS points, cannot generate map')
                return

            all_points = [{
                'latitude': float(self.latest_robot_gps.latitude),
                'longitude': float(self.latest_robot_gps.longitude),
            }]
            all_points.extend(self.nav_gps_points)

        self.logger.info('Generating map from GPS path...')

        # 记录 odom 坐标系的原点位置（机器人创建地图时的位置）
        # fusion_pose 中的 x,y 就是相对于 odom 原点的坐标
        with self.pose_lock:
            if self.latest_fusion_pose is not None:
                odom_origin_x = self.latest_fusion_pose.get('x', 0.0)
                odom_origin_y = self.latest_fusion_pose.get('y', 0.0)
            else:
                odom_origin_x = 0.0
                odom_origin_y = 0.0

        # 计算地图范围
        map_points = self._gps_to_relative_coords(all_points)
        robot_x, robot_y = map_points[0]  # 这是 GPS 原点（相对于机器人当前位置的 GPS 坐标）

        max_x_distance = 0.0
        max_y_distance = 0.0

        for i in range(len(map_points)):
            for j in range(i + 1, len(map_points)):
                p1 = map_points[i]
                p2 = map_points[j]

                x_dist = abs(p1[0] - p2[0])
                y_dist = abs(p1[1] - p2[1])

                if x_dist > max_x_distance:
                    max_x_distance = x_dist

                if y_dist > max_y_distance:
                    max_y_distance = y_dist

        padding = self.square_size
        map_width = max_x_distance + 2 * padding
        map_height = max_y_distance + 2 * padding

        grid_size_x = int(map_width / self.resolution)
        grid_size_y = int(map_height / self.resolution)
        grid_size = max(grid_size_x, grid_size_y)

        origin_x = robot_x - padding
        origin_y = robot_y + map_height - padding

        # 计算地图中心坐标（相对于 GPS 原点）
        map_center_x = origin_x + grid_size * self.resolution / 2.0
        map_center_y = origin_y + grid_size * self.resolution / 2.0

        # 计算 map 原点相对于 odom 原点的偏移
        # odom 原点 = 机器人当前位置 (odom_origin_x, odom_origin_y)
        # map 原点 = 地图中心 (map_center_x, map_center_y)
        # 注意：这里假设地图坐标系和 odom 坐标系对齐（都是 ENU）
        self.odom_offset_x = odom_origin_x - map_center_x
        self.odom_offset_y = odom_origin_y - map_center_y
        self.odom_offset_yaw = 0.0  # 假设地图和 odom 坐标系的朝向一致

        self.logger.info(f'odom origin: ({odom_origin_x:.2f}, {odom_origin_y:.2f})')
        self.logger.info(f'map center: ({map_center_x:.2f}, {map_center_y:.2f})')
        self.logger.info(f'map->odom offset: ({self.odom_offset_x:.2f}, {self.odom_offset_y:.2f})')

        grid = np.zeros((grid_size, grid_size), dtype=np.int8)

        centerline = self._interpolate_centerline(map_points[1:])
        road_mask = self._compute_road_mask(centerline, origin_x, origin_y + grid_size * self.resolution, grid_size)

        grid[road_mask] = 0
        grid[~road_mask] = 100

        lat_origin = float(self.latest_robot_gps.latitude)
        lon_origin = float(self.latest_robot_gps.longitude)

        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(lat_origin))

        metadata = MapMetadata(
            resolution=self.resolution,
            width=grid_size,
            height=grid_size,
            origin_x=origin_x,
            origin_y=origin_y - grid_size * self.resolution,
            robot_x=robot_x - origin_x,
            robot_y=origin_y + grid_size * self.resolution - robot_y,
            gps_points=self.nav_gps_points.copy(),
            origin_lat=lat_origin,
            origin_lon=lon_origin,
            meters_per_degree_lat=meters_per_degree_lat,
            meters_per_degree_lon=meters_per_degree_lon,
            odom_offset_x=self.odom_offset_x,
            odom_offset_y=self.odom_offset_y,
            odom_offset_yaw=self.odom_offset_yaw,
        )

        shared_map = get_shared_map()
        shared_map.set_map(grid, metadata)

        # 新地图生成后需要重新发完整的 OccupancyGrid，而非增量更新
        self.first_map_published = False

        # 重置 TF 发布标志，等待首次地图发布后发送 TF
        self.map_odom_tf_published = False

        # 发布地图原点GPS（供 tf_publisher 计算 map->odom TF）
        # 地图原点 = 地图中心 = (origin_x + grid_size*resolution/2, origin_y + grid_size*resolution/2)
        # 但我们需要发布的是地图中心对应的GPS坐标
        map_center_lat = lat_origin + (map_center_y / meters_per_degree_lat)
        map_center_lon = lon_origin + (map_center_x / meters_per_degree_lon)

        # 保存当前地图原点GPS，供持续发布
        self.current_map_origin_lat = map_center_lat
        self.current_map_origin_lon = map_center_lon

        map_origin_msg = String()
        map_origin_msg.data = json.dumps({
            'latitude': map_center_lat,
            'longitude': map_center_lon
        })
        self.map_origin_pub.publish(map_origin_msg)

        self.logger.info(
            f'Map generated and stored: {grid_size}x{grid_size}, resolution={self.resolution}m'
        )
        self.logger.info(f'Published map origin GPS: lat={map_center_lat:.8f}, lon={map_center_lon:.8f}')

    def compute_and_publish_nav_map_points(self):
        """计算导航点的地图坐标并发布"""
        shared_map = get_shared_map()
        map_info = shared_map.get_map_info()
        
        if map_info is None:
            self.logger.warning('No map info when computing nav map points')
            return

        with self.path_lock:
            if not self.nav_gps_points:
                return

            nav_map_points = []
            for wp in self.nav_gps_points:
                lat = float(wp['latitude'])
                lon = float(wp['longitude'])
                map_x, map_y = shared_map.gps_to_map(lat, lon)
                if map_x is None:
                    continue
                nav_map_points.append({'x': map_x, 'y': map_y})

            if not nav_map_points:
                self.logger.warning('Failed to compute any nav map points')
                return

            # 发布导航地图坐标
            msg = String()
            msg.data = json.dumps({
                'batch_id': self.batch_id,
                'batch_number': self.batch_counter,
                'points': nav_map_points,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })
            self.nav_map_points_pub.publish(msg)

            self.logger.info(f'Published {len(nav_map_points)} nav map points, batch_number: {self.batch_counter}')

            # 递增组号
            self.batch_counter += 1

    def local_costmap_callback(self, msg: String):
        """局部 costmap 回调，使用 map_pose 更新地图"""
        try:
            data = json.loads(msg.data)
            
            costmap_data = data.get('costmap', [])
            width = data.get('width', 0)
            height = data.get('height', 0)
            resolution = data.get('resolution', 0.1)
            
            if not costmap_data or width == 0 or height == 0:
                return
            
            costmap = np.array(costmap_data, dtype=np.int8).reshape((height, width))
            
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
        """旋转 costmap"""
        height, width = costmap.shape

        robot_grid_x = width // 2
        robot_grid_y = height - 1

        rotated = np.zeros_like(costmap)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        for y in range(height):
            for x in range(width):
                if costmap[y, x] == 100:
                    dx = x - robot_grid_x
                    dy = robot_grid_y - y

                    new_dx = dx * cos_a - dy * sin_a
                    new_dy = dx * sin_a + dy * cos_a

                    new_x = int(new_dx + robot_grid_x)
                    new_y = int(robot_grid_y - new_dy)

                    if 0 <= new_x < width and 0 <= new_y < height:
                        rotated[new_y, new_x] = 100

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
            origin_pose.position.y = metadata.origin_y - metadata.height * metadata.resolution
            origin_pose.position.z = 0.0
            origin_pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
            grid_msg.info.origin = origin_pose

            grid_data = np.zeros((metadata.height, metadata.width), dtype=np.int8)
            for row in range(metadata.height):
                for col in range(metadata.width):
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
        # OccupancyGrid 坐标系：y=0 在底部（地图原点），向上递增
        # 内部坐标系：row=0 在顶部，向下递增
        # 转换：OccupancyGrid row = metadata.height - 1 - internal_row
        #
        # update_msg.y 是更新区域左下角的 OccupancyGrid 行号
        # 内部 max_row（最大内部行 = 最靠底部）对应最小的 OccupancyGrid 行号
        update_msg = OccupancyGridUpdate()
        update_msg.header.stamp = self.get_clock().now().to_msg()
        update_msg.header.frame_id = 'map'
        update_msg.x = min_col
        update_msg.y = metadata.height - 1 - max_row   # 内部底边 → OccupancyGrid 底行
        update_msg.width = width
        update_msg.height = height

        # data 按 OccupancyGrid 顺序排列：从底行到顶行（内部从 max_row 到 min_row）
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

    def publish_map_odom_tf(self):
        """
        发布 map->odom 静态 TF 变换

        这个变换描述了 map 原点（地图中心）相对于 odom 原点的位置关系。
        由于 odom 原点（机器人启动位置）和 map 原点（地图中心）都是固定的，
        因此只需要发布一次静态变换即可。
        """
        if self.map_odom_tf_published:
            return

        if self.tf_broadcaster is None:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        try:
            # 创建 TransformStamped 消息
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'map'    # 父坐标系
            transform.child_frame_id = 'odom'    # 子坐标系

            # 设置平移：odom 相对于 map 的位置
            # 即：map -> (-offset_x, -offset_y) -> odom
            transform.transform.translation.x = -self.odom_offset_x
            transform.transform.translation.y = -self.odom_offset_y
            transform.transform.translation.z = 0.0

            # 设置旋转（假设地图和 odom 坐标系朝向一致）
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0

            # 发布变换
            self.tf_broadcaster.sendTransform(transform)
            self.map_odom_tf_published = True

            self.logger.info(
                f'Published map->odom TF: translation=({-self.odom_offset_x:.2f}, {-self.odom_offset_y:.2f}, 0.0), '
                f'rotation=(0, 0, 0, 1)'
            )

        except Exception as e:
            self.logger.error(f'Failed to publish map->odom TF: {e}')

    def update(self):
        """执行一次更新"""
        # 计算并发布 map_pose
        self.compute_and_publish_map_pose()

        # 发布地图
        self.publish_map()

        # 持续发布地图原点GPS（供 tf_publisher 计算 map->odom TF）
        self._publish_map_origin_gps()

    def _publish_map_origin_gps(self):
        """持续发布地图原点GPS"""
        if self.current_map_origin_lat is None or self.current_map_origin_lon is None:
            return

        map_origin_msg = String()
        map_origin_msg.data = json.dumps({
            'latitude': self.current_map_origin_lat,
            'longitude': self.current_map_origin_lon
        })
        self.map_origin_pub.publish(map_origin_msg)


def main(args=None):
    rclpy.init(args=args)

    node = MapNode()

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
