#!/usr/bin/env python3
"""
合并后的规划器节点（planner_node）

功能：
1. 接收 anav_v3 下发的一组导航规划 GPS 点
2. 读取机器狗最新 GPS，拼接成 “机器人GPS + 导航GPS点” 路径并生成共享地图
3. 计算地图相关的坐标系转换参数，存入共享内存
4. 将导航 GPS 点转换为一组导航地图坐标
5. 维护未到达指针（指向未到达导航地图坐标）
6. 周期性读取 EKF 输出的机器人地图坐标，做到达判定，更新未到达指针和任务完成标志
7. 以机器人当前地图坐标为起点、未到达指针指向的导航地图坐标为终点，在共享地图上做 A* 最短路
8. 对最短路进行稀疏化，并将稀疏后的路径发布到指定话题
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Pose, PoseStamped
import json
import threading
import math
import heapq
import numpy as np
import os
import logging
from datetime import datetime

from config_loader import get_config
from shared_map_storage import get_shared_map
from frequency_stats import FrequencyStats


class PlannerNode(Node):
    """
    合并后的规划器节点
    """

    def __init__(self):
        super().__init__('planner_node')

        # 加载配置
        config = get_config()
        planner_config = config.get('planner_node', {})

        # ------ 话题与参数 ------
        # 获取订阅和发布话题配置
        subscriptions = planner_config.get('subscriptions', {})
        publications = planner_config.get('publications', {})

        # 订阅话题
        self.robot_pose_topic = subscriptions.get('robot_pose_topic', '/map_pose')
        self.nav_map_points_topic = subscriptions.get('nav_map_points_topic', '/nav_map_points')

        # 发布话题
        self.path_topic = publications.get('path_topic', '/planned_path')

        # 规划更新参数
        self.frequency = planner_config.get('frequency', 5.0)
        self.pose_timeout = planner_config.get('pose_timeout', 2.0)
        self.max_distance_between = planner_config.get('max_distance_between', 0.5)
        self.allow_diagonal = planner_config.get('allow_diagonal', False)
        self.arrival_threshold = planner_config.get('arrival_threshold', 1.0)  # 到达阈值（米）

        # ------ 状态 ------
        # 导航地图坐标（从 map_node 接收）
        self.nav_map_points = []  # [{'x': map_x, 'y': map_y}, ...]
        # 未到达指针
        self.unreached_index = 0
        # 任务完成标志
        self.task_completed = False
        # 批次 ID
        self.batch_id = ""
        # 组号
        self.batch_number = -1

        # EKF 输出的机器人地图位姿
        self.robot_pose = None  # {'x':..., 'y':..., 'yaw':..., 'timestamp':..., 'valid':...}

        # 地图数据（从共享内存获取）
        # 互斥锁
        self.path_lock = threading.Lock()
        self.pose_lock = threading.Lock()

        # ------ 订阅者 ------
        # EKF 融合位姿 (PoseStamped)
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.robot_pose_topic,
            self.pose_callback,
            10
        )

        # 导航地图坐标订阅（从 map_node 获取）
        self.nav_map_points_sub = self.create_subscription(
            Path,
            self.nav_map_points_topic,
            self.nav_map_points_callback,
            10
        )

        # ------ 发布者 ------
        self.path_pub = self.create_publisher(
            Path,
            self.path_topic,
            10
        )

        # 定时器：按指定频率执行规划
        period = 1.0 / max(self.frequency, 1e-3)
        self.timer = self.create_timer(period, self.update)

        # 初始化日志（在订阅/发布创建之后）
        self._init_logger(planner_config.get('log_enabled', True))

        # 初始化频率统计
        self.freq_stats = FrequencyStats(
            node_name='planner_node',
            target_frequency=self.frequency,
            logger=self.logger,
            ros_logger=self.get_logger(),
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )

    # ==================== 日志 ====================

    def _init_logger(self, enabled: bool):
        """初始化文件日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'planner_node_log_{timestamp}.log')

        self.logger = logging.getLogger('planner_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if enabled:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # 终端输出初始化信息（同时写入文件日志）
        init_info = [
            'Planner Node initialized',
            f'  订阅机器人 pose: {self.pose_sub.topic}',
            f'  订阅导航地图点: {self.nav_map_points_sub.topic}',
            f'  发布路径: {self.path_pub.topic}',
            f'  工作频率: {self.frequency} Hz',
            f'  到达阈值: {self.arrival_threshold} m',
            f'  详细日志已写入: {log_file}',
        ]

        for line in init_info:
            self.logger.info(line)  # 写入文件
            self.get_logger().info(line)  # 输出到终端

    # ==================== 订阅回调 ====================

    def pose_callback(self, msg: PoseStamped):
        """接收 EKF 输出的机器人地图位姿"""
        try:
            with self.pose_lock:
                # 从 PoseStamped 提取位置和朝向
                self.robot_pose = {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'yaw': math.atan2(
                        2.0 * (msg.pose.orientation.w * msg.pose.orientation.z +
                               msg.pose.orientation.x * msg.pose.orientation.y),
                        1.0 - 2.0 * (msg.pose.orientation.y ** 2 + msg.pose.orientation.z ** 2)
                    ),
                    'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    'valid': True
                }
        except Exception as e:
            self.logger.error(f'Failed to parse fusion pose: {e}')

    def nav_map_points_callback(self, msg: Path):
        """
        接收导航地图坐标（从 map_node 订阅）
        
        消息格式: nav_msgs/Path
        - header.frame_id = 'map'
        - poses 包含导航地图坐标点
        """
        try:
            points = []
            for pose_stamped in msg.poses:
                points.append({
                    'x': pose_stamped.pose.position.x,
                    'y': pose_stamped.pose.position.y
                })
        except Exception as e:
            self.logger.error(f'Failed to parse nav map points: {e}')
            return

        with self.path_lock:
            self.nav_map_points = points
            self.batch_id = ''
            self.batch_number = -1
            self.unreached_index = 0
            self.task_completed = False

        self.logger.info(f'Received nav map points: {len(self.nav_map_points)} points')

    def path_callback(self, msg: String):
        """不再需要，从 map_node 订阅导航地图坐标"""
        pass

    # ==================== 规划主循环 ====================

    def update(self):
        """按固定频率执行一次规划"""
        # 记录频率统计
        self.freq_stats.tick()

        # 任务完成则空转
        with self.path_lock:
            if self.task_completed:
                return

        # 从共享内存读取地图数据
        shared_map = get_shared_map()
        if not shared_map.has_map():
            return

        map_data, metadata = shared_map.get_map()
        if map_data is None or metadata is None:
            return

        map_info = {
            'resolution': metadata.resolution,
            'width': metadata.width,
            'height': metadata.height,
            'origin_x': metadata.origin_x,
            'origin_y': metadata.origin_y,
        }

        # 读取机器人地图位姿
        with self.pose_lock:
            robot_pose = self.robot_pose

        if robot_pose is None or not robot_pose.get('valid', False):
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        pose_timestamp = robot_pose.get('timestamp', 0.0)
        if current_time - pose_timestamp > self.pose_timeout:
            self.logger.warning('Robot pose timestamp expired')
            return

        robot_x = float(robot_pose['x'])
        robot_y = float(robot_pose['y'])

        # 读取导航地图坐标与未到达指针
        with self.path_lock:
            if not self.nav_map_points:
                return

            if self.unreached_index >= len(self.nav_map_points):
                self.task_completed = True
                return

            # 只检查当前指针指向的点是否到达
            # 不再在所有未到达点中寻找最近点
            current_point = self.nav_map_points[self.unreached_index]
            gx = current_point['x'] if isinstance(current_point, dict) else current_point[0]
            gy = current_point['y'] if isinstance(current_point, dict) else current_point[1]
            dx = gx - robot_x
            dy = gy - robot_y
            dist_to_current = math.hypot(dx, dy)

            # 到达判定：只检查当前指针指向的点
            if dist_to_current <= self.arrival_threshold:
                self.logger.info(
                    f'Navigation point {self.unreached_index} reached (dist={dist_to_current:.3f}m), '
                    f'threshold={self.arrival_threshold:.3f}m'
                )
                # 如果是最后一个导航点，任务完成
                if self.unreached_index == len(self.nav_map_points) - 1:
                    self.task_completed = True
                    return
                # 否则将未到达指针移动到下一个点
                self.unreached_index += 1

            # 再次检查指针是否有效
            if self.unreached_index >= len(self.nav_map_points):
                self.task_completed = True
                return

            goal_point = self.nav_map_points[self.unreached_index]
            goal_map_x = goal_point['x'] if isinstance(goal_point, dict) else goal_point[0]
            goal_map_y = goal_point['y'] if isinstance(goal_point, dict) else goal_point[1]

        # 使用从 topic 接收的地图数据
        resolution = map_info['resolution']
        origin_x = map_info['origin_x']
        origin_y = map_info['origin_y']
        width = map_info['width']
        height = map_info['height']

        # 坐标转网格
        start_grid_x = int((robot_x - origin_x) / resolution)
        start_grid_y = int((robot_y - origin_y) / resolution)

        target_grid_x = int((goal_map_x - origin_x) / resolution)
        target_grid_y = int((goal_map_y - origin_y) / resolution)

        # 边界裁剪
        start_grid_x = max(0, min(width - 1, start_grid_x))
        start_grid_y = max(0, min(height - 1, start_grid_y))
        target_grid_x = max(0, min(width - 1, target_grid_x))
        target_grid_y = max(0, min(height - 1, target_grid_y))

        # A* 路径规划
        path = self.astar_planning(
            map_data,
            (start_grid_x, start_grid_y),
            (target_grid_x, target_grid_y),
            width,
            height,
            self.allow_diagonal,
        )

        if not path or len(path) <= 1:
            self.logger.warning('No path found')
            return

        self.logger.info(f'Path found: {len(path)} cells')

        # 稀疏化（不包含起点，包含终点）
        # 根据地图分辨率将距离转换为格子数
        max_cells = int(self.max_distance_between / resolution) if resolution > 0 else 5
        sparse_path = self.sparsify_path(
            path[1:],  # 去掉起点
            max_cells,
        )

        self.logger.info(f'Sparse path: {len(sparse_path)} waypoints')

        # 转换为世界坐标并发布
        waypoints = []
        for grid_x, grid_y in sparse_path:
            world_x = origin_x + (grid_x + 0.5) * resolution
            world_y = origin_y + (grid_y + 0.5) * resolution
            waypoints.append({'x': world_x, 'y': world_y})

        self.publish_path(waypoints)

    # ==================== A* 与稀疏化 ====================

    def astar_planning(self, map_data: np.ndarray, start: tuple, goal: tuple,
                       width: int, height: int, allow_diagonal: bool):
        """A* 路径规划"""
        if allow_diagonal:
            moves = [
                (0, 1), (1, 0), (0, -1), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ]
            move_costs = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
        else:
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            move_costs = [1, 1, 1, 1]

        # 起点 / 终点可行性
        if map_data[start[1], start[0]] > 0:
            self.logger.warning('Start position is obstacle')
            return None
        if map_data[goal[1], goal[0]] > 0:
            self.logger.warning('Goal position is obstacle')
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
                # 回溯路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            closed_set.add(current)

            for move, move_cost in zip(moves, move_costs):
                neighbor = (current[0] + move[0], current[1] + move[1])

                if neighbor[0] < 0 or neighbor[0] >= width or neighbor[1] < 0 or neighbor[1] >= height:
                    continue

                if map_data[neighbor[1], neighbor[0]] > 0:
                    continue

                if allow_diagonal and move[0] != 0 and move[1] != 0:
                    if map_data[current[1] + move[1], current[0]] > 0 or \
                       map_data[current[1], current[0] + move[0]] > 0:
                        continue

                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return None

    def heuristic(self, a: tuple, b: tuple) -> float:
        """A* 启发函数（欧几里得距离）"""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def sparsify_path(self, path: list, max_cells: int):
        """
        路径稀疏化

        从路径的起点开始（起点为第一个点），数 x/0.1 个点作为第二个点，
        然后从第二个点开始同理数 x/0.1 个点，最终得到第三个点，以此类推，
        直到数到终点停下来，终点作为稀疏化后的最后一个点。

        换言之：从第二个点开始，每隔 max_cells 个点取一个点，终点必须保留。
        例如：max_cells = 10（对应1米，分辨率0.1m），则保留第1、11、21...个点，以及终点。

        Args:
            path: 路径点列表（网格坐标），不包含起点
            max_cells: 每隔多少个格子取一个点

        Returns:
            稀疏化后的路径点列表
        """
        if len(path) <= 1:
            return path

        sparse = []

        # 从第二个点开始（index=1），每隔 max_cells 个点取一个
        # 起点已经在 path[0]，我们从 path[0] 开始数
        # 即保留 path[0], path[max_cells], path[2*max_cells], ... 直到终点
        for i in range(0, len(path), max_cells):
            sparse.append(path[i])

        # 确保终点一定被保留
        if sparse[-1] != path[-1]:
            sparse.append(path[-1])

        return sparse

    def publish_path(self, waypoints: list):
        """将map坐标系的路径转换为base_link坐标系并发布"""
        # 从 robot_pose 获取机器人在 map 坐标系下的位置和朝向
        with self.pose_lock:
            if self.robot_pose is None or not self.robot_pose.get('valid', False):
                self.logger.warning('No valid robot pose, cannot transform path')
                return
            robot_x = self.robot_pose['x']
            robot_y = self.robot_pose['y']
            robot_yaw = self.robot_pose['yaw']

        # 计算旋转矩阵（从 map 到 base_link）
        # 机器人朝向 robot_yaw 表示 base_link 相对于 map 的旋转
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        for wp in waypoints:
            # map坐标系下的点
            map_x = wp['x']
            map_y = wp['y']

            # 相对位置
            dx = map_x - robot_x
            dy = map_y - robot_y

            # 转换为 base_link 坐标系（绕机器人当前朝向旋转）
            # p_base = R^(-1) * (p_map - p_robot)
            # R^(-1) = R^T (旋转矩阵的逆等于其转置)
            base_x = cos_yaw * dx + sin_yaw * dy
            base_y = -sin_yaw * dx + cos_yaw * dy

            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = 'base_link'
            pose.pose.position.x = base_x
            pose.pose.position.y = base_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)
        self.logger.info(f'Published path with {len(waypoints)} waypoints in base_link frame')


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
