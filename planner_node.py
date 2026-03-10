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


class PlannerNode(Node):
    """
    合并后的规划器节点
    """

    def __init__(self):
        super().__init__('planner_node')

        # 加载配置
        config = get_config()
        planner_config = config.get('planner_node', {})

        # 初始化日志
        self._init_logger(planner_config.get('log_enabled', True))

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
        self.update_frequency = planner_config.get('update_frequency', 5.0)
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
        # EKF 融合位姿
        self.pose_sub = self.create_subscription(
            String,
            self.robot_pose_topic,
            self.pose_callback,
            10
        )

        # 导航地图坐标订阅（从 map_node 获取）
        self.nav_map_points_sub = self.create_subscription(
            String,
            self.nav_map_points_topic,
            self.nav_map_points_callback,
            10
        )

        # ------ 发布者 ------
        self.path_pub = self.create_publisher(
            String,
            self.path_topic,
            10
        )

        # 定时器：按指定频率执行规划
        period = 1.0 / max(self.update_frequency, 1e-3)
        self.timer = self.create_timer(period, self.update)

        self.logger.info('Planner Node initialized')
        self.logger.info(f'  Robot pose topic: {self.robot_pose_topic}')
        self.logger.info(f'  Nav map points topic: {self.nav_map_points_topic}')
        self.logger.info(f'  Path output topic: {self.path_topic}')
        self.logger.info(f'  Update frequency: {self.update_frequency} Hz')
        self.logger.info(f'  Arrival threshold: {self.arrival_threshold} m')

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

        self.logger.info(f'Planner Node started, log file: {log_file}')

    # ==================== 订阅回调 ====================

    def pose_callback(self, msg: String):
        """接收 EKF 输出的机器人地图位姿"""
        try:
            data = json.loads(msg.data)
            with self.pose_lock:
                if data.get('valid', False):
                    self.robot_pose = {
                        'x': data.get('x', 0.0),
                        'y': data.get('y', 0.0),
                        'yaw': data.get('yaw', 0.0),
                        'timestamp': data.get('timestamp', 0.0),
                        'valid': True
                    }
                else:
                    # 标记为无效
                    self.robot_pose = {
                        'x': data.get('x', 0.0),
                        'y': data.get('y', 0.0),
                        'yaw': data.get('yaw', 0.0),
                        'timestamp': data.get('timestamp', 0.0),
                        'valid': False
                    }
        except Exception as e:
            self.logger.error(f'Failed to parse fusion pose: {e}')

    def nav_map_points_callback(self, msg: String):
        """
        接收导航地图坐标（从 map_node 订阅）
        
        消息格式 (JSON):
        {
            "batch_id": "batch_xxx",
            "batch_number": 0,
            "points": [
                {"x": map_x1, "y": map_y1},
                {"x": map_x2, "y": map_y2},
                ...
            ],
            "timestamp": 1234567890.123
        }
        """
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.logger.error(f'Failed to parse nav map points: {e}')
            return

        with self.path_lock:
            self.nav_map_points = data.get('points', [])
            self.batch_id = data.get('batch_id', '')
            self.batch_number = data.get('batch_number', -1)
            self.unreached_index = 0
            self.task_completed = False

        self.logger.info(f'Received nav map points: {len(self.nav_map_points)} points, batch_number: {self.batch_number}')

    def path_callback(self, msg: String):
        """不再需要，从 map_node 订阅导航地图坐标"""
        pass

    # ==================== 规划主循环 ====================

    def update(self):
        """按固定频率执行一次规划"""
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

            # 在未到达的导航点中寻找离机器人最近的点
            nearest_idx = None
            nearest_dist = None
            for idx in range(self.unreached_index, len(self.nav_map_points)):
                point = self.nav_map_points[idx]
                # 支持字典格式 {'x': x, 'y': y} 和元组格式 (x, y)
                gx = point['x'] if isinstance(point, dict) else point[0]
                gy = point['y'] if isinstance(point, dict) else point[1]
                dx = gx - robot_x
                dy = gy - robot_y
                dist = math.hypot(dx, dy)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = idx

            if nearest_idx is None:
                return

            # 到达判定
            if nearest_dist is not None and nearest_dist <= self.arrival_threshold:
                self.logger.info(
                    f'Navigation point {nearest_idx} reached (dist={nearest_dist:.3f}m), '
                    f'threshold={self.arrival_threshold:.3f}m'
                )
                # 如果是最后一个导航点，任务完成，本周期直接退出
                if nearest_idx == len(self.nav_map_points) - 1:
                    self.task_completed = True
                    return
                # 否则将未到达指针移动到该点之后的一个点，继续以新目标做规划
                self.unreached_index = nearest_idx + 1

            # 以未到达指针指向的导航地图坐标作为终点
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
        resolution = self.map_info['resolution'] if self.map_info else 0.1
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

        保证任意相邻两个保留坐标之间的格子数 <= max_cells，
        并始终保留终点。
        """
        if len(path) <= 1:
            return path

        sparse = [path[0]]

        for i in range(1, len(path)):
            curr = path[i]
            last_kept = sparse[-1]

            cells = abs(curr[0] - last_kept[0]) + abs(curr[1] - last_kept[1])

            # 如果当前点距离上一个保留点太远，则保留前一个点
            if cells > max_cells:
                prev = path[i - 1]
                if prev != last_kept:
                    sparse.append(prev)

        # 始终保留终点
        if sparse[-1] != path[-1]:
            sparse.append(path[-1])

        return sparse

    def publish_path(self, waypoints: list):
        """发布稀疏后的路径"""
        msg = String()
        msg.data = json.dumps({
            'waypoints': waypoints,
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
        })
        self.path_pub.publish(msg)


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
