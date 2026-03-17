#!/usr/bin/env python3
"""
主启动文件
启动所有导航相关节点
"""

import signal
import os
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from datetime import datetime

from config_loader import get_config


class Nav2GPSNode(Node):
    """导航系统主节点，负责启动所有子节点"""

    def __init__(self):
        super().__init__('navigation_system')

        self.get_logger().info('Navigation System started')

        # 生成统一的启动时间戳
        self.start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.get_logger().info(f'Navigation session timestamp: {self.start_timestamp}')

        # 创建统一的日志文件夹
        self.log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'logs',
            f'navigation_{self.start_timestamp}'
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.get_logger().info(f'Log directory: {self.log_dir}')

        # 启动各个节点
        self.start_nodes()

    def start_nodes(self):
        """启动所有子节点"""
        config = get_config()
        is_test_mode = config.get('common.test_mode', False)

        if is_test_mode:
            self.get_logger().info('=== TEST MODE: controller_node disabled, use joystick to control ===')

        # ekf_fusion_node 负责：
        # 1. 接收 /utildar/robot_odom (机器狗自带里程计)
        # 2. 接收 /rtk_fix (GPS) 和 /rtk_imu (世界系朝向)
        # 3. 发布 utm -> map 静态 TF
        # 4. 发布 map -> odom 静态 TF (收到首个有效 GPS 后)
        # 5. 发布 odom -> base_link 动态 TF
        # 6. 发布 /navigation/map_pose
        self.get_logger().info('Starting ekf_fusion_node...')
        from ekf_fusion_node import EKFFusionNode
        self.ekf_fusion_node = EKFFusionNode(log_dir=self.log_dir, timestamp=self.start_timestamp)

        self.get_logger().info('Starting map_node...')
        from map_node import MapNode
        self.map_node = MapNode(log_dir=self.log_dir, timestamp=self.start_timestamp)

        self.get_logger().info('Starting lidar_costmap_node...')
        from lidar_costmap_node import LidarCostmapNode
        self.lidar_costmap_node = LidarCostmapNode(log_dir=self.log_dir, timestamp=self.start_timestamp)

        # self.get_logger().info('Starting lidar_360_fusion_node...')
        # from lidar_360_fusion_node import Lidar360FusionNode
        # self.lidar_360_fusion_node = Lidar360FusionNode()

        self.get_logger().info('Starting planner_node...')
        from planner_node import PlannerNode
        self.planner_node = PlannerNode(log_dir=self.log_dir, timestamp=self.start_timestamp)

        if not is_test_mode:
            self.get_logger().info('Starting controller_node...')
            from controller_node import ControllerNode
            self.controller_node = ControllerNode(log_dir=self.log_dir, timestamp=self.start_timestamp)

        self.get_logger().info('All nodes started successfully!')

    def destroy_all_nodes(self):
        """销毁所有子节点"""
        for attr in ('ekf_fusion_node', 'map_node', 'lidar_costmap_node',
                     'lidar_360_fusion_node', 'planner_node', 'controller_node'):
            node = getattr(self, attr, None)
            if node is not None:
                try:
                    node.destroy_node()
                except Exception:
                    pass


def main(args=None):
    rclpy.init(args=args)

    # 创建主节点
    nav_node = Nav2GPSNode()

    # 创建多线程执行器（需要足够的线程来运行所有节点）
    executor = MultiThreadedExecutor(num_threads=10)

    # 添加所有节点
    config = get_config()
    is_test_mode = config.get('common.test_mode', False)

    # executor.add_node(nav_node.odom_node)
    # executor.add_node(nav_node.tf_publisher)
    executor.add_node(nav_node.ekf_fusion_node)
    executor.add_node(nav_node.map_node)
    executor.add_node(nav_node.planner_node)
    executor.add_node(nav_node.lidar_costmap_node)
    # executor.add_node(nav_node.lidar_360_fusion_node)

    if not is_test_mode:
        executor.add_node(nav_node.controller_node)

    executor.add_node(nav_node)

    def shutdown(signum, frame):
        print('\n[Nav2GPS] Ctrl+C received, shutting down all nodes...')
        executor.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print('[Nav2GPS] Cleaning up...')
        nav_node.destroy_all_nodes()
        try:
            nav_node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
