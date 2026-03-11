#!/usr/bin/env python3
"""
主启动文件
启动所有导航相关节点
"""

import signal
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from config_loader import get_config


class Nav2GPSNode(Node):
    """导航系统主节点，负责启动所有子节点"""

    def __init__(self):
        super().__init__('navigation_system')

        self.get_logger().info('Navigation System started')

        # 启动各个节点
        self.start_nodes()

    def start_nodes(self):
        """启动所有子节点"""
        config = get_config()
        is_test_mode = config.get('common.test_mode', False)

        if is_test_mode:
            self.get_logger().info('=== TEST MODE: controller_node disabled, use joystick to control ===')

        self.get_logger().info('Starting odom_node...')
        from odom_node import OdomNode
        self.odom_node = OdomNode(config)
        self.get_logger().info('odom_node initialized')

        self.get_logger().info('Starting tf_publisher...')
        from tf_publisher import TFPublisher
        tf_config = config.get('tf_publisher', {})
        urdf_path = tf_config.get('urdf_path', '/home/unitree/navigation_system/URDF/GO2_URDF/urdf/go2_description.urdf')
        self.tf_publisher = TFPublisher(urdf_path)
        self.get_logger().info('tf_publisher initialized')

        self.get_logger().info('Starting ekf_fusion_node...')
        from ekf_fusion_node import EKFFusionNode
        self.ekf_fusion_node = EKFFusionNode()
        self.get_logger().info('ekf_fusion_node initialized')

        self.get_logger().info('Starting map_node...')
        from map_node import MapNode
        self.map_node = MapNode()
        self.get_logger().info('map_node initialized')

        self.get_logger().info('Starting lidar_costmap_node...')
        from lidar_costmap_node import LidarCostmapNode
        self.lidar_costmap_node = LidarCostmapNode()
        self.get_logger().info('lidar_costmap_node initialized')

        # self.get_logger().info('Starting lidar_360_fusion_node...')
        # from lidar_360_fusion_node import Lidar360FusionNode
        # self.lidar_360_fusion_node = Lidar360FusionNode()
        # self.get_logger().info('lidar_360_fusion_node initialized')

        self.get_logger().info('Starting planner_node...')
        from planner_node import PlannerNode
        self.planner_node = PlannerNode()
        self.get_logger().info('planner_node initialized')

        if not is_test_mode:
            self.get_logger().info('Starting controller_node...')
            from controller_node import ControllerNode
            self.controller_node = ControllerNode()
            self.get_logger().info('controller_node initialized')

        self.get_logger().info('All nodes started successfully!')

    def destroy_all_nodes(self):
        """销毁所有子节点"""
        for attr in ('odom_node', 'tf_publisher', 'ekf_fusion_node', 'map_node', 'lidar_costmap_node',
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

    executor.add_node(nav_node.odom_node)
    executor.add_node(nav_node.tf_publisher)
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
