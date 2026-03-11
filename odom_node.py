#!/usr/bin/env python3
"""
odom_node - 里程计坐标转换节点

功能：
1. 订阅 /utlidar/robot_odom（旧odom，Y轴朝北）
2. 记录 main.py 启动时的GPS作为新 odom 坐标系原点
3. 订阅地图原点GPS（由 map_node 发布）
4. 将旧 odom 转换到新 odom 并发布
5. 发布 map -> odom 的 TF 变换

坐标系关系：
    旧 odom (/utlidar/robot_odom)
         │
         │ 减去旧odom下机器狗初始位置和朝向
         ▼
    新 odom (/navigation/robot_odom，原点=main.py启动时GPS)
         │
         │ 减去新odom原点相对于地图中心的偏移
         ▼
    map (原点=地图中心)
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Quaternion, Pose, Point
from tf2_ros import TransformListener, Buffer
from std_msgs.msg import String
import math
import json
import logging
import os
import yaml
from datetime import datetime


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class OdomNode(Node):
    """
    里程计坐标转换节点
    
    解决的问题：
    - /utlidar/robot_odom 原点GPS未知
    - 新 odom 原点GPS = main.py 启动时机器狗GPS
    - 发布 map -> odom TF
    """

    def __init__(self, config=None):
        super().__init__('odom_node')

        # 加载配置
        if config is None:
            config = load_config()

        odom_config = config.get('odom_node', {})

        # ========== 话题名称 ==========
        old_odom_topic = odom_config.get('subscriptions', {}).get('old_odom_topic', '/utlidar/robot_odom')
        gps_topic = odom_config.get('subscriptions', {}).get('gps_topic', '/rtk_fix')
        map_origin_gps_topic = odom_config.get('subscriptions', {}).get('map_origin_gps_topic', '/map_origin_gps')
        new_odom_topic = odom_config.get('publications', {}).get('new_odom_topic', '/navigation/robot_odom')

        # ========== 订阅 ==========
        # 旧 odom
        self.odom_sub = self.create_subscription(
            Odometry,
            old_odom_topic,
            self.old_odom_callback,
            10
        )

        # RTK GPS（用于确定新 odom 原点）
        self.gps_sub = self.create_subscription(
            NavSatFix,
            gps_topic,
            self.gps_callback,
            10
        )

        # 地图原点GPS（由 map_node 发布）
        self.map_origin_sub = self.create_subscription(
            String,
            map_origin_gps_topic,
            self.map_origin_callback,
            10
        )

        # ========== 发布 ==========
        # 新 odom
        self.new_odom_pub = self.create_publisher(
            Odometry,
            new_odom_topic,
            10
        )

        # 新 odom 原点 GPS 发布者 (供 tf_publisher 计算 map->odom TF)
        new_odom_origin_gps_topic = odom_config.get('publications', {}).get('new_odom_origin_gps_topic', '/navigation/new_odom_origin_gps')
        self.new_odom_origin_pub = self.create_publisher(
            String,
            new_odom_origin_gps_topic,
            10
        )

        # ========== 状态变量 ==========
        # 新 odom 原点GPS（main.py启动时机器狗GPS）
        self.new_odom_origin_lat = None
        self.new_odom_origin_lon = None
        self.new_odom_origin_recorded = False

        # 旧 odom 下机器狗的初始位置和朝向
        self.old_odom_initial_x = None
        self.old_odom_initial_y = None
        self.old_odom_initial_yaw = None
        self.old_odom_initial_received = False

        # 地图原点GPS（地图中心）
        self.map_origin_lat = None
        self.map_origin_lon = None
        self.map_origin_received = False

        # GPS转ENU的参数
        self.meters_per_degree_lat = 111320.0
        self.meters_per_degree_lon = None

        # 计算出的偏移量
        # 新 odom 相对于旧 odom 的偏移
        self.new_odom_offset_x = 0.0  # 新odom原点相对于旧odom原点的x偏移
        self.new_odom_offset_y = 0.0  # 新odom原点相对于旧odom原点的y偏移
        self.new_odom_offset_yaw = 0.0  # 新odom坐标系Y轴相对于旧odom坐标系Y轴的旋转

        # 定时器：持续发布新 odom 原点 GPS (供 tf_publisher 计算 map->odom TF)
        self.origin_timer = self.create_timer(1.0, self._publish_new_odom_origin_gps)

        # 初始化日志（需要在订阅/发布创建之后）
        self._init_logger()

    def _init_logger(self):
        """初始化日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'odom_node_log_{timestamp}.log')

        self.logger = logging.getLogger('odom_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # 终端输出初始化信息（同时写入文件日志）
        init_info = [
            'Odom Node initialized',
            f'  旧 odom 话题: {self.odom_sub.topic}',
            f'  GPS 话题: {self.gps_sub.topic}',
            f'  发布 odom: {self.new_odom_pub.topic}',
            f'  详细日志已写入: {log_file}',
        ]

        for line in init_info:
            self.logger.info(line)  # 写入文件
            self.get_logger().info(line)  # 输出到终端

    def gps_callback(self, msg: NavSatFix):
        """接收 RTK GPS，记录新 odom 原点并持续发布"""
        if self.new_odom_origin_recorded:
            return

        self.new_odom_origin_lat = msg.latitude
        self.new_odom_origin_lon = msg.longitude

        # 计算经纬度转米的参数
        self.meters_per_degree_lon = 111320.0 * math.cos(math.radians(self.new_odom_origin_lat))

        self.new_odom_origin_recorded = True

        self.logger.info(f'Recorded new odom origin GPS: lat={self.new_odom_origin_lat:.8f}, lon={self.new_odom_origin_lon:.8f}')

        # 立即发布新 odom 原点 GPS
        self._publish_new_odom_origin_gps()

    def _publish_new_odom_origin_gps(self):
        """发布新 odom 原点 GPS"""
        if self.new_odom_origin_lat is None or self.new_odom_origin_lon is None:
            return

        msg = String()
        msg.data = json.dumps({
            'latitude': self.new_odom_origin_lat,
            'longitude': self.new_odom_origin_lon
        })
        self.new_odom_origin_pub.publish(msg)
        self.logger.info(f'Published new_odom origin GPS to topic')

    def map_origin_callback(self, msg: String):
        """接收地图原点GPS（供 tf_publisher 使用）"""
        try:
            data = json.loads(msg.data)
            self.map_origin_lat = data.get('latitude')
            self.map_origin_lon = data.get('longitude')

            if self.map_origin_lat is not None and self.map_origin_lon is not None:
                self.map_origin_received = True
                self.logger.info(f'Received map origin GPS: lat={self.map_origin_lat:.8f}, lon={self.map_origin_lon:.8f}')

        except Exception as e:
            self.logger.error(f'Failed to parse map origin: {e}')

    def old_odom_callback(self, msg: Odometry):
        """接收旧 odom 数据并转换到新 odom"""
        # 提取旧 odom 的位置
        old_x = msg.pose.pose.position.x
        old_y = msg.pose.pose.position.y

        # 从四元数提取 yaw（Y轴正方向）
        q = msg.pose.pose.orientation
        old_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # 检查是否需要记录旧 odom 初始位置
        if not self.old_odom_initial_received:
            self.old_odom_initial_x = old_x
            self.old_odom_initial_y = old_y
            self.old_odom_initial_yaw = old_yaw
            self.old_odom_initial_received = True

            self.logger.info(f'Recorded initial old_odom: x={old_x:.4f}, y={old_y:.4f}, yaw={math.degrees(old_yaw):.2f}deg')

            # 计算新 odom 相对于旧 odom 的偏移
            # 新odom原点 - 旧odom原点 = -初始位置（在旧odom坐标系下）
            self.new_odom_offset_x = -self.old_odom_initial_x
            self.new_odom_offset_y = -self.old_odom_initial_y
            # 新odom Y轴朝向与旧odom Y轴朝向相同（都默认朝北）
            self.new_odom_offset_yaw = -self.old_odom_initial_yaw

            self.logger.info(f'New odom offset (old -> new): x={self.new_odom_offset_x:.4f}, y={self.new_odom_offset_y:.4f}, yaw={math.degrees(self.new_odom_offset_yaw):.2f}deg')

            return

        # 转换到新 odom 坐标系
        # 旧 odom 下的坐标 -> 新 odom 下的坐标
        # 新位置 = 旧位置 - 旧初始位置
        new_x = old_x - self.old_odom_initial_x
        new_y = old_y - self.old_odom_initial_y
        new_yaw = old_yaw - self.old_odom_initial_yaw

        # 发布新 odom
        new_odom_msg = Odometry()
        new_odom_msg.header.stamp = msg.header.stamp
        new_odom_msg.header.frame_id = 'odom'  # 新 odom
        new_odom_msg.child_frame_id = 'base_link'

        new_odom_msg.pose.pose.position.x = new_x
        new_odom_msg.pose.pose.position.y = new_y
        new_odom_msg.pose.pose.position.z = 0.0

        new_odom_msg.pose.pose.orientation = self._yaw_to_quaternion(new_yaw)

        # 速度保持不变
        new_odom_msg.twist.twist = msg.twist.twist
        new_odom_msg.pose.covariance = msg.pose.covariance
        new_odom_msg.twist.covariance = msg.twist.covariance

        self.new_odom_pub.publish(new_odom_msg)

    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        """将 yaw 角转换为四元数"""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def get_status(self):
        """获取节点状态"""
        return {
            'new_odom_origin_recorded': self.new_odom_origin_recorded,
            'old_odom_initial_received': self.old_odom_initial_received,
            'map_origin_received': self.map_origin_received,
            'new_odom_origin_lat': self.new_odom_origin_lat,
            'new_odom_origin_lon': self.new_odom_origin_lon,
            'map_origin_lat': self.map_origin_lat,
            'map_origin_lon': self.map_origin_lon,
        }


def main(args=None):
    rclpy.init(args=args)

    config = load_config()
    node = OdomNode(config)

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        status = node.get_status()
        print(f"Odom Node Status:")
        print(f"  New odom origin recorded: {status['new_odom_origin_recorded']}")
        print(f"  Old odom initial received: {status['old_odom_initial_received']}")
        print(f"  Map origin received: {status['map_origin_received']}")
        if status['new_odom_origin_lat']:
            print(f"  New odom origin: lat={status['new_odom_origin_lat']:.8f}, lon={status['new_odom_origin_lon']:.8f}")
        if status['map_origin_lat']:
            print(f"  Map origin: lat={status['map_origin_lat']:.8f}, lon={status['map_origin_lon']:.8f}")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
