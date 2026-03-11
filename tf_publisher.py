#!/usr/bin/env python3
"""
tf_publisher - 统一 TF 发布节点

功能：
1. 订阅 map_node 发布的地图原点GPS坐标
2. 订阅 odom_node 发布的新odom原点GPS坐标
3. 发布 map -> odom 的静态 TF 变换
4. 订阅 /navigation/robot_odom，发布 odom -> base_link 的动态 TF
5. 从 URDF 解析并发布 base_link -> imu 的静态 TF

坐标系关系：
    map (原点=地图中心GPS)
         │
         │ map_offset (静态)
         ▼
    odom (原点=main.py启动时机器狗GPS)
         │
         │ 动态变换 (每帧更新)
         ▼
    base_link

    base_link (静态)
         │
         ▼
    imu
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import json
import math
import xml.etree.ElementTree as ET
import logging
import os
from datetime import datetime


class TFPublisher(Node):
    """统一 TF 发布节点"""

    def __init__(self, urdf_path: str = None):
        super().__init__('tf_publisher')

        # 默认 URDF 路径
        if urdf_path is None:
            urdf_path = '/home/unitree/navigation_system/URDF/GO2_URDF/urdf/go2_description.urdf'

        # 保存 URDF 路径供日志使用
        self.urdf_path = urdf_path

        # ========== TF 广播器 ==========
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # ========== 状态变量 ==========
        # 地图原点 GPS
        self.map_origin_lat = None
        self.map_origin_lon = None

        # 新 odom 原点 GPS
        self.new_odom_origin_lat = None
        self.new_odom_origin_lon = None

        # GPS 转 ENU 的参数
        self.meters_per_degree_lat = 111320.0
        self.meters_per_degree_lon = None

        # 标志位
        self.map_origin_received = False
        self.new_odom_origin_received = False
        self.map_odom_tf_published = False

        # ========== 订阅 ==========
        # 地图原点 GPS (由 map_node 持续发布)
        self.map_origin_sub = self.create_subscription(
            String,
            '/navigation/map_origin_gps',
            self.map_origin_callback,
            10
        )

        # 新 odom 原点 GPS (由 odom_node 持续发布)
        self.new_odom_origin_sub = self.create_subscription(
            String,
            '/navigation/new_odom_origin_gps',
            self.new_odom_origin_callback,
            10
        )

        # 新 odom 坐标 (由 odom_node 发布)
        self.new_odom_sub = self.create_subscription(
            Odometry,
            '/navigation/robot_odom',
            self.new_odom_callback,
            10
        )

        # 初始化日志（在订阅创建之后，发布静态TF之前）
        self._init_logger()

        # 发布静态变换 base_link -> imu
        self._publish_static_imu_tf(urdf_path)

    def _init_logger(self):
        """初始化日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'tf_publisher_log_{timestamp}.log')

        self.logger = logging.getLogger('tf_publisher')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # 终端输出初始化信息（同时写入文件日志）
        init_info = [
            f'TF Publisher 节点已启动',
            f'  URDF 路径: {self.urdf_path}',
            f'  详细日志已写入: {log_file}',
        ]

        for line in init_info:
            self.logger.info(line)  # 写入文件
            self.get_logger().info(line)  # 输出到终端

    def map_origin_callback(self, msg):
        """接收地图原点 GPS"""
        try:
            data = json.loads(msg.data)
            self.map_origin_lat = data.get('latitude')
            self.map_origin_lon = data.get('longitude')

            if self.map_origin_lat is not None and self.map_origin_lon is not None:
                self.map_origin_received = True
                self.logger.info(f'Received map origin GPS: lat={self.map_origin_lat:.8f}, lon={self.map_origin_lon:.8f}')
                self._try_publish_map_odom_tf()

        except Exception as e:
            self.logger.error(f'Failed to parse map origin: {e}')

    def new_odom_origin_callback(self, msg):
        """接收新 odom 原点 GPS"""
        try:
            data = json.loads(msg.data)
            self.new_odom_origin_lat = data.get('latitude')
            self.new_odom_origin_lon = data.get('longitude')

            if self.new_odom_origin_lat is not None and self.new_odom_origin_lon is not None:
                # 计算经纬度转米的参数
                self.meters_per_degree_lon = 111320.0 * math.cos(math.radians(self.new_odom_origin_lat))

                self.new_odom_origin_received = True
                self.logger.info(f'Received new_odom origin GPS: lat={self.new_odom_origin_lat:.8f}, lon={self.new_odom_origin_lon:.8f}')
                self._try_publish_map_odom_tf()

        except Exception as e:
            self.logger.error(f'Failed to parse new_odom origin: {e}')

    def new_odom_callback(self, msg):
        """接收新 odom 数据，发布动态 TF: odom -> base_link"""
        try:
            # 获取位置
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z

            # 获取朝向
            q = msg.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            # 发布动态 TF: odom -> base_link
            self._publish_odom_to_base_link(msg.header.stamp, x, y, z, yaw)

        except Exception as e:
            self.logger.warning(f'处理 odom 消息错误: {e}')

    def _try_publish_map_odom_tf(self):
        """尝试发布 map -> odom 静态 TF"""
        if not self.map_origin_received or not self.new_odom_origin_received:
            return

        if self.map_odom_tf_published:
            return

        # 计算偏移
        dlat = self.map_origin_lat - self.new_odom_origin_lat
        dlon = self.map_origin_lon - self.new_odom_origin_lon

        # 转换为米（ENU坐标系）
        # 纬度差 -> y方向（北）
        # 经度差 -> x方向（东）
        offset_x = dlon * self.meters_per_degree_lon
        offset_y = dlat * self.meters_per_degree_lat

        # map -> odom 的变换：odom 相对于 map 的位置
        # 即：map 坐标 + offset = odom 坐标
        # 所以 map -> odom 的 translation = offset
        map_to_odom_x = offset_x
        map_to_odom_y = offset_y

        # 由于新 odom、地图坐标系的 y 轴都朝北，故 yaw offset = 0
        map_to_odom_yaw = 0.0

        self._publish_map_odom_tf(map_to_odom_x, map_to_odom_y, map_to_odom_yaw)

    def _publish_map_odom_tf(self, x: float, y: float, yaw: float):
        """发布 map -> odom 静态 TF"""
        if self.map_odom_tf_published:
            return

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'odom'

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0

        transform.transform.rotation = self._yaw_to_quaternion(yaw)

        self.tf_broadcaster.sendTransform(transform)
        self.map_odom_tf_published = True

        self.logger.info(f'Published map->odom TF: x={x:.4f}, y={y:.4f}, yaw={math.degrees(yaw):.2f}deg')

    def _publish_odom_to_base_link(self, stamp, x: float, y: float, z: float, yaw: float):
        """发布动态 TF: odom -> base_link"""
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'

        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z

        transform.transform.rotation = self._yaw_to_quaternion(yaw)

        self.tf_broadcaster.sendTransform(transform)

    def _publish_static_imu_tf(self, urdf_path: str):
        """发布静态变换: base_link -> imu (从 URDF 解析)"""
        try:
            # 解析 URDF
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            # 查找 imu_joint
            imu_joint = None
            for joint in root.findall('.//joint'):
                if joint.get('name') == 'imu_joint':
                    imu_joint = joint
                    break

            if imu_joint is None:
                self.logger.warning('未找到 imu_joint，使用默认值')
                self._publish_imu_tf_default()
                return

            # 获取 origin
            origin = imu_joint.find('origin')
            if origin is None:
                self.logger.warning('imu_joint 无 origin，使用默认值')
                self._publish_imu_tf_default()
                return

            xyz_str = origin.get('xyz', '0 0 0')
            rpy_str = origin.get('rpy', '0 0 0')

            xyz = [float(x) for x in xyz_str.split()]
            rpy = [float(x) for x in rpy_str.split()]

            # 发布 TF
            self._send_static_imu_tf(xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2])

        except Exception as e:
            self.logger.error(f'解析 URDF 失败: {e}，使用默认值')
            self._publish_imu_tf_default()

    def _publish_imu_tf_default(self):
        """使用默认值发布 imu TF"""
        # URDF 默认值: origin xyz="-0.02557 0 0.04232" rpy="0 0 0"
        self._send_static_imu_tf(-0.02557, 0.0, 0.04232, 0.0, 0.0, 0.0)

    def _send_static_imu_tf(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
        """发送 base_link -> imu 静态 TF"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'imu'

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        # 欧拉角转四元数
        q = self._euler_to_quaternion(roll, pitch, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.static_tf_broadcaster.sendTransform(t)
        self.logger.info(f'Published static TF: base_link -> imu (x={x:.5f}, y={y:.5f}, z={z:.5f})')

    def _yaw_to_quaternion(self, yaw: float):
        """yaw 角转四元数"""
        q = TransformStamped()
        q.transform.rotation.x = 0.0
        q.transform.rotation.y = 0.0
        q.transform.rotation.z = math.sin(yaw / 2.0)
        q.transform.rotation.w = math.cos(yaw / 2.0)
        return q.transform.rotation

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float):
        """欧拉角转四元数"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ]
        return q


def main(args=None):
    rclpy.init(args=args)
    node = TFPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
