#!/usr/bin/env python3
"""
控制器节点 (controller_node)

负责：
1. 接收 planner_node 下发的路径，维护未到达指针
2. 从 map_node 获取机器狗的地图坐标
3. 从 lidar_costmap_node 获取障碍物观测
4. 读取机器狗的速度信息
5. 拼接状态并输入模型推理得到控制指令
6. 发布控制指令到机器狗

使用 PyTorch 加载并运行模型
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64MultiArray
import json
import threading
import time
import math
import numpy as np
import os
import logging
from datetime import datetime

from config_loader import get_config


class ControllerNode(Node):
    """
    控制器节点
    
    功能：
    1. 接收路径，维护未到达指针
    2. 获取机器狗位置、障碍物观测、速度
    3. 拼接状态，输入模型，得到控制指令
    4. 发布控制指令
    """

    def __init__(self):
        super().__init__('controller_node')

        # 加载配置文件
        config = get_config()
        
        # 获取 controller_node 配置
        controller_config = config.get('controller_node', {})

        # 初始化日志系统
        log_enabled = controller_config.get('log_enabled', True)
        self._init_logger(log_enabled)

        # 话题配置
        subscriptions = controller_config.get('subscriptions', {})
        publications = controller_config.get('publications', {})

        # 订阅话题
        self.path_topic = subscriptions.get('path_topic', '/planned_path')
        self.lidar_obs_topic = subscriptions.get('lidar_obs_topic', '/lidar_obs')
        self.map_pose_topic = subscriptions.get('map_pose_topic', '/map_pose')
        
        # 速度话题配置
        self.wheel_odom_topic = subscriptions.get('wheel_odom_topic', '/wheel_odom')
        self.imu_topic = subscriptions.get('imu_topic', '/imu/data')
        
        # 发布话题
        self.cmd_topic = publications.get('cmd_topic', '/cmd_vel')

        # 状态锁
        self.path_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.velocity_lock = threading.Lock()
        
        # 路径数据
        self.waypoints = []  # 路径点列表
        self.unreached_index = 0  # 未到达指针
        self.last_path_timestamp = 0.0
        
        # 机器狗地图坐标
        self.robot_pose = None  # {'x': x, 'y': y, 'yaw': yaw}
        
        # 障碍物观测
        self.latest_obs = None  # obs_min 数组
        
        # 速度
        self.velocity = {'v': 0.0, 'w': 0.0}
        
        # 上一次动作
        self.last_action = {'v': 0.0, 'w': 0.0}
        
        # 控制参数
        self.max_v = controller_config.get('max_v', 1.0)  # 最大线速度 m/s
        self.max_w = controller_config.get('max_w', 1.0)  # 最大角速度 rad/s
        self.arrival_threshold = controller_config.get('arrival_threshold', 0.5)  # 到达阈值 (米)
        
        # 加载模型
        self.model = None
        self._load_model(controller_config)

        # 创建订阅者 - 路径
        self.path_sub = self.create_subscription(
            String,
            self.path_topic,
            self.path_callback,
            10
        )
        
        # 创建订阅者 - 障碍物观测
        self.obs_sub = self.create_subscription(
            String,
            self.lidar_obs_topic,
            self.obs_callback,
            10
        )
        
        # 创建订阅者 - 机器狗地图坐标
        self.pose_sub = self.create_subscription(
            String,
            self.map_pose_topic,
            self.pose_callback,
            10
        )
        
        # 创建订阅者 - 速度（wheel_odom 获取线速度 v，imu 获取角速度 w）
        self.wheel_odom_sub = self.create_subscription(
            Odometry,
            self.wheel_odom_topic,
            self.wheel_odom_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            10
        )
        
        # 创建发布者 - 控制指令
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            self.cmd_topic,
            10
        )

        # 更新频率
        self.update_frequency = controller_config.get('update_frequency', 10.0)

        self.logger.info('Controller Node initialized')
        self.logger.info(f'  Path topic: {self.path_topic}')
        self.logger.info(f'  Lidar obs topic: {self.lidar_obs_topic}')
        self.logger.info(f'  Map pose topic: {self.map_pose_topic}')
        self.logger.info(f'  Wheel odom topic (v): {self.wheel_odom_topic}')
        self.logger.info(f'  IMU topic (w): {self.imu_topic}')
        self.logger.info(f'  Cmd topic: {self.cmd_topic}')
        self.logger.info(f'  Update frequency: {self.update_frequency} Hz')
        self.logger.info(f'  Max v: {self.max_v}, Max w: {self.max_w}')
        self.logger.info(f'  Arrival threshold: {self.arrival_threshold} m')

        # 定时器：按指定频率执行控制
        period = 1.0 / max(self.update_frequency, 1e-3)
        self.timer = self.create_timer(period, self.update)

    def _init_logger(self, enabled: bool):
        """初始化日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'controller_node_log_{timestamp}.log')

        self.logger = logging.getLogger('controller_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        if enabled:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

            self.logger.info(f'Controller Node started, log file: {log_file}')

    def _load_model(self, controller_config: dict):
        """加载强化学习模型"""
        model_path = controller_config.get('model_path', '')
        
        if not model_path:
            self.logger.error('No model path specified')
            return
        
        if not os.path.exists(model_path):
            self.logger.error(f'Model file not found: {model_path}')
            return
        
        try:
            import torch
            self.model = torch.jit.load(model_path)
            self.model.eval()
            self.logger.info(f'Loaded PyTorch model: {model_path}')
        except Exception as e:
            self.logger.error(f'Failed to load model: {e}')
            self.model = None

    def path_callback(self, msg: String):
        """接收路径"""
        try:
            data = json.loads(msg.data)
            waypoints = data.get('waypoints', [])
            timestamp = data.get('timestamp', 0.0)
            
            if not waypoints:
                return
            
            # 如果收到新路径，重置指针
            with self.path_lock:
                self.waypoints = waypoints
                self.unreached_index = 0
                self.last_path_timestamp = timestamp
            
            self.logger.info(f'Received new path with {len(waypoints)} waypoints')
            
        except Exception as e:
            self.logger.error(f'Failed to parse path: {e}')

    def obs_callback(self, msg: String):
        """接收障碍物观测"""
        try:
            data = json.loads(msg.data)
            obs_min = data.get('obs_min', [])
            
            with self.obs_lock:
                self.latest_obs = np.array(obs_min, dtype=np.float32)
                
        except Exception as e:
            self.logger.error(f'Failed to parse obs: {e}')

    def pose_callback(self, msg: String):
        """接收机器狗地图坐标"""
        try:
            data = json.loads(msg.data)
            
            with self.pose_lock:
                self.robot_pose = {
                    'x': data.get('x', 0.0),
                    'y': data.get('y', 0.0),
                    'yaw': data.get('yaw', 0.0)
                }
                
        except Exception as e:
            self.logger.error(f'Failed to parse pose: {e}')

    def wheel_odom_callback(self, msg: Odometry):
        """从 wheel_odom 获取线速度 v"""
        try:
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            v = math.sqrt(vx * vx + vy * vy)
            
            with self.velocity_lock:
                self.velocity['v'] = v
                
        except Exception as e:
            self.logger.error(f'Failed to parse wheel_odom: {e}')

    def imu_callback(self, msg: Imu):
        """从 imu 获取角速度 w"""
        try:
            w = msg.angular_velocity.z
            
            with self.velocity_lock:
                self.velocity['w'] = w
                
        except Exception as e:
            self.logger.error(f'Failed to parse imu: {e}')

    def compute_state(self) -> np.ndarray:
        """计算状态"""
        state_parts = []
        
        # 1. obs_min
        with self.obs_lock:
            if self.latest_obs is not None:
                obs_min = self.latest_obs
            else:
                obs_min = np.ones(20, dtype=np.float32)  # 默认值
        
        state_parts.append(obs_min)
        
        # 2. distance, sin, cos
        with self.path_lock:
            with self.pose_lock:
                if self.waypoints and self.unreached_index < len(self.waypoints) and self.robot_pose is not None:
                    # 获取当前目标点
                    target = self.waypoints[self.unreached_index]
                    target_x = target[0] if isinstance(target, (list, tuple)) else target['x']
                    target_y = target[1] if isinstance(target, (list, tuple)) else target['y']
                    
                    # 计算距离
                    dx = target_x - self.robot_pose['x']
                    dy = target_y - self.robot_pose['y']
                    distance = math.sqrt(dx * dx + dy * dy)
                    
                    # 计算 sin, cos（相对于机器狗朝向）
                    angle_to_target = math.atan2(dy, dx)
                    angle_diff = angle_to_target - self.robot_pose['yaw']
                    
                    # 归一化到 [-pi, pi]
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    
                    sin_val = math.sin(angle_diff)
                    cos_val = math.cos(angle_diff)
                else:
                    # 没有目标点时使用默认值
                    distance = 0.0
                    sin_val = 0.0
                    cos_val = 1.0
        
        state_parts.append(np.array([distance, sin_val, cos_val], dtype=np.float32))
        
        # 3. v, w
        with self.velocity_lock:
            v = self.velocity['v']
            w = self.velocity['w']
        
        state_parts.append(np.array([v, w], dtype=np.float32))
        
        # 4. last_action
        state_parts.append(np.array([self.last_action['v'], self.last_action['w']], dtype=np.float32))
        
        # 拼接所有部分
        state = np.concatenate(state_parts)
        
        return state

    def inference(self, state: np.ndarray) -> tuple:
        """模型推理"""
        if self.model is None:
            # 没有模型时返回默认动作
            return 0.0, 0.0
        
        try:
            state = state.reshape(1, -1).astype(np.float32)
            
            import torch
            with torch.no_grad():
                output = self.model(torch.from_numpy(state))
                output = output.cpu().numpy().flatten()
            
            # 输出范围 [-1, 1]
            model_v = float(output[0])
            model_w = float(output[1])
            
            # 限制范围
            model_v = max(-1.0, min(1.0, model_v))
            model_w = max(-1.0, min(1.0, model_w))
            
            return model_v, model_w
            
        except Exception as e:
            self.logger.error(f'Model inference failed: {e}')
            return 0.0, 0.0

    def map_output(self, model_v: float, model_w: float) -> tuple:
        """映射模型输出到实际控制指令"""
        # model_v: [-1, 1] -> [0, max_v]
        cmd_v = (model_v + 1.0) / 2.0 * self.max_v
        
        # model_w: [-1, 1] -> [-max_w, max_w]
        cmd_w = model_w * self.max_w
        
        return cmd_v, cmd_w

    def check_arrival(self) -> bool:
        """检查是否到达目标点"""
        with self.path_lock:
            with self.pose_lock:
                if not self.waypoints or self.unreached_index >= len(self.waypoints):
                    return False
                
                if self.robot_pose is None:
                    return False
                
                target = self.waypoints[self.unreached_index]
                target_x = target[0] if isinstance(target, (list, tuple)) else target['x']
                target_y = target[1] if isinstance(target, (list, tuple)) else target['y']
                
                dx = target_x - self.robot_pose['x']
                dy = target_y - self.robot_pose['y']
                distance = math.sqrt(dx * dx + dy * dy)
                
                return distance <= self.arrival_threshold

    def publish_cmd(self, cmd_v: float, cmd_w: float):
        """发布控制指令"""
        msg = Float64MultiArray()
        msg.data = [cmd_v, cmd_w]
        self.cmd_pub.publish(msg)
        
        # 更新 last_action
        self.last_action = {'v': cmd_v, 'w': cmd_w}

    def update(self):
        """执行一次控制循环"""
        
        # 检查是否有路径和位置
        with self.path_lock:
            # 指针指向最后一个目标点的后一位时，空操作（等待新路径）
            if self.unreached_index >= len(self.waypoints):
                # 发布停止指令
                self.publish_cmd(0.0, 0.0)
                return
        
        # 计算状态
        state = self.compute_state()
        
        # 模型推理
        model_v, model_w = self.inference(state)
        
        # 映射输出
        cmd_v, cmd_w = self.map_output(model_v, model_w)
        
        # 检查是否到达目标点
        if self.check_arrival():
            with self.path_lock:
                self.unreached_index += 1
                self.logger.info(f'Reached waypoint {self.unreached_index - 1}, advance to {self.unreached_index}')
        
        # 发布控制指令
        self.publish_cmd(cmd_v, cmd_w)


def main(args=None):
    rclpy.init(args=args)

    node = ControllerNode()

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
