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
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
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
from frequency_stats import FrequencyStats


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

        # 话题配置
        subscriptions = controller_config.get('subscriptions', {})
        publications = controller_config.get('publications', {})

        # 订阅话题
        self.path_topic = subscriptions.get('path_topic', '/planned_path')
        self.lidar_obs_topic = subscriptions.get('lidar_obs_topic', '/lidar_obs')
        self.map_pose_topic = subscriptions.get('map_pose_topic', '/map_pose')
        
        # 速度话题配置
        self.odom_topic = subscriptions.get('odom_topic', '/navigation/robot_odom')
        
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
        self.latest_obs = None  # obs_min_distance 数组
        
        # 速度
        self.velocity = {'v': 0.0, 'w': 0.0}
        
        # 上一次动作
        self.last_action = {'v': 0.0, 'w': 0.0}
        
        # 控制参数
        self.max_v = controller_config.get('max_v', 1.0)  # 最大线速度 m/s
        self.max_w = controller_config.get('max_w', 1.0)  # 最大角速度 rad/s
        self.arrival_threshold = controller_config.get('arrival_threshold', 0.5)  # 到达阈值 (米)

        # 创建订阅者 - 路径
        self.path_sub = self.create_subscription(
            Path,
            self.path_topic,
            self.path_callback,
            10
        )

        # 创建订阅者 - 障碍物观测
        self.obs_sub = self.create_subscription(
            LaserScan,
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

        # 创建订阅者 - 速度（从 odom 获取线速度 v 和角速度 w）
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        # 创建发布者 - 控制指令
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            self.cmd_topic,
            10
        )

        # 工作频率
        self.frequency = controller_config.get('frequency', 10.0)

        # 定时器：按指定频率执行控制
        period = 1.0 / max(self.frequency, 1e-3)
        self.timer = self.create_timer(period, self.update)

        # 初始化频率统计
        self.freq_stats = FrequencyStats(
            node_name='controller_node',
            target_frequency=self.frequency,
            logger=None,  # 会在 _init_logger 之后设置
            ros_logger=self.get_logger(),
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )

        # 初始化日志（在订阅/发布创建之后，加载模型之前）
        log_enabled = controller_config.get('log_enabled', True)
        self._init_logger(log_enabled)

        # 更新 logger 引用
        self.freq_stats.logger = self.logger

        # 加载模型
        self.model = None
        self._load_model(controller_config)

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

        # 终端输出初始化信息（同时写入文件日志）
        init_info = [
            'Controller Node initialized',
            f'  订阅路径: {self.path_sub.topic}',
            f'  订阅激光雷达障碍物: {self.obs_sub.topic}',
            f'  订阅地图 pose: {self.pose_sub.topic}',
            f'  订阅里程计(v,w): {self.odom_sub.topic}',
            f'  发布控制命令: {self.cmd_pub.topic}',
            f'  工作频率: {self.frequency} Hz',
            f'  最大速度: {self.max_v} m/s, 最大角速度: {self.max_w} rad/s',
            f'  详细日志已写入: {log_file}',
        ]

        for line in init_info:
            self.logger.info(line)  # 写入文件
            self.get_logger().info(line)  # 输出到终端

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
            # 使用 torch.load 加载 .pth 检查点文件
            # weights_only=False 允许加载完整的模型对象（非仅权重）
            checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
            
            # 检查加载的内容类型
            if isinstance(checkpoint, torch.nn.Module):
                # 如果直接保存的是完整模型
                self.model = checkpoint
                self.model.eval()
                self.logger.info(f'Loaded full PyTorch model: {model_path}')
            elif isinstance(checkpoint, dict):
                # 如果保存的是 state_dict，检查是否有 'model' 或 'actor' 键
                if 'model' in checkpoint:
                    self.logger.info('Checkpoint contains model state_dict')
                    # 需要根据具体模型架构加载，此处记录警告
                    self.logger.warning('state_dict detected but model architecture unknown - using as-is')
                    self.model = checkpoint['model'] if isinstance(checkpoint['model'], torch.nn.Module) else checkpoint
                elif 'actor' in checkpoint:
                    self.logger.info('Checkpoint contains actor state_dict')
                    self.model = checkpoint['actor']
                else:
                    self.logger.warning(f'Unknown checkpoint keys: {checkpoint.keys()}')
                    self.model = checkpoint
            else:
                self.model = checkpoint
                self.model.eval()
                
            self.logger.info(f'Successfully loaded PyTorch model from: {model_path}')
        except Exception as e:
            self.logger.error(f'Failed to load model: {e}')
            self.model = None

    def path_callback(self, msg: Path):
        """接收路径"""
        try:
            waypoints = []
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            for pose_stamped in msg.poses:
                wp = [
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y
                ]
                waypoints.append(wp)
            
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

    def obs_callback(self, msg: LaserScan):
        """接收障碍物观测"""
        try:
            # 从 LaserScan 获取 ranges
            obs_min_distance = np.array(msg.ranges, dtype=np.float32)
            
            # 处理无效值 (inf, nan)
            obs_min_distance = np.where(
                np.isfinite(obs_min_distance),
                obs_min_distance,
                msg.range_max  # 用最大距离替换无效值
            )
            
            with self.obs_lock:
                self.latest_obs = obs_min_distance
                
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

    def odom_callback(self, msg: Odometry):
        """从 odom 获取线速度 v 和角速度 w"""
        try:
            # 线速度
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            v = math.sqrt(vx * vx + vy * vy)
            
            # 角速度
            w = msg.twist.twist.angular.z
            
            with self.velocity_lock:
                self.velocity['v'] = v
                self.velocity['w'] = w
                
        except Exception as e:
            self.logger.error(f'Failed to parse odom: {e}')

    def compute_state(self) -> np.ndarray:
        """计算状态"""
        state_parts = []
        
        # 1. obs_min_distance
        with self.obs_lock:
            if self.latest_obs is not None:
                obs_min_distance = self.latest_obs
            else:
                obs_min_distance = np.ones(20, dtype=np.float32)  # 默认值
        
        state_parts.append(obs_min_distance)
        
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
        # 记录频率统计
        self.freq_stats.tick()

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
    period = 1.0 / node.frequency
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
