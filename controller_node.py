#!/usr/bin/env python3
"""
控制器节点 (controller_node)

负责：
1. 接收 planner_node 下发的路径（base_link坐标系）
2. 从 lidar_costmap_node 获取障碍物观测
3. 读取机器狗的速度信息
4. 拼接状态并输入模型推理得到控制指令
5. 发布控制指令到机器狗

使用 PyTorch 加载并运行模型
"""

import torch
import torch.nn
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
import threading
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

    def __init__(self, log_dir: str = None, timestamp: str = None):
        super().__init__('controller_node')

        # 使用传入的日志目录和时间戳，或生成新的
        self.log_dir = log_dir
        self.timestamp = timestamp if timestamp is not None else datetime.now().strftime('%Y%m%d_%H%M%S')

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

        # 超时配置 (秒)
        self.path_timeout = controller_config.get('path_timeout', 0.5)
        self.lidar_obs_timeout = controller_config.get('lidar_obs_timeout', 0.3)
        self.odom_timeout = controller_config.get('odom_timeout', 0.2)

        # 各传感器最后接收时间戳
        self._last_path_time = None
        self._last_obs_time = None
        self._last_odom_time = None

        # 发布话题
        self.cmd_topic = publications.get('cmd_topic', '/cmd_vel')

        # 状态锁
        self.path_lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.velocity_lock = threading.Lock()

        # 路径数据（base_link坐标系）
        self.waypoints = []  # 路径点列表
        self.last_path_timestamp = 0.0

        # 收到空路径后的停车标记
        self.stop_requested_by_empty_path = False

        # 障碍物观测
        self.latest_obs = None  # obs_min_distance 数组

        # 速度
        self.velocity = {'v': 0.0, 'w': 0.0}

        # 上一次动作
        self.last_action = {'v': 0.0, 'w': 0.0}

        # 控制参数
        self.max_v = controller_config.get('max_v', 1.0)  # 最大线速度 m/s
        self.max_w = controller_config.get('max_w', 1.0)  # 最大角速度 rad/s

        # 创建订阅者 - 路径
        self.path_sub = self.create_subscription(
            Path,
            self.path_topic,
            self.path_callback,
            1
        )

        # 创建订阅者 - 障碍物观测
        self.obs_sub = self.create_subscription(
            LaserScan,
            self.lidar_obs_topic,
            self.obs_callback,
            1
        )

        # 创建订阅者 - 速度（从 odom 获取线速度 v 和角速度 w）
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            1
        )

        # 创建发布者 - 控制指令
        self.cmd_pub = self.create_publisher(
            Float64MultiArray,
            self.cmd_topic,
            1
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
        if self.log_dir is not None:
            log_dir = self.log_dir
        else:
            ts = self.timestamp
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{ts}')
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'controller_node_log_{self.timestamp}.log')

        self.logger = logging.getLogger('controller_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        if enabled:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

        init_info = [
            'Controller Node initialized',
            f'  订阅路径: {self.path_sub.topic}',
            f'  订阅激光雷达障碍物: {self.obs_sub.topic}',
            f'  订阅里程计(v,w): {self.odom_sub.topic}',
            f'  发布控制命令: {self.cmd_pub.topic}',
            f'  工作频率: {self.frequency} Hz',
            f'  最大速度: {self.max_v} m/s, 最大角速度: {self.max_w} rad/s',
            f'  详细日志已写入: {log_file}',
        ]

        for line in init_info:
            self.logger.info(line)
            self.get_logger().info(line)

    def _load_model(self, controller_config: dict):
        """加载强化学习模型"""

        model_path = controller_config.get('model_path', '')
        device = controller_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        if not model_path:
            self.logger.error('No model path specified')
            return

        if not os.path.exists(model_path):
            self.logger.error(f'Model file not found: {model_path}')
            return

        try:
            checkpoint = torch.load(model_path, weights_only=False, map_location=device)

            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint.to(device)
                self.model.eval()
                self.logger.info(f'Loaded full PyTorch model: {model_path} on {device}')
            elif isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.logger.info('Checkpoint contains model state_dict')
                    self.logger.warning('state_dict detected but model architecture unknown - using as-is')
                    self.model = checkpoint['model'].to(device) if isinstance(checkpoint['model'], torch.nn.Module) else checkpoint
                elif 'actor' in checkpoint:
                    self.logger.info('Checkpoint contains actor state_dict')
                    self.model = checkpoint['actor'].to(device)
                else:
                    self.logger.warning(f'Unknown checkpoint keys: {checkpoint.keys()}')
                    self.model = checkpoint
            else:
                self.model = checkpoint.to(device) if hasattr(checkpoint, 'to') else checkpoint
                if hasattr(self.model, 'eval'):
                    self.model.eval()

            self.logger.info(f'Successfully loaded PyTorch model from: {model_path} on {device}')
        except Exception as e:
            self.logger.error(f'Failed to load model: {e}')
            self.model = None

    def path_callback(self, msg: Path):
        """接收路径（base_link坐标系）"""
        try:
            waypoints = []
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            for pose_stamped in msg.poses:
                wp = [
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y
                ]
                waypoints.append(wp)

            with self.path_lock:
                self.last_path_timestamp = timestamp
                self._last_path_time = self.get_clock().now()

                # 收到空路径：请求停车
                if not waypoints:
                    self.waypoints = []
                    self.stop_requested_by_empty_path = True
                    self.logger.info('Received empty path, stop requested')
                    return

                # 收到非空路径：清除空路径停车标记
                self.waypoints = waypoints
                self.stop_requested_by_empty_path = False

            self.logger.info(f'Received new path with {len(waypoints)} waypoints (base_link frame)')

        except Exception as e:
            self.logger.error(f'Failed to parse path: {e}')

    def obs_callback(self, msg: LaserScan):
        """接收障碍物观测"""
        try:
            obs_min_distance = np.array(msg.ranges, dtype=np.float32)

            obs_min_distance = np.where(
                np.isfinite(obs_min_distance),
                obs_min_distance,
                msg.range_max
            )

            with self.obs_lock:
                self.latest_obs = obs_min_distance
                self._last_obs_time = self.get_clock().now()

        except Exception as e:
            self.logger.error(f'Failed to parse obs: {e}')

    def odom_callback(self, msg: Odometry):
        """从 odom 获取线速度 v 和角速度 w"""
        try:
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            v = math.sqrt(vx * vx + vy * vy)

            w = msg.twist.twist.angular.z

            with self.velocity_lock:
                self.velocity['v'] = v
                self.velocity['w'] = w
                self._last_odom_time = self.get_clock().now()

        except Exception as e:
            self.logger.error(f'Failed to parse odom: {e}')

    def compute_state(self, timeout_status: dict) -> np.ndarray:
        """计算状态"""
        state_parts = []

        # 1. obs_min_distance
        with self.obs_lock:
            if self.latest_obs is not None and not timeout_status['lidar_obs']:
                obs_min_distance = self.latest_obs
            else:
                obs_min_distance = np.ones(20, dtype=np.float32)

        state_parts.append(obs_min_distance)

        # 2. distance, sin, cos
        with self.path_lock:
            if self.waypoints:
                target = self.waypoints[0]
                target_x = target[0] if isinstance(target, (list, tuple)) else target['x']
                target_y = target[1] if isinstance(target, (list, tuple)) else target['y']

                distance = math.sqrt(target_x * target_x + target_y * target_y)
                angle_to_target = math.atan2(target_y, target_x)

                sin_val = math.sin(angle_to_target)
                cos_val = math.cos(angle_to_target)
            else:
                distance = 0.0
                sin_val = 0.0
                cos_val = 1.0

        state_parts.append(np.array([distance, sin_val, cos_val], dtype=np.float32))

        # 3. v, w
        with self.velocity_lock:
            if not timeout_status['odom']:
                v = self.velocity['v']
                w = self.velocity['w']
            else:
                v = 0.0
                w = 0.0

        state_parts.append(np.array([v, w], dtype=np.float32))

        # 4. last_action
        state_parts.append(np.array([self.last_action['v'], self.last_action['w']], dtype=np.float32))

        state = np.concatenate(state_parts)
        return state

    def inference(self, state: np.ndarray) -> tuple:
        """模型推理"""
        if self.model is None:
            return None, None

        try:
            state = state.reshape(1, -1).astype(np.float32)
            state_tensor = torch.from_numpy(state)

            device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else torch.device('cpu')
            state_tensor = state_tensor.to(device)

            with torch.no_grad():
                output = self.model(state_tensor)
                output = output.cpu().numpy().flatten()

            model_v = float(output[0])
            model_w = float(output[1])

            model_v = max(-1.0, min(1.0, model_v))
            model_w = max(-1.0, min(1.0, model_w))

            return model_v, model_w

        except Exception as e:
            self.logger.error(f'Model inference failed: {e}')
            return None, None

    def map_output(self, model_v: float, model_w: float) -> tuple:
        """映射模型输出到实际控制指令"""
        cmd_v = (model_v + 1.0) / 2.0 * self.max_v
        cmd_w = model_w * self.max_w
        return cmd_v, cmd_w

    def publish_cmd(self, cmd_v: float, cmd_w: float):
        """发布控制指令"""
        msg = Float64MultiArray()
        msg.data = [cmd_v, cmd_w]
        self.cmd_pub.publish(msg)

        self.last_action = {'v': cmd_v, 'w': cmd_w}

    def _check_timeout(self) -> dict:
        """检查各传感器数据是否超时"""
        current_time = self.get_clock().now()
        timeout_status = {'path': False, 'lidar_obs': False, 'odom': False}

        if self._last_path_time is not None:
            elapsed = (current_time - self._last_path_time).nanoseconds / 1e9
            if elapsed > self.path_timeout:
                timeout_status['path'] = True

        if self._last_obs_time is not None:
            elapsed = (current_time - self._last_obs_time).nanoseconds / 1e9
            if elapsed > self.lidar_obs_timeout:
                timeout_status['lidar_obs'] = True

        if self._last_odom_time is not None:
            elapsed = (current_time - self._last_odom_time).nanoseconds / 1e9
            if elapsed > self.odom_timeout:
                timeout_status['odom'] = True

        return timeout_status

    def update(self):
        """执行一次控制循环"""
        self.freq_stats.tick()

        timeout_status = self._check_timeout()

        if timeout_status['path']:
            self.logger.warning('Path timeout - no path received recently')
        if timeout_status['lidar_obs']:
            self.logger.warning('Lidar obs timeout - no obstacle data received recently')
        if timeout_status['odom']:
            self.logger.warning('Odom timeout - no odometry data received recently')

        # 优先处理：收到空路径后持续发送零速度
        with self.path_lock:
            stop_requested = self.stop_requested_by_empty_path
            has_waypoints = bool(self.waypoints)

        if stop_requested:
            self.publish_cmd(0.0, 0.0)
            return

        # 没有路径时或路径超时时，先不发布控制
        if (not has_waypoints) or timeout_status['path']:
            return

        if self.model is None:
            return

        state = self.compute_state(timeout_status)

        model_v, model_w = self.inference(state)
        if model_v is None or model_w is None:
            return

        cmd_v, cmd_w = self.map_output(model_v, model_w)
        self.publish_cmd(cmd_v, cmd_w)


def main(args=None):
    rclpy.init(args=args)

    node = ControllerNode()

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