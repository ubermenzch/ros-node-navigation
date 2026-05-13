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
from geometry_msgs.msg import Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
from utils.time_utils import TimeUtils
import math
import numpy as np
import os
from datetime import datetime

from config_loader import get_config
from utils.frequency_stats import FrequencyStats
from utils.logger import NodeLogger



import torch.nn.functional as F
from torch import distributions as pyd

# ====== TanhTransform（保持一致）======
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

# ====== MLP（对齐你的 utils.mlp）======
def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU()]
    for _ in range(hidden_depth - 1):
        layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()]
    layers.append(torch.nn.Linear(hidden_dim, output_dim))
    return torch.nn.Sequential(*layers)

# ====== 🚀 部署专用 Actor ======
class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.log_std_bounds = [-5, 2]

        # 25维输入 → 1024 → 1024 → 4（2*action_dim）
        self.trunk = mlp(
            input_dim=25,
            hidden_dim=1024,
            output_dim=4,   # 2 * action_dim (v, w)
            hidden_depth=2
        )

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # 限制 log_std
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)

        # 🚀 关键：直接输出确定性动作
        return dist.mean



class ControllerNode(Node):
    """
    控制器节点

    功能：
    1. 接收路径，维护未到达指针
    2. 获取机器狗位置、障碍物观测、速度
    3. 拼接状态，输入模型，得到控制指令
    4. 发布控制指令
    """

    def __init__(self, log_dir: str = None, log_timestamp: str = None):
        super().__init__('controller_node')

        # 使用传入的日志目录和时间戳，或生成新的
        self.log_dir = log_dir
        self.log_timestamp = log_timestamp if log_timestamp is not None else datetime.now().strftime('%Y%m%d_%H%M%S')

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

        # 速度话题配置
        self.odom_topic = subscriptions.get('odom_topic', '/navigation/robot_odom')

        # 超时配置 (秒)
        self.path_timeout = controller_config.get('path_timeout', 0.5)
        self.lidar_obs_timeout = controller_config.get('lidar_obs_timeout', 0.3)
        self.odom_timeout = controller_config.get('odom_timeout', 0.2)

        # 各传感器最后接收时间戳（int64 纳秒）
        self._last_path_time = 0
        self._last_obs_time = 0
        self._last_odom_time = 0
        self._last_unavailable_log_time = 0

        # 发布话题
        self.cmd_topic = publications.get('cmd_topic', '/cmd_vel')
        self.guidance_marker_topic = publications.get(
            'guidance_marker_topic',
            '/navigation/controller_guidance'
        )
        self.guidance_marker_frame = controller_config.get('guidance_marker_frame', 'base_link')

        # 路径相关：planner 已经高频发布当前 base_link 坐标系路径
        self.current_path = []  # 当前直接用于控制的 base_link 路径

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
            Twist,
            self.cmd_topic,
            1
        )

        # 创建发布者 - RViz2控制指导可视化
        self.guidance_marker_pub = self.create_publisher(
            MarkerArray,
            self.guidance_marker_topic,
            1
        )

        # 工作频率
        self.frequency = controller_config.get('frequency', 10.0)

        # 初始化日志（在订阅/发布创建之后，加载模型之前）
        log_enabled = controller_config.get('log_enabled', True)
        self._init_logger(log_enabled)

        # 定时器：按指定频率执行控制
        period = 1.0 / max(self.frequency, 1e-3)
        self.timer = self.create_timer(period, self.update)

        # 初始化频率统计（在 logger 初始化之后创建，直接传入 logger）
        self.freq_stats = FrequencyStats(
            object_name='controller_node',
            target_frequency=self.frequency,
            node_logger=self.logger,
            window_size=10,
            warn_threshold=0.8,
            log_interval=5.0
        )

        # 加载模型
        self.model = None
        self._load_model(controller_config)

    def _init_logger(self, enabled: bool):
        """初始化日志系统"""
        self.node_logger = NodeLogger(
            node_name='controller_node',
            log_dir=self.log_dir,
            log_timestamp=self.log_timestamp,
            enabled=enabled,
            ros_logger=self.get_logger()
        )
        self.logger = self.node_logger

        init_info = [
            'Controller Node initialized',
            f'  订阅路径: {self.path_sub.topic}',
            f'  订阅激光雷达障碍物: {self.obs_sub.topic}',
            f'  订阅里程计(v,w): {self.odom_sub.topic}',
            f'  发布控制命令: {self.cmd_pub.topic}',
            f'  发布控制指导MarkerArray: {self.guidance_marker_pub.topic}',
            f'  工作频率: {self.frequency} Hz',
            f'  路径来源: 直接使用planner发布的base_link路径',
            f'  最大速度: {self.max_v} m/s, 最大角速度: {self.max_w} rad/s',
            f'  详细日志已写入: {self.node_logger.log_file}',
        ]

        self.node_logger.log_init(init_info)

    def _load_model(self, controller_config: dict):
        model_path = controller_config.get('model_path', '')
        device = controller_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(model_path):
            self.logger.error(f'Model file not found: {model_path}')
            return

        try:
            checkpoint = torch.load(model_path, map_location=device)

            # 🚀 创建模型结构
            self.model = Actor().to(device)

            # 🚀 加载权重
            if 'actor' in checkpoint:
                self.model.load_state_dict(checkpoint['actor'])
                self.logger.info('Loaded actor state_dict')
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                self.logger.info('Loaded model state_dict')
            else:
                # 直接就是 state_dict
                self.model.load_state_dict(checkpoint)
                self.logger.info('Loaded raw state_dict')

            self.model.eval()

            self.logger.info(f'Model loaded successfully on {device}')

        except Exception as e:
            self.logger.error(f'Failed to load model: {e}')
            self.model = None

    def path_callback(self, msg: Path):
        """接收路径（base_link坐标系）"""
        try:
            waypoints = []

            for pose_stamped in msg.poses:
                wp = [
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y
                ]
                waypoints.append(wp)

            self._last_path_time = TimeUtils.now_nanos()

            # 收到空路径：清空路径
            if not waypoints:
                self.current_path = []
                self.logger.info('Received empty path')
                return

            # planner 已经基于最新 map_pose 发布 base_link 路径，controller 直接使用
            self.current_path = waypoints
            self._log_current_target_distance('path_callback')

            self.logger.debug(f'Received new path with {len(waypoints)} waypoints (base_link frame)')

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

            self.latest_obs = obs_min_distance
            self._last_obs_time = TimeUtils.now_nanos()

        except Exception as e:
            self.logger.error(f'Failed to parse obs: {e}')

    def odom_callback(self, msg: Odometry):
        """从 odom 获取线速度 v 和角速度 w"""
        try:
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            v = math.sqrt(vx * vx + vy * vy)

            w = msg.twist.twist.angular.z

            self.velocity['v'] = v
            self.velocity['w'] = w
            self._last_odom_time = TimeUtils.now_nanos()

        except Exception as e:
            self.logger.error(f'Failed to parse odom: {e}')

    def compute_state(self, timeout_status: dict) -> np.ndarray:
        """计算状态"""
        state_parts = []

        # 1. obs_min_distance
        if self.latest_obs is not None and not timeout_status['lidar_obs']:
            obs_min_distance = self.latest_obs
        else:
            obs_min_distance = np.ones(20, dtype=np.float32)

        state_parts.append(obs_min_distance)

        # 2. distance, sin, cos
        # planner 下发的路径不包含起点，第一个点就是当前控制目标点
        if len(self.current_path) >= 1:
            target_x, target_y = self._current_target_xy()

            distance = math.sqrt(target_x * target_x + target_y * target_y)
            angle_to_target = math.atan2(target_y, target_x)

            sin_val = math.sin(angle_to_target)
            cos_val = math.cos(angle_to_target)
        else:
            distance = 0.0
            sin_val = 0.0
            cos_val = 1.0

        state_parts.append(np.array([distance, cos_val, sin_val ], dtype=np.float32))

        # 3. v, w
        if not timeout_status['odom']:
            v = self.velocity['v']
            w = self.velocity['w']
        else:
            v = 0.0
            w = 0.0

        # 4. last_action
        state_parts.append(np.array([self.last_action['v'], self.last_action['w']], dtype=np.float32))

        state = np.concatenate(state_parts)

        # 记录模型原始输入
        self.logger.info(f'Computed state: state={state}')
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

    def publish_cmd(self, cmd_v: float, cmd_w: float, zero_reason: str = None):
        """发布控制指令"""
        msg = Twist()
        msg.linear.x = cmd_v
        msg.angular.z = cmd_w
        self.cmd_pub.publish(msg)

        self.last_action = {'v': cmd_v, 'w': cmd_w}

        try:
            self.publish_guidance_markers(cmd_v, cmd_w)
        except Exception as e:
            self.logger.error(f'Failed to publish guidance markers: {e}')

        if zero_reason is not None:
            self.logger.warning(
                f'Published zero cmd_vel: linear.x={cmd_v:.3f}, '
                f'angular.z={cmd_w:.3f}, reason={zero_reason}'
            )
        else:
            self.logger.info(f'Published cmd_vel: linear.x={cmd_v:.3f}, angular.z={cmd_w:.3f}')

    def publish_guidance_markers(self, cmd_v: float, cmd_w: float):
        """发布RViz2可视化控制指导：线速度直箭头和角速度弯曲箭头。"""
        stamp = TimeUtils.nanos_to_stamp(TimeUtils.now_nanos())
        marker_array = MarkerArray()

        linear_arrow_points = self._build_linear_arrow_points(cmd_v)
        if linear_arrow_points:
            linear_arrow = self._make_guidance_marker(0, Marker.ARROW, stamp)
            linear_arrow.scale.x = 0.075
            linear_arrow.scale.y = 0.18
            linear_arrow.scale.z = 0.20
            linear_arrow.color.r = 0.0
            linear_arrow.color.g = 0.85
            linear_arrow.color.b = 1.0
            linear_arrow.color.a = 1.0
            linear_arrow.points = linear_arrow_points
            marker_array.markers.append(linear_arrow)
        else:
            marker_array.markers.append(self._delete_guidance_marker(0, Marker.ARROW, stamp))

        if abs(cmd_w) > 0.02:
            turn_arc = self._make_guidance_marker(1, Marker.LINE_STRIP, stamp)
            turn_arc.scale.x = 0.04
            turn_arc.color.r = 1.0
            turn_arc.color.g = 0.55
            turn_arc.color.b = 0.05
            turn_arc.color.a = 1.0
            turn_arc.points = self._build_turn_arc_points(cmd_w)
            marker_array.markers.append(turn_arc)

            turn_arrow = self._make_guidance_marker(2, Marker.ARROW, stamp)
            turn_arrow.scale.x = 0.055
            turn_arrow.scale.y = 0.15
            turn_arrow.scale.z = 0.16
            turn_arrow.color.r = 1.0
            turn_arrow.color.g = 0.55
            turn_arrow.color.b = 0.05
            turn_arrow.color.a = 1.0
            turn_arrow.points = self._build_trailing_arrow_points(turn_arc.points, arrow_length=0.18)
            marker_array.markers.append(turn_arrow)
        else:
            marker_array.markers.append(self._delete_guidance_marker(1, Marker.LINE_STRIP, stamp))
            marker_array.markers.append(self._delete_guidance_marker(2, Marker.ARROW, stamp))
        marker_array.markers.append(self._delete_guidance_marker(3, Marker.ARROW, stamp))
        marker_array.markers.append(self._delete_guidance_marker(4, Marker.TEXT_VIEW_FACING, stamp))

        self.guidance_marker_pub.publish(marker_array)

    def _make_guidance_marker(self, marker_id: int, marker_type: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.guidance_marker_frame
        marker.header.stamp = stamp
        marker.ns = 'controller_guidance'
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 500_000_000
        marker.frame_locked = True
        return marker

    def _delete_guidance_marker(self, marker_id: int, marker_type: int, stamp) -> Marker:
        marker = self._make_guidance_marker(marker_id, marker_type, stamp)
        marker.action = Marker.DELETE
        return marker

    @staticmethod
    def _point(x: float, y: float, z: float = 0.0) -> Point:
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point

    def _build_linear_arrow_points(self, cmd_v: float) -> list:
        if abs(cmd_v) < 0.02:
            return []

        max_v = max(abs(self.max_v), 1e-6)
        arrow_length = min(1.2, max(0.25, abs(cmd_v) / max_v * 1.2))
        direction = 1.0 if cmd_v >= 0.0 else -1.0
        return [
            self._point(0.0, 0.0, 0.12),
            self._point(direction * arrow_length, 0.0, 0.12)
        ]

    def _build_turn_arc_points(self, cmd_w: float) -> list:
        sign = 1.0 if cmd_w >= 0.0 else -1.0
        max_w = max(abs(self.max_w), 1e-6)
        span = min(math.pi * 0.85, max(math.pi * 0.25, abs(cmd_w) / max_w * math.pi * 0.85))
        radius = 0.55
        points = []

        for idx in range(18):
            theta = sign * span * idx / 17
            points.append(self._point(radius * math.cos(theta), radius * math.sin(theta), 0.22))

        return points

    def _build_trailing_arrow_points(self, points: list, arrow_length: float = 0.28) -> list:
        if len(points) < 2:
            return []

        end = points[-1]
        for start_candidate in reversed(points[:-1]):
            dx = end.x - start_candidate.x
            dy = end.y - start_candidate.y
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 0.05:
                continue

            length = min(arrow_length, distance)
            ux = dx / distance
            uy = dy / distance
            start = self._point(end.x - ux * length, end.y - uy * length, end.z)
            return [start, self._point(end.x, end.y, end.z)]

        return []

    def _current_target_xy(self):
        """返回当前路径首点在 base_link 坐标系下的 x/y。"""
        target = self.current_path[0]
        if isinstance(target, (list, tuple, np.ndarray)):
            return float(target[0]), float(target[1])
        return float(target.get('x', 0.0)), float(target.get('y', 0.0))

    def _log_current_target_distance(self, source: str):
        """记录当前路径首点到机器人自身(base_link原点)的直线距离。"""
        if not self.current_path:
            return

        target_x, target_y = self._current_target_xy()
        distance = math.sqrt(target_x * target_x + target_y * target_y)
        self.logger.debug(
            f'当前路径首点到自身直线距离: source={source}, '
            f'x={target_x:.3f}, y={target_y:.3f}, distance={distance:.3f} m'
        )

    def _check_timeout(self) -> dict:
        """检查各传感器数据是否缺失或超时"""
        current_nanos = TimeUtils.now_nanos()
        timeout_status = {
            'path': self._last_path_time <= 0,
            'lidar_obs': self._last_obs_time <= 0,
            'odom': self._last_odom_time <= 0,
        }

        if self._last_path_time > 0:
            elapsed = (current_nanos - self._last_path_time) / 1e9
            if elapsed > self.path_timeout:
                timeout_status['path'] = True

        if self._last_obs_time > 0:
            elapsed = (current_nanos - self._last_obs_time) / 1e9
            if elapsed > self.lidar_obs_timeout:
                timeout_status['lidar_obs'] = True

        if self._last_odom_time > 0:
            elapsed = (current_nanos - self._last_odom_time) / 1e9
            if elapsed > self.odom_timeout:
                timeout_status['odom'] = True

        return timeout_status

    def update(self):
        """执行一次控制循环"""

        timeout_status = self._check_timeout()

        if timeout_status['path'] or timeout_status['lidar_obs'] or timeout_status['odom']:
            current_nanos = TimeUtils.now_nanos()
            unavailable = [name for name, is_unavailable in timeout_status.items() if is_unavailable]
            if current_nanos - self._last_unavailable_log_time > 1_000_000_000:
                self.logger.warning(f'Controller inputs unavailable: {", ".join(unavailable)}')
                self._last_unavailable_log_time = current_nanos
            self.publish_cmd(0.0, 0.0, zero_reason=f'controller inputs unavailable: {", ".join(unavailable)}')
            return

        has_current_path = bool(self.current_path)

        if not has_current_path:
            self.publish_cmd(0.0, 0.0, zero_reason='path is empty')
            return

        if self.model is None:
            self.publish_cmd(0.0, 0.0, zero_reason='model is unavailable')
            return

        state = self.compute_state(timeout_status)

        model_v, model_w = self.inference(state)
        if model_v is None or model_w is None:
            self.publish_cmd(0.0, 0.0, zero_reason='model inference failed')
            return

        cmd_v, cmd_w = self.map_output(model_v, model_w)
        self.publish_cmd(cmd_v, cmd_w)

        self.freq_stats.tick()


def run_controller_node(log_dir: str = None, log_timestamp: str = None, args=None):
    """运行节点

    Args:
        log_dir: 日志目录
        log_timestamp: 日志时间戳
        args: ROS 参数
    """
    rclpy.init(args=args)

    node = ControllerNode(log_dir=log_dir, log_timestamp=log_timestamp)

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


def main():
    """独立运行入口（使用默认日志目录）"""
    run_controller_node()


if __name__ == '__main__':
    main()
