#!/usr/bin/env python3
"""
雷达360度融合节点
接收部分角度的雷达点云和IMU数据，将多帧点云拼接成360度点云后发布。

原理：
- 旋转雷达每帧只扫描约60-90度的扇区
- 使用IMU的航向角来跟踪雷达的旋转角度
- 将每帧点云转换到世界坐标系，累积多帧形成360度点云

输入:
- 雷达: /utlidar/cloud (PointCloud2) - deskewed后的点云
- IMU: /imu/data (Imu) - 用于获取航向角

输出:
- 360度点云: /utlidar/cloud_360 (PointCloud2)
- 可视化图像: 保存到指定目录
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Imu
import numpy as np
import math
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, List
import time
import logging
import os
from datetime import datetime
import threading

# 可视化相关导入
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，避免需要显示
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"警告: 可视化库导入失败: {e}")
    print("可视化功能将被禁用，请安装: pip install matplotlib")
    VISUALIZATION_AVAILABLE = False

from config_loader import get_config


def read_points_cloud2(msg: PointCloud2):
    """
    手动解析PointCloud2消息
    返回: [(x, y, z), ...] 列表
    """
    points = []
    
    # 获取字段偏移量
    field_offsets = {}
    for field in msg.fields:
        field_offsets[field.name] = field.offset
    
    # 解析每个点
    num_points = msg.width * msg.height
    
    for i in range(num_points):
        offset = i * msg.point_step
        
        # 检查是否有足够的数据
        if offset + msg.point_step > len(msg.data):
            break
        
        try:
            # 读取 x, y, z 字段
            x = struct.unpack('f', msg.data[offset:offset+4])[0]
            y = struct.unpack('f', msg.data[offset+4:offset+8])[0]
            z = struct.unpack('f', msg.data[offset+8:offset+12])[0]
            
            # 检查是否为有效点 (不是NaN或Inf)
            if not (math.isnan(x) or math.isnan(y) or math.isnan(z) or
                    math.isinf(x) or math.isinf(y) or math.isinf(z)):
                points.append([x, y, z])
        except:
            continue
    
    return points


def create_cloud2_msg(points, frame_id='odom', timestamp=None):
    """
    手动创建PointCloud2消息
    points: Nx3 或 Nx4 的numpy数组 (x, y, z, [intensity])
    """
    msg = PointCloud2()
    
    # 设置header
    if timestamp is None:
        timestamp = rclpy.time.Time(seconds=time.time())
    
    msg.header.stamp = timestamp
    msg.header.frame_id = frame_id
    
    # 点数
    num_points = len(points)
    msg.height = 1
    msg.width = num_points
    
    # 字段定义
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    
    # 点步长: 4字节(x) + 4字节(y) + 4字节(z) + 4字节(padding) + 4字节(intensity) = 20字节
    # 但为了对齐，通常使用32字节
    msg.point_step = 32
    msg.row_step = msg.point_step * num_points
    
    # 构建点数据
    data = bytearray(msg.row_step)
    
    for i, pt in enumerate(points):
        offset = i * msg.point_step
        
        x = float(pt[0]) if len(pt) > 0 else 0.0
        y = float(pt[1]) if len(pt) > 1 else 0.0
        z = float(pt[2]) if len(pt) > 2 else 0.0
        intensity = float(pt[3]) if len(pt) > 3 else 0.0
        
        # 写入数据 (小端序 float32)
        # 使用 memoryview 来避免类型问题
        mv = memoryview(data)
        struct.pack_into('f', data, offset, x)
        struct.pack_into('f', data, offset + 4, y)
        struct.pack_into('f', data, offset + 8, z)
        struct.pack_into('f', data, offset + 16, intensity)
    
    msg.data = bytes(data)
    msg.is_bigendian = False
    
    return msg


@dataclass
class TimestampedPointCloud:
    """带时间戳的点云"""
    timestamp: float
    points: np.ndarray  # Nx3 array of [x, y, z]
    yaw: float          # 雷达朝向 (弧度)


@dataclass 
class ImuState:
    """IMU状态"""
    timestamp: float
    yaw: float          # 航向角 (弧度)
    yaw_rate: float     # 角速度 (弧度/秒)


class Lidar360FusionNode(Node):
    """
    雷达360度融合节点
    
    使用IMU航向角将多帧部分扫描拼接成360度点云。
    添加可视化功能：每次生成360度点云时保存可视化图像。
    """

    def __init__(self):
        super().__init__('lidar_360_fusion_node')

        # 加载配置
        config = get_config()
        self.config = config.get('lidar_360_fusion_node', {})

        # 参数
        self.scan_period = self.config.get('scan_period', 0.065)  # 雷达扫描周期 (秒), 约15.4Hz
        self.num_frames_to_accumulate = self.config.get('num_frames_to_accumulate', 21)  # 累积帧数
        self.rotation_speed = self.config.get('rotation_speed', 259.4)  # 雷达旋转速度 (度/秒)
        self.min_distance = self.config.get('min_distance', 0.1)  # 最小距离 (米)
        self.max_distance = self.config.get('max_distance', 100.0)  # 最大距离 (米)
        self.pointcloud_timeout = self.config.get('pointcloud_timeout', 0.5)  # 点云超时 (秒)
        self.imu_timeout = self.config.get('imu_timeout', 0.2)  # IMU超时 (秒)

        # 可视化参数
        self.save_visualization = self.config.get('save_visualization', True)
        self.visualization_dir = self.config.get('visualization_dir', './lidar_visualizations')
        self.max_points_in_visualization = self.config.get('max_points_in_visualization', 20000)
        self.create_3d_visualization = self.config.get('create_3d_visualization', True)
        self.create_top_down_visualization = self.config.get('create_top_down_visualization', True)
        self.create_polar_visualization = self.config.get('create_polar_visualization', True)
        self.create_statistics_visualization = self.config.get('create_statistics_visualization', True)

        # 订阅话题
        self.scan_topic = self.config.get('scan_topic', '/utlidar/cloud')
        self.imu_topic = self.config.get('imu_topic', '/utlidar/imu')

        # 发布话题
        self.output_topic = self.config.get('output_topic', '/utlidar/cloud_360')

        # 初始化日志系统（在订阅/发布创建之后，以便输出配置信息）
        self._init_logger()

        # 订阅者
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.scan_topic,
            self.cloud_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            10
        )

        # 发布者
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10
        )

        # 数据缓冲区
        self.imu_history: Deque[ImuState] = deque(maxlen=50)
        self.accumulated_clouds: Deque[TimestampedPointCloud] = deque(maxlen=self.num_frames_to_accumulate * 2)

        # 状态
        self.last_imu_yaw = 0.0
        self.last_imu_time = None
        self.total_yaw_accumulated = 0.0  # 累积旋转角度
        self.last_publish_time = None
        self.reference_yaw = None  # 首次IMU航向作为参考

        # 初始化参考航向的标志
        self.reference_yaw_initialized = False

        # 上一次IMU yaw（用于检测航向变化）
        self.prev_imu_yaw = None

        # 上一帧点云的时间戳（用于计算帧间角度）
        self.prev_cloud_time = None

        # 可视化相关状态
        self.visualization_counter = 0
        self.visualization_lock = threading.Lock()
        self.is_saving_visualization = False

        # 创建可视化目录
        if self.save_visualization and VISUALIZATION_AVAILABLE:
            os.makedirs(self.visualization_dir, exist_ok=True)
            self.logger.info(f"可视化图像将保存到: {self.visualization_dir}")
        elif self.save_visualization and not VISUALIZATION_AVAILABLE:
            self.logger.warning("可视化功能被禁用，因为matplotlib不可用")

        # 初始化日志系统
        self._init_logger()

    def _init_logger(self):
        """初始化日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'navigation_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f'lidar_360_fusion_node_log_{timestamp}.log')

        self.logger = logging.getLogger('lidar_360_fusion_node')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        self.logger.info(f'Lidar360FusionNode started, log file: {log_file}')

        # 终端输出初始化信息
        self.get_logger().info(f'Lidar360FusionNode 初始化完成')
        self.get_logger().info(f'  订阅雷达话题: {self.scan_topic}')
        self.get_logger().info(f'  订阅IMU话题: {self.imu_topic}')
        self.get_logger().info(f'  发布360度点云: {self.output_topic}')
        self.get_logger().info(f'  累积帧数: {self.num_frames_to_accumulate}')
        self.get_logger().info(f'  IMU超时: {self.imu_timeout}秒')
        if self.save_visualization:
            self.get_logger().info(f'  可视化保存: 启用 (目录: {self.visualization_dir})')
        else:
            self.get_logger().info(f'  可视化保存: 禁用')
        self.get_logger().info(f'  详细日志已写入: {log_file}')

    def imu_callback(self, msg: Imu):
        """IMU数据回调"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # 从四元数提取航向角 (绕Z轴旋转)
        q = msg.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        
        # 初始化参考航向
        if not self.reference_yaw_initialized:
            self.reference_yaw = yaw
            self.reference_yaw_initialized = True
            self.logger.info(f'参考航向初始化: {math.degrees(yaw):.1f}°')
        
        # 计算相对参考航向的角度
        relative_yaw = yaw - self.reference_yaw
        
        # 计算角速度
        yaw_rate = 0.0
        if self.last_imu_time is not None:
            dt = timestamp - self.last_imu_time
            if dt > 0:
                yaw_rate = (yaw - self.last_imu_yaw) / dt
                # 处理角度跳变 (-PI 到 PI)
                if yaw_rate > math.pi:
                    yaw_rate -= 2 * math.pi
                elif yaw_rate < -math.pi:
                    yaw_rate += 2 * math.pi
        
        self.imu_history.append(ImuState(
            timestamp=timestamp,
            yaw=relative_yaw,
            yaw_rate=yaw_rate
        ))
        
        # 调试：打印IMU历史数量
        # self.logger.info(f'IMU历史: {len(self.imu_history)}条')
        
        self.last_imu_yaw = yaw
        self.last_imu_time = timestamp

    def cloud_callback(self, msg: PointCloud2):
        """雷达点云回调"""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # 解析点云
        points = self.parse_pointcloud(msg)
        
        if len(points) == 0:
            self.logger.warning(f'点云无有效点')
            return
        
        # 计算基于时间的雷达旋转角度（不依赖IMU）
        # 每帧旋转角度 = 时间差 × 旋转速度
        if self.prev_cloud_time is not None:
            time_delta = timestamp - self.prev_cloud_time
            # 角度增量 (度 -> 弧度)
            yaw_delta = math.radians(time_delta * self.rotation_speed)
            self.total_yaw_accumulated += yaw_delta
        else:
            yaw_delta = 0.0
        
        self.prev_cloud_time = timestamp
        
        # 使用累积的角度作为当前帧的朝向
        current_yaw = self.total_yaw_accumulated
        
        self.logger.info(f'处理点云: {len(points)}点, 帧间Δyaw={math.degrees(yaw_delta):.1f}°, 累积yaw={math.degrees(current_yaw):.1f}°')
        
        # 存储累积的点云（不转换坐标系，直接保留原始雷达坐标）
        # 原始点云已经是deskewed的，所以我们按雷达旋转角度来拼接
        self.accumulated_clouds.append(TimestampedPointCloud(
            timestamp=timestamp,
            points=points,  # 保留原始点云，不做坐标变换
            yaw=current_yaw
        ))
        
        self.logger.info(f'累积点云: 累积{len(self.accumulated_clouds)}帧, yaw={math.degrees(current_yaw):.1f}°')
        
        # 检查是否需要发布360度点云
        if self.should_publish():
            self.logger.info(f'发布360度点云!')
            self.publish_360_cloud()
            self.last_publish_time = timestamp
            self.total_yaw_accumulated = 0.0

    def parse_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        """解析PointCloud2消息为numpy数组"""
        try:
            points_list = read_points_cloud2(msg)
            
            if not points_list:
                return np.array([])
            
            points = np.array(points_list, dtype=np.float32)
            
            # 过滤距离
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
            mask = (distances >= self.min_distance) & (distances <= self.max_distance)
            points = points[mask]
            
            return points
        except Exception as e:
            self.logger.error(f'解析点云失败: {e}')
            return np.array([])

    def get_imu_at_time(self, timestamp: float) -> ImuState:
        """获取最近的IMU状态（不严格按时间匹配，适配bag播放场景）"""
        if not self.imu_history:
            return None
        
        # 取最近的IMU数据（不严格匹配时间，适配bag播放场景）
        # 因为bag播放时，点云时间戳是录制时间，IMU是实时时间
        closest = self.imu_history[-1]
        
        return closest

    def transform_to_world(self, points: np.ndarray, yaw: float) -> np.ndarray:
        """将点云从雷达坐标系转换到世界坐标系"""
        if len(points) == 0:
            return points
        
        # 旋转矩阵 (绕Z轴旋转)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        # 旋转
        x = points[:, 0]
        y = points[:, 1]
        
        world_x = x * cos_yaw - y * sin_yaw
        world_y = x * sin_yaw + y * cos_yaw
        world_z = points[:, 2]
        
        return np.column_stack([world_x, world_y, world_z])

    def should_publish(self) -> bool:
        """判断是否应该发布360度点云"""
        if len(self.accumulated_clouds) < 2:
            self.logger.info(f'未发布: 累积不足 ({len(self.accumulated_clouds)}帧)')
            return False
        
        # 方法1: 检查累积的角度范围
        if len(self.accumulated_clouds) >= self.num_frames_to_accumulate:
            self.logger.info(f'发布: 累积{len(self.accumulated_clouds)}帧 >= {self.num_frames_to_accumulate}')
            return True
        
        # 方法2: 检查累积的旋转角度
        if abs(self.total_yaw_accumulated) >= 2 * math.pi:
            self.logger.info(f'发布: 累积角度{math.degrees(self.total_yaw_accumulated):.1f}° >= 360°')
            return True
        
        self.logger.info(f'未发布: 累积{len(self.accumulated_clouds)}帧, 角度{math.degrees(self.total_yaw_accumulated):.1f}°')
        return False

    def publish_360_cloud(self):
        """发布360度融合点云，并保存可视化图像"""
        if not self.accumulated_clouds:
            return
        
        # 合并所有累积的点云（无需旋转，因为已经是odom坐标系）
        all_points = []
        all_yaws = []
        
        for cloud_data in self.accumulated_clouds:
            pts = cloud_data.points
            if len(pts) > 0:
                all_points.append(pts)
                all_yaws.append(cloud_data.yaw)
        
        if not all_points:
            return
        
        combined_points = np.vstack(all_points)
        
        # 降采样 (可选)
        # if len(combined_points) > 50000:
        #     indices = np.random.choice(len(combined_points), 50000, replace=False)
        #     combined_points = combined_points[indices]
        
        # 添加intensity字段 (使用距离作为intensity)
        intensities = np.sqrt(combined_points[:, 0]**2 + combined_points[:, 1]**2 + combined_points[:, 2]**2)
        mask = 0.1 < combined_points[:, 2]
        print(f"高度过滤: {np.sum(mask)} / {len(combined_points)} 点")
        
        # 构建点数据 (x, y, z, intensity)
        points_with_intensity = np.zeros((len(combined_points), 4), dtype=np.float32)
        points_with_intensity[:, 0:3] = combined_points
        points_with_intensity[:, 3] = intensities
        
        # 创建PointCloud2消息（使用base_link作为坐标系，方便rviz显示）
        timestamp = self.get_clock().now().to_msg()
        cloud_msg = create_cloud2_msg(points_with_intensity, frame_id='base_link', timestamp=timestamp)
        
        self.cloud_pub.publish(cloud_msg)
        
        self.logger.info(f'发布360度点云: {len(combined_points)} 点')
        
        # 保存可视化图像（异步进行，避免阻塞主线程）
        if self.save_visualization and VISUALIZATION_AVAILABLE and not self.is_saving_visualization:
            self.save_visualization_async(combined_points[mask], intensities[mask], all_yaws)

    def save_visualization_async(self, points: np.ndarray, intensities: np.ndarray, yaws: List[float]):
        """异步保存可视化图像"""
        if self.is_saving_visualization:
            self.logger.warning("已有可视化任务正在运行，跳过本次保存")
            return
            
        # 启动新线程保存可视化
        thread = threading.Thread(
            target=self._save_visualization_thread,
            args=(points.copy(), intensities.copy(), yaws.copy())
        )
        thread.daemon = True
        thread.start()

    def _save_visualization_thread(self, points: np.ndarray, intensities: np.ndarray, yaws: List[float]):
        """在新线程中保存可视化图像"""
        with self.visualization_lock:
            self.is_saving_visualization = True
            try:
                self._save_visualization_impl(points, intensities, yaws)
            except Exception as e:
                self.logger.error(f"保存可视化图像时出错: {e}")
            finally:
                self.is_saving_visualization = False

    def _save_visualization_impl(self, points: np.ndarray, intensities: np.ndarray, yaws: List[float]):
        """实际保存可视化图像的实现"""
        if not VISUALIZATION_AVAILABLE:
            return
            
        if len(points) == 0:
            self.logger.warning("点云为空，跳过可视化")
            return
        
        # 降采样用于可视化（如果点太多）
        if len(points) > self.max_points_in_visualization:
            indices = np.random.choice(len(points), self.max_points_in_visualization, replace=False)
            points = points[indices]
            intensities = intensities[indices]
        
        # 准备数据
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # 计算极坐标
        ranges = np.sqrt(x**2 + y**2 + z**2)
        angles = np.arctan2(y, x)  # 相对于雷达的方位角
        
        # 创建时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.visualization_counter += 1
        counter_str = f"{self.visualization_counter:04d}"
        
        # 创建综合可视化图形
        fig = self._create_comprehensive_visualization(
            x, y, z, ranges, angles, intensities, yaws, timestamp, counter_str
        )
        
        # 保存图像
        filename = os.path.join(self.visualization_dir, f"lidar_panorama_{timestamp}_{counter_str}.png")
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"可视化图像已保存: {filename}")
        
        # 单独保存3D可视化（可选）
        if self.create_3d_visualization:
            fig_3d = self._create_3d_visualization(x, y, z, intensities, timestamp, counter_str)
            filename_3d = os.path.join(self.visualization_dir, f"lidar_3d_{timestamp}_{counter_str}.png")
            fig_3d.savefig(filename_3d, dpi=150, bbox_inches='tight')
            plt.close(fig_3d)
            
        # 单独保存俯视图（可选）
        if self.create_top_down_visualization:
            fig_top = self._create_top_down_visualization(x, y, intensities, timestamp, counter_str)
            filename_top = os.path.join(self.visualization_dir, f"lidar_top_{timestamp}_{counter_str}.png")
            fig_top.savefig(filename_top, dpi=150, bbox_inches='tight')
            plt.close(fig_top)

    def _create_comprehensive_visualization(self, x, y, z, ranges, angles, intensities, yaws, timestamp, counter_str):
        """创建综合可视化图形（包含多个子图）"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'LiDAR 360° Panorama - {timestamp} (#{counter_str})', fontsize=16)
        
        # 1. 俯视图 (X-Y平面)
        if self.create_top_down_visualization:
            ax1 = plt.subplot(2, 3, 1)
            scatter1 = ax1.scatter(x, y, c=intensities, cmap='viridis', s=1, alpha=0.7)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('Top View (X-Y Plane)')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            plt.colorbar(scatter1, ax=ax1, label='Distance (m)')
        
        # 2. 侧视图 (X-Z平面)
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(x, z, c=intensities, cmap='plasma', s=1, alpha=0.7)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Side View (X-Z Plane)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Distance (m)')
        
        # 3. 极坐标图
        if self.create_polar_visualization:
            ax3 = plt.subplot(2, 3, 3, projection='polar')
            scatter3 = ax3.scatter(angles, ranges, c=intensities, cmap='hsv', s=1, alpha=0.7)
            ax3.set_title('Polar View (Range vs Angle)')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter3, ax=ax3, label='Distance (m)')
        
        # 4. 3D视图
        if self.create_3d_visualization:
            ax4 = plt.subplot(2, 3, 4, projection='3d')
            scatter4 = ax4.scatter(x, y, z, c=intensities, cmap='viridis', s=1, alpha=0.7)
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            ax4.set_zlabel('Z (m)')
            ax4.set_title('3D View')
        
        # 5. 距离分布直方图
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(ranges, bins=50, range=(0, np.max(ranges)+1), edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Range (m)')
        ax5.set_ylabel('Point Count')
        ax5.set_title('Range Distribution')
        ax5.grid(True, alpha=0.3)
        
        # 6. 角度分布直方图
        ax6 = plt.subplot(2, 3, 6)
        angles_deg = np.degrees(angles)
        ax6.hist(angles_deg, bins=36, range=(-180, 180), edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Angle (degrees)')
        ax6.set_ylabel('Point Count')
        ax6.set_title('Angle Distribution')
        ax6.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = (
            f"总点数: {len(x):,}\n"
            f"距离范围: [{np.min(ranges):.2f}, {np.max(ranges):.2f}] m\n"
            f"平均距离: {np.mean(ranges):.2f} m\n"
            f"X范围: [{np.min(x):.2f}, {np.max(x):.2f}] m\n"
            f"Y范围: [{np.min(y):.2f}, {np.max(y):.2f}] m\n"
            f"Z范围: [{np.min(z):.2f}, {np.max(z):.2f}] m\n"
            f"累积帧数: {len(yaws)}\n"
            f"覆盖角度: {np.degrees(max(yaws) - min(yaws)):.1f}°"
        )
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        return fig

    def _create_3d_visualization(self, x, y, z, intensities, timestamp, counter_str):
        """创建单独的3D可视化"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x, y, z, c=intensities, cmap='viridis', s=2, alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D LiDAR Point Cloud - {timestamp} (#{counter_str})')
        
        plt.colorbar(scatter, ax=ax, label='Distance (m)')
        plt.tight_layout()
        return fig

    def _create_top_down_visualization(self, x, y, intensities, timestamp, counter_str):
        """创建单独的俯视图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(x, y, c=intensities, cmap='viridis', s=2, alpha=0.7)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Top-Down LiDAR View - {timestamp} (#{counter_str})')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.colorbar(scatter, ax=ax, label='Distance (m)')
        plt.tight_layout()
        return fig

    def quaternion_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        """从四元数提取 yaw 角 (绕Z轴旋转)"""
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw


def main(args=None):
    rclpy.init(args=args)
    
    node = Lidar360FusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import matplotlib

    # 提高最大打开图形数量
    matplotlib.rcParams['figure.max_open_warning'] = 100
    main()
