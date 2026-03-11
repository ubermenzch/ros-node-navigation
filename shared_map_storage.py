#!/usr/bin/env python3
"""
共享内存模块 - 用于存储导航地图数据

提供线程安全的地图数据存储和访问，使用互斥锁实现互斥访问。
"""

import numpy as np
import threading
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class MapMetadata:
    """地图元数据

    坐标系：X轴正方向=正北方，Y轴正方向=正东方（ENU）
    """
    resolution: float      # 分辨率 (米/格)
    width: int            # 地图宽度 (格数)
    height: int           # 地图高度 (格数)
    origin_x: float       # 地图原点X坐标 (米) - 地图左下角（原点）
    origin_y: float       # 地图原点Y坐标 (米) - 地图左下角（原点）
    robot_x: float        # 机器狗在地图中的X坐标 (米) - 始终为0
    robot_y: float        # 机器狗在地图中的Y坐标 (米) - 始终为0
    gps_points: List[dict]  # GPS点列表 (原始GPS坐标)

    # 坐标系转换参数
    origin_lat: float = 0.0   # 地图原点对应的GPS纬度 (第一个GPS点)
    origin_lon: float = 0.0   # 地图原点对应的GPS经度 (第一个GPS点)
    meters_per_degree_lat: float = 111320.0  # 每度纬度对应的米数
    meters_per_degree_lon: float = 111320.0  # 每度经度对应的米数

    # odom坐标系转换
    odom_offset_x: float = 0.0  # 地图原点相对于odom原点的X偏移 (米)
    odom_offset_y: float = 0.0  # 地图原点相对于odom原点的Y偏移 (米)
    odom_offset_yaw: float = 0.0  # 地图原点相对于odom原点的航向角偏移 (弧度)


class SharedMapStorage:
    """
    共享地图存储类
    
    使用互斥锁实现线程安全的地图数据存储和访问。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式，确保只有一个共享存储实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 互斥锁，用于保护地图数据的访问
        self._data_lock = threading.RLock()
        
        # 地图数据
        self._map_data: Optional[np.ndarray] = None
        self._metadata: Optional[MapMetadata] = None
        
        # 标志位
        self._has_map = False

        # 记录更新的区域（用于发布 OccupancyGridUpdate）
        self._last_update_bbox = None  # (min_col, min_row, max_col, max_row)

    def get_last_update_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """获取上次更新的区域边界 (min_col, min_row, max_col, max_row)"""
        with self._data_lock:
            return self._last_update_bbox

    def clear_last_update_bbox(self):
        """清除上次更新的区域边界"""
        with self._data_lock:
            self._last_update_bbox = None
        
        self._initialized = True
    
    def set_map(self, map_data: np.ndarray, metadata: MapMetadata):
        """
        设置地图数据（线程安全）
        
        Args:
            map_data: 地图数据 numpy 数组
            metadata: 地图元数据
        """
        with self._data_lock:
            self._map_data = map_data.copy() if map_data is not None else None
            self._metadata = metadata
            self._has_map = self._map_data is not None
    
    def get_map(self) -> Tuple[Optional[np.ndarray], Optional[MapMetadata]]:
        """
        获取地图数据（线程安全）
        
        Returns:
            (map_data, metadata) 元组
        """
        with self._data_lock:
            if self._map_data is not None:
                return self._map_data.copy(), self._metadata
            return None, None
    
    def get_map_info(self) -> Optional[dict]:
        """
        获取地图信息（线程安全）

        Returns:
            地图信息字典，如果无地图则返回None
        """
        with self._data_lock:
            if not self._has_map or self._metadata is None:
                return None

            return {
                'has_map': self._has_map,
                'resolution': self._metadata.resolution,
                'width': self._metadata.width,
                'height': self._metadata.height,
                'origin_x': self._metadata.origin_x,
                'origin_y': self._metadata.origin_y,
                'robot_x': self._metadata.robot_x,
                'robot_y': self._metadata.robot_y,
                'gps_points': self._metadata.gps_points.copy() if self._metadata.gps_points else [],

                # GPS和地图坐标转换
                'origin_lat': self._metadata.origin_lat,
                'origin_lon': self._metadata.origin_lon,
                'meters_per_degree_lat': self._metadata.meters_per_degree_lat,
                'meters_per_degree_lon': self._metadata.meters_per_degree_lon,

                # odom坐标系转换
                'odom_offset_x': self._metadata.odom_offset_x,
                'odom_offset_y': self._metadata.odom_offset_y,
                'odom_offset_yaw': self._metadata.odom_offset_yaw,
            }
    
    def has_map(self) -> bool:
        """检查是否有地图数据"""
        with self._data_lock:
            return self._has_map
    
    def clear(self):
        """清除地图数据"""
        with self._data_lock:
            self._map_data = None
            self._metadata = None
            self._has_map = False

    # ==================== 坐标转换方法 ====================

    def gps_to_map(self, latitude: float, longitude: float) -> Tuple[float, float]:
        """
        GPS坐标转换为地图坐标 (米)

        坐标系：X轴正方向=正北方（纬度方向），Y轴正方向=正东方（经度方向）

        Args:
            latitude: GPS纬度
            longitude: GPS经度

        Returns:
            (x, y) 地图坐标 (米)，原点在机器狗当前位置
        """
        with self._data_lock:
            if self._metadata is None:
                return None, None

            delta_lat = latitude - self._metadata.origin_lat
            delta_lon = longitude - self._metadata.origin_lon

            # X轴=北方（纬度变化），Y轴=东方（经度变化）
            x = delta_lat * self._metadata.meters_per_degree_lat
            y = delta_lon * self._metadata.meters_per_degree_lon

            return x, y

    def map_to_gps(self, x: float, y: float) -> Tuple[float, float]:
        """
        地图坐标转换为GPS坐标

        坐标系：X轴正方向=正北方（纬度方向），Y轴正方向=正东方（经度方向）

        Args:
            x: 地图X坐标 (米) - 北方方向
            y: 地图Y坐标 (米) - 东方方向

        Returns:
            (latitude, longitude) GPS坐标
        """
        with self._data_lock:
            if self._metadata is None:
                return None, None

            lat = self._metadata.origin_lat + x / self._metadata.meters_per_degree_lat
            lon = self._metadata.origin_lon + y / self._metadata.meters_per_degree_lon

            return lat, lon

    def map_to_odom(self, map_x: float, map_y: float) -> Tuple[float, float]:
        """
        地图坐标转换为odom坐标

        公式: odom = map - odom_offset

        Args:
            map_x: 地图X坐标 (米)
            map_y: 地图Y坐标 (米)

        Returns:
            (odom_x, odom_y) odom坐标 (米)
        """
        with self._data_lock:
            if self._metadata is None:
                return None, None

            odom_x = map_x - self._metadata.odom_offset_x
            odom_y = map_y - self._metadata.odom_offset_y

            return odom_x, odom_y

    def odom_to_map(self, odom_x: float, odom_y: float) -> Tuple[float, float]:
        """
        odom坐标转换为地图坐标

        公式: map = odom + odom_offset

        Args:
            odom_x: odom X坐标 (米)
            odom_y: odom Y坐标 (米)

        Returns:
            (map_x, map_y) 地图坐标 (米)
        """
        with self._data_lock:
            if self._metadata is None:
                return None, None

            map_x = odom_x + self._metadata.odom_offset_x
            map_y = odom_y + self._metadata.odom_offset_y

            return map_x, map_y

    def gps_to_odom(self, latitude: float, longitude: float) -> Tuple[float, float]:
        """
        GPS坐标直接转换为odom坐标

        Args:
            latitude: GPS纬度
            longitude: GPS经度

        Returns:
            (odom_x, odom_y) odom坐标 (米)
        """
        map_x, map_y = self.gps_to_map(latitude, longitude)
        if map_x is None:
            return None, None
        return self.map_to_odom(map_x, map_y)

    def odom_to_gps(self, odom_x: float, odom_y: float) -> Tuple[float, float]:
        """
        odom坐标直接转换为GPS坐标

        Args:
            odom_x: odom X坐标 (米)
            odom_y: odom Y坐标 (米)

        Returns:
            (latitude, longitude) GPS坐标
        """
        map_x, map_y = self.odom_to_map(odom_x, odom_y)
        if map_x is None:
            return None, None
        return self.map_to_gps(map_x, map_y)

    def update_local_region(self, local_map: np.ndarray, center_x: float, center_y: float,
                           local_resolution: float, fill_value: int = 100):
        """
        更新共享地图的局部区域

        Args:
            local_map: 局部地图数据 (2D numpy数组)
            center_x: 局部地图中心在全局地图中的X坐标 (米)
            center_y: 局部地图中心在全局地图中的Y坐标 (米)
            local_resolution: 局部地图分辨率 (米/格)，必须与全局地图相同
            fill_value: 填充值，用于标记障碍物 (默认100)

        Returns:
            bool: 更新是否成功
        """
        with self._data_lock:
            if not self._has_map or self._map_data is None or self._metadata is None:
                return False

            # 验证分辨率一致
            if abs(self._metadata.resolution - local_resolution) > 1e-6:
                return False

            # 计算局部地图的边界（以格为单位）
            local_height, local_width = local_map.shape

            # 计算局部地图左上角在全局地图中的坐标（米）
            top_left_x = center_x - local_width * local_resolution / 2
            top_left_y = center_y + local_height * local_resolution / 2

            # 转换为全局地图的网格索引
            top_left_col = int((top_left_x - self._metadata.origin_x) / local_resolution)
            top_left_row = int((top_left_y - self._metadata.origin_y) / local_resolution)

            # 遍历局部地图，填充到全局地图
            updated = False
            min_updated_col = None
            min_updated_row = None
            max_updated_col = None
            max_updated_row = None

            for local_row in range(local_height):
                for local_col in range(local_width):
                    if local_map[local_row, local_col] == fill_value:
                        # 计算在全局地图中的位置
                        global_row = top_left_row - local_row
                        global_col = top_left_col + local_col

                        # 检查是否在全局地图范围内
                        if 0 <= global_row < self._metadata.height and \
                           0 <= global_col < self._metadata.width:
                            self._map_data[global_row, global_col] = fill_value
                            updated = True

                            # 记录更新的边界
                            if min_updated_col is None or global_col < min_updated_col:
                                min_updated_col = global_col
                            if min_updated_row is None or global_row < min_updated_row:
                                min_updated_row = global_row
                            if max_updated_col is None or global_col > max_updated_col:
                                max_updated_col = global_col
                            if max_updated_row is None or global_row > max_updated_row:
                                max_updated_row = global_row

            # 记录更新的区域
            if updated:
                self._last_update_bbox = (min_updated_col, min_updated_row, max_updated_col, max_updated_row)

            return updated

    def get_local_map_region(self, center_x: float, center_y: float,
                             half_width: float, half_height: float) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        获取共享地图的局部区域

        Args:
            center_x: 区域中心X坐标 (米)
            center_y: 区域中心Y坐标 (米)
            half_width: 区域半宽 (米)
            half_height: 区域半高 (米)

        Returns:
            (局部地图数据, 局部元数据)，如果失败则返回(None, None)
        """
        with self._data_lock:
            if not self._has_map or self._map_data is None or self._metadata is None:
                return None, None

            resolution = self._metadata.resolution

            # 计算区域边界（米）
            min_x = center_x - half_width
            max_x = center_x + half_width
            min_y = center_y - half_height
            max_y = center_y + half_height

            # 转换为网格索引
            min_col = int((min_x - self._metadata.origin_x) / resolution)
            max_col = int((max_x - self._metadata.origin_x) / resolution)
            min_row = int((self._metadata.origin_y - max_y) / resolution)
            max_row = int((self._metadata.origin_y - min_y) / resolution)

            # 裁剪到地图范围内
            min_col = max(0, min_col)
            max_col = min(self._metadata.width - 1, max_col)
            min_row = max(0, min_row)
            max_row = min(self._metadata.height - 1, max_row)

            if max_col < min_col or max_row < min_row:
                return None, None

            # 提取局部地图
            local_map = self._map_data[min_row:max_row+1, min_col:max_col+1].copy()

            # 计算局部地图的元数据
            local_origin_x = self._metadata.origin_x + min_col * resolution
            local_origin_y = self._metadata.origin_y + (self._metadata.height - max_row - 1) * resolution

            local_metadata = {
                'resolution': resolution,
                'width': local_map.shape[1],
                'height': local_map.shape[0],
                'origin_x': local_origin_x,
                'origin_y': local_origin_y,
            }

            return local_map, local_metadata


# 全局共享存储实例
shared_map_storage = SharedMapStorage()


def get_shared_map() -> SharedMapStorage:
    """获取共享地图存储实例"""
    return shared_map_storage
