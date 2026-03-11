#!/usr/bin/env python3
"""
高度地图订阅与可视化 - 简化版
订阅一次高度地图，转换为点云，保存可视化图像后退出。
"""

import sys
import time
import numpy as np

# 尝试导入Unitree相关模块
try:
    import unitree_sdk2py.go2
    from unitree_robot.channel.channel_subscriber import ChannelSubscriber
    from unitree_robot.channel.channel_factory import ChannelFactory
    import unitree_go.msg.dds_ as unitree_msgs
    UNITREE_AVAILABLE = True
except ImportError as e:
    print(f"错误: 无法导入Unitree SDK模块: {e}")
    print("请确保已安装Unitree Go2 SDK的Python绑定")
    sys.exit(1)

# 可视化相关导入
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"错误: 无法导入matplotlib: {e}")
    print("请运行: pip install matplotlib")
    sys.exit(1)

TOPIC_HEIGHTMAP = "rt/utlidar/height_map_array"


class SimpleHeightMapVisualizer:
    """简单高度地图可视化器 - 单次接收并保存"""
    
    def __init__(self, network_interface: str):
        """初始化"""
        # 初始化DDS通道
        ChannelFactory.Instance().Init(0, network_interface)
        
        # 创建订阅者
        self.subscriber = ChannelSubscriber(TOPIC_HEIGHTMAP)
        self.received_data = None
        self.received = False
        
        print(f"初始化完成，等待高度地图数据...")
    
    def _message_handler(self, message):
        """消息处理回调函数"""
        if self.received:
            return  # 只接收一次
            
        self.received_data = message
        self.received = True
        
        print("收到高度地图数据")
    
    def wait_for_data(self, timeout=10.0):
        """等待数据，最多等待timeout秒"""
        # 设置回调
        self.subscriber.InitChannel(self._message_handler)
        
        start_time = time.time()
        while not self.received:
            if time.time() - start_time > timeout:
                print(f"错误: 在{timeout}秒内未收到数据")
                return False
            time.sleep(0.1)
        
        return True
    
    def process_and_visualize(self, output_file="height_map_simple.png"):
        """处理数据并可视化"""
        if not self.received_data:
            print("错误: 没有收到数据")
            return False
        
        map_msg = self.received_data
        
        # 打印基本信息
        print(f"处理高度地图:")
        print(f"  宽度: {map_msg.width}")
        print(f"  高度: {map_msg.height}")
        print(f"  分辨率: {map_msg.resolution}")
        
        # 转换为点云
        width = map_msg.width
        height = map_msg.height
        resolution = map_msg.resolution
        origin_x = map_msg.origin[0]
        origin_y = map_msg.origin[1]
        height_data = map_msg.data
        
        point_cloud = []
        
        for iy in range(height):
            for ix in range(width):
                index = ix + width * iy
                z = height_data[index]
                
                # 跳过无效值
                if abs(z - 1.0e9) < 0.1:
                    continue
                
                x = ix * resolution + origin_x
                y = iy * resolution + origin_y
                point_cloud.append([x, y, z])
        
        if not point_cloud:
            print("错误: 没有有效的点云数据")
            return False
        
        points = np.array(point_cloud, dtype=np.float32)
        print(f"转换得到 {len(points)} 个点")
        
        # 创建简单可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. X-Y平面视图
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], 
                              c=points[:, 2], cmap='viridis', s=2, alpha=0.7)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Height Map - Top View')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        plt.colorbar(scatter1, ax=ax1, label='Height (m)')
        
        # 2. 3D视图
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                              c=points[:, 2], cmap='plasma', s=2, alpha=0.7)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Height (m)')
        ax2.set_title('Height Map - 3D View')
        
        # 添加统计信息
        stats_text = (
            f"点数: {len(points):,}\n"
            f"高度范围: [{np.min(points[:,2]):.2f}, {np.max(points[:,2]):.2f}] m\n"
            f"平均高度: {np.mean(points[:,2]):.2f} m\n"
            f"分辨率: {resolution} m"
        )
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle(f'Height Map Visualization - {time.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=12)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"可视化图像已保存: {output_file}")
        
        return True


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python subscribe_height_map_simple.py <network_interface> [output_file]")
        print("例如: python subscribe_height_map_simple.py eth0 height_map.png")
        sys.exit(1)
    
    network_interface = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "height_map.png"
    
    # 创建可视化器
    visualizer = SimpleHeightMapVisualizer(network_interface)
    
    # 等待数据
    if not visualizer.wait_for_data(timeout=5.0):
        sys.exit(1)
    
    # 处理并可视化
    if visualizer.process_and_visualize(output_file):
        print("完成!")
    else:
        print("失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
