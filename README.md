# ROS Node Navigation

## 1. 项目简介

本项目是一个基于 ROS 2 的多节点导航系统，面向机器狗/移动机器人在 GNSS、里程计、激光雷达和深度强化学习控制器共同参与下的自主导航任务。

系统主入口为 `multi_main.py`，运行时会以多进程方式启动各功能节点：

- `ekf_fusion_node.py`：融合 RTK/GNSS、RTK yaw 与里程计，输出 `map` 坐标系定位。
- `lidar_costmap_node.py`：处理激光雷达数据，生成局部障碍物观测和局部 costmap。
- `map_planner_node.py`：根据导航航点、定位和局部 costmap 维护地图并规划路径。
- `controller_node.py`：加载 SAC 控制模型，根据路径和障碍物观测发布速度控制指令。
- `front_video_recorder_node.py`：可选启动，保存宇树 Go2 前置摄像头 H.264 视频流。

## 2. 依赖

主要 ROS2（仅在foxy做测试） 消息/功能包依赖：

- `sensor_msgs`
- `nav_msgs`
- `geometry_msgs`
- `visualization_msgs`
- `map_msgs`
- `builtin_interfaces`
- `tf2_ros`

主要 Python 依赖：

- `numpy`
- `pyproj`
- `PyYAML`
- `torch`
- `psutil`
- `PyGObject` / `GStreamer`，仅前置视频录制节点需要

## 3. 运行方法

进入项目根目录后运行：

```bash
python3 multi_main.py
```

停止系统：

```text
Ctrl+C
```

## 4. 作者与许可证

维护者：`ubermenzch`

GitHub 仓库：

```text
https://github.com/ubermenzch/ros-node-navigation
```

