# ROS Node Navigation

## 1. 项目简介

本项目是一个基于 ROS 2 的多节点导航系统，面向机器狗/移动机器人在 GNSS、里程计、激光雷达和深度强化学习控制器共同参与下的自主导航任务。

系统主入口为 `multi_main.py`，运行时会以多进程方式启动各功能节点：

- `ekf_fusion_node.py`：融合 RTK/GNSS、RTK yaw 与里程计，输出 `map` 坐标系定位。
- `lidar_costmap_node.py`：处理激光雷达数据，生成局部障碍物观测和局部 costmap。
- `map_planner_node.py`：根据导航航点、定位和局部 costmap 维护地图并规划路径。
- `controller_node.py`：加载 SAC 控制模型，根据路径和障碍物观测发布速度控制指令。
- `front_video_recorder_node.py`：可选启动，保存宇树 Go2 前置摄像头 H.264 视频流。

运行时产生的日志和视频会自动创建到 `logs/` 与 `videos/` 目录，这些运行产物不会提交到 GitHub。

## 3. 运行环境

建议运行环境：

- Linux 系统。
- ROS 2 环境，需支持 `rclpy`。
- Python 3.8 或更高版本。
- 可用的 RTK/GNSS、里程计和激光雷达数据源。
- 如果启用前置视频录制，需要可访问宇树 Go2 前置摄像头 RTP/H.264 多播流。

主要 ROS 消息/功能包依赖：

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

控制器模型文件默认路径：

```text
models/ETH25/SAC_actor.pth
```

## 5. 配置说明

项目配置统一放在 `config.yaml` 中。修改配置后重新运行 `multi_main.py` 即可生效。

### 通用配置

`common`：

- `resolution`：地图分辨率，单位为米/格。
- `test_mode`：测试模式。设为 `true` 时不启动 `controller_node`，可使用遥控器手动控制机器人。

### EKF 融合节点

`ekf_fusion_node`：

- `frequency`：融合频率。
- `subscriptions`：配置 GNSS、里程计、RTK yaw 输入话题。
- `publications`：配置融合定位输出话题。
- `use_odom_yaw_as_world_yaw`：是否使用 odom yaw 作为世界系 yaw。
- `use_2d_ekf`：是否启用 2D EKF。
- `ekf_2d`：GNSS/yaw 方差、过程噪声、Mahalanobis 异常剔除等参数。
- `log_enabled`：是否启用该节点文件日志。

### 激光雷达 Costmap 节点

`lidar_costmap_node`：

- `subscriptions.scan_topic`：激光雷达输入话题。
- `publications`：障碍物距离、局部 costmap、体素点云输出话题。
- `scan_range`：雷达处理范围。
- `min_height` / `max_height`：高度过滤范围。
- `voxel_size` / `min_points_per_voxel`：体素滤波参数。
- `bin_num` / `ray_angle_step_deg`：强化学习观测输入的角度分区参数。

### 地图与路径规划节点

`map_planner_node`：

- `square_size` / `road_width` / `road_sample_step`：初始道路地图生成参数。
- `inflation_enabled` / `inflation_margin`：障碍物膨胀配置。
- `planning_frequency`：路径规划频率。
- `arrival_threshold`：航点到达阈值。
- `allow_diagonal_astar`：A* 是否允许斜向移动。
- `use_bidirectional_astar`：是否使用双向 A*。
- `max_astar_nodes`：单次 A* 最大扩展节点数。
- `subscriptions` / `publications`：规划节点输入输出话题。

### 控制器节点

`controller_node`：

- `subscriptions.path_topic`：规划路径输入话题。
- `subscriptions.lidar_obs_topic`：障碍物观测输入话题。
- `subscriptions.odom_topic`：里程计输入话题。
- `publications.cmd_topic`：速度控制指令输出话题。
- `model_path`：SAC actor 模型路径。
- `max_v` / `max_w`：最大线速度和最大角速度。
- `frequency`：控制频率。

### 前置视频录制节点

`front_video_recorder_node`：

- `enabled`：是否随 `multi_main.py` 启动视频录制节点。
- `iface` / `address` / `port`：摄像头 RTP/H.264 多播流参数。
- `container`：输出容器，支持 `mp4` 或 `mkv`。
- `output_dir`：视频输出目录，默认 `videos`，不存在会自动创建。
- `output_filename` / `output_path`：自定义输出文件名或完整输出路径。
- `overwrite_existing`：是否覆盖已有视频文件。
- `shutdown_timeout_sec` / `eos_timeout_sec`：停止录制时等待视频文件完成写入的超时时间。

## 6. 运行方法

进入项目根目录后运行：

```bash
python3 multi_main.py
```

启动后，程序会根据 `config.yaml` 自动启动各节点，并在终端打印本次运行的 session、日志目录、测试模式和视频录制状态。

停止系统：

```text
Ctrl+C
```

程序会尝试优雅关闭所有子进程。如果启用了视频录制节点，会等待 GStreamer 写入 EOS，以便视频文件正常保存。

运行输出：

- 日志目录：`logs/navigation_<timestamp>/`
- 视频目录：`videos/`

这两个目录都会在运行时自动创建，无需提前放入仓库。

## 10. 作者与许可证

维护者：`ubermenzch`

GitHub 仓库：

```text
https://github.com/ubermenzch/ros-node-navigation
```

许可证：当前仓库尚未提供 `LICENSE` 文件。公开使用、分发或二次开发前，建议先补充明确的开源许可证。
