#!/bin/bash
# 清理 navigation_system 启动的所有进程

echo "正在清理 navigation_system 相关进程..."

# 查找并杀死 main.py 进程
PID=$(ps aux | grep "python3 main.py" | grep -v grep | awk '{print $2}')

if [ -n "$PID" ]; then
    echo "找到 main.py 进程 (PID: $PID)，正在终止..."
    kill -2 $PID  # SIGINT，允许优雅退出
    
    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "进程已终止"
            break
        fi
        sleep 0.5
    done
    
    # 如果进程仍然存在，强制杀死
    if ps -p $PID > /dev/null 2>&1; then
        echo "进程未响应 SIGINT，强制杀死..."
        kill -9 $PID 2>/dev/null
    fi
else
    echo "未找到运行中的 main.py 进程"
fi

# 清理可能残留的 ROS2 节点进程
ROS_PIDS=$(ps aux | grep -E "(ekf_fusion_node|lidar_costmap_node|planner_node|navigation_system)" | grep -v grep | awk '{print $2}')
if [ -n "$ROS_PIDS" ]; then
    echo "发现残留进程，正在清理..."
    echo "$ROS_PIDS" | xargs -r kill -2 2>/dev/null
    sleep 1
    echo "$ROS_PIDS" | xargs -r kill -9 2>/dev/null
fi

echo "清理完成"
