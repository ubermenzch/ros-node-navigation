#!/usr/bin/env python3
"""
多进程导航系统启动文件

将每个节点运行在独立进程中，充分利用多核 CPU。
每个节点完全独立，拥有自己的执行器和进程空间。

进程分配：
- ekf_fusion_node:    EKF 定位融合（单线程）
- lidar_costmap_node: 激光雷达处理（单线程）
- map_planner_node:   地图+规划（3个线程组：数据/雷达/规划）
- front_video_recorder_node: 前置摄像头 H.264 录像（可选）
- controller_node:    控制（单线程）

使用方式：
  python3 multi_main.py

按 Ctrl+C 可优雅关闭所有节点。
"""

import os
import sys
import signal
import time
import multiprocessing as mp
from multiprocessing import Process
from datetime import datetime
from typing import List, Tuple


# ========== 工具函数 ==========

def get_project_root() -> str:
    """获取项目根目录"""
    return os.path.dirname(os.path.abspath(__file__))


def ensure_log_dir(log_dir: str) -> str:
    """确保日志目录存在，并返回规范化后的路径"""
    resolved_log_dir = os.path.abspath(os.path.expanduser(log_dir))
    os.makedirs(resolved_log_dir, exist_ok=True)
    return resolved_log_dir


# ========== 节点包装函数 ==========

def node_wrapper(
    node_module: str,
    run_func_name: str,
    node_name: str,
    log_dir: str,
    log_timestamp: str,
    init_delay: float = 0.0
) -> None:
    """
    节点包装函数，运行在子进程中

    Args:
        node_module: 模块名 (如 'ekf_fusion_node')
        run_func_name: 入口函数名 (如 'run_ekf_fusion_node')
        node_name: 节点名
        log_dir: 日志目录
        log_timestamp: 时间戳
        init_delay: 初始化延迟（秒），用于控制启动顺序
    """
    # 设置进程名
    mp.current_process().name = node_name
    log_dir = ensure_log_dir(log_dir)

    print(f"[{node_name}] starting in process {os.getpid()}", flush=True)

    # 等待延迟（用于控制启动顺序）
    if init_delay > 0:
        print(f"[{node_name}] waiting {init_delay}s before initialization...", flush=True)
        time.sleep(init_delay)

    try:
        # 动态导入并调用节点入口函数
        module = __import__(node_module, fromlist=[run_func_name])

        run_func = getattr(module, run_func_name, None)
        if run_func is None:
            print(
                f"[{node_name}] function '{run_func_name}' not found in module '{node_module}'",
                file=sys.stderr,
                flush=True,
            )
            return

        # 调用节点的入口函数
        run_func(log_dir=log_dir, log_timestamp=log_timestamp)

    except Exception as e:
        print(f"[{node_name}] crashed: {e}", file=sys.stderr, flush=True)
        raise
    finally:
        print(f"[{node_name}] exited", flush=True)


# ========== 进程管理器 ==========

class ProcessManager:
    """进程管理器 - 启动、监控、关闭所有节点进程"""

    def __init__(self):
        self.processes: List[Tuple[str, Process]] = []

        # 生成统一时间戳
        self.start_timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 创建统一日志目录
        project_root = get_project_root()
        self.log_dir: str = os.path.join(project_root, 'logs', f'navigation_{self.start_timestamp}')
        self.log_dir = ensure_log_dir(self.log_dir)

    def start_node(
        self,
        node_module: str,
        run_func_name: str,
        node_name: str,
        init_delay: float = 0.0
    ) -> Process:
        """启动单个节点进程"""
        p = Process(
            target=node_wrapper,
            args=(
                node_module,
                run_func_name,
                node_name,
                self.log_dir,
                self.start_timestamp,
                init_delay
            ),
            name=node_name
        )
        p.start()
        self.processes.append((node_name, p))
        print(f"  + {node_name:<20} PID: {p.pid}  (delay: {init_delay:.1f}s)")
        return p

    def _print_banner(self, is_test_mode: bool, front_video_recorder_enabled: bool) -> None:
        """打印启动横幅"""
        print(f"\n{'=' * 70}")
        print(f"       MULTI-PROCESS NAVIGATION SYSTEM")
        print(f"{'=' * 70}")
        print(f"  Session:     {self.start_timestamp}")
        print(f"  Log dir:     {self.log_dir}")
        print(f"  Test mode:   {'YES (controller disabled)' if is_test_mode else 'NO'}")
        print(f"  Front video recorder: {'YES' if front_video_recorder_enabled else 'NO'}")
        print(f"{'=' * 70}")
        print(f"  Starting nodes:")
        print()

    def _print_ready(self) -> None:
        """打印就绪信息"""
        print()
        print(f"{'=' * 70}")
        print(f"  All nodes started!")
        print(f"  Press Ctrl+C to gracefully shutdown all nodes")
        print(f"{'=' * 70}\n")

    def _print_status(self) -> None:
        """打印所有进程状态"""
        print(f"\n  {'Node':<25} {'PID':<8} {'Status'}")
        print(f"  {'-'*25} {'-'*8} {'-'*10}")
        for name, p in self.processes:
            status = "running" if p.is_alive() else f"exited({p.exitcode})"
            print(f"  {name:<25} {p.pid:<8} {status}")

    def start_all(self) -> None:
        """启动所有节点"""
        from utils.config_loader import get_config
        config = get_config()
        is_test_mode = config.get_bool('common.test_mode', False)
        front_video_recorder_enabled = config.get_bool('front_video_recorder_node.enabled', False)

        # 打印横幅
        self._print_banner(is_test_mode, front_video_recorder_enabled)

        # 定义节点启动信息
        # 每个节点: (module_name, run_func_name, display_name, init_delay)
        nodes = [
            ('ekf_fusion_node',    'run_ekf_fusion_node',    'ekf_fusion_node',    0.0),
            ('lidar_costmap_node',  'run_lidar_costmap_node', 'lidar_costmap_node', 0.5),
            ('map_planner_node',    'run_map_planner_node',   'map_planner_node',   1.0),
        ]

        if front_video_recorder_enabled:
            nodes.append(
                (
                    'front_video_recorder_node',
                    'run_front_video_recorder_node',
                    'front_video_recorder_node',
                    1.5
                )
            )

        if not is_test_mode:
            nodes.append(
                ('controller_node', 'run_controller_node', 'controller_node', 2.0)
            )

        # 依次启动节点
        for module, func, name, delay in nodes:
            self.start_node(module, func, name, init_delay=delay)

        # 打印就绪信息
        self._print_ready()

    def _get_shutdown_timeout(self, node_name: str) -> float:
        """获取节点关闭等待时间，录像节点需要等待 EOS 完成文件索引写入。"""
        if node_name != 'front_video_recorder_node':
            return 5.0

        try:
            from utils.config_loader import get_config
            return max(
                1.0,
                float(get_config().get('front_video_recorder_node.shutdown_timeout_sec', 10.0))
            )
        except Exception:
            return 10.0

    def shutdown(self, signum=None, frame=None) -> None:
        """关闭所有节点"""
        print("\n\nShutdown signal received...")
        print("Stopping all nodes...\n")

        # 优雅关闭每个进程
        for name, p in self.processes:
            if p.is_alive():
                print(f"  Stopping {name:<20} (PID: {p.pid})...")
                p.terminate()

        # 等待进程结束
        for name, p in self.processes:
            p.join(timeout=self._get_shutdown_timeout(name))
            if p.is_alive():
                print(f"  Force killing {name:<20} (PID: {p.pid})...")
                p.kill()
                p.join(timeout=2)

        # 打印最终状态
        print("\n  Final status:")
        self._print_status()
        print("\n  All nodes stopped.")

    def wait(self) -> None:
        """等待所有节点运行"""
        try:
            while True:
                time.sleep(2)

                # 检查进程状态
                all_dead = True
                for name, p in self.processes:
                    if p.is_alive():
                        all_dead = False
                    elif p.exitcode is not None and p.exitcode != 0:
                        print(f"\n  WARNING: {name} exited with code {p.exitcode}")

                if all_dead:
                    print("\n  All processes have exited.")
                    break

        except KeyboardInterrupt:
            self.shutdown()

    def get_cpu_usage(self) -> None:
        """获取各进程 CPU 使用情况（调试用）"""
        try:
            import psutil
            print("\n  CPU Usage per Process:")
            for name, p in self.processes:
                try:
                    proc = psutil.Process(p.pid)
                    cpu = proc.cpu_percent(interval=0.1)
                    print(f"    {name:<20}: {cpu:.1f}%")
                except:
                    pass
        except ImportError:
            pass


def main() -> None:
    """主入口"""
    print("Initializing Multi-Process Navigation System...")

    # 设置 multiprocessing 启动方式为 'spawn'
    # 这样可以确保跨平台兼容性，避免 fork 相关的问题
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # 已经设置过了

    manager = ProcessManager()

    # 设置信号处理
    signal.signal(signal.SIGINT, manager.shutdown)
    signal.signal(signal.SIGTERM, manager.shutdown)

    try:
        manager.start_all()
        manager.wait()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        manager.shutdown()
        raise


if __name__ == '__main__':
    main()
