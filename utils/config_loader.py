#!/usr/bin/env python3
"""
配置加载模块
从config.yaml加载配置参数，支持相对路径
"""

import os
import yaml
from typing import Any, Optional


class ConfigLoader:
    """配置加载器"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _get_config_path(self) -> str:
        """获取项目根目录下的配置文件路径"""
        project_root = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
        ))
        return os.path.join(project_root, 'config.yaml')

    def _load_config(self):
        """加载配置文件"""
        config_path = self._get_config_path()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分隔的键
        例如: get('ekf_fusion_node.frequency')
        """
        if self._config is None:
            self._load_config()

        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔配置值，兼容 YAML bool 和字符串形式的 true/false。"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ('true', '1', 'yes', 'on'):
                return True
            if normalized in ('false', '0', 'no', 'off'):
                return False
        return bool(value)

    def get_ekf_config(self) -> dict:
        """获取EKF融合节点配置"""
        return self._config.get('ekf_fusion_node', {})

    def get_common_config(self) -> dict:
        """获取通用配置"""
        return self._config.get('common', {})

    def reload(self):
        """重新加载配置"""
        self._config = None
        self._load_config()


def get_config() -> ConfigLoader:
    """获取配置加载器单例"""
    return ConfigLoader()


# 便捷函数
def get(key: str, default: Any = None) -> Any:
    """获取配置值的便捷函数"""
    return get_config().get(key, default)


def get_bool(key: str, default: bool = False) -> bool:
    """获取布尔配置值的便捷函数"""
    return get_config().get_bool(key, default)
