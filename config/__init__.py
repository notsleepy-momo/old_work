"""
配置加载模块

提供项目配置文件的统一加载接口，支持缓存以避免重复读取。

用法:
    from config import load_smart_devices, load_furniture_allowed

    mappings = load_smart_devices()           # -> Dict[str, dict]
    device_names = get_smart_device_names()   # -> Set[str]
    allowed = load_furniture_allowed()        # -> Dict[str, list]
"""
import os
import yaml
from typing import Dict, List, Set

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 缓存 ──
_smart_devices_cache: Dict = None
_furniture_allowed_cache: Dict[str, List[str]] = None


def _load_yaml(filename: str) -> dict:
    """加载 YAML 配置文件"""
    path = os.path.join(_CONFIG_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_smart_devices() -> Dict[str, dict]:
    """加载智能设备 → 房间类型映射

    Returns:
        { "Refrigerator": {"types": ["kitchen"], "priority": 5}, ... }
    """
    global _smart_devices_cache
    if _smart_devices_cache is None:
        data = _load_yaml('smart_devices.yaml')
        _smart_devices_cache = data.get('smart_devices', {})
    return _smart_devices_cache


def get_smart_device_names() -> Set[str]:
    """获取智能设备名称集合（用于名称保护判断）"""
    return set(load_smart_devices().keys())


def load_furniture_allowed() -> Dict[str, List[str]]:
    """加载家具命名允许列表

    Returns:
        { "bathroom": ["toilet", "washbasin", ...], "kitchen": [...], ... }
    """
    global _furniture_allowed_cache
    if _furniture_allowed_cache is None:
        data = _load_yaml('furniture_allowed.yaml')
        _furniture_allowed_cache = data.get('furniture_allowed', {})
    return _furniture_allowed_cache
