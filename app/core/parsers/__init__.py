"""
轨迹解析适配器注册表

统一入口：parse_track(content, file_type) -> List[TrackPoint]

- file_type 明确指定时直接路由到对应适配器
- 未指定/不匹配时按内容自动探测（can_handle）
- 新增格式：实现 TrackParserAdapter 并加入 ADAPTERS 列表即可
"""
from typing import List

from app.core.track_processor import TrackPoint
from app.core.parsers.base import TrackParserAdapter
from app.core.parsers.kml_adapter import KmlParserAdapter
from app.core.parsers.gpx_adapter import GpxParserAdapter

#: 已注册的解析适配器（顺序即自动探测优先级）
ADAPTERS = [
    KmlParserAdapter,
    GpxParserAdapter,
]


def get_adapter(file_type: str) -> TrackParserAdapter:
    """按格式名获取适配器实例"""
    key = (file_type or "").strip().lower()
    for adapter_cls in ADAPTERS:
        if adapter_cls.name == key:
            return adapter_cls()
    raise ValueError(f"不支持的轨迹格式: {file_type}，已注册: {[a.name for a in ADAPTERS]}")


def detect_adapter(content: str) -> TrackParserAdapter:
    """按内容自动探测适配器"""
    for adapter_cls in ADAPTERS:
        if adapter_cls.can_handle(content):
            return adapter_cls()
    raise ValueError("无法识别轨迹文件格式（已支持: KML/GPX）")


def parse_track(content: str, file_type: str = None) -> List[TrackPoint]:
    """
    解析轨迹文件为标准数据（TrackPoint 列表）

    Args:
        content: 文件内容字符串
        file_type: "kml" / "gpx"，为 None 时按内容自动探测
    """
    if file_type:
        adapter = get_adapter(file_type)
    else:
        adapter = detect_adapter(content)
    return adapter.parse(content)
