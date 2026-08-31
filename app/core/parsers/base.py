"""
轨迹解析适配器基类与标准数据定义

架构：各数据格式（KML/GPX/...）各自实现一个适配器，统一解析为
标准轨迹数据（TrackPoint 列表），下游分析只依赖标准数据，与来源格式解耦。

新增格式支持：实现 TrackParserAdapter 子类并在 registry 注册即可。
"""
from abc import ABC, abstractmethod
from typing import List

from app.core.track_processor import TrackPoint


class TrackParserAdapter(ABC):
    """轨迹文件解析适配器基类"""

    #: 适配器名称（如 "kml" / "gpx"），同时也是注册键
    name: str = ""

    @classmethod
    @abstractmethod
    def can_handle(cls, content: str) -> bool:
        """根据文件内容判断是否能处理该格式"""

    @abstractmethod
    def parse(self, content: str) -> List[TrackPoint]:
        """
        解析文件内容为标准轨迹数据

        Returns:
            TrackPoint 列表（按轨迹顺序；无时间信息时 timestamp 为 None）
        """


def parse_iso_datetime(text: str):
    """解析 KML/GPX 常见时间格式，失败返回 None"""
    from datetime import datetime, timezone

    if not text:
        return None
    t = text.strip()
    # 统一 Z 后缀为 +00:00（Python 3.11+ fromisoformat 兼容 Z，这里兼容旧版写法）
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
        # 统一转为 UTC naive，避免后续比较出现 aware/naive 混用
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return None
