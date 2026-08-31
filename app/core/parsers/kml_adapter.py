"""
KML 轨迹解析适配器

支持：
- gx:Track / gx:MultiTrack（两步路、奥维等 App 导出），逐点配对 <when> 时间戳
- 普通 Placemark LineString 的 <coordinates>（无时间戳）
- 命名空间前缀差异自动兼容（按 local-name 匹配）
"""
import logging
from typing import List, Optional
from xml.etree import ElementTree as ET

from geopy.distance import geodesic

from app.core.track_processor import TrackPoint
from app.core.parsers.base import TrackParserAdapter, parse_iso_datetime

logger = logging.getLogger(__name__)


def _local_name(tag: str) -> str:
    """去掉命名空间，返回标签 local-name（兼容 kml:/gx:/无前缀等各种写法）"""
    return tag.split("}")[-1] if "}" in tag else tag


class KmlParserAdapter(TrackParserAdapter):
    name = "kml"

    @classmethod
    def can_handle(cls, content: str) -> bool:
        head = content[:2048].lstrip().lower()
        return "<kml" in head

    def parse(self, content: str) -> List[TrackPoint]:
        root = ET.fromstring(content)
        points: List[TrackPoint] = []

        cumulative_distance = 0.0  # km，全局累计；每条独立线段首点不连接上一条末点

        def append_point(lat, lon, elev, timestamp, segment_index):
            nonlocal cumulative_distance
            # 计算距离：跨线段不连接（距离不累计），与旧解析行为一致
            if points and points[-1].segment_index == segment_index:
                prev = points[-1]
                cumulative_distance += geodesic(
                    (prev.latitude, prev.longitude), (lat, lon)
                ).km
            points.append(TrackPoint(
                latitude=lat,
                longitude=lon,
                elevation=elev,
                timestamp=timestamp,
                distance_from_start=cumulative_distance,
                segment_index=segment_index,
            ))

        # 方案 1：gx:Track / gx:MultiTrack（两步路、奥维等 App 导出格式）
        # <when> 与 <gx:coord> 按文档顺序逐点交替出现，按下标配对
        gx_tracks = [e for e in root.iter() if _local_name(e.tag) == "Track"]

        for segment_index, track in enumerate(gx_tracks):
            coords = []   # (lon, lat, elev)
            whens = []    # datetime or None
            for elem in track:
                name = _local_name(elem.tag)
                if name == "coord" and elem.text:
                    parts = elem.text.strip().split()
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            elev = float(parts[2]) if len(parts) > 2 else None
                            coords.append((lon, lat, elev))
                        except (ValueError, IndexError):
                            continue
                elif name == "when":
                    whens.append(parse_iso_datetime(elem.text))

            for i, (lon, lat, elev) in enumerate(coords):
                ts = whens[i] if i < len(whens) else None
                append_point(lat, lon, elev, ts, segment_index)

        # 方案 2：普通 LineString / MultiGeometry 的 <coordinates>
        # 格式：lng,lat(,elev) 逗号分隔、空白分点；无时间戳
        if not points:
            segment_index = 0
            for elem in root.iter():
                if _local_name(elem.tag) != "coordinates" or not elem.text:
                    continue
                coord_lines = [l for l in elem.text.strip().split() if "," in l]
                if len(coord_lines) < 2:
                    continue  # 单点 Placemark，跳过
                for line in coord_lines:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            elev = float(parts[2]) if len(parts) > 2 else None
                            append_point(lat, lon, elev, None, segment_index)
                        except (ValueError, IndexError):
                            continue
                segment_index += 1

        logger.info(
            f"KmlParserAdapter: 解析 {len(points)} 个轨迹点，"
            f"含时间戳 {sum(1 for p in points if p.timestamp)} 个"
        )
        return points
