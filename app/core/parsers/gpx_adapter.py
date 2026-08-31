"""
GPX 轨迹解析适配器（基于 gpxpy）

标准输出：每个 trkseg 为独立 segment_index，逐点携带时间戳（若有）。
"""
import logging
from typing import List

import gpxpy

from app.core.track_processor import TrackPoint
from app.core.parsers.base import TrackParserAdapter

logger = logging.getLogger(__name__)


class GpxParserAdapter(TrackParserAdapter):
    name = "gpx"

    @classmethod
    def can_handle(cls, content: str) -> bool:
        head = content[:2048].lstrip().lower()
        return head.startswith("<?xml") and "<gpx" in head

    def parse(self, content: str) -> List[TrackPoint]:
        gpx = gpxpy.parse(content)
        points: List[TrackPoint] = []
        cumulative_distance = 0.0  # km

        segment_index = 0
        for track in gpx.tracks:
            for segment in track.segments:
                prev_point = None
                for p in segment.points:
                    if prev_point:
                        cumulative_distance += geodesic_km(
                            (prev_point.latitude, prev_point.longitude),
                            (p.latitude, p.longitude),
                        )
                    points.append(TrackPoint(
                        latitude=p.latitude,
                        longitude=p.longitude,
                        elevation=p.elevation,
                        timestamp=p.time.replace(tzinfo=None) if p.time else None,
                        distance_from_start=cumulative_distance,
                        segment_index=segment_index,
                    ))
                    prev_point = p
                segment_index += 1

        logger.info(
            f"GpxParserAdapter: 解析 {len(points)} 个轨迹点，"
            f"含时间戳 {sum(1 for p in points if p.timestamp)} 个"
        )
        return points


def geodesic_km(a, b) -> float:
    from geopy.distance import geodesic
    return geodesic(a, b).km
