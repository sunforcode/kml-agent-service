"""
轨迹数据处理器 - 基于成熟开源库封装

核心依赖（不重复造轮子）:
- gpxpy: 解析 + 基础统计（get_moving_data）
- scipy.signal: 平滑滤波（Savitzky-Golay等）
- numpy: 梯度计算、向量运算
- geopy: 距离计算

设计原则:
1. 尽量使用开源库的原生方法
2. 只在必要时做轻量封装
3. 保持接口简洁，易于Agent调用
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
from scipy import signal
from geopy.distance import geodesic

import gpxpy
import gpxpy.gpx
from gpxpy.geo import Location

logger = logging.getLogger(__name__)


class SegmentType(str, Enum):
    CLIMB = "climb"
    DESCENT = "descent"
    FLAT = "flat"
    MIXED = "mixed"


@dataclass
class TrackPoint:
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    timestamp: Optional[datetime] = None
    distance_from_start: float = 0.0
    slope_degrees: float = 0.0


@dataclass
class ElevationStats:
    total_gain_m: float
    total_loss_m: float
    max_elevation: float
    min_elevation: float
    avg_elevation: float


@dataclass
class TrackSegment:
    start_index: int
    end_index: int
    segment_type: SegmentType
    distance_km: float
    elevation_gain_m: float
    elevation_loss_m: float
    avg_slope_degrees: float
    max_slope_degrees: float
    start_point: TrackPoint
    end_point: TrackPoint


class TrackProcessor:
    """
    轨迹处理核心类
    
    封装 gpxpy + scipy + numpy 的能力
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 高程平滑参数
        self.smooth_window = self.config.get("smooth_window", 7)
        self.smooth_polyorder = self.config.get("smooth_polyorder", 3)
        
        # 高程阈值（用于过滤GPS噪声）
        self.elevation_threshold_m = self.config.get("elevation_threshold_m", 3.0)
        
        # 坡度判定阈值
        self.flat_slope_threshold = self.config.get("flat_slope_threshold", 3.0)
        self.climb_slope_threshold = self.config.get("climb_slope_threshold", 5.0)
        
        # 分段参数
        self.min_segment_points = self.config.get("min_segment_points", 10)
        self.min_segment_distance_km = self.config.get("min_segment_distance_km", 0.3)

    # ============================================
    # 1. 解析层 - 使用 gpxpy
    # ============================================
    
    def parse_kml_or_gpx(self, content: str, file_type: str = "kml") -> List[TrackPoint]:
        """
        解析 KML 或 GPX 文件
        
        Args:
            content: 文件内容字符串
            file_type: "kml" 或 "gpx"
            
        Returns:
            TrackPoint 列表
        """
        points = []
        
        if file_type.lower() == "gpx":
            gpx = gpxpy.parse(content)
            
            # gpxpy 已经帮我们处理了 tracks, segments
            for track in gpx.tracks:
                for segment in track.segments:
                    cumulative_distance = 0.0
                    prev_point = None
                    
                    for i, p in enumerate(segment.points):
                        if prev_point:
                            # geopy 计算距离（比自己实现更准确）
                            cumulative_distance += geodesic(
                                (prev_point.latitude, prev_point.longitude),
                                (p.latitude, p.longitude)
                            ).km
                        
                        point = TrackPoint(
                            latitude=p.latitude,
                            longitude=p.longitude,
                            elevation=p.elevation,
                            timestamp=p.time,
                            distance_from_start=cumulative_distance
                        )
                        points.append(point)
                        prev_point = p
            
            # 还可以提取 waypoints
            # for wp in gpx.waypoints:
            #     ...
            
        elif file_type.lower() == "kml":
            # KML 解析稍微复杂，用 fastkml 或 minidom
            # 这里简单实现，实际项目可以用 fastkml
            from xml.etree import ElementTree as ET
            
            # 命名空间处理
            ns = {"kml": "http://www.opengis.net/kml/2.2"}
            
            root = ET.fromstring(content)
            
            # 查找 LineString 或 coordinates
            coordinates_elems = root.findall(".//kml:coordinates", ns)
            
            cumulative_distance = 0.0
            prev_lat_lon = None
            
            for coords_elem in coordinates_elems:
                if coords_elem.text:
                    coord_text = coords_elem.text.strip()
                    coord_lines = coord_text.split()
                    
                    for line in coord_lines:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                elev = float(parts[2]) if len(parts) > 2 else None
                                
                                if prev_lat_lon:
                                    cumulative_distance += geodesic(
                                        prev_lat_lon, (lat, lon)
                                    ).km
                                
                                point = TrackPoint(
                                    latitude=lat,
                                    longitude=lon,
                                    elevation=elev,
                                    distance_from_start=cumulative_distance
                                )
                                points.append(point)
                                prev_lat_lon = (lat, lon)
                                
                            except (ValueError, IndexError):
                                continue
        
        return points

    # ============================================
    # 2. 数据清洗与平滑 - 使用 scipy
    # ============================================
    
    def smooth_elevation(self, elevations: List[float]) -> np.ndarray:
        """
        使用 Savitzky-Golay 滤波器平滑高程数据
        
        为什么用 Savitzky-Golay？
        - 比简单移动平均更好地保留峰值和拐点
        - scipy 原生实现，无需自己造轮子
        
        Args:
            elevations: 原始高程列表
            
        Returns:
            平滑后的 numpy 数组
        """
        if not elevations or len(elevations) < self.smooth_window:
            return np.array(elevations, dtype=float)
        
        # 处理 None/NaN
        elev_array = np.array([
            e if e is not None else np.nan for e in elevations
        ], dtype=float)
        
        # 简单的线性插值填充缺失值
        # 更复杂的可以用 scipy.interpolate
        valid_mask = ~np.isnan(elev_array)
        if valid_mask.sum() > 1:
            x = np.arange(len(elev_array))
            elev_array = np.interp(
                x, x[valid_mask], elev_array[valid_mask]
            )
        
        # Savitzky-Golay 平滑
        # 参考: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
        try:
            smoothed = signal.savgol_filter(
                elev_array,
                window_length=self.smooth_window,
                polyorder=self.smooth_polyorder,
                mode="interp"
            )
            return smoothed
        except Exception as e:
            logger.warning(f"Savitzky-Golay 平滑失败，使用移动平均: {e}")
            # 降级方案：移动平均
            kernel = np.ones(self.smooth_window) / self.smooth_window
            return np.convolve(elev_array, kernel, mode="same")

    # ============================================
    # 3. 高程增益计算 - gpxpy原生 + 阈值过滤
    # ============================================
    
    def calculate_elevation_stats(
        self, 
        points: List[TrackPoint]
    ) -> ElevationStats:
        """
        计算高程统计（爬升、下降等）
        
        策略：
        1. 先用 scipy 平滑
        2. 再用"蓄水池"阈值算法过滤噪声
        
        为什么不直接用 gpxpy 的 get_uphill_downhill()?
        - gpxpy 也有算法，但我们可以结合平滑得到更准确的结果
        - 两者可以对比校准
        """
        if not points or len(points) < 2:
            return ElevationStats(0, 0, 0, 0, 0)
        
        # 提取高程
        elevations = [p.elevation for p in points if p.elevation is not None]
        if not elevations:
            return ElevationStats(0, 0, 0, 0, 0)
        
        # 1. 平滑处理（scipy）
        smoothed = self.smooth_elevation(elevations)
        
        # 2. 计算差分
        diffs = np.diff(smoothed)
        
        # 3. 阈值过滤（蓄水池算法）
        # 参考 GPS Visualizer 和 Strava 的做法
        total_gain = 0.0
        total_loss = 0.0
        
        pending_gain = 0.0
        pending_loss = 0.0
        
        for d in diffs:
            if d > 0:
                pending_gain += d
                pending_loss = 0.0
                
                # 超过阈值才计入
                if pending_gain >= self.elevation_threshold_m:
                    total_gain += pending_gain
                    pending_gain = 0.0
            elif d < 0:
                pending_loss += abs(d)
                pending_gain = 0.0
                
                if pending_loss >= self.elevation_threshold_m:
                    total_loss += pending_loss
                    pending_loss = 0.0
        
        # 极值计算
        max_elev = float(np.max(smoothed))
        min_elev = float(np.min(smoothed))
        avg_elev = float(np.mean(smoothed))
        
        return ElevationStats(
            total_gain_m=round(total_gain, 1),
            total_loss_m=round(total_loss, 1),
            max_elevation=round(max_elev, 1),
            min_elevation=round(min_elev, 1),
            avg_elevation=round(avg_elev, 1)
        )

    # ============================================
    # 4. 坡度计算 - numpy gradient
    # ============================================
    
    def calculate_slopes(self, points: List[TrackPoint]) -> List[float]:
        """
        计算每个点的坡度（角度）
        
        使用 numpy.gradient 进行中心差分，
        比简单的后向差分更准确。
        
        坡度 = arctan(高程变化 / 水平距离)
        """
        if not points or len(points) < 2:
            return []
        
        # 提取高程和距离
        elevations = np.array([
            p.elevation if p.elevation is not None else np.nan 
            for p in points
        ])
        
        distances = np.array([
            p.distance_from_start * 1000  # 转米
            for p in points
        ])
        
        # 填充缺失高程
        valid_mask = ~np.isnan(elevations)
        if valid_mask.sum() > 1:
            x = np.arange(len(elevations))
            elevations = np.interp(x, x[valid_mask], elevations[valid_mask])
        
        # 计算梯度（中心差分）
        # numpy.gradient 考虑了不等间距
        try:
            elev_grad = np.gradient(elevations, distances)
        except:
            # 降级方案
            elev_grad = np.diff(elevations) / np.maximum(np.diff(distances), 0.1)
            elev_grad = np.append(elev_grad, elev_grad[-1])
        
        # 转角度
        slopes = np.degrees(np.arctan(elev_grad))
        
        # 简单平滑（可选）
        if len(slopes) >= 5:
            slopes = signal.medfilt(slopes, kernel_size=5)
        
        return [round(float(s), 2) for s in slopes]

    # ============================================
    # 5. 智能分段 - 多策略组合
    # ============================================
    
    def segment_by_time(
        self, 
        points: List[TrackPoint],
        gap_hours: float = 6.0
    ) -> List[TrackSegment]:
        """
        按时间间隔分段（用于识别多日轨迹）
        
        原理：如果两个点时间差超过阈值，视为新的一天/路段
        """
        if not points or len(points) < 2:
            return []
        
        segments = []
        segment_start_idx = 0
        
        for i in range(1, len(points)):
            curr = points[i]
            prev = points[i-1]
            
            if curr.timestamp and prev.timestamp:
                time_diff = (curr.timestamp - prev.timestamp).total_seconds() / 3600
                
                if time_diff >= gap_hours:
                    # 创建一个分段
                    seg = self._create_segment(
                        points, segment_start_idx, i - 1
                    )
                    if seg:
                        segments.append(seg)
                    segment_start_idx = i
        
        # 最后一段
        if segment_start_idx < len(points) - 1:
            seg = self._create_segment(points, segment_start_idx, len(points) - 1)
            if seg:
                segments.append(seg)
        
        return segments

    def segment_by_slope_trend(
        self, 
        points: List[TrackPoint],
        slopes: Optional[List[float]] = None
    ) -> List[TrackSegment]:
        """
        按坡度趋势分段（上升/下降/平坦）
        
        原理：
        1. 坡度连续为正 -> 爬升段
        2. 坡度连续为负 -> 下降段
        3. 坡度接近0 -> 平坦段
        
        使用 numpy 的信号处理能力
        """
        if not points or len(points) < self.min_segment_points:
            return []
        
        # 计算坡度
        if slopes is None:
            slopes = self.calculate_slopes(points)
        
        if len(slopes) != len(points):
            return []
        
        slopes_array = np.array(slopes)
        
        # 标注每个点的趋势类型
        # 0=平坦, 1=爬升, -1=下降
        trend_labels = np.zeros(len(slopes_array), dtype=int)
        
        # 爬升
        trend_labels[slopes_array > self.climb_slope_threshold] = 1
        # 下降
        trend_labels[slopes_array < -self.climb_slope_threshold] = -1
        # 中间的是平坦（在 flat 阈值内）
        
        # 找变化点（使用 numpy 的差分）
        changes = np.where(np.diff(trend_labels) != 0)[0] + 1
        
        # 构建分段边界
        boundaries = np.concatenate([[0], changes, [len(points)]])
        
        segments = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1] - 1
            
            # 过滤过短的分段
            if (end_idx - start_idx + 1) < self.min_segment_points:
                continue
            
            seg_points = points[start_idx:end_idx+1]
            seg_dist = (
                seg_points[-1].distance_from_start - 
                seg_points[0].distance_from_start
            )
            
            if seg_dist < self.min_segment_distance_km:
                continue
            
            seg = self._create_segment(points, start_idx, end_idx, slopes_array)
            if seg:
                segments.append(seg)
        
        # 后处理：合并相邻的相同类型短分段
        segments = self._merge_adjacent_similar_segments(segments)
        
        return segments

    def _create_segment(
        self,
        points: List[TrackPoint],
        start_idx: int,
        end_idx: int,
        slopes_array: Optional[np.ndarray] = None
    ) -> Optional[TrackSegment]:
        """从点范围创建 TrackSegment"""
        if end_idx <= start_idx or end_idx >= len(points):
            return None
        
        sub_points = points[start_idx:end_idx+1]
        
        # 距离
        distance_km = (
            sub_points[-1].distance_from_start - 
            sub_points[0].distance_from_start
        )
        
        # 高程统计
        elev_stats = self.calculate_elevation_stats(sub_points)
        
        # 坡度统计
        if slopes_array is None:
            slopes = self.calculate_slopes(sub_points)
            slopes_array = np.array(slopes) if slopes else np.array([])
        
        sub_slopes = slopes_array[start_idx:end_idx+1] if len(slopes_array) > end_idx else np.array([])
        
        avg_slope = float(np.mean(sub_slopes)) if len(sub_slopes) else 0.0
        max_slope = float(np.max(np.abs(sub_slopes))) if len(sub_slopes) else 0.0
        
        # 判断类型
        if abs(avg_slope) < self.flat_slope_threshold:
            seg_type = SegmentType.FLAT
        elif avg_slope > self.climb_slope_threshold:
            seg_type = SegmentType.CLIMB
        elif avg_slope < -self.climb_slope_threshold:
            seg_type = SegmentType.DESCENT
        else:
            seg_type = SegmentType.MIXED
        
        return TrackSegment(
            start_index=start_idx,
            end_index=end_idx,
            segment_type=seg_type,
            distance_km=round(distance_km, 2),
            elevation_gain_m=elev_stats.total_gain_m,
            elevation_loss_m=elev_stats.total_loss_m,
            avg_slope_degrees=round(avg_slope, 2),
            max_slope_degrees=round(max_slope, 2),
            start_point=sub_points[0],
            end_point=sub_points[-1]
        )

    def _merge_adjacent_similar_segments(
        self, 
        segments: List[TrackSegment]
    ) -> List[TrackSegment]:
        """合并相邻的相同类型的短分段"""
        if len(segments) < 2:
            return segments
        
        merged = []
        current = segments[0]
        
        for seg in segments[1:]:
            # 如果类型相同且都比较短，合并
            if (
                seg.segment_type == current.segment_type and
                (current.distance_km < 1.0 or seg.distance_km < 1.0)
            ):
                # 合并成新的分段
                # 这里简化处理，实际可以重新计算统计
                current = TrackSegment(
                    start_index=current.start_index,
                    end_index=seg.end_index,
                    segment_type=current.segment_type,
                    distance_km=round(current.distance_km + seg.distance_km, 2),
                    elevation_gain_m=current.elevation_gain_m + seg.elevation_gain_m,
                    elevation_loss_m=current.elevation_loss_m + seg.elevation_loss_m,
                    avg_slope_degrees=round(
                        (current.avg_slope_degrees * current.distance_km + 
                         seg.avg_slope_degrees * seg.distance_km) /
                        (current.distance_km + seg.distance_km), 2
                    ),
                    max_slope_degrees=max(current.max_slope_degrees, seg.max_slope_degrees),
                    start_point=current.start_point,
                    end_point=seg.end_point
                )
            else:
                merged.append(current)
                current = seg
        
        merged.append(current)
        return merged

    # ============================================
    # 6. 综合分段入口
    # ============================================
    
    def segment(
        self,
        points: List[TrackPoint],
        strategy: str = "auto"
    ) -> List[TrackSegment]:
        """
        智能分段入口
        
        Args:
            strategy: 
                - "time": 仅按时间间隔
                - "slope": 仅按坡度趋势
                - "auto": 先按时间分段，再在每个时间分段内按坡度细分
        """
        if not points or len(points) < self.min_segment_points:
            return []
        
        if strategy == "time":
            return self.segment_by_time(points)
        
        elif strategy == "slope":
            return self.segment_by_slope_trend(points)
        
        elif strategy == "auto":
            # 1. 先按时间粗分（识别多日轨迹）
            time_segments = self.segment_by_time(points)
            
            if not time_segments:
                # 没有明显时间间隔，直接按坡度分
                return self.segment_by_slope_trend(points)
            
            # 2. 每个时间分段内再按坡度细分
            all_segments = []
            
            for time_seg in time_segments:
                sub_points = points[time_seg.start_index : time_seg.end_index + 1]
                slope_segments = self.segment_by_slope_trend(sub_points)
                
                # 调整索引
                for s in slope_segments:
                    s.start_index += time_seg.start_index
                    s.end_index += time_seg.start_index
                
                all_segments.extend(slope_segments)
            
            return all_segments
        
        return []


# ============================================
# 便捷函数
# ============================================

def create_default_processor() -> TrackProcessor:
    """创建使用默认配置的处理器"""
    return TrackProcessor({
        "smooth_window": 7,
        "smooth_polyorder": 3,
        "elevation_threshold_m": 3.0,
        "flat_slope_threshold": 3.0,
        "climb_slope_threshold": 5.0,
        "min_segment_points": 10,
        "min_segment_distance_km": 0.3
    })
