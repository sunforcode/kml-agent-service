"""
路径分段Agent (Segmentation Agent)

职责:
1. 分析轨迹坡度变化
2. 基于坡度和模式进行智能分段
3. 为每个路段计算统计数据
4. 评估路段难度
5. 估算路段行进时间

输入:
- track_points: 轨迹点列表
- basic_stats: 基础统计数据
- request: 可能包含分段策略参数

输出:
- segments: 路段列表

分段策略:
1. "time": 按时间间隔（识别多日轨迹）
2. "slope": 按坡度趋势（爬升/下降/平坦）
3. "auto": 时间 + 坡度（推荐）

核心依赖（开源库）:
- numpy: 梯度、信号分析
- scipy.signal: 平滑、中值滤波
- track_processor: 封装好的分段逻辑
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from app.agents.base import BaseAgent
from app.core.track_processor import (
    TrackProcessor,
    create_default_processor,
    SegmentType,
    TrackPoint
)

logger = logging.getLogger(__name__)


class SegmentationStrategy(str, Enum):
    TIME = "time"           # 仅按时间间隔
    SLOPE = "slope"         # 仅按坡度趋势
    AUTO = "auto"           # 时间粗分 + 坡度细分（推荐）


class SegmentationAgent(BaseAgent):
    """
    路径分段Agent
    
    负责将连续轨迹智能切分为有意义的路段。
    
    核心能力（基于开源库）:
    1. 时间分段: 检测时间间隔 > 阈值（如6小时）视为新段
    2. 坡度分段: 使用 numpy.gradient 检测上升/下降趋势变化
    3. 后处理: 合并过短的相邻同类型段
    
    所有算法尽量复用:
    - numpy 的差分、统计
    - scipy.signal 的滤波
    - track_processor 封装好的逻辑
    """
    
    def __init__(self, processor: Optional[TrackProcessor] = None):
        super().__init__(
            name="segmentation_agent",
            description="基于坡度变化进行智能路径分段"
        )
        self.processor = processor or create_default_processor()
        
        # 从 request 可以覆盖这些参数
        self.default_strategy = SegmentationStrategy.AUTO
        self.time_gap_hours = 6.0        # 超过6小时间隔视为新天
        self.min_segment_points = 10
        self.min_segment_distance_km = 0.3

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行路径分段
        
        工作流程:
        1. 从 state 获取 track_points
        2. 从 request 获取分段策略（可选）
        3. 使用 track_processor 进行分段
        4. 为每个段计算: 难度、估算时间、建议名称
        5. 组装结果（与原有格式兼容）
        """
        self.log_start()
        
        try:
            track_points_dicts = state.get("track_points", [])
            request = state.get("request", {})
            
            if not track_points_dicts:
                logger.warning("SegmentationAgent: 无轨迹点，返回空分段")
                return self._empty_result("无轨迹点数据")
            
            # 1. 解析分段策略
            strategy = self._parse_strategy(request)
            self._apply_config_from_request(request)
            
            # 2. 将 dict 转换回 TrackPoint 对象
            points = self._dicts_to_track_points(track_points_dicts)
            
            if len(points) < self.min_segment_points:
                return self._empty_result("轨迹点过少，无法分段")
            
            # 3. 执行分段（核心逻辑在 track_processor）
            #    track_processor 内部封装了:
            #    - numpy.gradient 计算坡度
            #    - scipy.signal 平滑
            #    - 阈值分段 + 后处理
            segments = self.processor.segment(
                points,
                strategy=strategy.value
            )
            
            if not segments:
                # 如果无法分段，退化为单个段
                single_segment = self._create_single_fallback_segment(points)
                if single_segment:
                    segments = [single_segment]
            
            # 4. 验证分段覆盖是否完整（调试日志）
            total_pts = len(points)
            if segments:
                covered_start = segments[0].start_index
                covered_end = segments[-1].end_index
                coverage_pct = (covered_end - covered_start + 1) / total_pts * 100
                logger.info(
                    f"SegmentationAgent: 分段数={len(segments)}, "
                    f"轨迹总点数={total_pts}, "
                    f"覆盖范围=[{covered_start}, {covered_end}], "
                    f"覆盖率={coverage_pct:.1f}%"
                )
                if covered_start > 0 or covered_end < total_pts - 1:
                    logger.warning(
                        f"SegmentationAgent: 轨迹覆盖不完整！"
                        f"前段遗漏={covered_start}点, 后段遗漏={total_pts - 1 - covered_end}点"
                    )

            # 5. 为每个段计算附加信息：难度、估算时间、建议名称
            enriched_segments = []
            for i, seg in enumerate(segments):
                enriched = self._enrich_segment(seg, points, i, len(segments))
                enriched_segments.append(enriched)
            
            # 6. 计算按天分段（如果有时间戳）
            day_segments = self._build_day_segments(points)

            # 7. 组装为 v2 格式（segment_schemes）
            result = self._assemble_result(enriched_segments, day_segments)

            self.log_complete(segments_count=len(enriched_segments))
            return result
            
        except Exception as e:
            self.log_error(e)
            return self._empty_result(f"分段失败: {str(e)}")

    def _parse_strategy(self, request: Dict[str, Any]) -> SegmentationStrategy:
        """
        从请求解析分段策略
        
        支持的参数:
        - segmentation_strategy: "time" / "slope" / "auto"
        - 也可以推断：如果有多日信号，建议用 time + slope
        """
        strategy_str = request.get("segmentation_strategy", "").strip().lower()
        
        if strategy_str == "time":
            return SegmentationStrategy.TIME
        if strategy_str == "slope":
            return SegmentationStrategy.SLOPE
        
        return SegmentationStrategy.AUTO

    def _apply_config_from_request(self, request: Dict[str, Any]):
        """
        从请求读取配置覆盖默认值
        
        支持:
        - time_gap_hours: 时间间隔阈值（小时）
        - min_segment_distance_km: 最小区段距离
        - elevation_threshold_m: 高程阈值
        - climb_slope_threshold: 爬升坡度阈值
        - flat_slope_threshold: 平坦坡度阈值
        """
        if "time_gap_hours" in request:
            try:
                self.time_gap_hours = float(request["time_gap_hours"])
            except (ValueError, TypeError):
                pass
        
        if "min_segment_distance_km" in request:
            try:
                self.min_segment_distance_km = float(request["min_segment_distance_km"])
            except (ValueError, TypeError):
                pass

    def _dicts_to_track_points(self, dicts: List[Dict[str, Any]]) -> List[TrackPoint]:
        """
        将字典列表转换回 TrackPoint 对象列表
        """
        points = []
        
        for d in dicts:
            lat = d.get("latitude")
            lon = d.get("longitude")
            
            if lat is None or lon is None:
                continue
            
            elev = d.get("elevation")
            
            timestamp = None
            ts_str = d.get("timestamp")
            if ts_str:
                try:
                    if isinstance(ts_str, str):
                        # 尝试常见格式
                        for fmt in [
                            "%Y-%m-%dT%H:%M:%S.%f",
                            "%Y-%m-%dT%H:%M:%S",
                            "%Y-%m-%d %H:%M:%S"
                        ]:
                            try:
                                timestamp = datetime.strptime(ts_str, fmt)
                                break
                            except ValueError:
                                continue
                except Exception:
                    pass
            
            dist_m = d.get("distance_from_start", 0.0)
            # 统一单位：track_processor 内部用 km
            distance_km = dist_m / 1000.0 if dist_m > 100 else dist_m
            
            point = TrackPoint(
                latitude=lat,
                longitude=lon,
                elevation=elev,
                timestamp=timestamp,
                distance_from_start=distance_km
            )
            points.append(point)
        
        return points

    def _create_single_fallback_segment(self, points: List[TrackPoint]) -> Optional[Any]:
        """
        当无法自动分段时，创建一个包含全部轨迹的单一"混合"段
        """
        if len(points) < 2:
            return None
        
        # 直接复用 _create_segment 逻辑
        return self.processor._create_segment(points, 0, len(points) - 1)

    def _get_segment_color(self, segment_type: str) -> str:
        """
        根据路段类型返回颜色
        - climb: 橙色 (#FF5722)
        - descent: 蓝色 (#2196F3)
        - flat: 绿色 (#4CAF50)
        - mixed: 紫色 (#9C27B0)
        """
        color_map = {
            "climb": "#FF5722",
            "descent": "#2196F3",
            "flat": "#4CAF50",
            "mixed": "#9C27B0"
        }
        return color_map.get(segment_type, "#9C27B0")

    def _enrich_segment(
        self, 
        segment, 
        all_points: List[TrackPoint],
        index: int, 
        total: int
    ) -> Dict[str, Any]:
        """
        为 TrackSegment 添加：
        - id: 唯一ID
        - name: 路段名称
        - sequence_number: 序号
        - color: 显示颜色
        - difficulty: 难度(1-5)
        - estimated_time: 估算时间(分钟)
        - start_point/end_point: 起止坐标对象
        - track_start_index/track_end_index: 轨迹点索引
        - segment_type: 路段类型
        - 额外信息: avg_slope_degrees, max_slope_degrees, confidence
        """
        
        # 1. 路段类型
        seg_type_map = {
            SegmentType.CLIMB: "climb",
            SegmentType.DESCENT: "descent",
            SegmentType.FLAT: "flat",
            SegmentType.MIXED: "mixed"
        }
        segment_type = seg_type_map.get(segment.segment_type, "mixed")
        
        # 2. 估算难度（基于坡度和爬升）
        difficulty = self._estimate_difficulty(
            segment.avg_slope_degrees,
            segment.max_slope_degrees,
            segment.elevation_gain_m,
            segment.distance_km,
            segment_type
        )
        
        # 3. 估算时间（Naismith 法则）
        time_minutes = self._estimate_time_naismith(
            segment.distance_km,
            segment.elevation_gain_m,
            segment.elevation_loss_m,
            difficulty
        )
        
        # 4. 建议名称
        name = self._suggest_name(
            segment, 
            index, 
            total, 
            segment_type, 
            all_points
        )
        
        # 5. 置信度（简单启发式）
        confidence = 0.8
        if segment.distance_km < 0.5:
            confidence -= 0.1
        if segment_type == "mixed":
            confidence -= 0.1
        
        # 6. 生成唯一ID
        segment_id = f"seg_{index + 1:03d}"
        
        # 7. 序号
        sequence_number = index + 1
        
        # 8. 颜色
        color = self._get_segment_color(segment_type)
        
        # 组装字典（与 walkbg/walkfg 数据模型对齐）
        return {
            # 唯一标识和排序
            "id": segment_id,
            "name": name,
            "sequence_number": sequence_number,
            "color": color,
            
            # 数值字段
            "distance": round(segment.distance_km, 2),
            "elevation_gain": round(segment.elevation_gain_m, 1),
            "elevation_loss": round(segment.elevation_loss_m, 1),
            "estimated_time": time_minutes,
            "difficulty": difficulty,
            
            # 轨迹点索引范围
            "track_start_index": segment.start_index,
            "track_end_index": segment.end_index,
            
            # 起止坐标（对象结构，与 walkbg SegmentDto / walkfg TrackPointVO 对齐）
            "start_point": {
                "latitude": segment.start_point.latitude,
                "longitude": segment.start_point.longitude,
                "elevation": segment.start_point.elevation
            },
            "end_point": {
                "latitude": segment.end_point.latitude,
                "longitude": segment.end_point.longitude,
                "elevation": segment.end_point.elevation
            },
            
            # 路段类型（坡度方向：climb/descent/flat/mixed）
            # 注意：与 walkfg RouteType 枚举概念不同，
            # RouteType 是路面材质(泥路/石路等)，这里是坡度方向
            "segment_type": segment_type,
            "slope_direction": segment_type,
            
            # 额外分析信息（可选，但有用）
            "avg_slope_degrees": round(segment.avg_slope_degrees, 2),
            "max_slope_degrees": round(segment.max_slope_degrees, 2),
            "confidence": round(max(0.5, min(1.0, confidence)), 2),
            
            # 可选字段（暂为空）
            "description": None,
            "notes": None
        }

    def _estimate_difficulty(
        self,
        avg_slope: float,
        max_slope: float,
        gain_m: float,
        distance_km: float,
        segment_type: str
    ) -> int:
        """
        估算路段难度（1-5）
        
        策略：
        1. 主要基于坡度和距离
        2. 下降比平路稍难
        3. 混合段按主要趋势
        
        参考：
        - 难度 1: 平缓，坡度 < 3°
        - 难度 2: 轻度，坡度 3-8°
        - 难度 3: 中等，坡度 8-15°
        - 难度 4: 较陡，坡度 15-25°
        - 难度 5: 极陡，坡度 > 25°
        """
        
        # 有效坡度取绝对值
        effective_slope = abs(avg_slope)
        
        # 基础难度
        if effective_slope < 3:
            base = 1
        elif effective_slope < 8:
            base = 2
        elif effective_slope < 15:
            base = 3
        elif effective_slope < 25:
            base = 4
        else:
            base = 5
        
        # 距离修正：长距离难度+1
        if distance_km > 10:
            base = min(5, base + 1)
        elif distance_km > 5:
            base = min(5, base + 0)  # 暂不额外+1，保持保守
        
        # 爬升修正：大爬升难度+1
        if segment_type == "climb" and gain_m > 800:
            base = min(5, base + 1)
        
        # 下降修正：长距离下降稍难
        if segment_type == "descent" and distance_km > 8:
            base = min(5, base + 1)
        
        return int(round(base))

    def _estimate_time_naismith(
        self,
        distance_km: float,
        gain_m: float,
        loss_m: float,
        difficulty: int
    ) -> int:
        """
        使用 Naismith 法则估算行进时间（分钟）
        
        Naismith 法则原始版:
        - 平地：每小时 5 公里（或每公里 12 分钟）
        - 爬升：每 600 米 增加 1 小时（每 100 米 +10 分钟）
        
        我们的调整版（更适合徒步）：
        - 基础速度：3-5 km/h，依难度
        - 爬升：每 100 米 +8-12 分钟
        - 下降：每 100 米 +3-5 分钟（陡坡下降更慢）
        """
        
        # 1. 基础速度（依难度）
        # 难度 1: ~5 km/h
        # 难度 2: ~4 km/h
        # 难度 3: ~3 km/h
        # 难度 4-5: ~2-2.5 km/h
        base_speed_kmh = {
            1: 5.0,
            2: 4.0,
            3: 3.2,
            4: 2.5,
            5: 2.0
        }.get(difficulty, 3.5)
        
        # 距离时间
        if base_speed_kmh > 0:
            time_min = (distance_km / base_speed_kmh) * 60
        else:
            time_min = distance_km * 20  # 保守值
        
        # 2. 爬升时间
        # 每 100 米爬升
        # 难度1-2: 8分钟 / 100m
        # 难度3-4: 10分钟 / 100m
        # 难度5: 12分钟 / 100m
        climb_per_100m_min = {
            1: 8,
            2: 9,
            3: 10,
            4: 11,
            5: 12
        }.get(difficulty, 10)
        
        climb_time = (gain_m / 100.0) * climb_per_100m_min
        time_min += climb_time
        
        # 3. 下降时间（通常比爬升快，但陡坡可能慢）
        # 缓坡下降：每100米 3分钟
        # 中等：5分钟
        # 陡坡：7-10分钟（难走）
        if difficulty <= 2:
            descent_per_100m_min = 3
        elif difficulty == 3:
            descent_per_100m_min = 5
        elif difficulty == 4:
            descent_per_100m_min = 7
        else:
            descent_per_100m_min = 10
        
        descent_time = (loss_m / 100.0) * descent_per_100m_min
        time_min += descent_time
        
        # 4. 最小时间保护
        time_min = max(1, int(round(time_min)))
        
        return time_min

    def _suggest_name(
        self,
        segment,
        index: int,
        total: int,
        segment_type: str,
        all_points: List[TrackPoint]
    ) -> str:
        """
        简单的路段名称建议
        
        策略：
        - 数字顺序：第N段
        - 类型描述：爬升/下降/平坦/混合
        - 可选：距离+爬升组合（如"2.5km+200m爬升"）
        """
        
        type_names = {
            "climb": "爬升段",
            "descent": "下降段",
            "flat": "平路段",
            "mixed": "混合段"
        }
        type_cn = type_names.get(segment_type, "路段")
        
        # 数字序号
        ordinal = index + 1
        
        # 距离与爬升描述
        dist_str = f"{segment.distance_km:.1f}km"
        if segment.elevation_gain_m > 0:
            detail_str = f"(+{int(segment.elevation_gain_m)}m)"
        elif segment.elevation_loss_m > 0:
            detail_str = f"(-{int(segment.elevation_loss_m)}m)"
        else:
            detail_str = ""
        
        # 如果是第一段或最后一段，可以加标识
        if ordinal == 1:
            prefix = "起点-"
        elif ordinal == total:
            prefix = "终点-"
        else:
            prefix = f"第{ordinal}段-"
        
        # 组合
        name = f"{prefix}{type_cn}{detail_str}"
        
        # 更简洁的备选（如果太长）
        if len(name) > 25:
            name = f"第{ordinal}段{type_cn}"
        
        return name

    def _assemble_result(self, enriched_segments: List[Dict], day_segments: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        组装 v2 格式结果：输出 segment_schemes 列表

        包含两个方案：
        - slope: 按坡度（主方案，已算法实现）
        - day: 按天（有时间戳时真实实现，无时间戳则为空）
        """
        schemes = [
            {
                "scheme_type": "slope",
                "label": "按坡度",
                "is_default": True,
                "segments": enriched_segments
            },
            {
                "scheme_type": "day",
                "label": "按天",
                "is_default": False,
                "segments": day_segments or []  # 无时间戳时为空列表
            },
            {
                "scheme_type": "terrain",
                "label": "按地形",
                "is_default": False,
                "segments": []  # TODO: 待 find_peaks 算法实现
            },
            {
                "scheme_type": "road_type",
                "label": "按路况",
                "is_default": False,
                "segments": []  # TODO: 待 OSM highway 数据接入
            }
        ]
        return {
            "segment_schemes": schemes,
            "current_step": "segmentation",
            "overall_progress": 30
        }

    def _build_day_segments(self, points: List[TrackPoint]) -> List[Dict[str, Any]]:
        """
        基于时间间隔计算按天分段列表

        过滤条件：轨迹点必须有时间戳，否则返回空列表。
        分天逐辑：相邻两点时间间隔 > time_gap_hours 就新开一天。
        """
        # 如果没有时间戳，返回空列表
        has_timestamps = any(p.timestamp is not None for p in points)
        if not has_timestamps:
            logger.info("SegmentationAgent: 轨迹无时间戳，按天方案 segments 为空")
            return []

        # 分天逻辑
        gap_threshold_seconds = self.time_gap_hours * 3600
        day_groups: List[List[int]] = [[0]]  # 第一天从第0个点开始

        for i in range(1, len(points)):
            prev = points[i - 1]
            curr = points[i]
            if prev.timestamp and curr.timestamp:
                diff = (curr.timestamp - prev.timestamp).total_seconds()
                if diff > gap_threshold_seconds:
                    day_groups.append([])  # 新开一天
            day_groups[-1].append(i)

        if len(day_groups) <= 1:
            # 只有一天，不需要按天分段
            return []

        # 为每天生成一个 segment
        day_segment_list = []
        for day_idx, idx_list in enumerate(day_groups):
            if not idx_list:
                continue
            start_i = idx_list[0]
            end_i = idx_list[-1]
            sp = points[start_i]
            ep = points[end_i]

            # 简单计算距离
            total_dist = (points[end_i].distance_from_start or 0) - (points[start_i].distance_from_start or 0)
            total_gain = sum(
                max(0, points[j].elevation - points[j - 1].elevation)
                for j in range(start_i + 1, end_i + 1)
                if points[j].elevation is not None and points[j - 1].elevation is not None
            )
            total_loss = sum(
                max(0, points[j - 1].elevation - points[j].elevation)
                for j in range(start_i + 1, end_i + 1)
                if points[j].elevation is not None and points[j - 1].elevation is not None
            )

            seg_id = f"day_{day_idx + 1:02d}"
            day_segment_list.append({
                "id": seg_id,
                "name": f"第{day_idx + 1}天",
                "sequence_number": day_idx + 1,
                "color": "#607D8B",  # 按天统一用灯笼蓝
                "distance": round(total_dist, 2),
                "elevation_gain": round(total_gain, 1),
                "elevation_loss": round(total_loss, 1),
                "estimated_time": None,
                "difficulty": None,
                "track_start_index": start_i,
                "track_end_index": end_i,
                "start_point": {
                    "latitude": sp.latitude,
                    "longitude": sp.longitude,
                    "elevation": sp.elevation
                },
                "end_point": {
                    "latitude": ep.latitude,
                    "longitude": ep.longitude,
                    "elevation": ep.elevation
                },
                "segment_type": "day",
                "description": None,
                "notes": None
            })

        return day_segment_list

    def _empty_result(self, message: str) -> Dict[str, Any]:
        """
        降级：返回空分段方案 + 警告
        """
        logger.warning(f"SegmentationAgent 降级: {message}")
        empty_schemes = [
            {"scheme_type": "slope", "label": "按坡度", "is_default": True, "segments": []},
            {"scheme_type": "day", "label": "按天", "is_default": False, "segments": []},
            {"scheme_type": "terrain", "label": "按地形", "is_default": False, "segments": []},
            {"scheme_type": "road_type", "label": "按路况", "is_default": False, "segments": []},
        ]
        return {
            "segment_schemes": empty_schemes,
            "current_step": "segmentation",
            "overall_progress": 30,
            "warnings": [{"level": "warning", "message": message}]
        }
