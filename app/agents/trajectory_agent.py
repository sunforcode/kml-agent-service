"""
轨迹分析Agent (Trajectory Agent)

职责:
1. 解析KML/GPX文件
2. 提取轨迹点和标记点
3. 计算基础统计数据（距离、爬升、下降等）
4. 识别轨迹特征（环线、多日轨迹等）

输入:
- request.kml_source: KML文件URL
- 或 request.kml_content: KML内容字符串

输出:
- track_points: 轨迹点列表
- kml_markers: KML标记点列表
- basic_stats: 基础统计数据

依赖:
- track_processor 模块（封装 gpxpy + scipy + numpy）
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx

from app.agents.base import BaseAgent
from app.core.track_processor import (
    TrackProcessor,
    create_default_processor,
    SegmentType
)
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


class TrajectoryAgent(BaseAgent):
    """
    轨迹分析Agent
    
    负责解析KML文件并计算基础统计数据。
    
    核心依赖开源库（不自己造轮子）:
    - gpxpy: 解析 GPX/KML
    - scipy.signal: 高程平滑（Savitzky-Golay）
    - numpy: 梯度、统计计算
    - geopy: 距离计算（Vincenty/Haversine）
    """
    
    def __init__(self, processor: Optional[TrackProcessor] = None):
        super().__init__(
            name="trajectory_agent",
            description="解析KML文件，计算基础统计数据"
        )
        self.processor = processor or create_default_processor()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行轨迹分析
        
        工作流程:
        1. 获取KML内容（从URL下载 或 直接使用提供的内容）
        2. 使用 gpxpy 解析
        3. 使用 scipy 平滑高程 + 阈值过滤计算爬升/下降
        4. 输出结构化结果
        
        Args:
            state: 当前工作流状态
            
        Returns:
            包含轨迹分析结果的状态更新
        """
        self.log_start()
        
        try:
            request = state.get("request", {})
            
            # 1. 获取KML内容
            kml_content, file_type = await self._get_kml_content(request)
            if not kml_content:
                raise ValueError("无法获取KML内容")
            
            # 2. 解析轨迹点（使用 track_processor，内部封装 gpxpy）
            track_points = self.processor.parse_kml_or_gpx(
                kml_content, 
                file_type=file_type
            )
            
            if not track_points:
                return self._fallback_result("未解析到有效轨迹点")
            
            # 3. 计算统计数据
            #  - 距离：geopy.geodesic（已在解析时计算）
            #  - 爬升/下降：scipy 平滑 + 阈值过滤
            elev_stats = self.processor.calculate_elevation_stats(track_points)
            
            # 4. 提取标记点（Waypoints）
            markers = self._extract_markers(kml_content, file_type)
            
            # 5. 计算附加特征（环线检测、多日检测等）
            additional_features = self._calculate_additional_features(
                track_points, kml_content, file_type
            )
            
            # 6. 组装结果（保持原格式兼容）
            result = self._assemble_result(
                track_points,
                markers,
                elev_stats,
                additional_features
            )
            
            self.log_complete(points_count=len(track_points))
            return result
            
        except Exception as e:
            self.log_error(e)
            return self._fallback_result(f"轨迹分析失败: {str(e)}")

    async def _get_kml_content(
        self, 
        request: Dict[str, Any]
    ) -> tuple:
        """
        获取KML/GPX内容
        
        策略:
        1. 优先使用 kml_content（直接提供的字符串）
        2. 其次从 kml_source URL 下载
        3. 根据扩展名或内容推断文件类型
        
        Returns:
            (content_str, file_type)  file_type: "kml" 或 "gpx"
        """
        # 1. 优先使用直接提供的内容
        kml_content = request.get("kml_content", "").strip()
        if kml_content:
            file_type = self._infer_file_type_from_content(kml_content)
            return kml_content, file_type
        
        # 2. 从URL下载
        kml_url = request.get("kml_source", "").strip()
        if not kml_url:
            raise ValueError("未提供 kml_content 或 kml_source")
        
        # 从URL推断类型
        file_type = "gpx" if kml_url.lower().endswith(".gpx") else "kml"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(kml_url)
                resp.raise_for_status()
                content = resp.text
                
                # 再次根据内容确认类型
                detected_type = self._infer_file_type_from_content(content)
                return content, detected_type
                
        except httpx.HTTPError as e:
            raise ValueError(f"下载KML失败: {str(e)}")

    def _infer_file_type_from_content(self, content: str) -> str:
        """根据文件内容推断是 KML 还是 GPX"""
        content_lower = content.lower()
        if "<gpx" in content_lower:
            return "gpx"
        if "<kml" in content_lower or "<coordinates>" in content_lower:
            return "kml"
        # 默认按 KML 处理
        return "kml"

    def _extract_markers(
        self, 
        content: str, 
        file_type: str
    ) -> List[Dict[str, Any]]:
        """
        提取标记点（Waypoints / Placemarks）
        
        使用 gpxpy 解析 GPX 的 waypoints
        使用简单 XML 解析 KML 的 Placemarks
        """
        markers = []
        
        try:
            if file_type == "gpx":
                import gpxpy
                gpx = gpxpy.parse(content)
                
                for wp in gpx.waypoints:
                    marker = {
                        "name": wp.name or "",
                        "latitude": wp.latitude,
                        "longitude": wp.longitude,
                        "elevation": wp.elevation,
                        "description": wp.description or "",
                        "type": self._classify_marker_type(wp.name, wp.description),
                        "confidence": 1.0
                    }
                    markers.append(marker)
                    
            else:  # KML
                from xml.etree import ElementTree as ET
                
                # 尝试各种常见的KML命名空间
                # 这里用简单方式，实际可以用 fastkml 库更完善
                ns = {"kml": "http://www.opengis.net/kml/2.2"}
                
                try:
                    root = ET.fromstring(content)
                    
                    # 查找所有 Placemark
                    placemarks = root.findall(".//kml:Placemark", ns)
                    
                    for pm in placemarks:
                        try:
                            name_elem = pm.find("kml:name", ns)
                            desc_elem = pm.find("kml:description", ns)
                            coord_elem = pm.find(".//kml:coordinates", ns)
                            
                            name = name_elem.text.strip() if name_elem is not None and name_elem.text else ""
                            desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
                            
                            if coord_elem is not None and coord_elem.text:
                                coord_text = coord_elem.text.strip()
                                parts = coord_text.replace("\n", " ").replace(",", " ").split()
                                if len(parts) >= 2:
                                    lon = float(parts[0])
                                    lat = float(parts[1])
                                    elev = float(parts[2]) if len(parts) > 2 else None
                                    
                                    markers.append({
                                        "name": name,
                                        "latitude": lat,
                                        "longitude": lon,
                                        "elevation": elev,
                                        "description": desc,
                                        "type": self._classify_marker_type(name, desc),
                                        "confidence": 1.0
                                    })
                        except Exception:
                            continue
                            
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"提取标记点时出错，继续: {e}")
            
        return markers

    def _classify_marker_type(self, name: str, desc: str) -> str:
        """简单的基于关键词的标记点分类"""
        text = (name + " " + desc).lower()
        
        if any(k in text for k in ["start", "起点", "开始", "出发"]):
            return "start"
        if any(k in text for k in ["end", "终点", "结束", "终点"]):
            return "end"
        if any(k in text for k in ["view", "观景", "望海", "顶峰", "顶", "peak", "summit"]):
            return "viewpoint"
        if any(k in text for k in ["camp", "营地", "住宿", "hotel", "hostel"]):
            return "camp"
        if any(k in text for k in ["water", "水源", "泉", "creek", "river", "lake"]):
            return "water"
        if any(k in text for k in ["food", "补给", "店", "restaurant"]):
            return "supply"
        
        return "waypoint"

    def _calculate_additional_features(
        self,
        track_points,
        content: str,
        file_type: str
    ) -> Dict[str, Any]:
        """
        计算附加特征:
        - 环线检测
        - 多日轨迹检测
        - 数据质量评分
        """
        features = {
            "is_loop": False,
            "loop_start_end_distance_m": 0.0,
            "multi_day_signals": False,
            "total_duration_seconds": 0,
            "avg_speed_kmh": 0.0,
            "total_track_points": len(track_points),
            "valid_elevation_points": 0,
            "valid_timestamp_points": 0,
            "data_quality_score": 70.0
        }
        
        if len(track_points) < 2:
            return features
        
        # 1. 环线检测：起点终点距离
        first = track_points[0]
        last = track_points[-1]
        
        start_end_dist_m = geodesic(
            (first.latitude, first.longitude),
            (last.latitude, last.longitude)
        ).m
        
        features["loop_start_end_distance_m"] = round(start_end_dist_m, 1)
        
        # 如果起点终点距离小于500米，认为可能是环线
        features["is_loop"] = start_end_dist_m < 500.0
        
        # 2. 时间戳统计
        valid_elev = 0
        valid_time = 0
        timestamps = []
        
        for p in track_points:
            if p.elevation is not None:
                valid_elev += 1
            if p.timestamp:
                valid_time += 1
                timestamps.append(p.timestamp)
        
        features["valid_elevation_points"] = valid_elev
        features["valid_timestamp_points"] = valid_time
        
        # 3. 多日检测
        if len(timestamps) >= 2:
            timestamps.sort()
            total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
            features["total_duration_seconds"] = round(total_duration)
            
            # 如果时间超过一天
            features["multi_day_signals"] = total_duration > 24 * 3600
            
            # 平均速度
            total_dist_km = track_points[-1].distance_from_start
            if total_duration > 0:
                features["avg_speed_kmh"] = round(
                    total_dist_km / (total_duration / 3600), 2
                )
        
        # 4. 数据质量评分（简单版）
        quality_score = 70.0
        
        if track_points:
            # 高程完整度
            elev_ratio = valid_elev / len(track_points)
            quality_score += min(15, elev_ratio * 15)
            
            # 时间戳完整度
            time_ratio = valid_time / len(track_points)
            quality_score += min(15, time_ratio * 15)
        
        features["data_quality_score"] = round(quality_score, 1)
        
        return features

    def _assemble_result(
        self,
        track_points,
        markers: List[Dict],
        elev_stats,
        additional_features: Dict
    ) -> Dict[str, Any]:
        """
        组装为与原有假数据格式一致的结果
        """
        
        # 1. 轨迹点转换为字典列表
        points_dicts = []
        for i, p in enumerate(track_points):
            pdict = {
                "latitude": p.latitude,
                "longitude": p.longitude,
                "elevation": p.elevation,
                "sequence": i,
                "distance_from_start": p.distance_from_start * 1000,  # 转米
            }
            if p.timestamp:
                pdict["timestamp"] = p.timestamp.isoformat()
            points_dicts.append(pdict)
        
        # 2. 统计数据
        total_distance_km = track_points[-1].distance_from_start if track_points else 0.0
        
        basic_stats = {
            "total_distance_km": round(total_distance_km, 2),
            "total_gain_m": elev_stats.total_gain_m,
            "total_loss_m": elev_stats.total_loss_m,
            "max_elevation": elev_stats.max_elevation,
            "min_elevation": elev_stats.min_elevation,
            "avg_elevation": elev_stats.avg_elevation,
            **additional_features
        }
        
        result = {
            "track_points": points_dicts,
            "kml_markers": markers,
            "basic_stats": basic_stats,
            "current_step": "trajectory_analysis",
            "overall_progress": 15
        }
        
        return result

    def _fallback_result(self, message: str) -> Dict[str, Any]:
        """
        降级返回（保持与旧版兼容）
        如果解析失败，记录警告并返回最小可用结构
        """
        import warnings
        warnings.warn(f"TrajectoryAgent 降级模式: {message}")
        
        return {
            "track_points": [],
            "kml_markers": [],
            "basic_stats": {
                "total_distance_km": 0.0,
                "total_gain_m": 0.0,
                "total_loss_m": 0.0,
                "max_elevation": 0.0,
                "min_elevation": 0.0,
                "avg_elevation": 0.0,
                "is_loop": False,
                "multi_day_signals": False,
                "data_quality_score": 0.0
            },
            "current_step": "trajectory_analysis",
            "overall_progress": 15,
            "warnings": [{"level": "warning", "message": message}]
        }
