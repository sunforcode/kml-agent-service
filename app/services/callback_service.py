"""
WalkBG 回调服务

职责:
1. 任务完成后回调 WalkBG 后端
2. 处理回调失败的重试
3. 记录回调日志
"""

import logging
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class CallbackService:
    """
    WalkBG 回调服务
    
    负责在 KML 分析任务完成后回调 WalkBG 后端。
    """
    
    def __init__(self):
        self.base_url = settings.walkbg_base_url
        self.callback_endpoint = settings.walkbg_callback_endpoint
        self.timeout = settings.walkbg_api_timeout
        self.enabled = settings.walkbg_callback_enabled
        
        self.full_callback_url = f"{self.base_url}{self.callback_endpoint}"
        
        logger.info(
            f"回调服务初始化: enabled={self.enabled}, "
            f"url={self.full_callback_url}"
        )
    
    async def send_callback(
        self,
        task_id: str,
        route_id: Optional[str],
        result: Dict[str, Any],
        status: str = "completed"
    ) -> bool:
        """
        发送回调到 WalkBG
        
        Args:
            task_id: 任务ID
            route_id: 关联的路线ID（可选）
            result: 分析结果
            status: 任务状态 (completed/failed)
            
        Returns:
            是否回调成功
        """
        if not self.enabled:
            logger.info(f"回调已禁用，跳过回调: task_id={task_id}")
            return True
        
        callback_payload = self._build_callback_payload(task_id, route_id, result, status)
        
        logger.info(f"发送回调到 WalkBG: task_id={task_id}, url={self.full_callback_url}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.full_callback_url,
                    json=callback_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code in [200, 201, 202]:
                    logger.info(f"回调成功: task_id={task_id}, status_code={response.status_code}")
                    return True
                else:
                    logger.warning(
                        f"回调返回非成功状态: task_id={task_id}, "
                        f"status_code={response.status_code}, response={response.text}"
                    )
                    return False
                    
        except httpx.TimeoutException:
            logger.error(f"回调超时: task_id={task_id}")
            return False
        except httpx.ConnectError as e:
            logger.error(f"回调连接失败: task_id={task_id}, error={str(e)}")
            return False
        except Exception as e:
            logger.error(f"回调失败: task_id={task_id}, error={str(e)}", exc_info=True)
            return False
    
    def _build_callback_payload(
        self,
        task_id: str,
        route_id: Optional[str],
        result: Dict[str, Any],
        status: str
    ) -> Dict[str, Any]:
        """
        构建回调请求体
        
        根据 KmlAnalysisCallbackRequest 的格式构建。
        """
        segments = self._convert_segments(result.get("segments", []))
        water_sources = self._convert_water_sources(result.get("water_sources", []))
        campsites = self._convert_campsites(result.get("campsites", []))
        supplies = self._convert_supplies(result.get("supplies", []))
        marker_points = self._convert_marker_points(result.get("marker_points", []))
        
        payload = {
            "task_id": task_id,
            "route_id": route_id,
            "source_kml_url": result.get("source_kml_url"),
            "analysis_timestamp": result.get("analysis_timestamp") or datetime.utcnow().isoformat(),
            "quality_score": result.get("quality_score"),
            "total_distance_km": result.get("total_distance_km"),
            "total_elevation_gain_m": result.get("total_elevation_gain_m"),
            "total_elevation_loss_m": result.get("total_elevation_loss_m"),
            "max_elevation": result.get("max_elevation"),
            "min_elevation": result.get("min_elevation"),
            "is_loop": result.get("is_loop"),
            "estimated_difficulty": result.get("estimated_difficulty"),
            "segments": segments,
            "water_sources": water_sources,
            "campsites": campsites,
            "supplies": supplies,
            "marker_points": marker_points,
            "generated_description": result.get("generated_description"),
            "generated_highlights": result.get("generated_highlights", []),
            "generated_difficulties": result.get("generated_difficulties", []),
            "generated_safety_notes": result.get("generated_safety_notes", []),
            "equipment_recommendations": result.get("equipment_recommendations", []),
            "warnings": result.get("warnings", [])
        }
        
        return payload
    
    def _convert_water_sources(self, water_sources: list) -> list:
        """
        转换 water_sources 为 CallbackWaterSourceDto 格式
        """
        converted = []
        
        for ws in water_sources:
            converted_ws = {
                "name": ws.get("name") or "未知水源",
                "latitude": float(ws.get("latitude", 0)),
                "longitude": float(ws.get("longitude", 0)),
                "elevation": ws.get("elevation"),
                "description": ws.get("description"),
                "source_type": ws.get("source_type") or "unknown",
                "reliability": float(ws.get("reliability", 0.5)),
                "notes": ws.get("notes")
            }
            converted.append(converted_ws)
        
        return converted
    
    def _convert_campsites(self, campsites: list) -> list:
        """
        转换 campsites 为 CallbackCampsiteDto 格式
        """
        converted = []
        
        for camp in campsites:
            converted_camp = {
                "name": camp.get("name") or "未知营地",
                "latitude": float(camp.get("latitude", 0)),
                "longitude": float(camp.get("longitude", 0)),
                "elevation": camp.get("elevation"),
                "description": camp.get("description"),
                "capacity": camp.get("capacity"),
                "has_water": camp.get("has_water"),
                "has_facilities": camp.get("has_facilities"),
                "notes": camp.get("notes")
            }
            converted.append(converted_camp)
        
        return converted
    
    def _convert_supplies(self, supplies: list) -> list:
        """
        转换 supplies 为 CallbackSupplyDto 格式
        """
        converted = []
        
        for supply in supplies:
            converted_supply = {
                "name": supply.get("name") or "未知补给点",
                "latitude": float(supply.get("latitude", 0)),
                "longitude": float(supply.get("longitude", 0)),
                "elevation": supply.get("elevation"),
                "description": supply.get("description"),
                "supply_type": supply.get("supply_type") or "unknown",
                "notes": supply.get("notes")
            }
            converted.append(converted_supply)
        
        return converted
    
    def _convert_marker_points(self, marker_points: list) -> list:
        """
        转换 marker_points 为 CallbackMarkerPointDto 格式
        """
        converted = []
        
        for mp in marker_points:
            converted_mp = {
                "name": mp.get("name") or "未知标记点",
                "latitude": float(mp.get("latitude", 0)),
                "longitude": float(mp.get("longitude", 0)),
                "elevation": mp.get("elevation"),
                "description": mp.get("description"),
                "type": mp.get("type") or "viewpoint",
                "image_url": mp.get("image_url"),
                "icon_url": mp.get("icon_url"),
                "notes": mp.get("notes")
            }
            converted.append(converted_mp)
        
        return converted
    
    def _convert_segments(self, segments: list) -> list:
        """
        转换 segments 为 CallbackSegmentDto 格式
        
        输入格式（来自 EnhancedRouteOutput）:
        {
            "name": "路段名称",
            "distance_km": 5.2,
            "elevation_gain_m": 300,
            "elevation_loss_m": 50,
            "estimated_time_minutes": 45,
            "difficulty": 2,
            "start_lat": 39.0123,
            "start_lon": 113.4567,
            "end_lat": 39.0456,
            "end_lon": 113.7890,
            ...
        }
        
        输出格式（CallbackSegmentDto）:
        {
            "id": "seg_xxx",
            "name": "路段名称",
            "sequence_number": 1,
            "color": "#FF5722",
            "distance": 5.2,
            "elevation_gain": 300.0,
            "elevation_loss": 50.0,
            "estimated_time": 45,
            "difficulty": 2,
            "start_point": {"latitude": ..., "longitude": ...},
            "end_point": {"latitude": ..., "longitude": ...},
            ...
        }
        """
        converted = []
        
        for i, seg in enumerate(segments):
            converted_seg = {
                "id": seg.get("id") or f"seg_{i}",
                "name": seg.get("name") or f"路段{i+1}",
                "sequence_number": i + 1,
                "color": seg.get("color") or self._get_segment_color(i),
                "description": seg.get("description"),
                "distance": float(seg.get("distance_km", 0)),
                "elevation_gain": float(seg.get("elevation_gain_m", 0)),
                "elevation_loss": float(seg.get("elevation_loss_m", 0)),
                "estimated_time": int(seg.get("estimated_time_minutes", 0)),
                "difficulty": int(seg.get("difficulty", 2)),
                "track_start_index": seg.get("track_start_index"),
                "track_end_index": seg.get("track_end_index"),
                "segment_type": seg.get("segment_type"),
                "slope_direction": seg.get("slope_direction"),
                "avg_slope_degrees": seg.get("avg_slope_degrees"),
                "max_slope_degrees": seg.get("max_slope_degrees"),
                "confidence": seg.get("confidence"),
                "notes": seg.get("notes")
            }
            
            start_lat = seg.get("start_lat")
            start_lon = seg.get("start_lon")
            if start_lat is not None and start_lon is not None:
                converted_seg["start_point"] = {
                    "latitude": float(start_lat),
                    "longitude": float(start_lon),
                    "elevation": seg.get("start_elevation")
                }
            
            end_lat = seg.get("end_lat")
            end_lon = seg.get("end_lon")
            if end_lat is not None and end_lon is not None:
                converted_seg["end_point"] = {
                    "latitude": float(end_lat),
                    "longitude": float(end_lon),
                    "elevation": seg.get("end_elevation")
                }
            
            converted.append(converted_seg)
        
        return converted
    
    def _get_segment_color(self, index: int) -> str:
        """
        根据索引生成路段颜色
        """
        colors = [
            "#FF5722",  # 橙色
            "#4CAF50",  # 绿色
            "#2196F3",  # 蓝色
            "#9C27B0",  # 紫色
            "#FF9800",  # 深橙色
            "#00BCD4",  # 青色
            "#E91E63",  # 粉色
            "#607D8B",  # 蓝灰色
        ]
        return colors[index % len(colors)]


callback_service = CallbackService()
