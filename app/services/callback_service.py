"""
WalkBG 回调服务

职责:
1. 任务完成后回调 WalkBG 后端
2. 处理回调失败的重试
3. 记录回调日志
"""

import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.config import settings
from app.services.task_service import _sanitize_for_json

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
        
        callback_payload = _sanitize_for_json(
            self._build_callback_payload(task_id, route_id, result, status)
        )
        
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
        """构建回调请求体"""
        segment_schemes = self._convert_segment_schemes(result.get("segment_schemes", []))
        poi_points = self._convert_poi_points(result.get("poi_points", []))

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
            "segment_schemes": segment_schemes,
            "poi_points": poi_points,
            "generated_description": result.get("generated_description"),
            "generated_highlights": result.get("generated_highlights", []),
            "generated_difficulties": result.get("generated_difficulties", []),
            "generated_safety_notes": result.get("generated_safety_notes", []),
            "equipment_recommendations": result.get("equipment_recommendations", []),
            "warnings": result.get("warnings", [])
        }
        
        return payload

    def _convert_segment_schemes(self, schemes: list) -> list:
        """转换 segment_schemes 为 CallbackSegmentSchemeDto 格式列表"""
        converted = []
        for scheme in schemes:
            converted_scheme = {
                "scheme_type": scheme.get("scheme_type", "slope"),
                "label": scheme.get("label", ""),
                "is_default": bool(scheme.get("is_default", False)),
                "segments": self._convert_segments(scheme.get("segments", []))
            }
            converted.append(converted_scheme)
        return converted

    def _convert_poi_points(self, poi_points: list) -> list:
        """转换 poi_points 为 CallbackPoiPointDto 格式列表"""
        converted = []
        for poi in poi_points:
            converted_poi = {
                "name": poi.get("name") or "未命名 POI",
                "latitude": float(poi.get("latitude", 0)),
                "longitude": float(poi.get("longitude", 0)),
                "elevation": poi.get("elevation"),
                "category": poi.get("category", "photo"),
                "sub_category": poi.get("sub_category"),
                "source": poi.get("source", "kml_marker"),
                "description": poi.get("description"),
                "confidence": poi.get("confidence"),
                "card_data": poi.get("card_data")
            }
            converted.append(converted_poi)
        return converted

    def _convert_segments(self, segments: list) -> list:
        """转换 segments 为 CallbackSegmentDto 格式"""
        converted = []
        
        for i, seg in enumerate(segments):
            converted_seg = {
                "id": seg.get("id") or f"seg_{i}",
                "name": seg.get("name") or f"路段{i+1}",
                "sequence_number": i + 1,
                "color": seg.get("color") or self._get_segment_color(i),
                "description": seg.get("description"),
                "distance": float(seg.get("distance", 0)),
                "elevation_gain": float(seg.get("elevation_gain", 0)),
                "elevation_loss": float(seg.get("elevation_loss", 0)),
                "estimated_time": int(seg.get("estimated_time", 0)),
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
            
            if seg.get("start_point") is not None:
                converted_seg["start_point"] = seg["start_point"]
            
            if seg.get("end_point") is not None:
                converted_seg["end_point"] = seg["end_point"]
            
            converted.append(converted_seg)
        
        return converted

    def _get_segment_color(self, index: int) -> str:
        """根据索引生成路段颜色"""
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
