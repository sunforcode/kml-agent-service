"""
WalkBG 回调服务

职责:
1. 任务完成后回调 WalkBG 后端
2. 处理回调失败的重试
3. 记录回调日志
"""

import asyncio
import logging
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.config import settings
from app.models.execution_event import ExecutionEvent, classify_safe_error
from app.models.response import EnhancedRouteOutput
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
        self.max_attempts = settings.walkbg_callback_max_attempts
        self.retry_base_delay = settings.walkbg_callback_retry_base_delay

        self.full_callback_url = f"{self.base_url}{self.callback_endpoint}"
        self.execution_event_endpoint = settings.walkbg_execution_event_endpoint
        self._event_delivery_queues: Dict[
            str, asyncio.Queue[ExecutionEvent | Dict[str, Any]]
        ] = {}
        self._event_delivery_tasks: Dict[str, asyncio.Task[None]] = {}

        logger.info(
            f"回调服务初始化: enabled={self.enabled}, "
            f"url={self.full_callback_url}, max_attempts={self.max_attempts}"
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

        # 分析结果是经过数分钟 LLM 计算得出的，且仅保存在本进程内存中，
        # 一次网络抖动就永久丢失代价过高，因此对可重试的失败做指数退避重试。
        last_error = "unknown"
        for attempt in range(1, self.max_attempts + 1):
            should_retry, last_error = await self._attempt_callback(
                task_id, callback_payload, attempt
            )
            if should_retry is None:
                return True
            if not should_retry:
                # 客户端错误（4xx）重试不会改变结果，直接放弃
                break
            if attempt < self.max_attempts:
                # 退避时长：base * 2^(attempt-1)，例如 2s / 4s / 8s
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"回调失败，{delay}s 后重试: task_id={task_id}, "
                    f"attempt={attempt}/{self.max_attempts}, error={last_error}"
                )
                await asyncio.sleep(delay)

        logger.error(
            f"回调最终失败，分析结果未能送达 WalkBG: task_id={task_id}, "
            f"attempts={self.max_attempts}, last_error={last_error}"
        )
        return False

    @property
    def pending_event_deliveries(self) -> int:
        return len(self._event_delivery_tasks)

    def dispatch_execution_event(
        self,
        task_id: str,
        event: ExecutionEvent | Dict[str, Any],
    ) -> None:
        """Enqueue best-effort delivery without blocking the workflow."""
        queue = self._event_delivery_queues.get(task_id)
        worker = self._event_delivery_tasks.get(task_id)
        if queue is None or worker is None or worker.done():
            queue = asyncio.Queue()
            worker = asyncio.create_task(self._deliver_execution_events(task_id, queue))
            self._event_delivery_queues[task_id] = queue
            self._event_delivery_tasks[task_id] = worker
        queue.put_nowait(event)

    async def _deliver_execution_events(
        self,
        task_id: str,
        queue: asyncio.Queue[ExecutionEvent | Dict[str, Any]],
    ) -> None:
        """Deliver one task's events serially in dispatch order."""
        while True:
            event = await queue.get()
            try:
                await self.send_execution_event(task_id, event)
            finally:
                queue.task_done()

    async def drain_execution_events(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> None:
        """Wait for one task's queue, then cancel its worker and clean up."""
        queue = self._event_delivery_queues.get(task_id)
        worker = self._event_delivery_tasks.get(task_id)
        if queue is None or worker is None:
            return

        try:
            if timeout is None:
                await queue.join()
            else:
                await asyncio.wait_for(queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            if self._event_delivery_tasks.get(task_id) is worker:
                self._event_delivery_tasks.pop(task_id, None)
                self._event_delivery_queues.pop(task_id, None)

    async def send_execution_event(
        self,
        task_id: str,
        event: ExecutionEvent | Dict[str, Any],
    ) -> bool:
        """Best-effort event delivery; failures never affect analysis execution."""
        if not self.enabled:
            return True

        try:
            event_model = event if isinstance(event, ExecutionEvent) else ExecutionEvent.model_validate(event)
            await self._post_execution_event(task_id, event_model.model_dump(mode="json"))
            return True
        except Exception as exc:
            logger.warning(
                "执行事件发送失败，分析继续: task_id=%s, error_type=%s",
                task_id,
                type(exc).__name__,
            )
            return False

    async def _post_execution_event(self, task_id: str, event: Dict[str, Any]) -> None:
        endpoint = self.execution_event_endpoint.format(task_id=task_id)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}{endpoint}",
                json={"task_id": task_id, "execution_event": event},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

    async def _attempt_callback(
        self,
        task_id: str,
        callback_payload: Dict[str, Any],
        attempt: int
    ) -> tuple[Optional[bool], str]:
        """
        尝试一次回调。

        Returns:
            (should_retry, error_message)
            should_retry 为 None 表示成功；True 表示可重试；False 表示不应重试。
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.full_callback_url,
                    json=callback_payload,
                    headers={"Content-Type": "application/json"}
                )

            if response.status_code in [200, 201, 202]:
                logger.info(
                    f"回调成功: task_id={task_id}, "
                    f"status_code={response.status_code}, attempt={attempt}"
                )
                return None, ""

            error = f"HTTP {response.status_code}: {response.text[:200]}"
            # 4xx 表示请求本身不被接受，重发相同内容仍会失败；
            # 5xx 与其他状态可能是服务端临时问题，值得重试。
            if 400 <= response.status_code < 500:
                logger.error(
                    f"回调被拒绝（不重试）: task_id={task_id}, {error}"
                )
                return False, error
            return True, error

        except httpx.TimeoutException:
            return True, "timeout"
        except httpx.ConnectError as e:
            return True, f"connect_error: {str(e)}"
        except httpx.HTTPError as e:
            return True, f"http_error: {str(e)}"
        except Exception as e:
            # 非网络异常（如序列化问题）重试无意义
            logger.error(
                f"回调出现非网络异常（不重试）: task_id={task_id}, error={str(e)}",
                exc_info=True
            )
            return False, str(e)
    
    def _build_callback_payload(
        self,
        task_id: str,
        route_id: Optional[str],
        result: Dict[str, Any],
        status: str
    ) -> Dict[str, Any]:
        """构建回调请求体；成功结果必须满足最终输出契约。"""
        if status == "completed":
            result = EnhancedRouteOutput.model_validate(result).model_dump(mode="json")

        segment_schemes = self._convert_segment_schemes(result.get("segment_schemes", []))
        poi_points = self._convert_poi_points(result.get("poi_points", []))

        payload = {
            "task_id": task_id,
            "route_id": route_id,
            "status": status,
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
            "generation_mode": result.get("generation_mode"),
            "degraded": bool(result.get("degraded", False)),
            "warnings": result.get("warnings", [])
        }
        if status == "failed":
            safe_error = classify_safe_error(RuntimeError(str(result.get("error") or "")))
            payload["error"] = safe_error["summary"]
            payload["warnings"] = [
                {"level": "error", "message": safe_error["summary"]}
            ]

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
        
        for seg in segments:
            converted_seg = {
                "id": seg["id"],
                "name": seg["name"],
                "sequence_number": int(seg["sequence_number"]),
                "color": seg["color"],
                "description": seg.get("description"),
                "distance": float(seg.get("distance", 0)),
                "elevation_gain": float(seg.get("elevation_gain", 0)),
                "elevation_loss": float(seg.get("elevation_loss", 0)),
                "estimated_time": int(seg.get("estimated_time", 0)),
                "difficulty": int(seg.get("difficulty", 2)),
                "scheme_type": seg.get("scheme_type", "slope"),
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
