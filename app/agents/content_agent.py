"""Generate route content from the configured LLM with a factual fallback."""

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.base import BaseAgent
from app.core.config import settings
from app.core.llm import get_llm

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = (
    "description",
    "highlights",
    "difficulties",
    "safety_notes",
    "equipment_recommendations",
)


class ContentAgent(BaseAgent):
    """Generate structured narrative content without inventing fallback facts."""

    def __init__(self):
        super().__init__("content_agent", "使用LLM生成自然语言内容")

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_start()
        if not state.get("request", {}).get("enable_content_generation", True):
            return {
                "generated_content": None,
                "current_step": "content_generation",
                "overall_progress": 65,
            }

        try:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY 未配置")
            response = await get_llm().ainvoke(self._messages(state))
            raw = response.content if hasattr(response, "content") else str(response)
            content = self._parse_content(raw)
            content["generation_mode"] = "llm"
            self.log_complete()
            return {
                "generated_content": content,
                "current_step": "content_generation",
                "overall_progress": 65,
            }
        except Exception as exc:
            logger.warning("内容生成降级为确定性模板: %s", exc)
            return {
                "generated_content": self._fallback_content(state),
                "current_step": "content_generation",
                "overall_progress": 65,
                "warnings": [{
                    "level": "warning",
                    "message": f"内容生成降级为确定性模板: {exc}",
                }],
            }

    def _messages(self, state: Dict[str, Any]) -> List[Any]:
        facts = {
            "basic_stats": state.get("basic_stats") or {},
            "segment_schemes": state.get("segment_schemes") or [],
            "poi_points": state.get("poi_points") or [],
        }
        system = (
            "你是徒步路线内容编辑。只能使用输入 JSON 中的事实，不得推断地名、天气、"
            "水源可靠性或危险点。只输出 JSON，且必须包含 description、highlights、"
            "difficulties、safety_notes、equipment_recommendations；后四项均为字符串数组。"
        )
        return [
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(facts, ensure_ascii=False)),
        ]

    def _parse_content(self, raw: str) -> Dict[str, Any]:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start < 0 or end <= 0:
            raise ValueError("模型响应不是 JSON")
        data = json.loads(raw[start:end])
        if not isinstance(data, dict) or any(key not in data for key in _REQUIRED_FIELDS):
            raise ValueError("模型响应缺少必需字段")
        if not isinstance(data["description"], str):
            raise ValueError("description 必须是字符串")
        for key in _REQUIRED_FIELDS[1:]:
            if not isinstance(data[key], list) or not all(isinstance(item, str) for item in data[key]):
                raise ValueError(f"{key} 必须是字符串数组")
        return {key: data[key] for key in _REQUIRED_FIELDS}

    def _fallback_content(self, state: Dict[str, Any]) -> Dict[str, Any]:
        stats = state.get("basic_stats") or {}
        schemes = state.get("segment_schemes") or []
        segment_count = sum(len(scheme.get("segments", [])) for scheme in schemes)
        poi_count = len(state.get("poi_points") or [])
        distance = stats.get("total_distance_km", 0)
        gain = stats.get("total_gain_m", 0)
        loss = stats.get("total_loss_m", 0)
        return {
            "description": (
                f"该轨迹总距离 {distance} 公里，累计爬升 {gain} 米，累计下降 {loss} 米；"
                f"当前包含 {segment_count} 个路段和 {poi_count} 个 POI。"
            ),
            "highlights": [],
            "difficulties": [],
            "safety_notes": ["内容仅基于轨迹统计，请结合现场情况判断。"],
            "equipment_recommendations": [],
            "generation_mode": "fallback",
        }
