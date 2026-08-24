"""Deterministic quality assessment based only on observed workflow state."""

from typing import Any, Dict

from app.agents.base import BaseAgent


class QualityAgent(BaseAgent):
    """Score track, segmentation, POI, content, and execution completeness."""

    def __init__(self):
        super().__init__("quality_agent", "评估分析结果的质量")

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_start()
        assessment = self._assess(state)
        self.log_complete(quality_score=assessment["overall_score"])
        return {
            "quality_assessment": assessment,
            "current_step": "quality_assessment",
            "overall_progress": 85,
        }

    def _assess(self, state: Dict[str, Any]) -> Dict[str, Any]:
        points = state.get("track_points") or []
        stats = state.get("basic_stats") or {}
        schemes = state.get("segment_schemes") or []
        pois = state.get("poi_points") or []
        content = state.get("generated_content") or {}
        warnings = state.get("warnings") or []
        errors = state.get("errors") or []

        point_count = len(points)
        elevation_ratio = min(1.0, stats.get("valid_elevation_points", 0) / point_count) if point_count else 0.0
        timestamp_ratio = min(1.0, stats.get("valid_timestamp_points", 0) / point_count) if point_count else 0.0
        trajectory_score = 50.0 if point_count >= 2 else 0.0
        trajectory_score += 25.0 * elevation_ratio + 25.0 * timestamp_ratio

        segment_count = sum(len(scheme.get("segments", [])) for scheme in schemes)
        segmentation_score = 100.0 if segment_count else 0.0
        poi_score = 100.0 if pois else 0.0
        mode = content.get("generation_mode")
        content_score = 100.0 if mode == "llm" else 50.0 if mode == "fallback" else 0.0
        execution_score = max(0.0, 100.0 - 20.0 * len(errors) - 5.0 * len(warnings))

        scores = {
            "trajectory_quality": round(trajectory_score, 1),
            "segmentation_quality": segmentation_score,
            "poi_quality": poi_score,
            "content_quality": content_score,
            "execution_quality": execution_score,
        }
        overall = round(
            scores["trajectory_quality"] * 0.4
            + scores["segmentation_quality"] * 0.25
            + scores["poi_quality"] * 0.1
            + scores["content_quality"] * 0.15
            + scores["execution_quality"] * 0.1,
            1,
        )

        missing = []
        if not points:
            missing.append("track_points")
        if elevation_ratio < 1:
            missing.append("complete_elevation_data")
        if timestamp_ratio < 1:
            missing.append("complete_timestamp_data")
        if not segment_count:
            missing.append("segment_schemes")
        if not pois:
            missing.append("poi_points")
        if mode != "llm":
            missing.append("llm_generated_content")

        return {
            "overall_score": overall,
            "dimension_scores": scores,
            "missing_data": missing,
            "needs_human_review": bool(missing or warnings or errors),
            "warning_count": len(warnings),
            "error_count": len(errors),
        }
