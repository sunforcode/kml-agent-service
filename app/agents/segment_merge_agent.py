"""
分段合并 Agent (Segment Merge Agent)

职责:
在 SegmentationAgent 完成坡度分段后，调用 LLM 判断哪些相邻连续段可以合并。
合并标准：从用户视角看，若相邻段类型相同、坡度走势相似、没有独立展示的必要，则合并为一段。
合并判断完全由 LLM 决策。

工作流程:
1. 读取 slope 方案分段列表
2. 构建 prompt（system 定义合并原则和输出格式，user 传入分段数据）
3. 调用 LLM，获取合并方案 JSON
4. 按方案执行合并并重新计算属性

输入:
- state["segment_schemes"]: 包含 slope 方案的分段列表

输出:
- state["segment_schemes"]: 合并后的 slope 方案（其他方案不变）
"""

import json
import logging
from typing import Dict, Any, List, Optional

from app.agents.base import BaseAgent
from app.core.config import settings

logger = logging.getLogger(__name__)


# 颜色映射（与 SegmentationAgent 保持一致）
SEGMENT_COLORS = {
    "climb": "#FF5722",
    "descent": "#2196F3",
    "flat": "#4CAF50",
    "mixed": "#9C27B0",
}


def _get_color(segment_type: str) -> str:
    return SEGMENT_COLORS.get(segment_type, "#9C27B0")


def _estimate_difficulty(avg_slope: float, gain_m: float, distance_km: float, segment_type: str) -> int:
    """估算难度（1-5），与 SegmentationAgent 逻辑一致"""
    effective_slope = abs(avg_slope)
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

    if distance_km > 10:
        base = min(5, base + 1)
    if segment_type == "climb" and gain_m > 800:
        base = min(5, base + 1)
    if segment_type == "descent" and distance_km > 8:
        base = min(5, base + 1)

    return int(round(base))


def _estimate_time_naismith(distance_km: float, gain_m: float, loss_m: float, difficulty: int) -> int:
    """Naismith 法则估算时间（分钟），与 SegmentationAgent 逻辑一致"""
    base_speed = {1: 5.0, 2: 4.0, 3: 3.2, 4: 2.5, 5: 2.0}.get(difficulty, 3.5)
    time_min = (distance_km / base_speed) * 60

    climb_rate = {1: 8, 2: 9, 3: 10, 4: 11, 5: 12}.get(difficulty, 10)
    time_min += (gain_m / 100.0) * climb_rate

    descent_rate = {1: 3, 2: 3, 3: 5, 4: 7, 5: 10}.get(difficulty, 5)
    time_min += (loss_m / 100.0) * descent_rate

    return max(1, int(round(time_min)))


def _suggest_name(index: int, total: int, segment_type: str, distance_km: float,
                  gain_m: float, loss_m: float) -> str:
    """生成路段名称"""
    type_names = {
        "climb": "爬升段",
        "descent": "下降段",
        "flat": "平路段",
        "mixed": "混合段",
    }
    type_cn = type_names.get(segment_type, "路段")
    ordinal = index + 1

    if gain_m > 0:
        detail = f"(+{int(gain_m)}m)"
    elif loss_m > 0:
        detail = f"(-{int(loss_m)}m)"
    else:
        detail = ""

    if ordinal == 1:
        prefix = "起点-"
    elif ordinal == total:
        prefix = "终点-"
    else:
        prefix = f"第{ordinal}段-"

    name = f"{prefix}{type_cn}{detail}"
    if len(name) > 25:
        name = f"第{ordinal}段{type_cn}"
    return name


def _merge_group(segs: List[Dict[str, Any]], new_index: int, new_total: int) -> Dict[str, Any]:
    """
    将一组段合并为一个新段，重新计算所有属性。
    """
    if not segs:
        raise ValueError("合并段列表不能为空")
    if len(segs) == 1:
        s = dict(segs[0])
        s["id"] = f"seg_{new_index + 1:03d}"
        s["sequence_number"] = new_index + 1
        s["name"] = _suggest_name(
            new_index, new_total,
            s.get("segment_type", "mixed"),
            s.get("distance", 0),
            s.get("elevation_gain", 0),
            s.get("elevation_loss", 0),
        )
        return s

    total_distance = sum(s.get("distance", 0) for s in segs)
    total_gain = sum(s.get("elevation_gain", 0) for s in segs)
    total_loss = sum(s.get("elevation_loss", 0) for s in segs)

    track_start = min(s.get("track_start_index", 0) for s in segs)
    track_end = max(s.get("track_end_index", 0) for s in segs)

    start_point = segs[0].get("start_point")
    end_point = segs[-1].get("end_point")

    types = set(s.get("segment_type", "mixed") for s in segs)
    merged_type = list(types)[0] if len(types) == 1 else "mixed"

    if total_distance > 0:
        weighted_slope = sum(
            s.get("avg_slope_degrees", 0) * s.get("distance", 0)
            for s in segs
        ) / total_distance
    else:
        weighted_slope = segs[0].get("avg_slope_degrees", 0)

    max_slope = max(abs(s.get("max_slope_degrees", 0)) for s in segs)
    if any(s.get("avg_slope_degrees", 0) < 0 for s in segs) and weighted_slope < 0:
        max_slope = -max_slope

    difficulty = _estimate_difficulty(weighted_slope, total_gain, total_distance, merged_type)
    estimated_time = _estimate_time_naismith(total_distance, total_gain, total_loss, difficulty)

    min_conf = min(s.get("confidence", 0.8) for s in segs)
    confidence = round(max(0.5, min_conf - 0.05), 2)

    name = _suggest_name(new_index, new_total, merged_type, total_distance, total_gain, total_loss)
    color = _get_color(merged_type)

    return {
        "id": f"seg_{new_index + 1:03d}",
        "name": name,
        "sequence_number": new_index + 1,
        "color": color,
        "distance": round(total_distance, 2),
        "elevation_gain": round(total_gain, 1),
        "elevation_loss": round(total_loss, 1),
        "estimated_time": estimated_time,
        "difficulty": difficulty,
        "track_start_index": track_start,
        "track_end_index": track_end,
        "start_point": start_point,
        "end_point": end_point,
        "segment_type": merged_type,
        "slope_direction": merged_type,
        "avg_slope_degrees": round(weighted_slope, 2),
        "max_slope_degrees": round(max_slope, 2),
        "confidence": confidence,
        "description": None,
        "notes": None,
    }


class SegmentMergeAgent(BaseAgent):
    """
    分段合并 Agent

    在 SegmentationAgent 完成分段后，调用 LLM 判断哪些相邻连续段可以合并，
    使最终 slope 方案的分段粒度对用户有意义。
    """

    def __init__(self):
        super().__init__(
            name="segment_merge_agent",
            description="AI 评估并合并相邻的可合并分段"
        )
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未配置")

        from langchain_openai import ChatOpenAI
        kwargs: Dict[str, Any] = {
            "model": settings.openai_model,
            "temperature": 0,
            "max_tokens": 256,
            "api_key": api_key,
        }
        base_url = settings.openai_base_url
        if base_url:
            kwargs["base_url"] = base_url
        self._llm = ChatOpenAI(**kwargs)
        return self._llm

    # =========================================================================
    # 主入口
    # =========================================================================

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.log_start()

        try:
            segment_schemes = state.get("segment_schemes", [])
            if not segment_schemes:
                logger.info("SegmentMergeAgent: 无分段方案，跳过")
                return {"current_step": "segment_merge", "overall_progress": 35}

            slope_idx, slope_scheme = next(
                ((i, s) for i, s in enumerate(segment_schemes) if s.get("scheme_type") == "slope"),
                (None, None),
            )

            if slope_scheme is None:
                logger.info("SegmentMergeAgent: 未找到 slope 方案，跳过")
                return {"current_step": "segment_merge", "overall_progress": 35}

            segments = slope_scheme.get("segments", [])
            if len(segments) <= 1:
                logger.info("SegmentMergeAgent: slope 方案段数 ≤ 1，无需合并")
                return {"current_step": "segment_merge", "overall_progress": 35}

            logger.info(f"SegmentMergeAgent: slope 方案共 {len(segments)} 段，当前跳过 LLM 合并，透传原始分段")

            self.log_complete(
                original_count=len(segments),
                merged_count=len(segments),
            )
            return {
                "current_step": "segment_merge",
                "overall_progress": 35,
            }

        except Exception as e:
            self.log_error(e)
            return {
                "current_step": "segment_merge",
                "overall_progress": 35,
                "warnings": [{
                    "level": "warning",
                    "message": f"SegmentMergeAgent 执行失败，分段保持原样: {str(e)}",
                }],
            }

    # =========================================================================
    # LLM 合并
    # =========================================================================

    async def _llm_merge(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """调用 LLM 获取合并方案并执行"""
        try:
            llm = self._get_llm()
        except RuntimeError as e:
            logger.info(f"SegmentMergeAgent: {e}，跳过合并")
            return segments

        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = (
            "你是一个徒步路线分段优化专家。给定一条路线的坡度分段列表（含序号/类型/距离/爬升/下降/平均坡度），"
            "目标是合并后让每段都具有清晰的地形含义，最终段数控制在 6~14 段之间。\n\n"
            "【必须合并的情况（优先级高）】\n"
            "1. 相邻段类型相同（同为 climb、同为 descent、同为 flat），无论距离长短，都应合并为一段。\n"
            "2. 短距离噪点段（dist_km < 1.5）且坡度变化不显著（avg_slope 绝对值 < 5°），"
            "若被夹在两个同方向的主段之间，必须吸收进主段（climb→小混合→climb 视为一段 climb）。\n"
            "3. 方向相同的段被一个极短过渡段（dist_km < 1.0）隔开，整体视为同一趋势，应合并。\n\n"
            "【可以合并的情况】\n"
            "4. climb 和 descent 交替出现，但其中某段极短（dist_km < 1.0）且爬升/下降量不大（< 80m），"
            "说明只是小路口或短暂起伏，可并入相邻更长的同向段。\n\n"
            "【必须保留、不得合并的情况】\n"
            "5. 方向明确反转（爬升→下降 或 下降→爬升），且两段距离都 ≥ 1.5km、爬升/下降量都 ≥ 100m，"
            "说明经过了真正的山顶或垭口，必须独立保留。\n"
            "6. 独立的平路段：前后分别是爬升段和下降段（或方向相反），且该平路段 dist_km ≥ 1.5km，"
            "必须独立保留，不得并入任何一侧。\n"
            "7. 长段（dist_km ≥ 5km）且爬升/下降量 ≥ 400m 的段，本身已足够有意义，不得拆分或过度合并。\n\n"
            "严格按如下 JSON 格式输出，不要包含任何其他文字：\n"
            '{"merge_groups": [[seq1, seq2], [seq3, seq4, seq5]]}\n\n'
            "说明：\n"
            "- merge_groups 中每个子数组为需合并的连续段序号（1-based），长度 ≥ 2\n"
            "- 未出现在 merge_groups 中的段保持不变\n"
            '- 若无需合并，返回 {"merge_groups": []}'
        )

        seg_data = [
            {
                "seq": s.get("sequence_number"),
                "type": s.get("segment_type"),
                "dist_km": s.get("distance"),
                "gain_m": s.get("elevation_gain"),
                "loss_m": s.get("elevation_loss"),
                "avg_slope": s.get("avg_slope_degrees"),
            }
            for s in segments
        ]

        logger.info(f"SegmentMergeAgent: 合并前原始分段共 {len(seg_data)} 段")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(seg_data, ensure_ascii=False)),
        ]

        try:
            response = await llm.ainvoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            merge_groups = self._parse_response(raw, total_segs=len(segments))
        except Exception as e:
            logger.warning(f"SegmentMergeAgent: LLM 调用失败，保留原始分段: {e}")
            return segments

        if merge_groups is None:
            logger.warning(f"SegmentMergeAgent: LLM 输出解析失败，保留原始分段。原始输出: {raw[:200]}")
            return segments

        if not merge_groups:
            logger.info("SegmentMergeAgent: LLM 判断无需合并")
            return segments

        logger.info(f"SegmentMergeAgent: LLM 返回 {len(merge_groups)} 组合并方案")
        return self._apply_merge(segments, merge_groups)

    def _parse_response(self, raw: str, total_segs: int) -> Optional[List[List[int]]]:
        """解析 LLM 输出，返回 merge_groups；解析失败返回 None"""
        try:
            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return None
            data = json.loads(raw[start:end])

            merge_groups_raw = data.get("merge_groups")
            if not isinstance(merge_groups_raw, list):
                return None
            if len(merge_groups_raw) == 0:
                return []

            validated = []
            seen: set = set()
            for group in merge_groups_raw:
                if not isinstance(group, list) or len(group) < 2:
                    continue
                seqs = sorted(int(x) for x in group)
                # 序号范围检查
                if any(s < 1 or s > total_segs for s in seqs):
                    continue
                # 连续性检查
                if not all(seqs[i + 1] == seqs[i] + 1 for i in range(len(seqs) - 1)):
                    continue
                # 重叠检查
                if any(s in seen for s in seqs):
                    continue
                seen.update(seqs)
                validated.append(seqs)

            return validated

        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def _apply_merge(
        self, segments: List[Dict[str, Any]], merge_groups: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """按合并方案执行合并，返回新的段列表"""
        seq_to_group: Dict[int, int] = {}
        for gid, group in enumerate(merge_groups):
            for seq in group:
                seq_to_group[seq] = gid

        slots: List[List[Dict[str, Any]]] = []
        processed: set = set()

        for seg in segments:
            seq = seg.get("sequence_number", 0)
            if seq in seq_to_group:
                gid = seq_to_group[seq]
                if gid not in processed:
                    group_segs = [
                        s for s in segments
                        if s.get("sequence_number") in merge_groups[gid]
                    ]
                    slots.append(group_segs)
                    processed.add(gid)
            else:
                slots.append([seg])

        total = len(slots)
        result = [_merge_group(slot, new_index=i, new_total=total) for i, slot in enumerate(slots)]

        logger.info(f"SegmentMergeAgent: 合并完成，{len(segments)} 段 → {len(result)} 段")
        return result
