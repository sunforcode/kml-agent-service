"""
POI 位置合并 AI 判定服务

库的定位是"经纬度 + 该位置的信息"。新 POI 是否与库内某个条目是同一位置，
不由代码写死距离阈值策略，而是：代码只召回候选（同批次内按距离粗筛），
最终是否合并/命中交给 LLM 判定。

约定：
- 召回半径 RECALL_RADIUS_METERS 只是性能边界（避免把整库塞给 LLM），不是合并策略；
- LLM 解析失败时确定性回退为"不命中"（宁可新增/保持草稿，也不错合并）。
"""

import json
import math
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.llm import get_llm

# 候选召回半径（米）：只影响送给 LLM 的候选范围，不决定合并结果
RECALL_RADIUS_METERS = 500.0

# 每批送 LLM 判定的 POI 数量
CHUNK_SIZE = 20

_SYSTEM_PROMPT = (
    "你是徒步路线 POI 位置判定助手。给你一个新路线上的 POI，以及全局 POI 库中"
    "距离它较近的候选条目（附距离，单位米）。请判断该 POI 与哪个候选是"
    "同一个地理位置（如同一座山峰、同一座寺庙、同一段补给点；环线起终点相距很近时也算同一位置）。\n"
    "判定依据优先级：坐标距离 > 海拔一致性 > 名称相似性（名称仅作参考，名称不同不代表不是同一位置）。\n"
    "严格只输出 JSON 数组，格式：\n"
    '[{"index": 0, "library_id": "pl_xxx", "reason": "距东台候选仅32米，海拔一致，判定为同一位置"}]\n'
    "若无合适候选则 library_id 填 null。index 必须与输入一一对应，不得增删。"
)


class _LLMResolveItem(BaseModel):
    """LLM 返回的单条判定结果"""

    index: int
    library_id: Optional[str] = None
    reason: str = ""


def haversine_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _recall_candidates(
    poi: dict, library: List[dict]
) -> List[dict]:
    """召回距离 RECALL_RADIUS_METERS 内的库条目（附带距离字段，可排除指定 id）"""
    exclude = poi.get("_exclude_id")
    candidates = []
    for item in library:
        if exclude and item.get("id") == exclude:
            continue
        dist = haversine_meters(
            poi["latitude"], poi["longitude"], item["latitude"], item["longitude"]
        )
        if dist <= RECALL_RADIUS_METERS:
            cand = dict(item)
            cand["_distance"] = round(dist, 1)
            candidates.append(cand)
    candidates.sort(key=lambda c: c["_distance"])
    return candidates[:5]  # 最多给 5 个最近的候选


def _parse_llm_json_array(text: str) -> List[_LLMResolveItem]:
    text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("LLM 输出中未找到 JSON 数组")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("LLM 输出不是数组")
    items = []
    for raw in data:
        if isinstance(raw, dict):
            try:
                items.append(_LLMResolveItem(**raw))
            except Exception:
                continue
    return items


async def _resolve_chunk(
    pois: List[dict], library: List[dict]
) -> Tuple[List[_LLMResolveItem], bool]:
    """判定一批 POI；失败时该批全部回退为不命中"""
    lines = []
    for i, poi in enumerate(pois):
        parts = [f"index={i}", f'name="{poi["name"]}"']
        if poi.get("elevation") is not None:
            parts.append(f"elevation={poi['elevation']}m")
        lines.append("- " + ", ".join(parts))

        cands = poi.get("_candidates") or []
        if cands:
            lines.append("  候选:")
            for c in cands:
                desc = f'    - id={c["id"]}, name="{c["name"]}", category={c.get("category", "")}, 距离={c["_distance"]}m'
                if c.get("elevation") is not None:
                    desc += f", elevation={c['elevation']}m"
                lines.append(desc)
        else:
            lines.append("  候选: 无")

    try:
        response = await get_llm().ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content="\n".join(lines))]
        )
        parsed = _parse_llm_json_array(response.content or "")
        if not parsed:
            raise ValueError("LLM 返回空数组")
        return parsed, False
    except Exception as exc:  # noqa: BLE001 - 失败确定性回退为不命中
        print(f"[poi_resolve] LLM 判定失败，本批 {len(pois)} 个回退为不命中: {exc}")
        return (
            [_LLMResolveItem(index=i, library_id=None, reason="LLM 判定失败，保守不合并") for i in range(len(pois))],
            True,
        )


async def resolve_poi_matches(
    pois: List[dict], library: List[dict]
) -> Tuple[List[dict], bool]:
    """
    判定每个 POI 是否与库内条目为同一位置。

    Args:
        pois: [{name, latitude, longitude, elevation?}, ...]
        library: [{id, name, latitude, longitude, category?, elevation?}, ...]

    Returns:
        (results, degraded)：results 与 pois 等长且顺序一致，
        每项 {index, library_id(可空), reason}
    """
    # 先召回候选
    enriched: List[dict] = []
    lib_ids = {item.get("id") for item in library}
    for i, poi in enumerate(pois):
        p = dict(poi)
        p["_index"] = i
        p["_exclude_id"] = poi.get("exclude_id")
        cands = _recall_candidates(p, library)
        # 过滤掉非法 id
        p["_candidates"] = [c for c in cands if c.get("id") in lib_ids]
        enriched.append(p)

    results: Dict[int, _LLMResolveItem] = {}
    degraded = False

    # 无候选的直接不命中（无需 LLM）；有候选的分批判定
    pending: List[dict] = []
    for p in enriched:
        if not p["_candidates"]:
            results[p["_index"]] = _LLMResolveItem(
                index=p["_index"], library_id=None, reason="附近无库内候选"
            )
        else:
            pending.append(p)

    for offset in range(0, len(pending), CHUNK_SIZE):
        chunk = pending[offset:offset + CHUNK_SIZE]
        judgements, chunk_degraded = await _resolve_chunk(chunk, library)
        degraded = degraded or chunk_degraded
        got = {j.index for j in judgements}
        for j in judgements:
            # index 是批内下标，映射回全局下标
            global_idx = chunk[j.index]["_index"] if j.index < len(chunk) else None
            if global_idx is None:
                continue
            # 校验 library_id 确实在该 POI 的候选里，防止 LLM 幻觉
            cand_ids = {c["id"] for c in chunk[j.index]["_candidates"]}
            if j.library_id and j.library_id not in cand_ids:
                j = _LLMResolveItem(
                    index=j.index, library_id=None, reason=f"LLM 返回的候选不在召回范围内，忽略"
                )
            results[global_idx] = j
        for k in range(len(chunk)):
            gi = chunk[k]["_index"]
            if gi not in results:
                results[gi] = _LLMResolveItem(
                    index=k, library_id=None, reason="LLM 未返回该项，保守不合并"
                )
                degraded = True

    out = []
    for p in enriched:
        j = results[p["_index"]]
        out.append({
            "index": p["_index"],
            "library_id": j.library_id,
            "reason": j.reason or "",
        })
    return out, degraded
