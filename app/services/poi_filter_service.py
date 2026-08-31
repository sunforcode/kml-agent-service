"""
POI LLM 筛选服务

对路线解析出的 POI 列表做一次 LLM 质量筛选：
- 判断每个 POI 是否与徒步路线相关（保留/剔除）
- 规范化类别（限定在系统支持的类别枚举内）
- 给出判断理由，供人工确认时参考

约定：LLM 只做筛选判断，不虚构位置与内容；
任一分块解析失败时确定性回退为"保留原样"，不丢数据。
"""

import json
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.llm import get_llm
from app.models.request import PoiFilterItem

# 与 walkbg poi_points.category 保持一致
VALID_CATEGORIES = {
    "water", "camp", "supply", "photo", "pass",
    "valley", "weather", "danger", "start", "end",
}

# 每次送入 LLM 的 POI 数量（防止超上下文/超时）
CHUNK_SIZE = 30

_SYSTEM_PROMPT = (
    "你是徒步路线 POI 质量审核助手。给你一组从徒步路线 KML 轨迹文件中提取的 POI（兴趣点），"
    "请逐个判断：\n"
    "1. action：该点是否值得保留——keep 表示对徒步者有实际意义（如山峰、寺庙、补给点、水源、"
    "垭口、营地、观景点、起点终点等）；reject 表示无意义或误导（如重复点、公路桩号、"
    "无名的道路交叉口、与徒步无关的设施等）。\n"
    "2. category：规范化类别，只能取以下之一："
    "water(水源)/camp(营地)/supply(补给点)/photo(拍照打卡点)/pass(垭口)/"
    "valley(河谷)/weather(气象点)/danger(危险点)/start(起点)/end(终点)。\n"
    "3. reason：一句话中文理由。\n"
    "严格只输出 JSON 数组，不要输出任何其他文字。数组元素格式：\n"
    '[{"index": 0, "action": "keep", "category": "pass", "reason": "东台望海峰，五台之一"}]\n'
    "index 必须与输入列表的下标一一对应，不得增删。"
)


class _LLMFilterItem(BaseModel):
    """LLM 返回的单条筛选结果（宽松解析用）"""

    index: int
    action: str = "keep"
    category: str = ""
    reason: str = ""


def _parse_llm_json_array(text: str) -> List[_LLMFilterItem]:
    """从 LLM 输出中提取 JSON 数组并解析；失败抛 ValueError"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("LLM 输出中未找到 JSON 数组")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("LLM 输出不是数组")
    items = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        try:
            items.append(_LLMFilterItem(**raw))
        except Exception:
            continue
    return items


def _build_chunk_prompt(chunk: List[PoiFilterItem], offset: int) -> HumanMessage:
    lines = []
    for i, poi in enumerate(chunk):
        parts = [f"index={offset + i}", f'name="{poi.name}"']
        if poi.category:
            parts.append(f"category={poi.category}")
        if poi.elevation is not None:
            parts.append(f"elevation={poi.elevation}m")
        if poi.description:
            desc = poi.description[:80]
            parts.append(f'description="{desc}"')
        lines.append("- " + ", ".join(parts))
    return HumanMessage(content="\n".join(lines))


async def _filter_chunk(chunk: List[PoiFilterItem], offset: int) -> List[_LLMFilterItem]:
    """筛选一个分块；失败时整块回退为保留"""
    try:
        response = await get_llm().ainvoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                _build_chunk_prompt(chunk, offset),
            ]
        )
        parsed = _parse_llm_json_array(response.content or "")
        if not parsed:
            raise ValueError("LLM 返回空数组")
        return parsed
    except Exception as exc:  # noqa: BLE001 - 解析/网络失败统一确定性回退
        print(f"[poi_filter] 分块 {offset}~{offset + len(chunk) - 1} LLM 筛选失败，回退为保留: {exc}")
        return [
            _LLMFilterItem(index=offset + i, action="keep", category="", reason="LLM 筛选失败，默认保留")
            for i in range(len(chunk))
        ]


async def filter_pois_by_llm(pois: List[PoiFilterItem]) -> tuple[List[dict], bool]:
    """
    对 POI 列表做 LLM 筛选。

    Returns:
        (results, degraded)
        results: 与输入等长且顺序一致，每项为
            {index, action, category, reason}
        degraded: 任一分块发生回退时为 True
    """
    all_judgements: List[_LLMFilterItem] = []
    degraded = False

    for offset in range(0, len(pois), CHUNK_SIZE):
        chunk = pois[offset:offset + CHUNK_SIZE]
        judgements = await _filter_chunk(chunk, offset)
        got_indices = {j.index for j in judgements}
        expected = set(range(offset, offset + len(chunk)))
        if not expected.issubset(got_indices):
            # 缺失的下标补默认保留
            for i in sorted(expected - got_indices):
                judgements.append(
                    _LLMFilterItem(index=i, action="keep", category="", reason="LLM 未返回该项，默认保留")
                )
            degraded = True
        all_judgements.extend(judgements)

    by_index = {j.index: j for j in all_judgements}

    results: List[dict] = []
    for i, poi in enumerate(pois):
        j = by_index.get(i)
        if j is None:
            action, category, reason = "keep", poi.category, "无判断结果，默认保留"
            degraded = True
        else:
            action = "keep" if j.action == "keep" else "reject"
            reason = j.reason or ""
            # 类别规范化：非法类别回退原类别
            category = j.category if j.category in VALID_CATEGORIES else poi.category
        results.append({
            "index": i,
            "action": action,
            "category": category,
            "reason": reason,
        })

    return results, degraded
